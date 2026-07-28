/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"
#include <libxs/libxs_math.h>

/* Embedded kernel source (generated at build time via tool_opencl.sh) */
#include "ozaki_kernels.h"

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_INT8)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_OZAKI2_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI2_INT8)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_GEMM3M)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_GEMM3M)"
#endif

/**
 * Compute units per work-group required to consider the device saturated during
 * tile selection, i.e. the divisor in nwg_min = nunits / OZAKI_TILE_SAT.
 * Measured on PVC: below this the largest tile starves the device, above it the
 * extra parallelism no longer pays for the lost reuse.
 *
 * A divisor only transfers across devices that report comparable compute-unit
 * granularity. PVC exposes many lean units (~448), so nunits/16 still demands
 * ~28 work-groups; an H100 exposes 114 fat SMs and the same divisor accepts 7,
 * i.e. 0.06 work-groups per SM -- the heuristic then picks the largest tile for
 * its reuse and leaves most of the device idle. NVIDIA therefore uses a floor of
 * one work-group per SM (OZAKI_TILE_SAT_NV), which measured 13-81% faster than
 * the tile the divisor selected; PVC keeps the divisor it was tuned with.
 */
#if !defined(OZAKI_TILE_SAT)
# define OZAKI_TILE_SAT 16
#endif
#if !defined(OZAKI_TILE_SAT_NV)
# define OZAKI_TILE_SAT_NV 1
#endif
/**
 * Sub-groups (warps) per work-group that NVIDIA tile selection aims for. At a
 * fixed tile area on H100/n=4096 the measurement is monotone in work-group
 * size: 4 warps 6158-6529 GFLOPS, 8 warps 6145-6275, 16 warps 5103. A fat
 * work-group holds its SM for its whole lifetime, so smaller ones give the
 * scheduler more independent blocks to overlap.
 */
#if !defined(OZAKI_WGS_MAX_NV)
# define OZAKI_WGS_MAX_NV 4
#endif


/* Internal helpers */
static const uint16_t ozaki_u8_moduli[] = {211, 199, 163, 256, 251, 223, 197, 167, 243, 227, 193, 169, 241, 229, 191, 173, 239, 233, 181, 179};
static const uint16_t ozaki_i8_moduli[] = {101, 97, 59, 128, 127, 103, 89, 61, 125, 107, 83, 67, 121, 109, 81, 71, 119, 113, 79, 73};

static void ozaki_print_opt(FILE* stream, const char* name, int val)
{
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}


/**
 * Emit fractional-CRT (OZAKI_FRACCRT) reconstruction tables as -D flags for the
 * active moduli set (nprimes entries of modtab). Computes, without bignum:
 *   k_i     = (prod_{j!=i} m_j mod m_i)^{-1} mod m_i
 *   climb[i][l] = l-th base-256 limb of 1/m_i
 *   M       = prod m_i  as a double-double (Mh, Ml) via compensated product
 * L (limb count) is fixed at OZ2G_FRAC_L; double-double reconstruction is exact
 * whenever |x| > M * 2^-53, which holds for the real Ozaki-2 magnitude range.
 */
static size_t ozaki_emit_fraccrt(char* buf, size_t size, const uint16_t* modtab, int nprimes, int frac_l)
{
  size_t off = 0;
  double mh = 1.0, ml = 0.0;
  int i, l;
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, " -DOZ2G_FRAC_L=%d -DOZ2G_FRAC_K={", frac_l);
  for (i = 0; i < nprimes; ++i) {
    uint32_t prod_mod = 1;
    int j;
    for (j = 0; j < nprimes; ++j) {
      if (j != i) prod_mod = (uint32_t)(((uint64_t)prod_mod * (uint32_t)modtab[j]) % (uint32_t)modtab[i]);
    }
    off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "%s%u", (0 != i) ? "," : "",
      (unsigned)libxs_mod_inverse_u32(prod_mod, (uint32_t)modtab[i]));
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "} -DOZ2G_FRAC_CLIMB={");
  for (i = 0; i < nprimes; ++i) {
    uint32_t rem = 1;
    for (l = 0; l < frac_l; ++l) {
      rem <<= 8;
      off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "%s%u",
        (0 != i || 0 != l) ? "," : "", (unsigned)(rem / (uint32_t)modtab[i]));
      rem %= (uint32_t)modtab[i];
    }
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "}");
  for (i = 0; i < nprimes; ++i) {
    const double p = (double)modtab[i];
    double perr;
    const double ph = libxs_two_product(mh, p, &perr);
    const double e = perr + ml * p;
    double serr;
    mh = libxs_two_sum(ph, e, &serr);
    ml = serr;
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off,
    " -DOZ2G_FRAC_MH=%.20e -DOZ2G_FRAC_ML=%.20e", mh, ml);
  return off;
}


/**
 * Emit leaf fractional-CRT tables (OZAKI_FRACCRT=2): per-group fractional
 * reconstruction feeding the exact hierarchical level-2 combine. Reuses the
 * shared per-prime limb table OZ2G_FRAC_CLIMB and emits, per prime, the
 * group-relative inverse OZ2G_FRAC_KG_i = (M_g/m_i)^{-1} mod m_i (M_g the
 * product of that prime's group), plus each group product M_g as a
 * double-double (OZ2G_FRAC_GMH/GML). Each group value V_g = x mod M_g is a
 * non-negative integer below M_g < 2^53, so leaf reconstruction is exact for
 * all group values -- the hierarchy keeps exactness across the full range.
 */
static size_t ozaki_emit_fraccrt2(char* buf, size_t size, const uint16_t* modtab, int nprimes, int frac_l, int hier_gs)
{
  const int ngroups = (nprimes + hier_gs - 1) / hier_gs;
  size_t off = 0;
  int i, l, g;
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, " -DOZ2G_FRAC_L=%d -DOZ2G_FRAC_CLIMB={", frac_l);
  for (i = 0; i < nprimes; ++i) {
    uint32_t rem = 1;
    for (l = 0; l < frac_l; ++l) {
      rem <<= 8;
      off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "%s%u",
        (0 != i || 0 != l) ? "," : "", (unsigned)(rem / (uint32_t)modtab[i]));
      rem %= (uint32_t)modtab[i];
    }
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "} -DOZ2G_FRAC_KG={");
  for (g = 0; g < ngroups; ++g) {
    const int lo = g * hier_gs;
    const int hi = (lo + hier_gs <= nprimes) ? (lo + hier_gs) : nprimes;
    for (i = lo; i < hi; ++i) {
      uint32_t prod_mod = 1;
      int j;
      for (j = lo; j < hi; ++j) {
        if (j != i) prod_mod = (uint32_t)(((uint64_t)prod_mod * (uint32_t)modtab[j]) % (uint32_t)modtab[i]);
      }
      off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "%s%u", (0 != i) ? "," : "",
        (unsigned)libxs_mod_inverse_u32(prod_mod, (uint32_t)modtab[i]));
    }
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "} -DOZ2G_FRAC_GMH={");
  for (g = 0; g < ngroups; ++g) {
    const int lo = g * hier_gs;
    const int hi = (lo + hier_gs <= nprimes) ? (lo + hier_gs) : nprimes;
    double mh = 1.0;
    for (i = lo; i < hi; ++i) mh *= (double)modtab[i];
    off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "%s%.20e", (0 != g) ? "," : "", mh);
  }
  off += (size_t)LIBXS_SNPRINTF(buf + off, size - off, "}");
  return off;
}


ozaki_tile_t ozaki_tile_select(const ozaki_context_t* ctx, int M, int N, int rtm, int rtn)
{
  const int xmx_m = OZAKI_XMX_M(ctx);
  const int xmx_n = OZAKI_XMX_N(ctx);
  const int gm = xmx_m * LIBXS_MAX(rtm, 1); /* tm granularity */
  const int gn = xmx_n * LIBXS_MAX(rtn, 1); /* tn granularity */
  /**
   * Total work-items are invariant to the tile size: every sub-group computes
   * gm x gn outputs regardless. The tile only controls (a) how many
   * work-groups the problem splits into and (b) how far M/N are padded up to
   * a tile multiple. Large tiles win on operand reuse but produce too few
   * work-groups to fill the device at small M/N, and waste work on padding
   * when M/N are not tile multiples. Candidates must therefore fit the
   * work-group size bound and still saturate the compute units; among those,
   * the best reuse per unit of padded work wins.
   */
  const int nwg_min = (0 < ctx->nunits) ? (ctx->nunits / ctx->tile_sat) : 32;
  /**
   * Work-group size ceiling, distinct from the hardware bound in max_wgs.
   * Occupancy is per work-group: a fat work-group occupies an SM/slice for its
   * whole life, so fewer, larger ones leave less for the scheduler to overlap.
   * Measured on H100 at n=4096, holding the tile area fixed: WGS=128 reached
   * 6158-6529 GFLOPS, WGS=256 6145-6275, WGS=512 only 5103 -- monotone in WGS,
   * independent of aspect ratio. The reuse objective below cannot see this (it
   * scores tile shape, not residency), hence an explicit cap.
   */
  const size_t wgs_max = (0 < ctx->wgs_max) ? (size_t)ctx->wgs_max : ctx->max_wgs;
  ozaki_tile_t tile;
  tile.m = gm;
  tile.n = gn;
  if (0 != ctx->tm_req && 0 != ctx->tn_req) {
    /**
     * Round to the sub-tile granularity: NTM/NTN truncate, so a non-multiple
     * would leave the tail of the tile uncovered and part of C unwritten.
     */
    tile.m = (ctx->tm / gm) * gm;
    tile.n = (ctx->tn / gn) * gn;
    if (tile.m < gm) tile.m = gm;
    if (tile.n < gn) tile.n = gn;
  }
  else {
    double best_score = -1.0;
    int cm;
    for (cm = gm; cm <= ctx->tm; cm += gm) {
      int cn;
      for (cn = gn; cn <= ctx->tn; cn += gn) {
        const size_t wgs = (size_t)ctx->sg * ((size_t)(cm / gm) * (cn / gn));
        const int nwg = LIBXS_UPDIV(M, cm) * LIBXS_UPDIV(N, cn);
        if (wgs <= ctx->max_wgs && wgs <= wgs_max && nwg >= nwg_min) {
          /**
           * A tile performs cm*cn*K MACs while loading (cm+cn)*K operand
           * elements, so cm*cn/(cm+cn) is its arithmetic intensity -- maximal
           * for square tiles, which is why an area-only objective would pick
           * degenerate aspect ratios. Divided by the padded work, this favors
           * the largest tile that both stays square and divides M and N.
           */
          const double padded = (double)LIBXS_UP(M, cm) * LIBXS_UP(N, cn);
          const double score = ((double)cm * cn / (cm + cn)) / padded;
          if (score > best_score) {
            tile.m = cm;
            tile.n = cn;
            best_score = score;
          }
        }
      }
    }
  }
  return tile;
}


int ozaki_init(ozaki_context_t* ctx, int tm, int tn, int use_double, int kind, int verbosity, int ndecomp, int ozflags, int oztrim,
  int ozgroups, int maxk, int profiling)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const int gpu = (CL_DEVICE_TYPE_GPU == devinfo->type ? 1 : 0);
  int result = EXIT_SUCCESS;
  int nv, has_fp64;
  int wg, sg, use_i8;
  int nslices, nprimes, oztrim_crt;
  const char* env;
  memset(ctx, 0, sizeof(*ctx));

  if (0 >= kind) kind = 2;

  /* CRT (kind=2): no triangular/symmetrize (no cross-prime products) */
  if (2 == kind) {
    if (0 > ozflags) ozflags = 0; /* CRT does not use triangular/symmetrize */
  }

  if (0 > verbosity || 2 < verbosity) {
    char name[256] = "";
    libxstream_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /**
   * FP64 is needed for the fp64 interface, and independently by Scheme 2 even
   * in fp32: Garner/Horner and the fractional reconstruction accumulate CRT
   * values far beyond 2^24, so their intermediates cannot be demoted to float
   * without losing the exactness the scheme exists to provide. Scheme 1 uses
   * no double at all, hence a device without cl_khr_fp64 can still run fp32
   * Scheme 1 -- kind 2 (and the adaptive kind 3, which may select it) cannot.
   */
  { const char *const fp64_ext[] = {"cl_khr_fp64"};
    has_fp64 = (EXIT_SUCCESS == libxstream_opencl_device_ext(device, fp64_ext, 1));
    if (0 == has_fp64) {
      if (0 != use_double) {
        fprintf(stderr, "ERROR OZAKI: FP64 requested but device does not support cl_khr_fp64\n");
        result = EXIT_FAILURE;
      }
      else if (2 == kind) {
        fprintf(stderr, "ERROR OZAKI: Scheme 2 needs cl_khr_fp64 for CRT reconstruction"
                        " (device lacks it); use OZAKI=1\n");
        result = EXIT_FAILURE;
      }
      else if (3 == kind && (0 > verbosity || 0 < verbosity)) {
        fprintf(stderr, "INFO OZAKI: no cl_khr_fp64, adaptive selection restricted to Scheme 1\n");
      }
    }
  }

  /**
   * Scheme 2 signed i8 fallback: OZAKI_I8=1 uses moduli<=128 (legacy).
   * Default (u8): moduli<=256, fewer primes for same cumulative product.
   */
  {
    const char *const env_i8 = getenv("OZAKI_I8");
    use_i8 = (NULL != env_i8 && 0 != atoi(env_i8));
  }
  /**
   * Compute nslices (Scheme 1) and nprimes (Scheme 2) independently.
   * Both are needed for adaptive scheme selection.
   */
  {
    const int u8_def = use_double ? 16 : 9;
    const int i8_def = use_double ? 19 : 10;
    nslices = use_double ? 8 : 4;
    nprimes = (0 != use_i8) ? i8_def : u8_def;
    if (0 < ndecomp) {
      if (1 == kind) nslices = ndecomp;
      else if (ndecomp != u8_def && ndecomp != i8_def) nprimes = ndecomp;
    }
    { /* Scheme 2: Convert trim levels to input mantissa bits. */
      const int mant = use_double ? 52 : 23;
      const int max_levels = mant / 2;
      oztrim_crt = (3 == kind) ? 0 : LIBXS_MIN(oztrim, max_levels) * 2;
      if (0 < oztrim_crt) {
        static const int cumbits_u8[20] = {7, 15, 22, 30, 38, 46, 54, 61, 69, 77, 84, 92, 100, 107, 115, 122, 130, 138, 146, 153};
        static const int cumbits_i8[20] = {6, 13, 19, 26, 33, 39, 46, 52, 59, 65, 72, 78, 85, 92, 98, 104, 111, 118, 124, 130};
        const int* cumbits = (0 != use_i8) ? cumbits_i8 : cumbits_u8;
        const int req = 2 * (mant - oztrim_crt) + 23;
        int np;
        for (np = 0; np < 20 && cumbits[np] < req; ++np);
        nprimes = (np < 20) ? np + 1 : 20;
      }
    }
    if (0 < maxk && 2 == kind) {
      static const int cumbits_u8[20] = {7, 15, 22, 30, 38, 46, 54, 61, 69, 77, 84, 92, 100, 107, 115, 122, 130, 138, 146, 153};
      static const int cumbits_i8[20] = {6, 13, 19, 26, 33, 39, 46, 52, 59, 65, 72, 78, 85, 92, 98, 104, 111, 118, 124, 130};
      const int* cumbits = (0 != use_i8) ? cumbits_i8 : cumbits_u8;
      const int mant = use_double ? 52 : 23;
      int lgk = 0, req_bits, np_k;
      uint64_t kk = (uint64_t)maxk - 1;
      while (kk > 0) { ++lgk; kk >>= 1; }
      req_bits = 2 * (mant - oztrim_crt + 1) + lgk + 1;
      for (np_k = 0; np_k < 20 && cumbits[np_k] < req_bits; ++np_k);
      np_k = (np_k < 20) ? np_k + 1 : 20;
      if (np_k < nprimes) {
        if (0 > verbosity || 2 < verbosity) {
          fprintf(stderr, "INFO OZAKI: bounded-K=%d reduces primes %d -> %d\n", maxk, nprimes, np_k);
        }
        nprimes = np_k;
      }
    }
    { /* Scheme 1: cutoff = 2*(nslices-1) - oztrim must stay >= 0 */
      const int max_trim = 2 * (nslices - 1);
      if (oztrim > max_trim) oztrim = max_trim;
    }
    ctx->nslices = nslices;
    ctx->nprimes = nprimes;
    ndecomp = (2 == kind) ? nprimes : nslices;
  } /* ndecomp_auto */
  if (2 == kind && 20 < ndecomp) ndecomp = 20;
  if (0 > ozflags) ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;

  ctx->use_double = use_double;
  ctx->kind = kind;
  ctx->ozflags = ozflags;
  ctx->oztrim = oztrim;
  ctx->ndecomp = ndecomp;
  ctx->verbosity = verbosity;

  nv = (int)devinfo->nv;

  /* Environment-driven tuning */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0);
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)devinfo->wgsize[2]);
  if (0 >= sg) sg = (int)devinfo->wgsize[1]; /* fallback: preferred WG multiple */
  if (0 >= sg) sg = 16; /* last resort */
  { /* NV MMA: enabled when NV>=3 (SM>=80 Ampere+, set via LIBXSTREAM_NV). */
    const int nv_mma = (3 <= nv && 0 != gpu) ? 1 : 0;
    if (0 != nv_mma) {
      sg = 32;
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: NV_MMA enabled (SG=32, m16n8k32)\n");
      }
    }
    else if (16 != sg) {
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: SG forced to 16\n");
      }
      sg = 16;
    }
    ctx->nv_mma = nv_mma;
  }
  ctx->sg = sg;

  /* GEMM-mode kernels (tiled K-loop path) */
  ctx->kern_preprocess_a = NULL;
  ctx->kern_preprocess_b = NULL;
  ctx->kernel_registry = NULL;
  ctx->kern_scale_beta = NULL;
  ctx->kern_crt_preprocess_a = NULL;
  ctx->kern_crt_preprocess_b = NULL;
  ctx->crt_registry = NULL;
  ctx->kern_crt_scale_beta = NULL;
  /**
   * output tile sizes: fit SG * NTM * NTN <= max_wgs.
   * tm must be multiple of 8*RTM, tn must be multiple of 16*RTN.
   * Large GRF halves effective max work-group size.
   */
  if (EXIT_SUCCESS == result) {
    const int bm_pre = 16, bn_pre = 16, bk_pre = 32;
    char build_params[2048];
    char build_options[128];
    const int mant_bits = use_double ? 52 : 23;
    const int bias_plus_mant = use_double ? 1075 : 150;
    int rtm = 0, rtn = 0, rtn_req = 0, ku_req, biggrf, hier;
    size_t max_wgs;
    int v;
    {
      const char *const env_hier = getenv("OZAKI_HIER");
      hier = (NULL != env_hier) ? (0 != atoi(env_hier) ? 1 : 0) : (2 == kind ? 1 : 0);
    }
    /**
     * Ozaki-local 256-GRF decision (per-kernel, not global).
     * LIBXSTREAM_BIGGRF: explicit user override for all kernels.
     * OZAKI_BIGGRF: Ozaki-specific override.
     * Default: auto-enable for Intel GPUs, but HIER prefers GRF128
     * (halved private arrays make 2x occupancy the better trade-off).
     */
    env = getenv("OZAKI_BIGGRF");
    if (NULL != env) {
      biggrf = (0 != atoi(env));
    }
    else if (NULL != getenv("LIBXSTREAM_BIGGRF")) {
      biggrf = (0 != devinfo->biggrf);
    }
    else {
      biggrf = (0 != hier && 0 != devinfo->intel && 0 != gpu) ? 0 : (0 != devinfo->intel && 0 != gpu);
    }
    LIBXS_SNPRINTF(build_options, sizeof(build_options), "-cl-fast-relaxed-math -cl-denorms-are-zero%s",
      (0 != biggrf && 0 != devinfo->intel && 0 == devinfo->biggrf) ? " -cl-intel-256-GRF-per-thread" : "");
    max_wgs = (0 != biggrf) ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
    /* Read optional user overrides for register tiling factors. */
    env = getenv("OZAKI_RTM");
    if (NULL != env && 0 < atoi(env)) rtm = atoi(env);
    env = getenv("OZAKI_RTN");
    if (NULL != env && 0 < atoi(env)) {
      rtn = atoi(env);
      rtn_req = rtn; /* explicit request applies to both schemes */
    }
    /**
     * Choose defaults when not explicitly set:
     *  256-GRF: RTM=4 RTN=2 (8 accumulators, measured sweet spot)
     *  128-GRF: RTM=2 RTN=2 (4 accumulators)
     *  Other vendors:  RTM=1 RTN=1 (conservative)
     */
    env = getenv("OZAKI_KU");
    ku_req = (NULL != env && 0 < atoi(env)) ? atoi(env) : 0;
    {
      /* dp4a benefits from a deeper K-unroll, but only at RTN>1 (see below). */
      const int ku_default = (0 == devinfo->intel && 0 == ctx->nv_mma && 2 <= nv && 0 != gpu && 2 == kind) ? 4 : 2;
      int ku = (0 != ku_req) ? ku_req : ku_default;
      if (0 != ctx->nv_mma && ku > 1) ku = 1;
      ctx->ku = ku;
    }
    env = getenv("OZAKI_RC");
    {
      int rc = (NULL != env && 0 < atoi(env)) ? atoi(env) : 8;
      ctx->rc = (rc <= 4) ? 4 : 8;
    }
    env = getenv("OZAKI_PB");
    {
      int pb = (NULL != env && 0 < atoi(env)) ? atoi(env) : 1;
      ctx->pb = pb;
    }
    env = getenv("OZAKI_XOVER");
    ctx->xover = (NULL != env && 0 < atof(env)) ? atof(env) : 128.0;
    ctx->hier = hier;
    ctx->maxk = maxk;
    if (0 == rtm) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtm = (0 != biggrf) ? 4 : 2;
      }
      else if (0 != ctx->nv_mma && 0 != gpu) {
        rtm = 2;
      }
      else rtm = 1; /* NV dp4a: RTM>1 replicates the A-load 16x, measured 4x slower */
    }
    if (0 == rtn) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtn = 2;
      }
      else if (0 != ctx->nv_mma && 0 != gpu) {
        rtn = 2;
      }
      else if (2 <= nv && 0 != gpu && 2 == kind) {
        rtn = 2;
      }
      else rtn = 1;
    }
    /* Sanitize: round down to nearest power of two. */
    v = rtm;
    rtm = 1;
    while (v > 1) {
      v >>= 1;
      rtm <<= 1;
    }
    v = rtn;
    rtn = 1;
    while (v > 1) {
      v >>= 1;
      rtn <<= 1;
    }
    /**
     * Explicit tm/tn bypass size-aware selection. Caller-supplied values (0 =
     * auto) take precedence; OZAKI_TM/OZAKI_TN are read here so that every
     * driver shares one authoritative override path. Otherwise tm/tn below are
     * the largest legal tile, i.e. the upper bound that ozaki_tile_select()
     * shrinks per call based on M and N.
     */
    if (0 >= tm) {
      env = getenv("OZAKI_TM");
      if (NULL != env) tm = atoi(env);
    }
    if (0 >= tn) {
      env = getenv("OZAKI_TN");
      if (NULL != env) tn = atoi(env);
    }
    ctx->tm_req = (0 < tm) ? tm : 0;
    ctx->tn_req = (0 < tn) ? tn : 0;
    if (0 >= tm) tm = 256;
    if (0 >= tn) tn = 256;
    /**
     * Clamp tiling factors so at least one sub-tile remains per dimension.
     * XMX_M=8, XMX_N=16 for dp4a/DPAS/scalar; XMX_M=16, XMX_N=8 for NV_MMA.
     */
    { const int xmx_m = OZAKI_XMX_M(ctx);
      const int xmx_n = OZAKI_XMX_N(ctx);
      while (rtm > 1 && tm / (xmx_m * rtm) < 1) rtm >>= 1;
      while (rtn > 1 && tn / (xmx_n * rtn) < 1) rtn >>= 1;
      /**
       * Shrink tile to satisfy work-group size constraint.
       * WGS = SG * NTM * NTN = SG * (BM/(XMX_M*RTM)) * (BN/(XMX_N*RTN)).
       * Scheme 2 may dispatch with crt_rtm < rtm, which raises WGS for the
       * same tile; ozaki_tile_select() re-checks the bound with the actual
       * register tiling, so this clamp only establishes the Scheme-1 maximum.
       */
      /**
       * Halving must not break sub-tile granularity: NTM/NTN are integer
       * divisions, so a tm that is a multiple of xmx_m*rtm before the shift
       * need not be one after (MMA, rtm=2: 160 -> 80, NTM truncates 80/32 to
       * 2 and 16 rows of the tile are never covered -- silently missing part
       * of C). Round the halved extent down to the granularity and stop when
       * it can no longer shrink.
       */
      { const int gm = xmx_m * rtm, gn = xmx_n * rtn;
        while ((size_t)sg * ((size_t)(tm / gm) * (tn / gn)) > max_wgs && (tm > gm || tn > gn)) {
          if (tm >= tn && tm > gm) tm = (tm / 2 / gm) * gm;
          else if (tn > gn) tn = (tn / 2 / gn) * gn;
          else break;
          if (tm < gm) tm = gm;
          if (tn < gn) tn = gn;
        }
      }
    }
    /**
     * dp4a register budget: OZAKI_DPAS_TILED keeps 8*RTN B-words, 64*RTM
     * A-words and 8*PB*RTM*RTN accumulator words live per work-item. NVIDIA
     * caps a thread at 255 registers and spills beyond that, which costs far
     * more than the tiling gains. Shrink RTN then RTM until the estimate fits.
     */
    if (0 == devinfo->intel && 0 == ctx->nv_mma && 2 <= nv && 0 != gpu) {
      const int budget = 224; /* leave headroom for addressing and the epilogue */
      int nreg = 64 * rtm + 8 * rtn + 8 * ctx->pb * rtm * rtn;
      while (budget < nreg && (1 < rtn || 1 < rtm)) {
        if (1 < rtn) rtn >>= 1;
        else rtm >>= 1;
        nreg = 64 * rtm + 8 * rtn + 8 * ctx->pb * rtm * rtn;
      }
      /**
       * The deeper K-unroll only pays when RTN>1 amortizes it over two B
       * columns; at RTN==1 it measured 21% slower than KU=2. RTN may have just
       * been reduced above, so re-derive KU here unless it was requested.
       */
      if (0 == ku_req && 2 > rtn && 2 < ctx->ku) ctx->ku = 2;
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: dp4a register estimate %d (RTM=%d RTN=%d PB=%d KU=%d)\n",
          nreg, rtm, rtn, ctx->pb, ctx->ku);
      }
    }
    ctx->max_wgs = max_wgs;
    { /* Compute units: saturation target for size-aware tile selection. */
      cl_uint nunits = 0;
      if (EXIT_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, NULL)) {
        nunits = 0;
      }
      ctx->nunits = (int)nunits;
    }
    /**
     * Saturation floor for tile selection. See OZAKI_TILE_SAT: the divisor is
     * only portable between devices of similar compute-unit granularity, so
     * NVIDIA asks for one work-group per SM instead of nunits/16.
     */
    ctx->tile_sat = (0 == devinfo->intel && 0 != gpu) ? OZAKI_TILE_SAT_NV : OZAKI_TILE_SAT;
    { const char *const env_sat = getenv("OZAKI_TILE_SAT");
      if (NULL != env_sat && 0 < atoi(env_sat)) ctx->tile_sat = atoi(env_sat);
    }
    /**
     * Occupancy cap on the work-group size (0 = hardware bound only). NVIDIA
     * measured monotonically better with smaller work-groups at fixed tile
     * area, so cap at OZAKI_WGS_MAX_NV warps' worth; PVC is left on max_wgs
     * because its tile selection was tuned against that bound.
     */
    ctx->wgs_max = (0 == devinfo->intel && 0 != gpu) ? (sg * OZAKI_WGS_MAX_NV) : 0;
    { const char *const env_wgs = getenv("OZAKI_WGS_MAX");
      if (NULL != env_wgs && 0 <= atoi(env_wgs)) ctx->wgs_max = atoi(env_wgs);
    }
    { /* Scheme 1: always compile preprocessing + create registry (for adaptive) */
      const int sq_jit = ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE);
      const int cutoff_jit = 2 * (nslices - 1) - oztrim;
      size_t goff = 0;
      goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff,
        "-DBK=%d -DKU=%d -DRC=%d -DSG=%d -DINTEL=%d -DNV=%d"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DRTM=%d -DRTN=%d"
        " -DOZAKI_SQ=%d -DCONSTANT=global",
        bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, nv,
        nslices, use_double, mant_bits, bias_plus_mant, bm_pre, bn_pre, bk_pre, rtm, rtn, sq_jit);
      if (0 != ctx->nv_mma) {
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DNV_MMA=1");
      }
      env = getenv("OZAKI_PREFETCH");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DOZAKI_PREFETCH=1");
      }
      env = getenv("OZAKI_SCALAR_ACC");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DOZAKI_SCALAR_ACC=1");
      }
      env = getenv("OZAKI_LU");
      { const int lu = (NULL != env) ? atoi(env) : 0;
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DLU=%d", lu);
      }
      LIBXS_UNUSED(goff);
      memcpy(ctx->base_flags, build_params, sizeof(ctx->base_flags));
      LIBXS_SNPRINTF(ctx->base_options, sizeof(ctx->base_options), "%s", build_options);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      { /* Compile preprocessing + scale_beta (shared, tile/cutoff-independent) */
        char pp_flags[sizeof(build_params) + 64];
        cl_program program = NULL;
        LIBXS_SNPRINTF(pp_flags, sizeof(pp_flags), "%s -DBM=%d -DBN=%d -DOZAKI_CUTOFF=%d", build_params, tm, tn, cutoff_jit);
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8, "ozaki1", pp_flags, build_options, NULL, NULL, NULL, 0, &program);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_a_dense", &ctx->kern_preprocess_a);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_b_dense", &ctx->kern_preprocess_b);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "scale_beta", &ctx->kern_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
      }
      ctx->kernel_registry = libxs_registry_create();
      if (EXIT_SUCCESS != result) {
        if (NULL != ctx->kern_preprocess_a) {
          clReleaseKernel(ctx->kern_preprocess_a);
          ctx->kern_preprocess_a = NULL;
        }
        if (NULL != ctx->kern_preprocess_b) {
          clReleaseKernel(ctx->kern_preprocess_b);
          ctx->kern_preprocess_b = NULL;
        }
        if (NULL != ctx->kern_scale_beta) {
          clReleaseKernel(ctx->kern_scale_beta);
          ctx->kern_scale_beta = NULL;
        }
      }
    }
    /**
     * Scheme 2: compile the CRT kernels even when Scheme 1 was requested, so
     * adaptive dispatch can reach them. Skipped without fp64 -- the
     * reconstruction is double-only, so the build would fail and take the
     * usable Scheme-1 path down with it. ozaki_gemm falls back on a NULL
     * crt_registry, and kind 2 already failed above.
     */
    if (0 != has_fp64) {
      /**
       * Fractional-CRT: mode 1 replaces the whole reconstruction with a flat
       * fractional sum (needs raw per-prime residues, so the flat path;
       * opt-in due to a magnitude domain bound). Mode 2 applies fractional
       * reconstruction only per hierarchical group (each group product is
       * below 2^53, so leaf reconstruction is exact for all group values) and
       * keeps the exact hierarchical level-2 combine, so it is exact
       * everywhere. Tables are generated per active moduli set.
       * Mode 2 is the default because it is exact over the whole CRT range
       * like Garner, keeps the same group-at-a-time storage (and hence
       * occupancy), and is faster; OZAKI_FRACCRT=0 selects Garner.
       */
      const char *const env_fraccrt = getenv("OZAKI_FRACCRT");
      const char *const env_skip = getenv("OZAKI_SKIP_GARNER");
      const int fraccrt_req = (NULL != env_fraccrt) ? atoi(env_fraccrt) : 2;
      const int fraccrt = (1 == fraccrt_req || 2 == fraccrt_req) ? fraccrt_req : 0;
      const int crt_hier = (1 == fraccrt) ? 0 : (0 != ctx->hier || 3 == kind || 2 == fraccrt);
      const int crt_rtm = (0 != crt_hier && 0 != biggrf && 0 == ctx->hier) ? LIBXS_MAX(rtm / 2, 1) : rtm;
      /**
       * MMA gives a sub-tile 16 rows but only 8 columns, so reaching a square
       * register tile needs twice the column tiling. Scheme 2 measured +36% at
       * RTN=4 (n=4096: 4300 -> 5828), while Scheme 1 measured -26% there and
       * peaks at RTN=2 -- it runs a pair loop over slices and is bound by the
       * per-pair epilogue rather than by column reuse. Hence per-scheme.
       */
      const int crt_rtn = (0 != ctx->nv_mma && 0 != gpu && 0 == rtn_req) ? 4 : rtn;
      char crt_build_options[128];
      size_t coff = 0;
      if (0 != fraccrt) {
        /**
         * Fractional CRT relies on error-free transformations (two_sum,
         * two_product), which relaxed math is allowed to algebraically
         * simplify away. Keep strict floating-point semantics.
         */
        LIBXS_SNPRINTF(crt_build_options, sizeof(crt_build_options), "-cl-denorms-are-zero");
      }
      else if (0 != crt_hier && 0 != biggrf && 0 == ctx->hier) {
        LIBXS_SNPRINTF(crt_build_options, sizeof(crt_build_options),
          "-cl-fast-relaxed-math -cl-denorms-are-zero");
      }
      else {
        LIBXS_SNPRINTF(crt_build_options, sizeof(crt_build_options), "%s", build_options);
      }
      coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
        "-DBK=%d -DKU=%d -DRC=%d -DSG=%d -DINTEL=%d -DNV=%d"
        " -DNPRIMES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d -DMANT_TRUNC=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DKGROUPS=%d -DRTM=%d -DRTN=%d -DPB=%d"
        " -DCONSTANT=global",
        bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, nv,
        nprimes, use_double, mant_bits, bias_plus_mant - oztrim_crt, oztrim_crt, bm_pre, bn_pre, bk_pre,
        (1 < ozgroups) ? ozgroups : 0, crt_rtm, crt_rtn, ctx->pb);
      if (0 != ctx->nv_mma) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DNV_MMA=1");
      }
      if (0 == use_i8) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_U8=1");
      }
      /**
       * Pre-interleave B for the NVIDIA paths so each operand is one aligned
       * uint: dp4a gets 8 loads per column instead of 32 strided byte gathers,
       * and the MMA b-fragment becomes 2 loads instead of 8. Intel keeps the
       * plain layout because DPAS transforms on read; OZAKI_BVNNI=0 opts out.
       */
      env = getenv("OZAKI_BVNNI");
      if (0 == devinfo->intel && 2 <= nv && 0 != gpu &&
          (NULL == env || 0 != atoi(env)))
      {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_BVNNI=1");
      }
      if (1 == fraccrt) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_FRACCRT=1");
        coff += ozaki_emit_fraccrt(build_params + coff, sizeof(build_params) - coff,
          (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli, nprimes, 14);
      }
      else if (2 == fraccrt) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_FRACCRT=2");
        coff += ozaki_emit_fraccrt2(build_params + coff, sizeof(build_params) - coff,
          (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli, nprimes, 11, 4);
      }
      if (NULL != env_skip && 0 != atoi(env_skip)) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DSKIP_GARNER=1");
      }
      if (0 != crt_hier) {
        const uint16_t* modtab = (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli;
        const int hier_gs = 4, ngroups = LIBXS_UPDIV(nprimes, hier_gs);
        uint32_t gp[5];
        uint64_t l2b[5];
        int gi, use_tree;
        for (gi = 0; gi < ngroups; ++gi) {
          const int lo = gi * hier_gs, hi = (lo + hier_gs <= nprimes) ? lo + hier_gs : nprimes;
          uint32_t p = 1;
          int k;
          for (k = lo; k < hi; ++k) p *= (uint32_t)modtab[k];
          gp[gi] = p;
          l2b[gi] = (uint64_t)(-1) / (uint64_t)p;
        }
        /**
         * Tree-merge level 2 is implemented for at most 2 groups; requesting it
         * for more would build a kernel that leaves the result unassigned, so
         * the request is clamped rather than honored.
         */
        env = getenv("OZAKI_HIER_L2");
        use_tree = (NULL != env) ? (0 != atoi(env) ? 1 : 0) : (ngroups <= 2 ? 1 : 0);
        if (0 != use_tree && 2 < ngroups) {
          if (0 > verbosity || 2 < verbosity) {
            fprintf(stderr, "INFO OZAKI: tree-merge level 2 needs <=2 groups (have %d), using Garner\n", ngroups);
          }
          use_tree = 0;
        }
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
          " -DOZAKI_HIER=1 -DHIER_NGROUPS_ACTUAL=%d -DOZAKI_HIER_L2=%d", ngroups, use_tree);
        for (gi = 0; gi < ngroups; ++gi) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
            " -DHIER_GPROD_%d=%uu -DHIER_L2B_%d=%luul", gi, (unsigned)gp[gi], gi, (unsigned long)l2b[gi]);
        }
        if (0 == use_tree) {
          int gj;
          for (gi = 0; gi < ngroups; ++gi) {
            for (gj = gi + 1; gj < ngroups; ++gj) {
              coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
                " -DHIER_L2INV_%d_%d=%uu", gi, gj,
                (unsigned)libxs_mod_inverse_u32(gp[gi] % gp[gj], gp[gj]));
            }
          }
        }
        else {
          uint64_t gprod[5];
          gprod[0] = (uint64_t)gp[0];
          if (ngroups >= 2) {
            coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
              " -DHIER_TREE_INV_0_1=%uu", (unsigned)libxs_mod_inverse_u32(gp[0] % gp[1], gp[1]));
            gprod[0] = (uint64_t)gp[0] * (uint64_t)gp[1];
            coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
              " -DHIER_TREE_PROD_01=%luul", (unsigned long)gprod[0]);
          }
        }
      }
      env = getenv("OZAKI_LU");
      { const int lu = (NULL != env) ? atoi(env) : 0;
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DLU=%d", lu);
      }
      LIBXS_UNUSED(coff);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      /**
       * Base flags for the tile-specialized CRT registry: BM/BN and
       * OZAKI_BOUNDS are appended per specialization by ozaki_get_crt_kernel.
       */
      memcpy(ctx->crt_flags, build_params, sizeof(ctx->crt_flags));
      LIBXS_SNPRINTF(ctx->crt_options, sizeof(ctx->crt_options), "%s", crt_build_options);
      ctx->crt_registry = libxs_registry_create();
      {
        char base_flags[sizeof(build_params) + 64];
        cl_program program = NULL;
        LIBXS_SNPRINTF(base_flags, sizeof(base_flags), "%s -DBM=%d -DBN=%d -DOZAKI_BOUNDS=1", build_params, tm, tn);
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8, "ozaki2", base_flags, crt_build_options, NULL, NULL, NULL, 0, &program);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_a_crt_dense", &ctx->kern_crt_preprocess_a);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_b_crt_dense", &ctx->kern_crt_preprocess_b);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "scale_beta", &ctx->kern_crt_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
      }
      ctx->crt_rtm = crt_rtm;
      ctx->crt_rtn = crt_rtn;
      if (EXIT_SUCCESS != result) {
        if (NULL != ctx->kern_crt_preprocess_a) {
          clReleaseKernel(ctx->kern_crt_preprocess_a);
          ctx->kern_crt_preprocess_a = NULL;
        }
        if (NULL != ctx->kern_crt_preprocess_b) {
          clReleaseKernel(ctx->kern_crt_preprocess_b);
          ctx->kern_crt_preprocess_b = NULL;
        }
        if (NULL != ctx->kern_crt_scale_beta) {
          clReleaseKernel(ctx->kern_crt_scale_beta);
          ctx->kern_crt_scale_beta = NULL;
        }
      }
    }

    /* Initialize complex GEMM block-embedding kernels (precision-agnostic, always compiled) */
    if (EXIT_SUCCESS == result) {
      cl_program program_3m = NULL;
      char build_params_3m[512];

      LIBXS_SNPRINTF(build_params_3m, sizeof(build_params_3m), "-DUSE_DOUBLE=%d", use_double ? 1 : 0);

      result = libxstream_opencl_program(
        0, OPENCL_KERNELS_SOURCE_GEMM3M, "zgemm_block", build_params_3m, build_options, NULL, NULL, NULL, 0, &program_3m);

      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(program_3m, "zgemm_block_construct_a", &ctx->kern_zgemm_block_construct_a);
      }
      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(program_3m, "zgemm_block_construct_b_n", &ctx->kern_zgemm_block_construct_b_n);
      }
      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(program_3m, "zgemm_block_construct_b_t", &ctx->kern_zgemm_block_construct_b_t);
      }
      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(program_3m, "zgemm_block_finalize", &ctx->kern_zgemm_block_finalize);
      }
      if (NULL != program_3m) clReleaseProgram(program_3m);

      /* Block-embedding kernel failure is non-fatal - just disables complex GEMM */
      if (EXIT_SUCCESS != result) {
        if (2 < verbosity) {
          if (NULL == program_3m) {
            fprintf(stderr, "WARN OZAKI: block-embedding kernel compilation failed (fp=%d), complex GEMM disabled\n",
              use_double ? 64 : 32);
          }
          else {
            fprintf(stderr, "WARN OZAKI: block-embedding kernel query failed (fp=%d), complex GEMM disabled\n",
              use_double ? 64 : 32);
          }
        }
        if (NULL != ctx->kern_zgemm_block_construct_a) {
          clReleaseKernel(ctx->kern_zgemm_block_construct_a);
          ctx->kern_zgemm_block_construct_a = NULL;
        }
        if (NULL != ctx->kern_zgemm_block_construct_b_n) {
          clReleaseKernel(ctx->kern_zgemm_block_construct_b_n);
          ctx->kern_zgemm_block_construct_b_n = NULL;
        }
        if (NULL != ctx->kern_zgemm_block_construct_b_t) {
          clReleaseKernel(ctx->kern_zgemm_block_construct_b_t);
          ctx->kern_zgemm_block_construct_b_t = NULL;
        }
        if (NULL != ctx->kern_zgemm_block_finalize) {
          clReleaseKernel(ctx->kern_zgemm_block_finalize);
          ctx->kern_zgemm_block_finalize = NULL;
        }
        result = EXIT_SUCCESS; /* non-fatal */
      }
    }

    if (EXIT_SUCCESS == result) {
      ctx->tm = tm;
      ctx->tn = tn;
      ctx->rtm = rtm;
      ctx->rtn = rtn;
      ctx->biggrf = biggrf;
      ctx->bm_pre = bm_pre;
      ctx->bn_pre = bn_pre;
      ctx->bk_pre = bk_pre;
    }
    else if (0 != verbosity) {
      fprintf(stderr, "ERROR OZAKI: kernel build failed\n");
    }
  } /* end if (EXIT_SUCCESS == result) for kernel initialization */


  /**
   * OZAKI_CACHE: preprocessing cache bitmask (1=A, 2=B, -1 or 3=both).
   * Default off: cache assumes matrix content at a given pointer is unchanged
   * between calls. Applications that modify matrices in-place must either
   * disable cache (0) or ensure cached matrices are truly constant.
   * The fingerprint check catches some modifications but is not exhaustive.
   */
  {
    const char *const env_cache = getenv("OZAKI_CACHE");
    const int cache = (NULL != env_cache ? atoi(env_cache) : 0);
    ctx->cache.flags = (0 == cache ? 0 : (0 > cache ? 3 : cache));
  }

  /* Report compiled kernel info */
  if (EXIT_SUCCESS == result && (0 > verbosity || 2 < verbosity)) {
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "kind", kind);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "intel", (int)devinfo->intel);
    ozaki_print_opt(stderr, "nv", nv);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "tm", ctx->tm);
    ozaki_print_opt(stderr, "tn", ctx->tn);
    ozaki_print_opt(stderr, "rtm", ctx->rtm);
    if (ctx->crt_rtm != ctx->rtm) ozaki_print_opt(stderr, "crt_rtm", ctx->crt_rtm);
    ozaki_print_opt(stderr, "rtn", ctx->rtn);
    if (ctx->crt_rtn != ctx->rtn) ozaki_print_opt(stderr, "crt_rtn", ctx->crt_rtn);
    if (0 != devinfo->intel) {
      const int crt_grf128 = (0 != ctx->crt_rtm && ctx->crt_rtm < ctx->rtm);
      ozaki_print_opt(stderr, "grf", ctx->biggrf ? 256 : 128);
      if (0 != crt_grf128) ozaki_print_opt(stderr, "crt_grf", 128);
    }
    ozaki_print_opt(stderr, "ndecomp", ndecomp);
    ozaki_print_opt(stderr, "trim", oztrim);
    if (2 == kind) {
      const char *const e_i8 = getenv("OZAKI_I8");
      fprintf(stderr, " u8=%d", (NULL == e_i8 || 0 == atoi(e_i8)) ? 1 : 0);
      ozaki_print_opt(stderr, "kgroups", ozgroups);
      ozaki_print_opt(stderr, "pb", ctx->pb);
      ozaki_print_opt(stderr, "hier", ctx->hier);
    }
    ozaki_print_opt(stderr, "cache", ctx->cache.flags);
    if (3 == kind) fprintf(stderr, " xover=%g", ctx->xover);
    fprintf(stderr, "\n");
  }

  /* Create persistent helper streams and synchronization events */
  {
    const int sflags = (0 != profiling) ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT;
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(&ctx->stream_a, "ozaki_a", sflags);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(&ctx->stream_b, "ozaki_b", sflags);
    }
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_b);

  return result;
}


void ozaki_destroy(ozaki_context_t* ctx)
{
  if (NULL != ctx) {
    if (NULL != ctx->kern_preprocess_a) {
      clReleaseKernel(ctx->kern_preprocess_a);
    }
    if (NULL != ctx->kern_preprocess_b) {
      clReleaseKernel(ctx->kern_preprocess_b);
    }
    if (NULL != ctx->kern_scale_beta) {
      clReleaseKernel(ctx->kern_scale_beta);
    }
    if (NULL != ctx->kernel_registry) {
      const void* rkey = NULL;
      size_t cursor = 0;
      ozaki_kernel_set_t* kset = (ozaki_kernel_set_t*)libxs_registry_begin(
        ctx->kernel_registry, &rkey, &cursor);
      while (NULL != kset) {
        if (NULL != kset->kern_fused) clReleaseKernel(kset->kern_fused);
        kset = (ozaki_kernel_set_t*)libxs_registry_next(
          ctx->kernel_registry, &rkey, &cursor);
      }
      libxs_registry_destroy(ctx->kernel_registry);
    }
    if (NULL != ctx->kern_crt_preprocess_a) {
      clReleaseKernel(ctx->kern_crt_preprocess_a);
    }
    if (NULL != ctx->kern_crt_preprocess_b) {
      clReleaseKernel(ctx->kern_crt_preprocess_b);
    }
    if (NULL != ctx->crt_registry) {
      const void* rkey = NULL;
      size_t cursor = 0;
      ozaki_crt_kernel_set_t* kset = (ozaki_crt_kernel_set_t*)libxs_registry_begin(
        ctx->crt_registry, &rkey, &cursor);
      while (NULL != kset) {
        if (NULL != kset->kern_fused) clReleaseKernel(kset->kern_fused);
        kset = (ozaki_crt_kernel_set_t*)libxs_registry_next(
          ctx->crt_registry, &rkey, &cursor);
      }
      libxs_registry_destroy(ctx->crt_registry);
    }
    if (NULL != ctx->kern_crt_scale_beta) {
      clReleaseKernel(ctx->kern_crt_scale_beta);
    }
    if (NULL != ctx->kern_zgemm_block_construct_a) {
      clReleaseKernel(ctx->kern_zgemm_block_construct_a);
    }
    if (NULL != ctx->kern_zgemm_block_construct_b_n) {
      clReleaseKernel(ctx->kern_zgemm_block_construct_b_n);
    }
    if (NULL != ctx->kern_zgemm_block_construct_b_t) {
      clReleaseKernel(ctx->kern_zgemm_block_construct_b_t);
    }
    if (NULL != ctx->kern_zgemm_block_finalize) {
      clReleaseKernel(ctx->kern_zgemm_block_finalize);
    }

    /**
     * Quiesce cache: NULL pointers under lock (prevents new hits),
     * then wait for in-flight gemm threads to finish using cached buffers.
     */
    {
      void *sa_sl, *sa_ex, *sb_sl, *sb_ex;
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->cache.lock);
      sa_sl = ctx->cache.a.d_slices;
      ctx->cache.a.d_slices = NULL;
      sa_ex = ctx->cache.a.d_exp;
      ctx->cache.a.d_exp = NULL;
      sb_sl = ctx->cache.b.d_slices;
      ctx->cache.b.d_slices = NULL;
      sb_ex = ctx->cache.b.d_exp;
      ctx->cache.b.d_exp = NULL;
      ctx->cache.flags = 0;
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->cache.lock);
      while (0 != ctx->cache.nusers) LIBXS_SYNC_PAUSE;
      OZAKI_DEV_FREE(sa_sl);
      OZAKI_DEV_FREE(sa_ex);
      OZAKI_DEV_FREE(sb_sl);
      OZAKI_DEV_FREE(sb_ex);
    }

    if (NULL != libxstream_opencl_config.pool_dev) {
      const int verbosity = libxs_get_verbosity();
      if (0 > LIBXS_MIN(ctx->verbosity, verbosity) || 2 < LIBXS_MAX(ctx->verbosity, verbosity)) {
        libxs_malloc_pool_info_t info;
        if (EXIT_SUCCESS == libxs_malloc_pool_info(libxstream_opencl_config.pool_dev, &info)) {
          const int peak = (int)LIBXS_UPDIV(info.peak, (size_t)1 << 20);
          const int size = (int)LIBXS_UPDIV(info.size, (size_t)1 << 20);
          printf("POOL: peak_mb=%i size_mb=%i nmallocs=%lu\n", peak, size, (unsigned long int)info.nmallocs);
        }
      }
    }
    /* Destroy persistent synchronization events */
    if (NULL != ctx->evt_prep_a) libxstream_event_destroy(ctx->evt_prep_a);
    if (NULL != ctx->evt_prep_b) libxstream_event_destroy(ctx->evt_prep_b);
    /* Destroy persistent helper streams */
    if (NULL != ctx->stream_a) libxstream_stream_destroy(ctx->stream_a);
    if (NULL != ctx->stream_b) libxstream_stream_destroy(ctx->stream_b);
    LIBXS_MEMZERO(ctx);
  }
}


void ozaki_invalidate_cache(ozaki_context_t* ctx, const void* a, const void* b)
{
  if (NULL != ctx) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->cache.lock);
    /* Invalidate A cache entry if it matches the given pointer */
    if (NULL != a && a == ctx->cache.a.ptr) {
      ctx->cache.a.ptr = NULL;
      ctx->cache.a.dim = 0;
      ctx->cache.a.K = 0;
      ctx->cache.a.ld = 0;
      ctx->cache.a.trans = 0;
    }
    /* Invalidate B cache entry if it matches the given pointer */
    if (NULL != b && b == ctx->cache.b.ptr) {
      ctx->cache.b.ptr = NULL;
      ctx->cache.b.dim = 0;
      ctx->cache.b.K = 0;
      ctx->cache.b.ld = 0;
      ctx->cache.b.trans = 0;
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->cache.lock);
  }
}
