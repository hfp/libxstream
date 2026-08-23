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
 * i.e. 0.06 work-groups per SM - the heuristic then picks the largest tile for
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
 * Leaf group size of the hierarchical CRT. At most 4, because the level-2
 * datapath is 32-bit and a group product must fit uint32; below that, prefer a
 * size that leaves no group holding a single prime - at nprimes=9 the 4,4,1
 * split costs the reconstruction 1.21 ms where three full groups of 3 take 0.58.
 *
 * That preference is bounded by OZAKI_HIER_NGROUPS_MAX: nprimes=17 is the only
 * fp64-legal count whose divisor rule asks for 3, and the resulting six groups
 * indexed past the kernel's five-group tables, returning rsq=0 with exit code 0.
 * A one-prime tail group is a performance defect; a sixth group is a wrong
 * answer, so the bound wins.
 */
static int ozaki_hier_gs(int nprimes)
{
  const int gs_min = LIBXS_UPDIV(nprimes, OZAKI_HIER_NGROUPS_MAX);
  int gs = (1 != (nprimes % 4)) ? 4 : ((1 != (nprimes % 3)) ? 3 : ((1 != (nprimes % 2)) ? 2 : 4));
  if (gs < gs_min) gs = gs_min;
  return gs;
}


/**
 * One step of register-tile growth, rows before columns: a square register tile
 * has the best reuse per accumulator, and both schemes measured their optimum at
 * 4 rows - Scheme 1 then wants 4x4 where it has the registers for it (256-GRF),
 * while Scheme 2 stays at 4x2 and loses 2.5x at 4x4. Growing rows first reaches
 * both from their respective base without a per-scheme rule.
 */
static ozaki_tile_t ozaki_rtile_grow(int rtm, int rtn)
{
  ozaki_tile_t result;
  result.m = (4 > rtm) ? (2 * rtm) : rtm;
  result.n = (4 > rtm) ? rtn : (2 * rtn);
  return result;
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
 * all group values - the hierarchy keeps exactness across the full range.
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
   * 6158-6529 GFLOPS, WGS=256 6145-6275, WGS=512 only 5103 - monotone in WGS,
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
    /**
     * Two warp groups per work-group halve the residue-plane traffic but also
     * halve the work-group count, and only the first of those scales with the
     * problem: measured +17.8% at n=8192 and +5.5% at n=4096 against one warp
     * group, but -1.6% at n=1024, where 64 work-groups no longer cover 114 SMs.
     * Fall back to one warp group below the saturation floor. This is the only
     * size-dependent choice the wgmma path makes, and it is free because the CRT
     * registry is keyed on the tile, so both variants coexist per shape.
     */
    if (0 != ctx->wgmma && 64 < tile.m && nwg_min > (LIBXS_UPDIV(M, tile.m) * LIBXS_UPDIV(N, tile.n))) {
      tile.m = 64;
    }
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
           * elements, so cm*cn/(cm+cn) is its arithmetic intensity - maximal
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


ozaki_tile_t ozaki_rtile_select(const ozaki_context_t* ctx, int M, int N, int crt)
{
  ozaki_tile_t result;
  result.m = (0 != crt) ? ctx->crt_rtm : ctx->rtm;
  result.n = (0 != crt) ? ctx->crt_rtn : ctx->rtn;
  { const int big_m = (0 != crt) ? ctx->crt_rtm_big : ctx->rtm_big;
    const int big_n = (0 != crt) ? ctx->crt_rtn_big : ctx->rtn_big;
    if ((big_m != result.m || big_n != result.n) && 0 < ctx->nunits) {
      /**
       * The promoted pair covers twice the output per sub-group, so the tile grid
       * it can form is half as fine and the smallest useful problem twice as
       * large. Require one tile per compute unit at that granularity - a stricter
       * floor than ozaki_tile_select's nunits/tile_sat, which only has to keep a
       * tile grid busy rather than justify making the grid coarser. Calibrated on
       * PVC (448 units): it promotes from n=1024, where fp64 Scheme 2 gains 14%,
       * and holds off at n=512, where promoting costs 1.5x.
       *
       * The symmetrized pair loop needs a coarser floor than either scheme's
       * plain tile grid, because the mirror product accumulates alongside the
       * pair and the promoted tile therefore holds twice the live state. On PVC
       * its crossover sits between n=1024, where the base pair leads by 1.30x in
       * fp64 (1.23 against 1.59 ms) and 1.24x in fp32, and n=1280, where the
       * promoted pair leads by 1.49x (2.05 against 3.07 ms). Three tiles per unit
       * separates them; one tile per unit would promote from n=677 and lose at
       * both measured sizes below the crossover.
       */
      const int gm = OZAKI_XMX_M(ctx) * big_m, gn = OZAKI_XMX_N(ctx) * big_n;
      const int units = (0 == crt && 0 != (ctx->ozflags & OZAKI_SYMMETRIZE)) ? 3 * ctx->nunits : ctx->nunits;
      if ((LIBXS_UPDIV(M, gm) * LIBXS_UPDIV(N, gn)) >= units) {
        result.m = big_m;
        result.n = big_n;
      }
    }
  }
  return result;
}


int ozaki_npanel(const ozaki_context_t* ctx, int M, int N, int tm, int tn)
{
  const int nblk_m = LIBXS_UPDIV(M, tm);
  const int nwg_min = (0 < ctx->nunits) ? (ctx->nunits / ctx->tile_sat) : 32;
  /**
   * Tiles per panel needed to keep the device busy: a panel is one GEMM
   * launch, so if its tile count falls below the saturation floor the pipeline
   * wins latency but loses more throughput than it hides.
   */
  const int ntile_min = LIBXS_UPDIV(nwg_min, nblk_m);
  int width = N;
  if (1 != ctx->npanel && tn < N) {
    if (0 < ctx->npanel) { /* explicit request: honor, but keep tiles whole */
      width = LIBXS_UP(ctx->npanel, tn);
    }
    else {
      /**
       * Aim for OZAKI_NSLOTS panels so every slot is used once and the
       * pipeline reaches steady state, but never below the saturation floor.
       */
      const int ntile_n = LIBXS_UPDIV(LIBXS_UPDIV(N, tn), OZAKI_NSLOTS);
      width = (ntile_n < ntile_min ? ntile_min : ntile_n) * tn;
    }
    if (width > N) width = N;
    if (width < tn) width = tn;
  }
  return width;
}


int ozaki_init(ozaki_context_t* ctx, int tm, int tn, int use_double, int kind, int verbosity, int ndecomp, int ozflags, int oztrim,
  int ozgroups, int maxk)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const int gpu = (CL_DEVICE_TYPE_GPU == devinfo->type ? 1 : 0);
  int result = EXIT_SUCCESS;
  int nv, has_fp64, crt;
  int wg, sg, use_i8;
  int nslices, nprimes, oztrim_crt;
  const char* env;
  memset(ctx, 0, sizeof(*ctx));

  /**
   * Scheme request: 1 or 2 force that scheme, anything else asks for adaptive.
   * The default for a caller that expresses no preference lives HERE and nowhere
   * else, because a driver-side default is not a default but silent policy: two
   * drivers over the same library disagreed on it (one passed 1, the other 2)
   * while a third opinion sat in this function, so the same shape on the same
   * device ran a different scheme depending on which binary asked. A driver may
   * still force a scheme; what it may not do is decide what "unspecified" means.
   * The per-call choice under adaptive is ozaki_gemm's, which is where the
   * device-side knowledge is; the CPU-side equivalent belongs to the caller.
   */
  if (0 >= kind || 3 < kind) kind = 3;
  crt = (1 != kind); /* CRT participates: forced (2) or possible (3) */

  /* CRT: no triangular/symmetrize (no cross-prime products). Not under adaptive,
   * where Scheme 1 may still run and the crossover counts its pairs. */
  if (2 == kind) {
    if (0 > ozflags) ozflags = 0;
  }

  if (0 > verbosity || 2 < verbosity) {
    char name[256] = "";
    libxstream_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /**
   * FP64 is needed for the fp64 interface only. CRT values exceed 2^24, but
   * Garner/Horner keep them in ulong and convert to real_t once, so fp32
   * Scheme 2 is exact without double; only the fractional variants use
   * double unconditionally (error-free transformations), and they fall back
   * to Garner when fp64 is absent. Scheme 1 uses no double at all.
   */
  { const char *const fp64_ext[] = {"cl_khr_fp64"};
    has_fp64 = (EXIT_SUCCESS == libxstream_opencl_device_ext(device, fp64_ext, 1));
    if (0 == has_fp64) {
      if (0 != use_double) {
        fprintf(stderr, "ERROR OZAKI: FP64 requested but device does not support cl_khr_fp64\n");
        result = EXIT_FAILURE;
      }
      else if (0 != crt && (0 > verbosity || 0 < verbosity)) {
        fprintf(stderr, "INFO OZAKI: no cl_khr_fp64, Scheme 2 uses Garner reconstruction\n");
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
    if (0 < maxk && 0 != crt) {
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
    ndecomp = (0 != crt) ? nprimes : nslices;
  } /* ndecomp_auto */
  if (0 != crt && 20 < ndecomp) ndecomp = 20;
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
    ctx->nv = (0 != gpu) ? nv : 0;
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
    int rtm = 0, rtn = 0, rtm_req = 0, rtn_req = 0, ku_req, biggrf, hier, wgmma, wgmma_rs = 0;
    int fraccrt, crt_hier, unfuse_pre;
    size_t max_wgs;
    int v;
    /**
     * Configurations where Scheme 2 cannot be dispatched, mirroring ozaki_gemm:
     * kind==1 asks for Scheme 1, and under kind==3 fp32 on a non-NVIDIA device
     * selects it outright. The Scheme-2 knobs must not shape the Scheme-1 build
     * there - hier below picks the GRF mode, which fixes the register tiling and
     * the work-group ceiling of every kernel, so an unreachable scheme would
     * otherwise compile the one that runs at half its row tiling.
     */
    const int sch1_only = (1 == kind || (3 == kind && 0 == use_double && 0 == ctx->nv));
    {
      const char *const env_hier = getenv("OZAKI_HIER");
      hier = (NULL != env_hier) ? (0 != atoi(env_hier) ? 1 : 0) : (0 != crt ? 1 : 0);
    }
    /**
     * Ozaki-local 256-GRF decision (per-kernel, not global).
     * LIBXSTREAM_BIGGRF: explicit user override for all kernels.
     * OZAKI_BIGGRF: Ozaki-specific override.
     * Default: auto-enable for Intel GPUs, but HIER prefers GRF128
     * (halved private arrays make 2x occupancy the better trade-off).
     *
     * HIER is a Scheme-2 property, so it only speaks for a build Scheme 2 can
     * reach: where the scheme is settled as 1 at init, GRF256 stands. Both
     * schemes reachable per call (fp64 under kind==3) still resolve in Scheme
     * 2's favor, which is one GRF mode serving two programs and wants the
     * per-scheme split rather than a different tie-break.
     */
    env = getenv("OZAKI_BIGGRF");
    if (NULL != env) {
      biggrf = (0 != atoi(env));
    }
    else if (NULL != getenv("LIBXSTREAM_BIGGRF")) {
      biggrf = (0 != devinfo->biggrf);
    }
    else {
      biggrf = (0 != devinfo->intel && 0 != gpu && (0 == hier || 0 != sch1_only));
    }
    LIBXS_SNPRINTF(build_options, sizeof(build_options), "-cl-fast-relaxed-math -cl-denorms-are-zero%s",
      (0 != biggrf && 0 != devinfo->intel && 0 == devinfo->biggrf) ? " -cl-intel-256-GRF-per-thread" : "");
    max_wgs = (0 != biggrf) ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
    /* Read optional user overrides for register tiling factors. */
    env = getenv("OZAKI_RTM");
    if (NULL != env && 0 < atoi(env)) {
      rtm = atoi(env);
      rtm_req = rtm; /* explicit request applies to both schemes */
    }
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
      const int ku_default = (0 == devinfo->intel && 0 == ctx->nv_mma && 2 <= nv && 0 != gpu && 0 != crt) ? 4 : 2;
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
    /**
     * Scheme-1/2 crossover weight: the reconstruction cost expressed in units of
     * one prime's int8 GEMM pass, so that `xover * P^2 / K` is comparable to a pair
     * count (see ozaki_gemm). It is reached only for fp64 on a non-NVIDIA device -
     * every other case decides the scheme without it - so this default is the Intel
     * fp64 tuning and nothing else.
     *
     * Calibrated rather than fitted, because the term has a measurable meaning: on
     * PVC at n=4096, K=4096 the unfused Garner reconstruction is 2.11 ms against a
     * 19.95 ms GEMM of 16 passes, i.e. 1.69 passes, and xover = 1.69 * K / P^2 = 27.
     * One point suffices because the ratio is independent of M and N (both terms
     * scale with M*N) and the passes are equal - measured, 16 primes to 8 halves the
     * GEMM exactly. The previous 128 dated from before the unfused epilogue halved
     * the reconstruction, and it put the switch at K<683, which mispredicted the one
     * shape that could show it: at K=256 it chose Scheme 1 and paid 1.4x.
     *
     * Two limits worth knowing. The rule assumes a Scheme-1 pass costs the same as a
     * Scheme-2 pass, and it does not - fitting the K-slope of Scheme 1 at n=4096
     * gives about 0.57 of a Scheme-2 pass, whose operands are residue planes and
     * which mod-reduces per prime - so the rule leans toward Scheme 2 at small K,
     * which is the side every direct comparison on PVC supports. And nothing below
     * K=256 has been measured, so the resulting boundary (K<144) is derived, not
     * observed.
     */
    env = getenv("OZAKI_XOVER");
    ctx->xover = (NULL != env && 0 < atof(env)) ? atof(env) : 27.0;
    ctx->hier = hier;
    ctx->maxk = maxk;
    /**
     * N-panel width (0 = auto, 1 = disabled). Independent of maxk, which sizes
     * the K pass and feeds the bounded-K prime reduction above: a pipelining
     * granularity must not silently change nprimes, so the two stay separate.
     *
     * Disabled by default: whether pipelining pays is shape-dependent (measured
     * on PVC, up to 12% slower on square shapes and up to 7% faster where N is
     * much larger than M), and the automatic width does not account for shape.
     * Enabling it by default would trade one set of shapes for another rather
     * than improve the default, so the choice is left to the caller until the
     * width can be selected per shape.
     */
    env = getenv("OZAKI_NPANEL");
    ctx->npanel = (NULL != env) ? atoi(env) : 1;
    /**
     * Scheme-1 slice blocking (1 = unblocked).  The kernel shares each loaded
     * A/B fragment across all pairs of an OZAKI_SB-wide slice block, trading
     * OZAKI_SB^2 concurrent accumulator sets for an OZAKI_SB-fold cut in load
     * messages.
     *
     * Opt-in: blocking only pays where registers are spare, which on PVC means
     * 256-GRF.  There it is decisive - at a fixed RTM=2 and 128x128 tile the
     * gemm_fused kernel goes 1676 -> 3018 GFLOPS at n=6144 (1.8x) - but under
     * the default GRF128 the row tiling is already 2, so the SB^2 accumulator
     * sets spill and the same kernel drops 2203 -> 547 GFLOPS.  fp32 gains
     * nothing either way: nslices=4 leaves too little pair redundancy.
     */
    env = getenv("OZAKI_SB");
    { const int sb = (NULL != env && 0 < atoi(env)) ? atoi(env) : 1;
      /* The kernel indexes whole blocks, so OZAKI_SB must divide NSLICES. */
      ctx->sb = (1 < sb && sb <= nslices && 0 == (nslices % sb)) ? sb : 1;
      if (1 < sb && ctx->sb != sb && 0 != verbosity) {
        fprintf(stderr, "INFO OZAKI: OZAKI_SB=%d does not divide nslices=%d - ignored\n", sb, nslices);
      }
    }
    if (0 == rtm) {
      if (0 != devinfo->intel && 0 != gpu) {
        /**
         * Slice blocking holds SB^2 accumulator sets, so it halves RTM. The
         * symmetrized pair loop takes the same base and grows into 4x2 by
         * problem size instead of standing on it: 4x2 is 1.45x (fp32) and 1.53x
         * (fp64) off the base pair at n=256 on PVC, and pinning either value
         * costs one end of the range. The square loop keeps 4x2 as its base
         * because its promotion goes on to 4x4, which the mirror cannot afford.
         */
        rtm = (0 != biggrf && 1 == ctx->sb && 0 == (ozflags & OZAKI_SYMMETRIZE)) ? 4 : 2;
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
      else if (2 <= nv && 0 != gpu && 0 != crt) {
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
    /**
     * Reconstruction choice, settled here rather than next to the CRT build below
     * because warp-group MMA requires the hierarchical variant and its geometry has
     * to be decided before the work-group clamp. Fractional CRT: mode 1 replaces
     * the whole reconstruction with a flat fractional sum (needs raw per-prime
     * residues, so the flat path; opt-in due to a magnitude domain bound). Mode 2
     * applies fractional reconstruction only per hierarchical group (each group
     * product is below 2^53, so leaf reconstruction is exact for all group values)
     * and keeps the exact hierarchical level-2 combine, so it is exact everywhere.
     * Mode 2 is the default because it is exact over the whole CRT range like
     * Garner, keeps the same group-at-a-time storage (and hence occupancy), and is
     * faster; OZAKI_FRACCRT=0 selects Garner.
     *
     * Except under NV MMA, where the ranking inverts: the fractional reconstruction
     * spends fp64 registers per accumulator, and the MMA path holds twice as many
     * accumulators per thread (crt_rtn=8 below), so Garner's integer epilogue wins
     * - measured n=4096 20.9 vs 24.6 ms (+18%), K=1024 8.3 vs 11.2 (+35%), and
     * still +4% at n=12288 where the kernel is loop-bound. Same result to the last
     * bit either way.
     */
    /**
     * The same inversion happens for a second, device-independent reason, which is
     * why the unfused epilogue has to be decided before this and not after: once
     * reconstruction is a separate pass, only HIER_NGROUPS group values are live
     * and the fp64 registers the fractional variant spends buy nothing, while its
     * double-double arithmetic still has to be paid per output. Measured on PVC at
     * n=4096, reconstruction alone is 2.09 ms with Garner against 3.50 fractional
     * for an identical GEMM, i.e. the Intel default inverts with the epilogue:
     * fused prefers fractional (26.5 vs 31.8 ms) and unfused prefers Garner
     * (22.0 vs 23.4). Same result to the last bit either way.
     */
    { const char *const env_unfuse = getenv("OZAKI_UNFUSE");
      const int unfuse_dfl = (0 != gpu) ? 1 : 0;
      const int unfuse_req = (NULL != env_unfuse) ? (0 != atoi(env_unfuse)) : unfuse_dfl;
      /* Hierarchical CRT is required, but crt_hier below needs fraccrt first, so
       * this mirrors the part of its condition that does not depend on fraccrt. */
      unfuse_pre = (0 != unfuse_req && 1 == ctx->pb && 2 > ozgroups && (0 != ctx->hier || 3 == kind)) ? 1 : 0;
      /**
       * K-grouping is the one of those conditions a caller is likely to request
       * without knowing what it costs: it forfeits the unfused epilogue, and that
       * epilogue is worth more than the grouping ever was. Measured on PVC at
       * n=2048, fp64: 2.99 ms against 6.45 with OZAKI_GROUPS=4, and the gap widens
       * with K (18.7 vs 45.0 ms at K=16384) rather than closing.
       */
      if (0 != unfuse_req && 1 < ozgroups && 0 != crt && 0 != verbosity) {
        fprintf(stderr, "WARN OZAKI: OZAKI_GROUPS=%d disables the unfused epilogue - expect ~2x\n", ozgroups);
      }
    }
    { const char *const env_fraccrt = getenv("OZAKI_FRACCRT");
      const int fraccrt_dfl = (0 != unfuse_pre || (0 != ctx->nv_mma && 0 != gpu)) ? 0 : 2;
      const int fraccrt_req = (NULL != env_fraccrt) ? atoi(env_fraccrt) : fraccrt_dfl;
      /* Fractional CRT is double-only: without fp64 fall back to Garner. */
      fraccrt = (0 == has_fp64) ? 0 : ((1 == fraccrt_req || 2 == fraccrt_req) ? fraccrt_req : 0);
      crt_hier = (1 == fraccrt) ? 0 : (0 != ctx->hier || 3 == kind || 2 == fraccrt);
      /**
       * OZAKI_HIER=0 asks for flat reconstruction, which the per-group fractional
       * CRT and the adaptive kind both require the hierarchy for - and turning the
       * hierarchy off also drops the unfused epilogue, whose default in turn flips
       * the fractional-CRT default. Three changes from one knob, none of them the
       * one it names, so say so instead of silently doing something else. Flat
       * Garner needs OZAKI_HIER=0 with OZAKI_FRACCRT=0 and kind 1 or 2.
       */
      if (NULL != getenv("OZAKI_HIER") && 0 == ctx->hier && 0 != crt_hier && 0 != verbosity) {
        fprintf(stderr, "WARN OZAKI: OZAKI_HIER=0 needs OZAKI_FRACCRT=0 and OZAKI=2 to select flat reconstruction\n");
      }
    }
    /* Residue element type, needed this early because the wgmma splice names it. */
    ctx->u8 = (0 == use_i8) ? 1 : 0;
    /**
     * Warp-group MMA: the default on Hopper (NV>=4), off elsewhere, and
     * OZAKI_WGMMA=0 opts out. A warp group computes m64 x OZAKI_WGMMA_N, which
     * fixes the geometry rather than preferring it: SG=32, RTM=1, RTN=N/8, and
     * KU>=2 so a staging round covers at least 64 bytes of K. OZAKI_WGMMA_M picks
     * how many warp groups a work-group runs. Decided here because the
     * work-group-size clamp below, the K-padding and the tile request all have to
     * see it, and it overrides an explicit OZAKI_TM/OZAKI_TN for the same reason --
     * which is also why the reachability probe has to run here and not on first
     * use: by the time a kernel is built, the geometry is no longer negotiable.
     */
    { const char *const env_wgmma = getenv("OZAKI_WGMMA");
      wgmma = ((NULL == env_wgmma || 0 != atoi(env_wgmma)) && 4 <= nv && 0 != ctx->nv_mma && 0 != gpu && 32 == sg &&
                1 == ctx->pb && 0 != crt_hier && 2 > ozgroups && 1 != fraccrt)
                ? 1
                : 0;
      if (0 != wgmma) {
        /**
         * Tile width: n128 measured 1476 int8 TOPS against n64's 1449 as a pure
         * instruction ceiling, but it also halves the global traffic per output
         * because the same A tile feeds twice the columns. n64 stays selectable
         * for the register-pressure trade (64 accumulators per thread instead of
         * 32). OZAKI_WGMMA_N picks the width; anything else falls back to 128.
         */
        const char *const env_wn = getenv("OZAKI_WGMMA_N");
        const int wn = (NULL != env_wn && 64 == atoi(env_wn)) ? 64 : 128;
        /**
         * Rows come from warp groups, not from registers: two warp groups per
         * work-group (BM=128, 256 work-items) share one staged B tile, which is
         * what cuts the residue-plane traffic - at BM=64/BN=128 a full GEMM reads
         * nprimes*(M*K*(N/BN) + K*N*(M/BM)) = 25.8 GB at n=4096, at BM=128 17.2 GB
         * - while each work-item still holds the same RTN*XMX_FRAG accumulators.
         * The price is shared memory, doubled for A, hence occupancy - and the
         * halved work-group count, which ozaki_tile_select() gives back below the
         * saturation floor. Measured against one warp group: +17.8% at n=8192,
         * +5.5% at n=4096, +4.3% at n=2048, bit-identical throughout.
         */
        const char *const env_wm = getenv("OZAKI_WGMMA_M");
        const int wm_req = (NULL != env_wm) ? atoi(env_wm) : 0;
        const int wm = (64 == wm_req || 256 == wm_req) ? wm_req : 128;
        /**
         * Operand form, the default being RS: A goes straight from global memory
         * into registers in the fragment layout the instruction expects (mma.sync's,
         * repeated per warp - see OZAKI_WGMMA_ALOAD), so it is neither copied nor
         * staged, the tile's shared memory halves and the round loses half its
         * copies. Measured against the SS form at the shipped defaults, n=4096
         * 6.92 -> 5.45 ms and n=8192 60.3 -> 44.0 (+27% and +37%), bit-identical.
         * OZAKI_WGMMA_RS=0 selects SS, which stages both operands.
         */
        const char *const env_rs = getenv("OZAKI_WGMMA_RS");
        const int wrs = (NULL != env_rs) ? (0 != atoi(env_rs)) : 1;
        /**
         * Staging depth: WBK = KU * BK bytes of K per round, hence KU wgmma issues
         * between one barrier and the next. KU=2 is the minimum (64 bytes of K) and
         * it is also the worst - at n=4096 the barrier is not amortized and the
         * whole port loses to mma.sync (16.3 against 15.4 ms). Depth is not free:
         * shared memory is 2*(BM + BN)*KU*BK bytes double-buffered, or 2*BN*KU*BK
         * with A in registers, and once that exceeds half of the SM's share only one
         * work-group stays resident.
         *
         * That is what sets the two defaults: SS peaks at KU=8 (128 KB at BM=BN=128,
         * already one work-group per SM), while RS pays only for B and can afford
         * KU=16 for the same footprint - measured 5.93 -> 5.45 ms at n=4096 and
         * 47.8 -> 44.0 at n=8192. The deeper default costs where the tile grid no
         * longer fills the device, because there the second resident work-group is
         * worth more than the depth (1024x1024x4096: 0.68 at KU=8 against 0.78), and
         * the cost tracks M*N rather than K - at M=N=4096 KU=16 leads even at
         * K=1024. OZAKI_KU remains the knob for that regime.
         */
        const int wku = (2 <= ku_req) ? ku_req : ((0 != wrs) ? 16 : 8);
        const size_t lbytes = (size_t)2 * ((0 != wrs) ? wn : (wm + wn)) * wku * bk_pre;
        wgmma = (EXIT_SUCCESS == ozaki_wgmma_probe(ctx, wn, wku * bk_pre, lbytes, wrs)) ? 1 : 0;
        if (0 == wgmma) {
          if (0 != verbosity) {
            fprintf(stderr, "INFO OZAKI: warp-group MMA not reachable on this device - using mma.sync\n");
          }
        }
        else {
          if (0 != verbosity && ((0 < tm && wm != tm) || (0 < tn && wn != tn))) {
            fprintf(stderr, "INFO OZAKI: OZAKI_WGMMA implies a %ix%i tile - OZAKI_TM/OZAKI_TN ignored\n", wm, wn);
          }
          tm = wm;
          tn = wn;
          ctx->ku = wku;
          wgmma_rs = wrs;
        }
      }
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
       * 2 and 16 rows of the tile are never covered - silently missing part
       * of C). Round the halved extent down to the granularity and stop when
       * it can no longer shrink.
       */
      /**
       * Size the ceiling for the coarsest register tiling Scheme 1 may dispatch,
       * not for its base: a promoted pair needs fewer sub-tiles over the same
       * extent, so a ceiling derived from the base caps the promoted call at a
       * tile shaped for the finer pair. On PVC that is the whole regression -
       * fp64 n=4096 measures 46.3 ms at 128x256 against 49.0 at the 128x128 the
       * base-sized ceiling allows. Widening it is safe because ozaki_tile_select
       * re-checks the work-group bound per call with the tiling actually in use.
       * Only the symmetrized loop, whose base moved, is sized this way; the
       * square loop keeps its ceiling until the same sweep has been run for it.
       */
      { const ozaki_tile_t rmax = ozaki_rtile_grow(rtm, rtn);
        const int wide = (0 != (ozflags & OZAKI_SYMMETRIZE) && 0 == rtm_req && 0 == rtn_req
          && 1 == ctx->sb && 0 != devinfo->intel && 0 != gpu && 0 == ctx->nv_mma);
        const int gm = xmx_m * (0 != wide ? rmax.m : rtm), gn = xmx_n * (0 != wide ? rmax.n : rtn);
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
      /**
       * The two flag bits reach the kernel separately, because they name two
       * independent properties of the pair loop and the host implements them that
       * way (ozaki1_int8.c: sb_start, do_mirror). A single combined value, used as
       * a boolean, made the device read every non-zero OZAKI_FLAGS as the full S^2
       * loop and OZAKI_FLAGS=0 as triangular+symmetrize - the documented meaning
       * inverted, so a host-versus-device comparison of the pair loop compared
       * opposite configurations and the default ran the loop the flag disclaims.
       */
      const int tri_jit = (0 != (ozflags & OZAKI_TRIANGULAR)) ? 1 : 0;
      const int sym_jit = (0 != (ozflags & OZAKI_SYMMETRIZE)) ? 1 : 0;
      const int cutoff_jit = 2 * (nslices - 1) - oztrim;
      size_t goff = 0;
      goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff,
        "-DBK=%d -DKU=%d -DRC=%d -DSG=%d -DINTEL=%d -DNV=%d"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DOZAKI_SB=%d"
        " -DOZAKI_TRI=%d -DOZAKI_SYM=%d -DCONSTANT=global",
        bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, nv,
        nslices, use_double, mant_bits, bias_plus_mant, bm_pre, bn_pre, bk_pre, ctx->sb, tri_jit, sym_jit);
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
        LIBXS_SNPRINTF(pp_flags, sizeof(pp_flags), "%s -DBM=%d -DBN=%d -DRTM=%d -DRTN=%d -DOZAKI_CUTOFF=%d",
          build_params, tm, tn, rtm, rtn, cutoff_jit);
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
     * adaptive dispatch can reach them. Without fp64 this needs fp32 operands
     * and Garner reconstruction: only the fractional variants carry
     * unconditional double (the error-free transformations and the leaf
     * fractional sum), whereas Garner/Horner accumulate in ulong and convert
     * to float. An fp64 device is therefore required only for the fp64
     * interface, not for fp32 Scheme 2. ozaki_gemm falls back on a NULL
     * crt_registry when the build is skipped.
     */
    if (0 != has_fp64 || 0 == use_double) {
      /* fraccrt/crt_hier were settled above, where the wgmma geometry needs them. */
      const char *const env_skip = getenv("OZAKI_SKIP_GARNER");
      /**
       * Scheme 2 has no pair loop, so slice blocking never applies to it and it
       * must not inherit the halved RTM that blocking imposes on Scheme 1.
       * rtm_crt_base is the row tiling Scheme 1 would have used unblocked.
       */
      /**
       * Warp-group MMA: RTM=1 and RTN = BN/8, because a warp group covers 64 rows
       * and the whole tile width in one instruction. Only the default configuration
       * is supported - hierarchical CRT, no K-grouping, PB=1 - which the gate
       * above enforces and the kernel restates with #error.
       */
      const int rtm_crt_base = (1 < ctx->sb && 0 == rtm_req) ? rtm * 2 : rtm;
      const int crt_rtm = (0 != wgmma)
                            ? 1
                            : ((0 != crt_hier && 0 != biggrf && 0 == ctx->hier) ? LIBXS_MAX(rtm_crt_base / 2, 1)
                                                                               : rtm_crt_base);
      /**
       * MMA gives a sub-tile 16 rows but only 8 columns, so reaching a square
       * register tile needs twice the column tiling. Scheme 1 measured -26% at
       * RTN=4 and peaks at RTN=2 - it runs a pair loop over slices and is bound
       * by the per-pair epilogue rather than by column reuse. Hence per-scheme.
       *
       * Scheme 2 wants 8: OZAKI_WGS_MAX_NV holds NVIDIA to 4 warps, which fixes
       * NTM*NTN <= 4 sub-tiles per work-group, so the column tiling is the only
       * way to spend that budget on a larger tile (RTN=4 selects 64x64, RTN=8
       * selects 64x128 at the same 128 threads). Measured against RTN=4 with the
       * same (Garner) epilogue: +38% at n=4096 (28.9 -> 20.9 ms) and +9% at
       * K=1024, bit-identical since int32 accumulation is exact. The gain is
       * bounded by K, which has to amortize the doubled per-thread prologue and
       * epilogue, so short K pays: -6% at K=512 and -21% at K=256. RTN is a JIT
       * constant while K is per-call, so trading the short-K case for the long
       * one is the only choice available without keying the CRT registry on the
       * register tiling as well. Both together (this and the Garner default
       * above) beat the previous defaults at every measured shape.
       */
      const int crt_rtn = (0 != wgmma) ? (ctx->tn_req / 8) : ((0 != ctx->nv_mma && 0 != gpu && 0 == rtn_req) ? 8 : rtn);
      char crt_build_options[128];
      size_t coff = 0;
      int bkmajor, bblock;
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
        " -DKGROUPS=%d -DPB=%d"
        " -DCONSTANT=global",
        bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, nv,
        nprimes, use_double, mant_bits, bias_plus_mant - oztrim_crt, oztrim_crt, bm_pre, bn_pre, bk_pre,
        (1 < ozgroups) ? ozgroups : 0, ctx->pb);
      if (0 != ctx->nv_mma) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DNV_MMA=1");
      }
      if (0 == use_i8) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_U8=1");
      }
      if (0 != wgmma) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_WGMMA=1");
        if (0 != wgmma_rs) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_WGMMA_RS=1");
        }
      }
      ctx->wgmma = wgmma;
      ctx->wgmma_rs = (0 != wgmma) ? wgmma_rs : 0;
      /**
       * Work-group rasterization width (0 = the launch order). The resident
       * work-groups otherwise form a column strip of the tile grid and share one B
       * panel while each reads its own A panel, which is why only the A-term of the
       * residue traffic measures. Walking the grid in strips makes a wave cover a
       * block instead.
       */
      { const char *const env_swizzle = getenv("OZAKI_SWIZZLE");
        const int swizzle = (NULL != env_swizzle && 0 < atoi(env_swizzle)) ? atoi(env_swizzle) : 0;
        if (0 != swizzle) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
            " -DOZAKI_SWIZZLE=%i", swizzle);
        }
      }
      /**
       * Unfused reconstruction: the default on any GPU and OZAKI_UNFUSE=0 opts out.
       * It requires the hierarchical epilogue (the reduce kernel implements only
       * that one), PB=1 and no K-grouping, because the unfused prime loop stores one
       * prime per pass and has nowhere to accumulate a partial K-group. Those
       * conditions were already settled above, where the fractional-CRT default
       * needs to know whether the epilogue will be a separate pass.
       *
       * The fused epilogue has to keep every output's group values live across the
       * prime loop, which is 2 KB per work-item of dynamically indexed arrays and
       * therefore served from L2 rather than from registers. Storing a residue byte
       * per prime instead and reconstructing in a second pass, with the output loop
       * outermost, keeps only HIER_NGROUPS group values live. Measured at n=4096:
       * 13.08 ms fused against 7.87 + 0.66, i.e. +53%, and the GEMM alone beats
       * even its own loop-only time (8.62 ms with the epilogue compiled out),
       * because the frame was costing the K-loop as well. Bit-identical, and it
       * gains most where the epilogue floor dominated: +158% at n=257, +76% in
       * fp32. It also helps the mma.sync path (+11%), so it is not wgmma-specific.
       *
       * It is not NVIDIA-specific either, which is what makes "any GPU" the right
       * default rather than a vendor list: the frame exists because the prime loop
       * is outermost, which every backend shares, and the reduce kernel reads the
       * fragment mapping each one already defines. Measured on PVC (DPAS, fp64,
       * with the Garner epilogue it selects above): n=257 0.66 -> 0.15 + 0.13 ms,
       * n=512 0.73 -> 0.30, n=1024 1.23 -> 0.93, n=2048 4.05 -> 3.64, n=4096
       * 26.5 -> 22.0, fp32 n=4096 15.8 -> 12.6, bit-identical throughout.
       *
       * The price is scratch memory, NPRIMES bytes per output element, which is
       * twice the size of C in fp64.
       */
      { ctx->unfuse = (0 != unfuse_pre && 0 != crt_hier) ? 1 : 0;
        if (0 != ctx->unfuse) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_UNFUSE=1");
        }
        else if (0 != unfuse_pre && 0 != verbosity) {
          fprintf(stderr, "INFO OZAKI: OZAKI_UNFUSE needs hierarchical CRT, PB=1 and no K-grouping - disabled\n");
        }
      }
      /**
       * B layout for the NVIDIA paths, three mutually exclusive variants (Intel
       * keeps the plain layout because DPAS transforms on read):
       *
       * OZAKI_BBLOCK blocks 16 consecutive K-values of a column, which is the only
       * layout that lets the warp-group staging move 16 bytes per copy while both
       * producer and consumer stay coalesced. That removes three quarters of the
       * copies, and copy count is what the loop is bound by: the GEMM measures 8.03
       * -> 6.89 ms at n=4096 while B preprocessing pays 0.53 -> 0.73, so +11% net.
       * The default wherever warp-group MMA runs, and warp-group only: the older
       * paths have no branch for it. See ozaki_common.cl.
       *
       * OZAKI_BKMAJOR transposes B to [N_pad][K_pad]. Warp-group MMA requires it
       * (both operands K-major, staged through shared memory), and it also suits
       * the older paths better than the interleave - see ozaki_common.cl.
       *
       * OZAKI_BVNNI pre-interleaves so each operand is one aligned uint: dp4a
       * gets 8 loads per column instead of 32 strided byte gathers, and the MMA
       * b-fragment becomes 2 loads instead of 8. OZAKI_BVNNI=0 opts out.
       */
      env = getenv("OZAKI_BBLOCK");
      bblock = (0 != wgmma && (NULL == env || 0 != atoi(env)));
      env = getenv("OZAKI_BKMAJOR");
      bkmajor = (0 == bblock && 0 == devinfo->intel && 2 <= nv && 0 != gpu && NULL != env && 0 != atoi(env));
      if (0 != bblock) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_BBLOCK=1");
      }
      else if (0 != bkmajor) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_BKMAJOR=1");
      }
      else {
        env = getenv("OZAKI_BVNNI");
        if (0 == devinfo->intel && 2 <= nv && 0 != gpu && (NULL == env || 0 != atoi(env))) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_BVNNI=1");
        }
      }
      if (1 == fraccrt) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_FRACCRT=1");
        coff += ozaki_emit_fraccrt(build_params + coff, sizeof(build_params) - coff,
          (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli, nprimes, 14);
      }
      else if (2 == fraccrt) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_FRACCRT=2");
        coff += ozaki_emit_fraccrt2(build_params + coff, sizeof(build_params) - coff,
          (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli, nprimes, 11, ozaki_hier_gs(nprimes));
      }
      if (NULL != env_skip && 0 != atoi(env_skip)) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DSKIP_GARNER=1");
      }
      if (0 != crt_hier) {
        const uint16_t* modtab = (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli;
        /* Leaf group size and the group count the kernel is compiled for; see ozaki_hier_gs. */
        const int hier_gs = ozaki_hier_gs(nprimes);
        const int ngroups = LIBXS_UPDIV(nprimes, hier_gs);
        uint32_t gp[OZAKI_HIER_NGROUPS_MAX];
        uint64_t l2b[OZAKI_HIER_NGROUPS_MAX];
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
          " -DOZAKI_HIER=1 -DHIER_GS=%d -DHIER_NGROUPS_ACTUAL=%d -DOZAKI_HIER_L2=%d", hier_gs, ngroups, use_tree);
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
       * Base flags for the tile-specialized CRT registry: BM/BN, RTM/RTN and
       * OZAKI_BOUNDS are appended per specialization by ozaki_get_crt_kernel.
       */
      memcpy(ctx->crt_flags, build_params, sizeof(ctx->crt_flags));
      LIBXS_SNPRINTF(ctx->crt_options, sizeof(ctx->crt_options), "%s", crt_build_options);
      ctx->crt_registry = libxs_registry_create();
      {
        char base_flags[sizeof(build_params) + 64];
        cl_program program = NULL;
        LIBXS_SNPRINTF(base_flags, sizeof(base_flags), "%s -DBM=%d -DBN=%d -DRTM=%d -DRTN=%d -DOZAKI_BOUNDS=1",
          build_params, tm, tn, crt_rtm, crt_rtn);
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
      /**
       * Growable under the same conditions as Scheme 1, plus: not the wgmma
       * geometry, and rows only. Scheme 2 loses 2.5x at a 4x4 register tile on
       * DPAS - it keeps a group-value frame per work-item across the prime loop,
       * so the columns it can afford are bounded by registers rather than by
       * reuse - hence promote only while there are rows left to grow.
       */
      ctx->crt_rtm_big = crt_rtm;
      ctx->crt_rtn_big = crt_rtn;
      if (0 == rtm_req && 0 == rtn_req && 0 == wgmma && 4 > crt_rtm
        && 0 != devinfo->intel && 0 != gpu && 0 == ctx->nv_mma)
      {
        const ozaki_tile_t big = ozaki_rtile_grow(crt_rtm, crt_rtn);
        if (tm >= OZAKI_XMX_M(ctx) * big.m && tn >= OZAKI_XMX_N(ctx) * big.n) {
          ctx->crt_rtm_big = big.m;
          ctx->crt_rtn_big = big.n;
        }
      }
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
      /**
       * Growable only where it was measured: an Intel GPU, no explicit request,
       * and no second accumulator set already claiming the registers it needs -
       * which rules out slice blocking (OZAKI_SB^2 sets). The symmetrized pair
       * loop grows too, but only one step, since its base is 2x2 and the ladder
       * reaches 4x2 and stops there; what it must not reach is 4x4, where the
       * mirror product accumulating alongside the pair measures 99.1 ms against
       * 46.5 at 256-GRF fp64 n=4096, while the square loop goes 50.1 -> 36.0 on
       * the same step - which is why the square loop bases at 4x2 instead.
       * NVIDIA keeps its tuned pair until the same sweep has been run there. The
       * ceiling tile must admit one sub-tile of the coarser granularity, otherwise
       * the promoted pair has no legal tile at all.
       */
      ctx->rtm_big = rtm;
      ctx->rtn_big = rtn;
      if (0 == rtm_req && 0 == rtn_req && 1 == ctx->sb
        && 0 != devinfo->intel && 0 != gpu && 0 == ctx->nv_mma)
      {
        const ozaki_tile_t big = ozaki_rtile_grow(rtm, rtn);
        if (tm >= OZAKI_XMX_M(ctx) * big.m && tn >= OZAKI_XMX_N(ctx) * big.n) {
          ctx->rtm_big = big.m;
          ctx->rtn_big = big.n;
        }
      }
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
  /**
   * Scratch arena budget in MB (see ozaki_scratch_t): OZAKI_ARENA=0 disables it
   * and restores per-call allocation, a positive value caps it - a call needing
   * more than the cap falls back per buffer rather than failing - and unset
   * enables it only where libxstream has no device pool of its own.
   *
   * That default is deliberate rather than cautious. Where the pool exists
   * (it is gated on USM, so in practice Intel) per-call allocation is already
   * cheap, so the arena would buy nothing while holding a large block - the
   * operand planes, the residue planes and the device copy of C - for the life
   * of the context, memory the pool would otherwise recycle for its other
   * consumers. Where the pool does not exist (NVIDIA) the same per-call
   * allocation is what dominates the wall clock, so the arena is the difference
   * between 93 ms and 6.7 ms per call at n=4096 on a GH200, and 18% on an H100.
   *
   * The two parts charge for it differently - the first write to a fresh buffer
   * on one, the release on the other - so what the arena removes is the create
   * and destroy itself rather than any one slow call. A positive OZAKI_ARENA
   * enables it either way, which is how the path stays testable on a device that
   * has a pool, and what such a device needs if its allocation is not in fact
   * cheap: a pool proves that allocations are recycled, nothing more.
   *
   * The pool is a proxy for "allocation is already cheap", not a coincidence: it
   * can never exist on NVIDIA, because the SVM capability query is skipped there
   * by an explicit vendor workaround and the Intel USM entry points are absent,
   * which is why the churn this arena removes was only ever measured on NVIDIA.
   * The proxy is imperfect in one direction - an Intel device whose USM entry
   * points do not resolve gets an arena it does not need, since that runtime
   * pools internally anyway - which costs footprint rather than speed, and
   * OZAKI_ARENA=0 settles it. It is preferred over testing the vendor because a
   * mechanism outlives a vendor list.
   *
   * Requesting the arena does not switch the pool off, and must not: the pool is
   * created by libxstream and used by libxstream_mem_* itself, so it belongs to
   * every consumer in the process rather than to this sample. Nothing is paid
   * twice for having both - the arena bypasses the pool for the buffers it
   * carves, and what still goes through the pool here (the cached operand
   * planes) persists across calls instead of churning. What both do cost is
   * footprint, and the cap above is the lever for that.
   */
  { const char *const env_arena = getenv("OZAKI_ARENA");
    const int arena = (NULL != env_arena) ? atoi(env_arena) : -1;
    if (0 > arena) {
      ctx->scratch.limit = (NULL != libxstream_opencl_config.pool_dev) ? 0 : ((size_t)-1);
    }
    else ctx->scratch.limit = (size_t)arena << 20;
    ctx->scratch.owned = 1;
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
    /* rtm/rtn print as base..promoted where the promotion applies (per call). */
    ozaki_print_opt(stderr, "rtm", ctx->rtm);
    if (ctx->rtm_big != ctx->rtm) fprintf(stderr, "..%d", ctx->rtm_big);
    if (ctx->crt_rtm != ctx->rtm || ctx->crt_rtm_big != ctx->rtm_big) {
      ozaki_print_opt(stderr, "crt_rtm", ctx->crt_rtm);
      if (ctx->crt_rtm_big != ctx->crt_rtm) fprintf(stderr, "..%d", ctx->crt_rtm_big);
    }
    ozaki_print_opt(stderr, "rtn", ctx->rtn);
    if (ctx->rtn_big != ctx->rtn) fprintf(stderr, "..%d", ctx->rtn_big);
    if (ctx->crt_rtn != ctx->rtn || ctx->crt_rtn_big != ctx->rtn_big) {
      ozaki_print_opt(stderr, "crt_rtn", ctx->crt_rtn);
      if (ctx->crt_rtn_big != ctx->crt_rtn) fprintf(stderr, "..%d", ctx->crt_rtn_big);
    }
    if (1 < ctx->sb) ozaki_print_opt(stderr, "sb", ctx->sb);
    if (0 != devinfo->intel) {
      const int crt_grf128 = (0 != ctx->crt_rtm && ctx->crt_rtm < ctx->rtm);
      ozaki_print_opt(stderr, "grf", ctx->biggrf ? 256 : 128);
      if (0 != crt_grf128) ozaki_print_opt(stderr, "crt_grf", 128);
    }
    ozaki_print_opt(stderr, "ndecomp", ndecomp);
    ozaki_print_opt(stderr, "trim", oztrim);
    if (0 != crt) {
      const char *const e_i8 = getenv("OZAKI_I8");
      fprintf(stderr, " u8=%d", (NULL == e_i8 || 0 == atoi(e_i8)) ? 1 : 0);
      ozaki_print_opt(stderr, "kgroups", ozgroups);
      ozaki_print_opt(stderr, "pb", ctx->pb);
      ozaki_print_opt(stderr, "hier", ctx->hier);
      ozaki_print_opt(stderr, "wgmma", ctx->wgmma);
      ozaki_print_opt(stderr, "wgmma_rs", ctx->wgmma_rs);
      ozaki_print_opt(stderr, "unfuse", ctx->unfuse);
    }
    ozaki_print_opt(stderr, "cache", ctx->cache.flags);
    if (0 == ctx->scratch.limit) fprintf(stderr, " arena=off");
    else if (((size_t)-1) == ctx->scratch.limit) fprintf(stderr, " arena=on");
    else fprintf(stderr, " arena=%uMB", (unsigned int)(ctx->scratch.limit >> 20));
    if (3 == kind) fprintf(stderr, " xover=%g", ctx->xover);
    fprintf(stderr, "\n");
  }

  /**
   * Create persistent helper streams and synchronization events. No profiling
   * flag: LIBXSTREAM_PROFILE enables queue timestamps by itself, so asking for
   * them here as well only offered a second, weaker switch for the same thing.
   */
  {
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(&ctx->stream_a, "ozaki_a", LIBXSTREAM_STREAM_DEFAULT);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(&ctx->stream_b, "ozaki_b", LIBXSTREAM_STREAM_DEFAULT);
    }
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_b);
  { int si;
    for (si = 0; si < OZAKI_NSLOTS && EXIT_SUCCESS == result; ++si) {
      result = libxstream_event_create(&ctx->evt_slot[si]);
    }
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_panel);

  return result;
}


void ozaki_destroy(ozaki_context_t* ctx)
{
  if (NULL != ctx) {
    if (0 != ctx->scratch.owned && NULL != ctx->scratch.ptr) {
      libxstream_mem_dev_deallocate_hint(ctx->scratch.ptr);
      ctx->scratch.ptr = NULL;
    }
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
    { int si;
      for (si = 0; si < OZAKI_NSLOTS; ++si) {
        if (NULL != ctx->evt_slot[si]) libxstream_event_destroy(ctx->evt_slot[si]);
      }
    }
    if (NULL != ctx->evt_panel) libxstream_event_destroy(ctx->evt_panel);
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
