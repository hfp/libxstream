/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"
#include <libxs_math.h>

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


/* Wrapped allocator for libxs_malloc_xpool: delegates to device allocator. */
static void* ozaki_dev_allocate(size_t size, const void* extra)
{
  void* result = NULL;
  LIBXS_UNUSED(extra);
  libxstream_mem_allocate(&result, size);
  return result;
}

/* Wrapped deallocator: syncs all streams before freeing device memory.
 * Only called on the pool grow path (when a larger buffer is needed). */
static void ozaki_dev_deallocate(void* pointer, const void* extra)
{
  const ozaki_context_t* ctx = (const ozaki_context_t*)extra;
  if (NULL != ctx->stream) libxstream_stream_sync(ctx->stream);
  if (NULL != ctx->stream_a) libxstream_stream_sync(ctx->stream_a);
  if (NULL != ctx->stream_b) libxstream_stream_sync(ctx->stream_b);
  libxstream_mem_deallocate(pointer);
}


/* Internal helpers */
static const uint16_t ozaki_u8_moduli[] = {211, 199, 163, 256, 251, 223, 197, 167, 243, 227, 193, 169, 241, 229, 191, 173, 239, 233, 181, 179};
static const uint16_t ozaki_i8_moduli[] = {101, 97, 59, 128, 127, 103, 89, 61, 125, 107, 83, 67, 121, 109, 81, 71, 119, 113, 79, 73};

static void ozaki_print_opt(FILE* stream, const char* name, int val)
{
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}


int ozaki_init(ozaki_context_t* ctx, int tm, int tn, int use_double, int kind, int verbosity, int ndecomp, int ozflags, int oztrim,
  int ozgroups, int maxk, int profiling)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const int gpu = (CL_DEVICE_TYPE_GPU == devinfo->type ? 1 : 0);
  int result = EXIT_SUCCESS;
  int wg, sg, use_i8;
  int nslices, nprimes, oztrim_crt;
  const char* env;
  memset(ctx, 0, sizeof(*ctx));

  if (0 >= kind) kind = 2;

  /* CRT (kind=2): no XMX support (scalar only), no triangular/symmetrize */
  if (2 == kind) {
    if (0 > ozflags) ozflags = 0; /* CRT does not use triangular/symmetrize */
  }

  if (0 > verbosity || 2 < verbosity) {
    char name[256] = "";
    libxstream_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char *const fp64_ext[] = {"cl_khr_fp64"};
    result = libxstream_opencl_device_ext(device, fp64_ext, 1);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR OZAKI: FP64 requested but device does not support cl_khr_fp64\n");
    }
  }

  /* Scheme 2 signed i8 fallback: OZAKI_I8=1 uses moduli<=128 (legacy).
   * Default (u8): moduli<=256, fewer primes for same cumulative product. */
  {
    const char *const env_i8 = getenv("OZAKI_I8");
    use_i8 = (NULL != env_i8 && 0 != atoi(env_i8));
  }
  { /* Compute nslices (Scheme 1) and nprimes (Scheme 2) independently.
     * Both are needed for adaptive scheme selection. */
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
      oztrim_crt = LIBXS_MIN(oztrim, max_levels) * 2;
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

  /* Environment-driven tuning */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0);
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)devinfo->wgsize[2]);
  if (0 >= sg) sg = (int)devinfo->wgsize[1]; /* fallback: preferred WG multiple */
  if (0 >= sg) sg = 16; /* last resort */
  /* Tile addressing requires SG=16 (XMX_N=16 columns per sub-group).
   * Intel DPAS and 2D block I/O also mandate SG=16. */
  if (16 != sg) {
    if (0 > verbosity || 2 < verbosity) {
      fprintf(stderr, "INFO OZAKI: SG forced to 16\n");
    }
    sg = 16;
  }
  ctx->sg = sg;

  /* GEMM-mode kernels (tiled K-loop path) */
  ctx->kern_preprocess_a = NULL;
  ctx->kern_preprocess_b = NULL;
  ctx->kernel_registry = NULL;
  ctx->kern_scale_beta = NULL;
  ctx->kern_crt_preprocess_a = NULL;
  ctx->kern_crt_preprocess_b = NULL;
  ctx->kern_crt_fused = NULL;
  ctx->kern_crt_scale_beta = NULL;
  if (EXIT_SUCCESS == result) { /* output tile sizes: fit SG * NTM * NTN <= max_wgs.
     * tm must be multiple of 8*RTM, tn must be multiple of 16*RTN.
     * Large GRF halves effective max work-group size. */
    const int bm_pre = 16, bn_pre = 16, bk_pre = 32;
    char build_params[1024];
    char build_options[128];
    const int mant_bits = use_double ? 52 : 23;
    const int bias_plus_mant = use_double ? 1075 : 150;
    int rtm = 0, rtn = 0, biggrf, hier;
    size_t max_wgs;
    int v;
    {
      const char *const env_hier = getenv("OZAKI_HIER");
      hier = (NULL != env_hier) ? (0 != atoi(env_hier) ? 1 : 0) : (2 == kind ? 1 : 0);
    }
    /* Ozaki-local 256-GRF decision (per-kernel, not global).
     * LIBXSTREAM_BIGGRF: explicit user override for all kernels.
     * OZAKI_BIGGRF: Ozaki-specific override.
     * Default: auto-enable for Intel GPUs, but HIER prefers GRF128
     * (halved private arrays make 2x occupancy the better trade-off). */
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
    if (NULL != env && 0 < atoi(env)) rtn = atoi(env);
    /* Choose defaults when not explicitly set:
     *  256-GRF: RTM=4 RTN=2 (8 accumulators, measured sweet spot)
     *  128-GRF: RTM=2 RTN=2 (4 accumulators)
     *  Other vendors:  RTM=1 RTN=1 (conservative) */
    env = getenv("OZAKI_KU");
    {
      int ku = (NULL != env && 0 < atoi(env)) ? atoi(env) : 2;
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
    ctx->hier = hier;
    ctx->maxk = maxk;
    if (0 == rtm) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtm = (0 != biggrf) ? 4 : 2;
      }
      else if (2 <= devinfo->nv && 0 != gpu && 2 == kind) {
        rtm = 2; /* Ozaki-2 CRT benefits from register tiling on NV */
      }
      else rtm = 1;
    }
    if (0 == rtn) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtn = 2;
      }
      else if (2 <= devinfo->nv && 0 != gpu && 2 == kind) {
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
    if (0 >= tm) tm = 256;
    if (0 >= tn) tn = 256;
    /* Clamp tiling factors so at least one sub-tile remains per dimension.
     * XMX_M=8, XMX_N=16 for all paths (Intel DPAS, NVIDIA dp4a, scalar). */
    while (rtm > 1 && tm / (8 * rtm) < 1) rtm >>= 1;
    while (rtn > 1 && tn / (16 * rtn) < 1) rtn >>= 1;
    /* Shrink tile to satisfy work-group size constraint.
     * WGS = SG * NTM * NTN = SG * (BM/(8*RTM)) * (BN/(16*RTN)). */
    { const size_t xmx_area = (size_t)(8 * rtm) * (16 * rtn);
      while ((size_t)sg * ((size_t)tm * tn / xmx_area) > max_wgs && (tm > 8 * rtm || tn > 16 * rtn)) {
        if (tm >= tn) tm /= 2;
        else tn /= 2;
      }
    }
    { /* Scheme 1: always compile preprocessing + create registry (for adaptive) */
      const int sq_jit = ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE);
      const int cutoff_jit = 2 * (nslices - 1) - oztrim;
      size_t goff = 0;
      goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff,
        "-DBM=%d -DBN=%d -DBK=%d -DKU=%d -DRC=%d -DSG=%d -DINTEL=%d -DNV=%d"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DRTM=%d -DRTN=%d"
        " -DOZAKI_SQ=%d -DCONSTANT=global",
        tm, tn, bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, (int)devinfo->nv,
        nslices, use_double, mant_bits, bias_plus_mant, bm_pre, bn_pre, bk_pre, rtm, rtn, sq_jit);
      env = getenv("OZAKI_PREFETCH");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DOZAKI_PREFETCH=1");
      }
      env = getenv("OZAKI_SCALAR_ACC");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(build_params + goff, sizeof(build_params) - goff, " -DOZAKI_SCALAR_ACC=1");
      }
      LIBXS_UNUSED(goff);
      memcpy(ctx->base_flags, build_params, sizeof(ctx->base_flags));
      LIBXS_SNPRINTF(ctx->base_options, sizeof(ctx->base_options), "%s", build_options);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      /* Compile preprocessing + scale_beta (shared, cutoff-independent) */
      {
        char pp_flags[sizeof(build_params) + 32];
        cl_program program = NULL;
        LIBXS_SNPRINTF(pp_flags, sizeof(pp_flags), "%s -DOZAKI_CUTOFF=%d", build_params, cutoff_jit);
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
    { /* Scheme 2: always compile CRT kernels (for adaptive) */
      const int crt_hier = (0 != ctx->hier || 3 == kind);
      const int crt_rtm = (0 != crt_hier && 0 != biggrf && 0 == ctx->hier) ? LIBXS_MAX(rtm / 2, 1) : rtm;
      char crt_build_options[128];
      size_t coff = 0;
      if (0 != crt_hier && 0 != biggrf && 0 == ctx->hier) {
        LIBXS_SNPRINTF(crt_build_options, sizeof(crt_build_options),
          "-cl-fast-relaxed-math -cl-denorms-are-zero");
      }
      else {
        LIBXS_SNPRINTF(crt_build_options, sizeof(crt_build_options), "%s", build_options);
      }
      coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
        "-DBM=%d -DBN=%d -DBK=%d -DKU=%d -DRC=%d -DSG=%d -DINTEL=%d -DNV=%d"
        " -DNPRIMES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d -DMANT_TRUNC=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DKGROUPS=%d -DRTM=%d -DRTN=%d -DPB=%d"
        " -DCONSTANT=global",
        tm, tn, bk_pre, ctx->ku, ctx->rc, sg, (int)devinfo->intel, (int)devinfo->nv,
        nprimes, use_double, mant_bits, bias_plus_mant - oztrim_crt, oztrim_crt, bm_pre, bn_pre, bk_pre,
        (1 < ozgroups) ? ozgroups : 0, crt_rtm, rtn, ctx->pb);
      if (0 == use_i8) {
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff, " -DOZAKI_U8=1");
      }
      if (0 != crt_hier) {
        const uint16_t* modtab = (0 == use_i8) ? ozaki_u8_moduli : ozaki_i8_moduli;
        const int hier_gs = 4, ngroups = (nprimes + hier_gs - 1) / hier_gs;
        uint32_t gp[5];
        uint64_t l2b[5];
        int gi, gj;
        for (gi = 0; gi < ngroups; ++gi) {
          const int lo = gi * hier_gs, hi = (lo + hier_gs <= nprimes) ? lo + hier_gs : nprimes;
          uint32_t p = 1;
          int k;
          for (k = lo; k < hi; ++k) p *= (uint32_t)modtab[k];
          gp[gi] = p;
          l2b[gi] = (uint64_t)(-1) / (uint64_t)p;
        }
        coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
          " -DOZAKI_HIER=1 -DHIER_NGROUPS_ACTUAL=%d", ngroups);
        for (gi = 0; gi < ngroups; ++gi) {
          coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
            " -DHIER_GPROD_%d=%uu -DHIER_L2B_%d=%luul", gi, (unsigned)gp[gi], gi, (unsigned long)l2b[gi]);
        }
        for (gi = 0; gi < ngroups; ++gi) {
          for (gj = gi + 1; gj < ngroups; ++gj) {
            coff += (size_t)LIBXS_SNPRINTF(build_params + coff, sizeof(build_params) - coff,
              " -DHIER_L2INV_%d_%d=%uu", gi, gj,
              (unsigned)libxs_mod_inverse_u32(gp[gi] % gp[gj], gp[gj]));
          }
        }
      }
      LIBXS_UNUSED(coff);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      {
        cl_program program = NULL;
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8, "ozaki2", build_params, crt_build_options, NULL, NULL, NULL, 0, &program);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_a_crt_dense", &ctx->kern_crt_preprocess_a);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "preprocess_b_crt_dense", &ctx->kern_crt_preprocess_b);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "gemm_crt_fused", &ctx->kern_crt_fused);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(program, "scale_beta", &ctx->kern_crt_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
      }
      ctx->crt_rtm = crt_rtm;
      if (EXIT_SUCCESS != result) {
        if (NULL != ctx->kern_crt_preprocess_a) {
          clReleaseKernel(ctx->kern_crt_preprocess_a);
          ctx->kern_crt_preprocess_a = NULL;
        }
        if (NULL != ctx->kern_crt_preprocess_b) {
          clReleaseKernel(ctx->kern_crt_preprocess_b);
          ctx->kern_crt_preprocess_b = NULL;
        }
        if (NULL != ctx->kern_crt_fused) {
          clReleaseKernel(ctx->kern_crt_fused);
          ctx->kern_crt_fused = NULL;
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

  /* Device memory pool for async buffer reuse across ozaki_gemm calls.
   * Controlled via OZAKI_DEVPOOL environment variable.
   * Uses libxs_malloc_xpool with wrapped allocator/deallocator: the deallocator
   * syncs all streams before freeing (grow path only).
   * LIBXS_MALLOC_NATIVE preserves the allocator's exact pointer (no inline
   * metadata) so USM/SVM device pointers remain valid for the OpenCL runtime.
   * Requires USM shared or SVM; falls back to direct allocation otherwise. */
  ctx->devpool = NULL;
  { const char *const devpool_env = getenv("OZAKI_DEVPOOL");
    if (NULL == devpool_env || 0 != atoi(devpool_env)) {
      int pool_ok = 0;
#if (1 >= LIBXSTREAM_USM)
      if (NULL != devinfo->clSharedMemAllocINTEL) pool_ok = 1;
      else
#endif
#if (0 != LIBXSTREAM_USM)
        if (0 != devinfo->usm)
        pool_ok = 1;
      else
#endif
      {
        LIBXS_UNUSED(devinfo);
      }
      if (0 != pool_ok) {
        ctx->devpool = libxs_malloc_xpool(ozaki_dev_allocate, ozaki_dev_deallocate, 1);
      }
    }
  }

  /* OZAKI_CACHE: preprocessing cache bitmask (1=A, 2=B, -1 or 3=both).
   * Default off: cache assumes matrix content at a given pointer is unchanged
   * between calls. Applications that modify matrices in-place must either
   * disable cache (0) or ensure cached matrices are truly constant.
   * The fingerprint check catches some modifications but is not exhaustive. */
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
    ozaki_print_opt(stderr, "nv", (int)devinfo->nv);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "tm", ctx->tm);
    ozaki_print_opt(stderr, "tn", ctx->tn);
    ozaki_print_opt(stderr, "rtm", ctx->rtm);
    if (ctx->crt_rtm != ctx->rtm) ozaki_print_opt(stderr, "crt_rtm", ctx->crt_rtm);
    ozaki_print_opt(stderr, "rtn", ctx->rtn);
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
  if (NULL != ctx->devpool) libxs_malloc_arg((libxs_malloc_pool_t*)ctx->devpool, ctx);

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
    if (NULL != ctx->kern_crt_fused) {
      clReleaseKernel(ctx->kern_crt_fused);
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

    { /* Quiesce cache: NULL pointers under lock (prevents new hits),
       * then wait for in-flight gemm threads to finish using cached buffers. */
      libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
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
      /* Drain active users that grabbed pointers before the NULL.
       * LIBXS_SYNC_CYCLE only tests the low bit (lock semantics),
       * so spin explicitly on the full counter value. */
      while (0 != ctx->cache.nusers) LIBXS_SYNC_PAUSE;
      OZAKI_DEV_FREE(sa_sl);
      OZAKI_DEV_FREE(sa_ex);
      OZAKI_DEV_FREE(sb_sl);
      OZAKI_DEV_FREE(sb_ex);
    }

    /* Free pool before helper streams: the pool deallocator may sync streams
     * on the grow path.  Clear ctx->stream (caller-owned, possibly already
     * destroyed) so the deallocator skips it during teardown. */
    ctx->stream = NULL;
    if (NULL != ctx->devpool) {
      libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
      const int verbosity = libxs_get_verbosity();
      if (0 > LIBXS_MIN(ctx->verbosity, verbosity) || 2 < LIBXS_MAX(ctx->verbosity, verbosity)) {
        libxs_malloc_pool_info_t info;
        if (EXIT_SUCCESS == libxs_malloc_pool_info(pool, &info)) {
          const int peak = (int)LIBXS_UPDIV(info.peak, (size_t)1 << 20);
          const int size = (int)LIBXS_UPDIV(info.size, (size_t)1 << 20);
          printf("POOL: peak_mb=%i size_mb=%i nmallocs=%lu\n", peak, size, (unsigned long int)info.nmallocs);
        }
      }
      libxs_free_pool(pool);
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
