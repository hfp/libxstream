/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"

/* Embedded kernel source (generated at build time via tool_opencl.sh) */
#include "ozaki_kernels.h"

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_INT8)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_OZAKI2_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI2_INT8)"
#endif

#if !defined(OZAKI_TINYTC_BM)
# define OZAKI_TINYTC_BM 256
#endif
#if !defined(OZAKI_TINYTC_BN)
# define OZAKI_TINYTC_BN 128
#endif
#define OZAKI_TINYTC_STR_(X) #X
#define OZAKI_TINYTC_STR(X) OZAKI_TINYTC_STR_(X)
#define OZAKI_TINYTC_CLX(TY) \
  "kernels/ozaki1_" TY "_" \
  OZAKI_TINYTC_STR(OZAKI_TINYTC_BM) "x" \
  OZAKI_TINYTC_STR(OZAKI_TINYTC_BN) ".clx"

#if defined(OZAKI_DEVPOOL)
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
  if (NULL != ctx->stream)   libxstream_stream_sync(ctx->stream);
  if (NULL != ctx->stream_a) libxstream_stream_sync(ctx->stream_a);
  if (NULL != ctx->stream_b) libxstream_stream_sync(ctx->stream_b);
  libxstream_mem_deallocate(pointer);
}
#endif

/* Embed precompiled TinyTC SPIR-V kernels (one per precision).
 * Filenames encode BM x BN so the Makefile can parse them back.
 * Build with -DOZAKI_TINYTC_EMBED after running kernels/ozaki1.sh. */
#if defined(LIBXS_INCBIN) && defined(OZAKI_TINYTC_EMBED)
LIBXS_INCBIN(ozaki_tinytc_f64, OZAKI_TINYTC_CLX("f64"), 16);
LIBXS_INCBIN(ozaki_tinytc_f32, OZAKI_TINYTC_CLX("f32"), 16);
#endif

/* Embedded production TinyTC kernels (specialized per ndecomp/trim/scheme) */
#include "ozaki_tinytc.h"


/* Internal helpers */
static void ozaki_print_opt(FILE* stream, const char* name, int val) {
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}


int ozaki_init(ozaki_context_t* ctx, int tm, int tn,
               int use_double, int kind, int verbosity,
               int ndecomp, int ozflags, int oztrim,
               int ozgroups)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const char* env;
  const int gpu = (CL_DEVICE_TYPE_GPU == devinfo->type ? 1 : 0);
  int wg, sg, use_xmx;
  int result = EXIT_SUCCESS;
  memset(ctx, 0, sizeof(*ctx));

  if (0 >= kind) kind = 1;

  /* CRT (kind=2): no XMX support (scalar only), no triangular/symmetrize */
  if (2 == kind) {
    if (0 > ozflags) ozflags = 0; /* CRT does not use triangular/symmetrize */
  }

  if (0 > verbosity || 2 < verbosity) {
    char name[256] = "";
    libxstream_opencl_device_name(
      device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char* const fp64_ext[] = { "cl_khr_fp64" };
    result = libxstream_opencl_device_ext(device, fp64_ext, 1);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr,
        "ERROR OZAKI: FP64 requested but device does not support cl_khr_fp64\n");
    }
  }

  /* Detect hardware matrix multiply support */
  { const char* const xmx_exts[] = {
      "cl_intel_subgroup_matrix_multiply_accumulate",
      "cl_intel_subgroup_2d_block_io" };
    env = getenv("OZAKI_XMX");
    if (NULL != env) {
      use_xmx = atoi(env);
    }
    else {
      use_xmx = (EXIT_SUCCESS == libxstream_opencl_device_ext(
                   device, xmx_exts, 2)) ? 1 : 0;
    }
  }

  { const int ndecomp_auto = (0 >= ndecomp);
    if (ndecomp_auto) {
      /* Scheme 1: mantissa slicing - 7 bits/slice
       *   FP32 (23-bit): 4 slices minimum, 5 for safety margin
       *   FP64 (52-bit): 8 slices
       * Scheme 2: CRT - formula req = 2*mant + 23 (accumulation headroom)
       *   FP32: 10 primes (68 bits, sufficient for typical K < 4M)
       *   FP64: 19 primes (124 bits, sufficient for typical K < 8M) */
      ndecomp = (2 == kind ? (use_double ? 19 : 10) : (use_double ? 8 : 5));
    }
    if (2 == kind) {
      /* Scheme 2: Convert trim levels to bits (unified semantics with Scheme 1).
       * Each level = 7 bits. Max levels: 7 (fp64), 3 (fp32). */
      const int mant = use_double ? 52 : 23;
      const int max_levels = mant / 7;  /* 7 for fp64, 3 for fp32 */
      const int oztrim_bits = LIBXS_MIN(oztrim, max_levels) * 7;

      if (0 < oztrim_bits && 0 != ndecomp_auto) {
        /* floor(cumulative log2) of CRT moduli products */
        static const int cumbits[20] = {
          7, 13, 20, 27, 34, 41, 48, 55, 61, 68,
          75, 81, 87, 94, 100, 106, 112, 118, 124, 130
        };
        const int req = 2 * (mant - oztrim_bits) + 23;
        int np;
        for (np = 0; np < 20 && cumbits[np] < req; ++np) {}
        ndecomp = (np < 20) ? np + 1 : 20;
      }
      /* Store as bits for kernel compilation */
      oztrim = oztrim_bits;
    }
    else if (1 == kind) {
      /* Scheme 1: cutoff = 2*(ndecomp-1) - oztrim must stay >= 0 */
      const int max_trim = 2 * (ndecomp - 1);
      if (oztrim > max_trim) oztrim = max_trim;
    }
  }
  if (2 == kind && 20 < ndecomp) ndecomp = 20;
  if (0 > ozflags) ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;

  ctx->use_double = use_double;
  ctx->use_xmx = use_xmx;
  ctx->kind = kind;
  ctx->ozflags = ozflags;
  ctx->oztrim  = oztrim;
  ctx->ndecomp = ndecomp;
  ctx->verbosity = verbosity;

  /* Environment-driven tuning */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0);
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)devinfo->wgsize[2]);

  /* 2D block I/O and SG=16 DPAS both require sub-group size 16 */
  if (use_xmx && 16 != sg) {
    if (0 > verbosity || 2 < verbosity) {
      fprintf(stderr, "INFO OZAKI: SG forced to 16 for XMX\n");
    }
    sg = 16;
  }
  ctx->sg = sg;

  /* GEMM-mode kernels (tiled K-loop path) */
  ctx->kern_preprocess_a = NULL;
  ctx->kern_preprocess_b = NULL;
  ctx->kern_fused = NULL;
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
    const int mant_bits      = use_double ? 52 : 23;
    const int bias_plus_mant = use_double ? 1075 : 150;
    int rtm = 0, rtn = 0, biggrf;
    size_t max_wgs;
    int v;
    const char* tinytc_source = NULL;
    size_t tinytc_source_kind = 0;
    char tinytc_path[512];
    int tinytc_avail = 0;
    /* Ozaki-local 256-GRF decision (per-kernel, not global).
     * LIBXSTREAM_BIGGRF: explicit user override for all kernels.
     * OZAKI_BIGGRF: Ozaki-specific override.
     * Default: auto-enable for Intel GPUs. */
    env = getenv("OZAKI_BIGGRF");
    if (NULL != env) {
      biggrf = (0 != atoi(env));
    }
    else if (NULL != getenv("LIBXSTREAM_BIGGRF")) {
      biggrf = (0 != devinfo->biggrf);
    }
    else {
      biggrf = (0 != devinfo->intel && 0 != gpu);
    }
    LIBXS_SNPRINTF(build_options, sizeof(build_options),
      "-cl-fast-relaxed-math -cl-denorms-are-zero%s",
      (0 != biggrf && 0 != devinfo->intel && 0 == devinfo->biggrf)
        ? " -cl-intel-256-GRF-per-thread" : "");
    max_wgs = (0 != biggrf)
      ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
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
    { int ku = (NULL != env && 0 < atoi(env)) ? atoi(env) : 2;
      ctx->ku = ku;
    }
    env = getenv("OZAKI_RC");
    { int rc = (NULL != env && 0 < atoi(env)) ? atoi(env) : 8;
      ctx->rc = (rc <= 4) ? 4 : 8;
    }
    env = getenv("OZAKI_PB");
    { int pb = (NULL != env && 0 < atoi(env)) ? atoi(env) : 1;
      ctx->pb = pb;
    }
    if (0 == rtm) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtm = (0 != biggrf) ? 4 : 2;
      }
      else rtm = 1;
    }
    if (0 == rtn) {
      if (0 != devinfo->intel && 0 != gpu) {
        rtn = 2;
      }
      else rtn = 1;
    }
    /* Sanitize: round down to nearest power of two. */
    v = rtm; rtm = 1; while (v > 1) { v >>= 1; rtm <<= 1; }
    v = rtn; rtn = 1; while (v > 1) { v >>= 1; rtn <<= 1; }
    if (0 >= tm) tm = 256;
    if (0 >= tn) tn = 256;
    /* Clamp tiling factors so at least one sub-tile remains per dimension. */
    while (rtm > 1 && tm / (8 * rtm) < 1) rtm >>= 1;
    while (rtn > 1 && tn / (16 * rtn) < 1) rtn >>= 1;
    /* Shrink tile to satisfy work-group size constraint. */
    while ((size_t)tm * tn / (8 * rtm * rtn) > max_wgs && (tm > 8 * rtm || tn > 16 * rtn)) {
      if (tm >= tn) tm /= 2; else tn /= 2;
    }
    if (1 == kind) {
      size_t goff = 0;
      goff += (size_t)LIBXS_SNPRINTF(
        build_params + goff, sizeof(build_params) - goff,
        "-DBM=%d -DBN=%d -DBK=%d -DKU=%d -DRC=%d -DSG=16"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DRTM=%d -DRTN=%d"
        " -DCONSTANT=global",
        tm, tn, bk_pre, ctx->ku, ctx->rc,
        ndecomp, use_double,
        mant_bits, bias_plus_mant,
        bm_pre, bn_pre, bk_pre,
        rtm, rtn);
      if (use_xmx) {
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DUSE_XMX=1");
      }
      env = getenv("OZAKI_PREFETCH");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DOZAKI_PREFETCH=1");
      }
      env = getenv("OZAKI_BOUNDS");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DOZAKI_BOUNDS=1");
      }
      env = getenv("OZAKI_SCALAR_ACC");
      if (NULL != env && '1' == *env) {
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DOZAKI_SCALAR_ACC=1");
      }
      { /* TinyTC kernel selection: disabled by default.
         * OZAKI_TINYTC=1 tries specialized embedded .clx then general,
         * OZAKI_TINYTC=<path> loads from file,
         * OZAKI_TINYTC=0 or unset disables. */
        const char* tinytc_env = getenv("OZAKI_TINYTC");
#if defined(LIBXS_INCBIN) && defined(OZAKI_TINYTC_EMBED)
        if (NULL != tinytc_env && '0' != *tinytc_env) {
          const int sq_prod = (0 != (ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE)));
          const struct ozaki_tinytc_prod_entry* e;
          for (e = ozaki_tinytc_prod; NULL != e->data; ++e) {
            if (e->use_double == use_double
              && e->ndecomp == ndecomp
              && e->oztrim == oztrim
              && (0 != e->sq) == (0 != sq_prod))
            {
              tinytc_source = e->data;
              tinytc_source_kind =
                (size_t)(e->data_end - e->data);
              tinytc_avail = 1;
              break;
            }
          }
          if (0 == tinytc_avail) {
            tinytc_source = use_double
              ? ozaki_tinytc_f64 : ozaki_tinytc_f32;
            tinytc_source_kind = (size_t)(use_double
              ? (ozaki_tinytc_f64_end - ozaki_tinytc_f64)
              : (ozaki_tinytc_f32_end - ozaki_tinytc_f32));
            tinytc_avail = 1;
          }
        }
        else
#endif
        {
          LIBXS_SNPRINTF(tinytc_path, sizeof(tinytc_path),
            "%s", tinytc_env);
          tinytc_source = tinytc_path;
          tinytc_source_kind = 1;
          tinytc_avail = 1;
        }
      }
      { const int cutoff_jit = 2 * (ndecomp - 1) - oztrim;
        const int sq_jit = ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE);
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DOZAKI_CUTOFF=%d -DOZAKI_SQ=%d", cutoff_jit, sq_jit);
      }
      LIBXS_UNUSED(goff);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      { cl_program program = NULL;
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "ozaki1", build_params, build_options,
          NULL, NULL, NULL, 0, &program);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "preprocess_a_dense",
            &ctx->kern_preprocess_a);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "preprocess_b_dense",
            &ctx->kern_preprocess_b);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "gemm_fused", &ctx->kern_fused);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "scale_beta",
            &ctx->kern_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
      }
      /* Build bounds-checked fused kernel for non-tile-aligned sizes */
      if (EXIT_SUCCESS == result) {
        cl_program program_b = NULL;
        char bp_bounds[sizeof(build_params) + 20];
        LIBXS_SNPRINTF(bp_bounds, sizeof(bp_bounds),
          "%s -DOZAKI_BOUNDS=1", build_params);
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "ozaki1_b", bp_bounds, build_options,
          NULL, NULL, NULL, 0, &program_b);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program_b, "gemm_fused", &ctx->kern_fused_bounds);
        }
        if (NULL != program_b) clReleaseProgram(program_b);
      }
      if (EXIT_SUCCESS != result) {
        if (NULL != ctx->kern_preprocess_a) {
          clReleaseKernel(ctx->kern_preprocess_a);
          ctx->kern_preprocess_a = NULL;
        }
        if (NULL != ctx->kern_preprocess_b) {
          clReleaseKernel(ctx->kern_preprocess_b);
          ctx->kern_preprocess_b = NULL;
        }
        if (NULL != ctx->kern_fused) {
          clReleaseKernel(ctx->kern_fused);
          ctx->kern_fused = NULL;
        }
        if (NULL != ctx->kern_fused_bounds) {
          clReleaseKernel(ctx->kern_fused_bounds);
          ctx->kern_fused_bounds = NULL;
        }
        if (NULL != ctx->kern_scale_beta) {
          clReleaseKernel(ctx->kern_scale_beta);
          ctx->kern_scale_beta = NULL;
        }
      }
    }
    /* TinyTC SPIR-V kernel: use libxstream_opencl_program to handle both
     * embedded binary (source_kind>1) and file path (source_kind=1). */
    ctx->kern_tinytc = NULL;
    ctx->prog_tinytc = NULL;
    if (1 == kind && 0 != tinytc_avail && NULL != tinytc_source) {
      cl_program prog = NULL;
      int tc_res = libxstream_opencl_program(
        tinytc_source_kind, tinytc_source, "ozaki1_tinytc",
        NULL /*build_params*/, build_options,
        NULL /*try*/, NULL /*try_ok*/,
        NULL /*exts*/, 0 /*num_exts*/, &prog);
      if (EXIT_SUCCESS == tc_res && NULL != prog) {
        tc_res = libxstream_opencl_kernel_query(
          prog, "ozaki1", &ctx->kern_tinytc);
        if (EXIT_SUCCESS != tc_res) {
          tc_res = libxstream_opencl_kernel_query(
            prog, "gemm_fused", &ctx->kern_tinytc);
        }
        if (EXIT_SUCCESS == tc_res) {
          ctx->prog_tinytc = prog;
        }
        else {
          clReleaseProgram(prog);
        }
      }
      if (NULL != ctx->kern_tinytc
        && (0 > verbosity || 2 < verbosity))
      {
        fprintf(stderr,
          "INFO OZAKI: TinyTC kernel loaded%s%s\n",
          1 == tinytc_source_kind ? " from " : " (embedded)",
          1 == tinytc_source_kind ? tinytc_path : "");
      }
    }
    if (2 == kind) {
      size_t coff = 0;
      coff += (size_t)LIBXS_SNPRINTF(
        build_params + coff, sizeof(build_params) - coff,
        "-DBM=%d -DBN=%d -DBK=%d -DKU=%d -DRC=%d -DSG=16"
        " -DNPRIMES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d -DMANT_TRUNC=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DKGROUPS=%d -DRTM=%d -DRTN=%d -DPB=%d"
        " -DCONSTANT=global",
        tm, tn, bk_pre, ctx->ku, ctx->rc,
        ndecomp, use_double,
        mant_bits, bias_plus_mant - oztrim, oztrim,
        bm_pre, bn_pre, bk_pre,
        (2 == kind && 1 < ozgroups) ? ozgroups : 0,
        rtm, rtn, ctx->pb);
      if (use_xmx) {
        coff += (size_t)LIBXS_SNPRINTF(
          build_params + coff, sizeof(build_params) - coff,
          " -DUSE_XMX=1");
      }
      LIBXS_UNUSED(coff);
      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", build_params);
      }
      { cl_program program = NULL;
        result = libxstream_opencl_program(
          0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8,
          "ozaki2", build_params, build_options,
          NULL, NULL, NULL, 0, &program);
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "preprocess_a_crt_dense",
            &ctx->kern_crt_preprocess_a);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "preprocess_b_crt_dense",
            &ctx->kern_crt_preprocess_b);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "gemm_crt_fused",
            &ctx->kern_crt_fused);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "scale_beta",
            &ctx->kern_crt_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
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
    else if (1 != kind) {
      fprintf(stderr, "ERROR OZAKI: unsupported kind=%d\n", kind);
      result = EXIT_FAILURE;
    }

    /* Initialize complex GEMM 3M kernels (precision-agnostic, always compiled) */
    if (EXIT_SUCCESS == result) {
      cl_program program_3m = NULL;
      char build_params_3m[512];
      FILE* kern_file;
      char* source_3m = NULL;
      size_t source_size = 0;
      int try_ok_3m = 0;
      char try_msg_3m[4096];
      try_msg_3m[0] = '\0';

      /* Build params and options with include paths for zgemm3m.cl
       * Need both relative paths since cwd could be ozaki/ or samples/ */
      LIBXS_SNPRINTF(build_params_3m, sizeof(build_params_3m),
        "-DUSE_DOUBLE=%d -I../../include -I../../../libxstream/include",
        use_double ? 1 : 0);

      /* Read zgemm3m.cl kernel source from file */
      kern_file = fopen("samples/ozaki/kernels/zgemm3m.cl", "r");
      if (NULL == kern_file) kern_file = fopen("kernels/zgemm3m.cl", "r");
      if (NULL != kern_file) {
        fseek(kern_file, 0, SEEK_END);
        source_size = (size_t)ftell(kern_file);
        fseek(kern_file, 0, SEEK_SET);
        source_3m = (char*)malloc(source_size + 1);
        if (NULL != source_3m) {
          const size_t nread = fread(source_3m, 1, source_size, kern_file);
          source_3m[nread] = '\0';
          result = libxstream_opencl_program(
            0, source_3m, "zgemm3m",
            build_params_3m, build_options,
            try_msg_3m, &try_ok_3m, NULL, sizeof(try_msg_3m), &program_3m);
          free(source_3m);
        }
        fclose(kern_file);
      }
      else if (2 < verbosity) {
        fprintf(stderr, "WARN OZAKI: zgemm3m.cl not found, complex GEMM disabled\n");
      }

      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(
          program_3m, "zgemm3m_deinterleave",
          &ctx->kern_zgemm3m_deinterleave);
      }
      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(
          program_3m, "zgemm3m_matadd",
          &ctx->kern_zgemm3m_matadd);
      }
      if (NULL != program_3m && EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel_query(
          program_3m, "zgemm3m_finalize",
          &ctx->kern_zgemm3m_finalize);
      }
      if (NULL != program_3m) clReleaseProgram(program_3m);

      /* 3M kernel failure is non-fatal - just disables complex GEMM */
      if (EXIT_SUCCESS != result) {
        if (2 < verbosity) {
          if (NULL == program_3m) {
            fprintf(stderr,
              "WARN OZAKI: 3M kernel compilation failed (fp=%d), complex GEMM disabled\n",
              use_double ? 64 : 32);
            if (3 < verbosity && 0 < try_msg_3m[0]) {
              fprintf(stderr, "  Build details: try_ok=%d\n", try_ok_3m);
              fprintf(stderr, "  Build log: %s\n", try_msg_3m);
            }
          }
          else {
            fprintf(stderr,
              "WARN OZAKI: 3M kernel query failed (fp=%d), complex GEMM disabled\n",
              use_double ? 64 : 32);
          }
        }
        if (NULL != ctx->kern_zgemm3m_deinterleave) {
          clReleaseKernel(ctx->kern_zgemm3m_deinterleave);
          ctx->kern_zgemm3m_deinterleave = NULL;
        }
        if (NULL != ctx->kern_zgemm3m_matadd) {
          clReleaseKernel(ctx->kern_zgemm3m_matadd);
          ctx->kern_zgemm3m_matadd = NULL;
        }
        if (NULL != ctx->kern_zgemm3m_finalize) {
          clReleaseKernel(ctx->kern_zgemm3m_finalize);
          ctx->kern_zgemm3m_finalize = NULL;
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

#if defined(OZAKI_DEVPOOL)
  /* Create device memory pool for async buffer reuse across ozaki_gemm calls.
   * Uses libxs_malloc_xpool with wrapped allocator/deallocator: the deallocator
   * syncs all streams before freeing (grow path only).
   * LIBXS_MALLOC_NATIVE preserves the allocator's exact pointer (no inline
   * metadata) so USM/SVM device pointers remain valid for the OpenCL runtime.
   * Requires USM shared or SVM; falls back to direct allocation otherwise. */
  ctx->devpool = NULL;
  { int pool_ok = 0;
#if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clSharedMemAllocINTEL) pool_ok = 1;
    else
#endif
#if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm) pool_ok = 1;
    else
#endif
    {
      LIBXS_UNUSED(devinfo);
    }
    if (0 != pool_ok) {
      ctx->devpool = libxs_malloc_xpool(
        ozaki_dev_allocate, ozaki_dev_deallocate, 1);
    }
  }
#endif

  /* OZAKI_CACHE: preprocessing cache bitmask (1=A, 2=B, 3=both) */
  { const char* env_cache = getenv("OZAKI_CACHE");
    ctx->cache.flags = (NULL != env_cache) ? atoi(env_cache) : 3;
  }

  /* OZAKI_PROFILE: kernel execution-time profiling */
  ctx->hist = NULL;
  ctx->profile = 0;
  { const char* env_prof = getenv("OZAKI_PROFILE");
    ctx->profile = (NULL == env_prof ? 0 : atoi(env_prof));
    if (0 != ctx->profile) {
      const libxs_hist_update_t update[] = { libxs_hist_update_avg };
      ctx->hist = libxs_hist_create(3, 1, update);
      if (NULL == ctx->hist) ctx->profile = 0;
    }
  }

  /* Report compiled kernel info */
  if (EXIT_SUCCESS == result && (0 > verbosity || 2 < verbosity)) {
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "kind", kind);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "xmx", use_xmx);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "tm", ctx->tm);
    ozaki_print_opt(stderr, "tn", ctx->tn);
    ozaki_print_opt(stderr, "rtm", ctx->rtm);
    ozaki_print_opt(stderr, "rtn", ctx->rtn);
    if (0 != devinfo->intel) {
      ozaki_print_opt(stderr, "grf", ctx->biggrf ? 256 : 128);
    }
    ozaki_print_opt(stderr, "ndecomp", ndecomp);
    ozaki_print_opt(stderr, "trim", oztrim);
    if (2 == kind) {
      ozaki_print_opt(stderr, "kgroups", ozgroups);
      ozaki_print_opt(stderr, "pb", ctx->pb);
    }
    ozaki_print_opt(stderr, "cache", ctx->cache.flags);
    fprintf(stderr, "\n");
  }

  /* Create persistent helper streams and synchronization events */
  { const int sflags = (NULL != ctx->hist
      ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT);
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(
        &ctx->stream_a, "ozaki_a", sflags);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_create(
        &ctx->stream_b, "ozaki_b", sflags);
    }
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_b);
#if defined(OZAKI_DEVPOOL)
  if (NULL != ctx->devpool) libxs_malloc_arg((libxs_malloc_pool_t*)ctx->devpool, ctx);
#endif

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
    if (NULL != ctx->kern_fused) {
      clReleaseKernel(ctx->kern_fused);
    }
    if (NULL != ctx->kern_fused_bounds) {
      clReleaseKernel(ctx->kern_fused_bounds);
    }
    if (NULL != ctx->kern_scale_beta) {
      clReleaseKernel(ctx->kern_scale_beta);
    }
    if (NULL != ctx->kern_tinytc) {
      clReleaseKernel(ctx->kern_tinytc);
    }
    if (NULL != ctx->prog_tinytc) {
      clReleaseProgram(ctx->prog_tinytc);
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
    if (NULL != ctx->kern_zgemm3m_deinterleave) {
      clReleaseKernel(ctx->kern_zgemm3m_deinterleave);
    }
    if (NULL != ctx->kern_zgemm3m_matadd) {
      clReleaseKernel(ctx->kern_zgemm3m_matadd);
    }
    if (NULL != ctx->kern_zgemm3m_finalize) {
      clReleaseKernel(ctx->kern_zgemm3m_finalize);
    }

    { /* Quiesce cache: NULL pointers under lock (prevents new hits),
       * then wait for in-flight gemm threads to finish using cached buffers. */
#if defined(OZAKI_DEVPOOL)
      libxs_malloc_pool_t* const pool = ctx->devpool;
#endif
      void *sa_sl, *sa_ex, *sb_sl, *sb_ex;
      LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      sa_sl = ctx->cache.a.d_slices; ctx->cache.a.d_slices = NULL;
      sa_ex = ctx->cache.a.d_exp;    ctx->cache.a.d_exp = NULL;
      sb_sl = ctx->cache.b.d_slices; ctx->cache.b.d_slices = NULL;
      sb_ex = ctx->cache.b.d_exp;    ctx->cache.b.d_exp = NULL;
      ctx->cache.flags = 0;
      LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);
      /* Drain active users that grabbed pointers before the NULL.
       * LIBXS_SYNC_CYCLE only tests the low bit (lock semantics),
       * so spin explicitly on the full counter value. */
      while (0 != ctx->cache.nusers) LIBXS_SYNC_PAUSE;
#if defined(OZAKI_DEVPOOL)
      OZAKI_DEV_FREE(sa_sl); OZAKI_DEV_FREE(sa_ex);
      OZAKI_DEV_FREE(sb_sl); OZAKI_DEV_FREE(sb_ex);
      (void)pool;
#else
      if (NULL != sa_sl) libxstream_mem_deallocate(sa_sl);
      if (NULL != sa_ex) libxstream_mem_deallocate(sa_ex);
      if (NULL != sb_sl) libxstream_mem_deallocate(sb_sl);
      if (NULL != sb_ex) libxstream_mem_deallocate(sb_ex);
#endif
    }

#if defined(OZAKI_DEVPOOL)
    /* Free pool before helper streams: the pool deallocator may sync streams
     * on the grow path.  Clear ctx->stream (caller-owned, possibly already
     * destroyed) so the deallocator skips it during teardown. */
    ctx->stream = NULL;
    if (NULL != ctx->devpool) {
      libxs_malloc_pool_t *const pool = (libxs_malloc_pool_t*)ctx->devpool;
      const int verbosity = libxs_get_verbosity();
      if  (0 > LIBXS_MIN(ctx->verbosity, verbosity)
        || 2 < LIBXS_MAX(ctx->verbosity, verbosity))
      {
        libxs_malloc_pool_info_t info;
        if (EXIT_SUCCESS == libxs_malloc_pool_info(pool, &info)) {
          const int peak = (int)LIBXS_UPDIV(info.peak, (size_t)1 << 20);
          const int size = (int)LIBXS_UPDIV(info.size, (size_t)1 << 20);
          printf("POOL: peak_mb=%i size_mb=%i nmallocs=%lu\n",
            peak, size, (unsigned long int)info.nmallocs);
        }
      }
      libxs_free_pool(pool);
    }
#endif
    /* Destroy persistent synchronization events */
    if (NULL != ctx->evt_prep_a) libxstream_event_destroy(ctx->evt_prep_a);
    if (NULL != ctx->evt_prep_b) libxstream_event_destroy(ctx->evt_prep_b);
    /* Destroy persistent helper streams */
    if (NULL != ctx->stream_a) libxstream_stream_destroy(ctx->stream_a);
    if (NULL != ctx->stream_b) libxstream_stream_destroy(ctx->stream_b);
    /* Report and destroy profiling histogram */
    if (NULL != ctx->hist) {
      libxs_hist_print(stderr, ctx->hist, "OZAKI PROF (GFLOPS/s)", NULL);
      libxs_hist_destroy(ctx->hist);
    }
    LIBXS_MEMZERO(ctx);
  }
}


void ozaki_invalidate_cache(ozaki_context_t* ctx, const void* a, const void* b)
{
  if (NULL != ctx) {
    LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
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
    LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);
  }
}
