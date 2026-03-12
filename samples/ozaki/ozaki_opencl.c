/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"

/* Embedded kernel source (generated at build time via acc_opencl.sh) */
#include "ozaki_kernels.h"

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_INT8)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_OZAKI2_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI2_INT8)"
#endif

#if defined(OZAKI_DEVPOOL)
# define OZAKI_DEV_ALLOC(PTR, SIZE) ( \
  (NULL != pool) \
    ? ((*(PTR) = libxs_malloc(pool, SIZE, LIBXS_MALLOC_NATIVE)) != NULL ? EXIT_SUCCESS : EXIT_FAILURE) \
    : libxstream_mem_allocate((void**)(PTR), SIZE))
# define OZAKI_DEV_FREE(PTR) do { \
  if (NULL != (PTR)) { \
    if (NULL != pool) libxs_free(PTR); else libxstream_mem_deallocate(PTR); \
  } \
} while (0)
#else
# define OZAKI_DEV_ALLOC(PTR, SIZE) \
  libxstream_mem_allocate((void**)(PTR), SIZE)
# define OZAKI_DEV_FREE(PTR) do { \
  if (NULL != (PTR)) libxstream_mem_deallocate(PTR); \
} while (0)
#endif

#if defined(OZAKI_DEVPOOL)
/* Wrapped allocator for libxs_malloc_xpool: delegates to device allocator. */
static void* ozaki_dev_allocate(size_t size, const void* extra)
{
  void* result = NULL;
  (void)extra;
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


/* Internal helpers */
static void ozaki_print_opt(FILE* stream, const char* name, int val) {
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}


int ozaki_init(ozaki_context_t* ctx, int bm, int bn, int bk,
               int use_double, int kind, int verbosity,
               int nslices, int batch_k,
               int ozflags, int oztrim)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  const char* env;
  int wg, sg, gpu, use_xmx, result;
  memset(ctx, 0, sizeof(*ctx));

  if (0 >= kind) kind = 1;

  /* CRT (kind=2): no XMX support (scalar only), no triangular/symmetrize */
  if (2 == kind) {
    if (0 > ozflags) ozflags = 0; /* CRT does not use triangular/symmetrize */
  }

  gpu = (CL_DEVICE_TYPE_GPU == devinfo->type) ? 1 : 0;

  if (0 > verbosity || 2 < verbosity) {
    char name[256] = "";
    libxstream_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char* const fp64_ext[] = {"cl_khr_fp64"};
    if (EXIT_SUCCESS != libxstream_opencl_device_ext(device, fp64_ext, 1)) {
      if (0 > verbosity || 1 < verbosity) {
        fprintf(stderr, "WARN: device does not support cl_khr_fp64, falling back to float\n");
      }
      use_double = 0;
    }
  }

  /* Detect hardware matrix multiply support (before choosing defaults) */
  { const char* const xmx_exts[] = {
      "cl_intel_subgroup_matrix_multiply_accumulate",
      "cl_intel_subgroup_2d_block_io"
    };
    env = getenv("OZAKI_XMX");
    if (NULL != env) {
      use_xmx = atoi(env);
    }
    else {
      use_xmx = (EXIT_SUCCESS == libxstream_opencl_device_ext(
                   device, xmx_exts, 2)) ? 1 : 0;
    }
  }

  /* Environment-driven block/batch overrides (0 = caller wants auto) */
  env = getenv("OZAKI_BM");
  if (NULL != env && 0 >= bm) { int v = atoi(env); if (0 < v) bm = v; }
  env = getenv("OZAKI_BN");
  if (NULL != env && 0 >= bn) { int v = atoi(env); if (0 < v) bn = v; }
  env = getenv("OZAKI_BK");
  if (NULL != env && 0 >= bk) { int v = atoi(env); if (0 < v) bk = v; }
  env = getenv("OZAKI_BATCH_K");
  if (NULL != env && 0 >= batch_k) { int v = atoi(env); if (0 < v) batch_k = v; }

  /* Choose smart defaults: XMX-friendly when hardware is available.
   * XMX requires BK==32 (int8), BM divisible by 8, BN divisible by 16. */
  if (0 >= bm) bm = 16;
  if (0 >= bn) bn = 16;
  if (0 >= bk) bk = (use_xmx ? 32 : 16);
  if (0 >= nslices) nslices = (2 == kind ? 17 : 8); /* CRT: 17 primes default */
  if (0 >= batch_k) batch_k = 16; /* number of BK-sized panels grouped per launch */
  if (0 > ozflags) ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;

  /* Validate XMX constraints against final block sizes.
   * DPAS SG=16: XMX_N=16, so BN must be divisible by 16. */
  if (use_xmx && (32 != bk || 0 != (bm % 8) || 0 != (bn % 16))) {
    if (0 > verbosity || 1 < verbosity) {
      fprintf(stderr, "WARN OZAKI: XMX disabled (BK=%d, BM=%d, BN=%d)\n",
              bk, bm, bn);
    }
    use_xmx = 0;
  }

  ctx->bm = bm;
  ctx->bn = bn;
  ctx->bk = bk;
  ctx->batch_k = batch_k;
  ctx->use_double = use_double;
  ctx->use_xmx = use_xmx;
  ctx->kind = kind;
  ctx->ozflags = ozflags;
  ctx->oztrim  = oztrim;
  /* For kind==2 (CRT): kgroup = 2^oztrim, clamped to [1, batch_k].
   * Larger kgroup amortises Garner across more K sub-panels. */
  { int kg = 1;
    if (2 == kind && oztrim > 0) {
      int i;
      for (i = 0; i < oztrim && kg < batch_k; ++i) kg *= 2;
    }
    ctx->kgroup = kg;
    /* KGROUP>1 accumulates dot products across panels, requiring the CRT
     * modulus M_crt to cover KGROUP * BK * (2^53)^2.  With 17 primes
     * M_crt ~ 2^112 — barely enough for one BK=32 panel.  Use the 18th
     * prime (61, already in the tables) whenever KGROUP > 1. */
    if (2 == kind && kg > 1 && nslices < 18) nslices = 18;
  }
  ctx->nslices = nslices;
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
  ctx->kern_fused_sym = NULL;
  ctx->kern_scale_beta = NULL;
  ctx->kern_crt_preprocess_a = NULL;
  ctx->kern_crt_preprocess_b = NULL;
  ctx->kern_crt_fused = NULL;
  ctx->kern_crt_scale_beta = NULL;
  { if (2 != kind) {
      /* GEMM-mode block sizes: fit SG * (gbm/8) * (gbn/16) <= max_wgs.
       * gbm must be multiple of 8, gbn must be multiple of 16.
       * Large GRF halves effective max work-group size. */
      const size_t max_wgs = (0 != devinfo->biggrf)
        ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
      int gbm = 256, gbn = 256;
      const int bm_pre = 16, bn_pre = 16, bk_pre = 32;
      char gemm_params[1024];
      char gemm_options[512];
      const int mant_bits      = use_double ? 52 : 23;
      const int bias_plus_mant = use_double ? 1075 : 150;
      int ntm, ntn;
      size_t goff = 0;
      while ((size_t)gbm * gbn / 8 > max_wgs && (gbm > 8 || gbn > 16)) {
        if (gbm >= gbn) gbm /= 2; else gbn /= 2;
      }
      ntm = gbm / 8; ntn = gbn / 16;

      LIBXS_SNPRINTF(gemm_options, sizeof(gemm_options),
        "-cl-fast-relaxed-math -cl-denorms-are-zero");

      goff += (size_t)LIBXS_SNPRINTF(gemm_params + goff, sizeof(gemm_params) - goff,
        "-DGPU -DBM=%d -DBN=%d -DBK=%d -DSG=16"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DTRIANGULAR=%d -DCONSTANT=global",
        gbm, gbn, bk_pre,
        nslices, use_double,
        mant_bits, bias_plus_mant,
        bm_pre, bn_pre, bk_pre,
        (ozflags & OZAKI_TRIANGULAR) ? 1 : 0);
      if (use_xmx) {
        goff += (size_t)LIBXS_SNPRINTF(gemm_params + goff, sizeof(gemm_params) - goff,
          " -DUSE_XMX=1");
      }
      (void)goff;

      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", gemm_params);
      }

      result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
        "preprocess_a_dense", gemm_params, gemm_options,
        NULL, NULL, NULL, 0, &ctx->kern_preprocess_a);
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "preprocess_b_dense", gemm_params, gemm_options,
          NULL, NULL, NULL, 0, &ctx->kern_preprocess_b);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "gemm_fused", gemm_params, gemm_options,
          NULL, NULL, NULL, 0, &ctx->kern_fused);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "gemm_fused_sym", gemm_params, gemm_options,
          NULL, NULL, NULL, 0, &ctx->kern_fused_sym);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8,
          "scale_beta", gemm_params, gemm_options,
          NULL, NULL, NULL, 0, &ctx->kern_scale_beta);
      }
      if (EXIT_SUCCESS == result) {
        ctx->gbm = gbm;
        ctx->gbn = gbn;
        ctx->bm_pre = bm_pre;
        ctx->bn_pre = bn_pre;
        ctx->bk_pre = bk_pre;
        if (0 > verbosity || 2 < verbosity) {
          fprintf(stderr, "INFO OZAKI: GBM=%d GBN=%d NTM=%d NTN=%d\n",
            gbm, gbn, ntm, ntn);
        }
      }
      else {
        if (0 != verbosity) {
          fprintf(stderr, "ERROR OZAKI: GEMM kernel build failed\n");
        }
        /* Clean up any partially-built kernels */
        if (NULL != ctx->kern_preprocess_a) { clReleaseKernel(ctx->kern_preprocess_a); ctx->kern_preprocess_a = NULL; }
        if (NULL != ctx->kern_preprocess_b) { clReleaseKernel(ctx->kern_preprocess_b); ctx->kern_preprocess_b = NULL; }
        if (NULL != ctx->kern_fused)        { clReleaseKernel(ctx->kern_fused);        ctx->kern_fused = NULL; }
        if (NULL != ctx->kern_fused_sym)    { clReleaseKernel(ctx->kern_fused_sym);    ctx->kern_fused_sym = NULL; }
        if (NULL != ctx->kern_scale_beta)        { clReleaseKernel(ctx->kern_scale_beta);        ctx->kern_scale_beta = NULL; }
      }
    }
    /* CRT GEMM (kind==2): single fused kernel per tile, all primes internal */
    if (2 == kind) {
      /* Large GRF halves effective max work-group size. */
      const size_t max_wgs = (0 != devinfo->biggrf)
        ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
      int gbm = 256, gbn = 256;
      const int bm_pre = 16, bn_pre = 16, bk_pre = 32;
      char crt_params[1024];
      char crt_options[512];
      const int mant_bits      = use_double ? 52 : 23;
      const int bias_plus_mant = use_double ? 1075 : 150;
      int ntm, ntn;
      /* KGROUP_CRT: intermediate mod reduction every kgroup_crt*BK steps.
       * 0 = no intermediate reductions (safe for K <= ~133K).
       * For large K, set KGROUP_CRT = 4096/BK = 128 (safe up to any K). */
      int kgroup_crt = 0;
      size_t coff = 0;
      while ((size_t)gbm * gbn / 8 > max_wgs && (gbm > 8 || gbn > 16)) {
        if (gbm >= gbn) gbm /= 2; else gbn /= 2;
      }
      ntm = gbm / 8; ntn = gbn / 16;

      { const char* env_kg = getenv("OZAKI_KGROUP_CRT");
        if (NULL != env_kg) kgroup_crt = atoi(env_kg);
      }

      LIBXS_SNPRINTF(crt_options, sizeof(crt_options),
        "-cl-fast-relaxed-math -cl-denorms-are-zero");

      coff += (size_t)LIBXS_SNPRINTF(crt_params + coff, sizeof(crt_params) - coff,
        "-DGPU -DBM=%d -DBN=%d -DBK=%d -DSG=16"
        " -DNPRIMES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DKGROUP_CRT=%d -DCONSTANT=global",
        gbm, gbn, bk_pre,
        nslices, use_double,
        mant_bits, bias_plus_mant,
        bm_pre, bn_pre, bk_pre,
        kgroup_crt);
      if (use_xmx) {
        coff += (size_t)LIBXS_SNPRINTF(crt_params + coff, sizeof(crt_params) - coff,
          " -DUSE_XMX=1");
      }
      (void)coff;

      if (0 > verbosity || 2 < verbosity) {
        fprintf(stderr, "INFO OZAKI: %s\n", crt_params);
      }

      result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8,
        "preprocess_a_crt_dense", crt_params, crt_options,
        NULL, NULL, NULL, 0, &ctx->kern_crt_preprocess_a);
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8,
          "preprocess_b_crt_dense", crt_params, crt_options,
          NULL, NULL, NULL, 0, &ctx->kern_crt_preprocess_b);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8,
          "gemm_crt_fused", crt_params, crt_options,
          NULL, NULL, NULL, 0, &ctx->kern_crt_fused);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_kernel(0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8,
          "scale_beta", crt_params, crt_options,
          NULL, NULL, NULL, 0, &ctx->kern_crt_scale_beta);
      }
      if (EXIT_SUCCESS == result) {
        ctx->gbm = gbm;
        ctx->gbn = gbn;
        ctx->bm_pre = bm_pre;
        ctx->bn_pre = bn_pre;
        ctx->bk_pre = bk_pre;
        if (0 > verbosity || 2 < verbosity) {
          fprintf(stderr, "INFO OZAKI: GBM=%d GBN=%d NTM=%d NTN=%d\n",
            gbm, gbn, ntm, ntn);
        }
      }
      else {
        if (0 != verbosity) {
          fprintf(stderr, "ERROR OZAKI: CRT-GEMM kernel build failed\n");
        }
        if (NULL != ctx->kern_crt_preprocess_a) { clReleaseKernel(ctx->kern_crt_preprocess_a); ctx->kern_crt_preprocess_a = NULL; }
        if (NULL != ctx->kern_crt_preprocess_b) { clReleaseKernel(ctx->kern_crt_preprocess_b); ctx->kern_crt_preprocess_b = NULL; }
        if (NULL != ctx->kern_crt_fused)        { clReleaseKernel(ctx->kern_crt_fused);        ctx->kern_crt_fused = NULL; }
        if (NULL != ctx->kern_crt_scale_beta)   { clReleaseKernel(ctx->kern_crt_scale_beta);   ctx->kern_crt_scale_beta = NULL; }
      }
    }
  }

  /* Report compiled kernel info */
  if (EXIT_SUCCESS == result && (0 > verbosity || 2 < verbosity)) {
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "kind", kind);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "xmx", use_xmx);
    ozaki_print_opt(stderr, "gemm", NULL != ctx->kern_fused
      ? 1 : (NULL != ctx->kern_crt_fused ? 2 : 0));
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "nslices", nslices);
    ozaki_print_opt(stderr, "batch_k", batch_k);
    if (2 == kind) ozaki_print_opt(stderr, "kgroup", ctx->kgroup);
    fprintf(stderr, "\n");
  }

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
    { (void)devinfo; }
    if (0 != pool_ok) {
      ctx->devpool = libxs_malloc_xpool(
        ozaki_dev_allocate, ozaki_dev_deallocate, 1);
    }
  }
#endif

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

  /* Create persistent helper streams and synchronization events */
  { const int sflags = (NULL != ctx->hist ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT);
    if (EXIT_SUCCESS == result) result = libxstream_stream_create(&ctx->stream_a, "ozaki_a", sflags);
    if (EXIT_SUCCESS == result) result = libxstream_stream_create(&ctx->stream_b, "ozaki_b", sflags);
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_b);
#if defined(OZAKI_DEVPOOL)
  if (NULL != ctx->devpool) libxs_malloc_arg((libxs_malloc_pool_t*)ctx->devpool, ctx);
#endif
  if (EXIT_SUCCESS != result) ozaki_destroy(ctx);

  return result;
}


void ozaki_destroy(ozaki_context_t* ctx)
{
  if (NULL != ctx) {
    if (NULL != ctx->kern_preprocess_a) clReleaseKernel(ctx->kern_preprocess_a);
    if (NULL != ctx->kern_preprocess_b) clReleaseKernel(ctx->kern_preprocess_b);
    if (NULL != ctx->kern_fused)        clReleaseKernel(ctx->kern_fused);
    if (NULL != ctx->kern_fused_sym)    clReleaseKernel(ctx->kern_fused_sym);
    if (NULL != ctx->kern_scale_beta)        clReleaseKernel(ctx->kern_scale_beta);
    if (NULL != ctx->kern_crt_preprocess_a) clReleaseKernel(ctx->kern_crt_preprocess_a);
    if (NULL != ctx->kern_crt_preprocess_b) clReleaseKernel(ctx->kern_crt_preprocess_b);
    if (NULL != ctx->kern_crt_fused)        clReleaseKernel(ctx->kern_crt_fused);
    if (NULL != ctx->kern_crt_scale_beta)   clReleaseKernel(ctx->kern_crt_scale_beta);

#if defined(OZAKI_DEVPOOL)
    /* Free pool before helper streams: the pool deallocator may sync streams
     * on the grow path.  Clear ctx->stream (caller-owned, possibly already
     * destroyed) so the deallocator skips it during teardown. */
    ctx->stream = NULL;
    if (NULL != ctx->devpool) {
      libxs_malloc_pool_t *const pool = (libxs_malloc_pool_t*)ctx->devpool;
      if (0 > ctx->verbosity || 2 < ctx->verbosity) {
        libxs_malloc_pool_info_t info;
        if (EXIT_SUCCESS == libxs_malloc_pool_info(pool, &info)) {
          const int size = (int)LIBXS_UPDIV(info.size, (size_t)1 << 20);
          printf("POOL: size_mb=%i nmallocs=%lu\n", size,
            (unsigned long int)info.nmallocs);
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


int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream,
               char transa, char transb,
               int M, int N, int K,
               double alpha, const void* a, int lda,
                             const void* b, int ldb,
               double beta,        void* c, int ldc)
{
  const libxstream_opencl_stream_t* str = stream;
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);

  /* Persistent helper streams and events from context */
  libxstream_stream_t *stream_a = ctx->stream_a;
  libxstream_stream_t *stream_b = ctx->stream_b;
  libxstream_event_t *evt_prep_a = ctx->evt_prep_a;
  libxstream_event_t *evt_prep_b = ctx->evt_prep_b;
  size_t c_nbytes;
  int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;

#if defined(OZAKI_DEVPOOL)
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
  ctx->stream = stream; /* expose to deallocate wrapper */
#endif


  /* GEMM path (Scheme 1): full-split-then-tiled-GEMM.
   * Preprocesses entire K dimension up front into dense per-slice
   * int8 matrices, then runs a proper tiled GEMM per slice pair. */
  if (NULL != ctx->kern_fused && 0 < K) {
    const int nslices_g = ctx->nslices;
    const int BK_PRE = ctx->bk_pre;
    const int BM_PRE = ctx->bm_pre;
    const int BN_PRE = ctx->bn_pre;
    const int GBM = ctx->gbm, GBN = ctx->gbn;
    /* Pad K to multiple of BK_PRE (=32) and ensure >= 64 for 2D block I/O */
    int K_pad = ((K + BK_PRE - 1) / BK_PRE) * BK_PRE;
    const int M_pad = ((M + BM_PRE - 1) / BM_PRE) * BM_PRE;
    int N_pad = ((N + BN_PRE - 1) / BN_PRE) * BN_PRE;
    const int nblk_gm = (M + GBM - 1) / GBM;
    const int nblk_gn = (N + GBN - 1) / GBN;
    const int ntm = GBM / 8, ntn = GBN / 16;
    const int triangular = (ctx->ozflags & OZAKI_TRIANGULAR) ? 1 : 0;
    const int symmetrize = (ctx->ozflags & OZAKI_SYMMETRIZE) ? 1 : 0;
    const int cutoff = 2 * (nslices_g - 1) - ctx->oztrim;
    /* Dense slice buffer sizes */
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    void *h_as = NULL, *h_expa = NULL, *h_bs = NULL, *h_expb = NULL;
    int sa, sb, first_pair;
    int n_profiled = 0;
    cl_event *evt_prof = NULL;

    if (K_pad < 64) K_pad = 64;
    if (N_pad < 64) N_pad = 64;

    as_size   = (size_t)nslices_g * M_pad * K_pad;
    bs_size   = (size_t)nslices_g * K_pad * N_pad;
    expa_size = (size_t)M * sizeof(cl_int);   /* int for atomic_max */
    expb_size = (size_t)N * sizeof(cl_int);
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    /* Allocate device memory (skip full-matrix buffers for host-preprocessed sides) */
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_as, as_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);

    /* H2D transfers (skip full-matrix H2D for host-preprocessed sides) */
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    /* Zero exponent arrays and slice buffers (skip for host-preprocessed sides) */
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_zero(d_expb_g, 0, expb_size, stream_b);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs, 0, bs_size, stream_b);
    }

    /* Wait for H2D to finish before preprocessing */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

    /* Profiling: allocate event array (pre_a + pre_b + slice_pairs) */
    LIBXS_ASSERT(0 == ctx->profile || NULL != ctx->hist);
    if (NULL != ctx->hist) {
      evt_prof = (cl_event*)calloc((size_t)(nslices_g * nslices_g) + 2, sizeof(cl_event));
      if (EXIT_SUCCESS == result) result = libxstream_stream_set_profiling(stream);
    }

    /* Phase 1: Preprocess A (host callback or GPU kernel on stream_a) */
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
      h_as = calloc(as_size, 1);
      h_expa = calloc(expa_size, 1);
      if (NULL != h_as && NULL != h_expa) {
        ctx->host_preprocess_a(a, lda, ta, M, K, K_pad, M_pad,
          nslices_g, ctx->use_xmx, h_as, h_expa);
        result = libxstream_mem_copy_h2d(h_as, d_as, as_size, stream_a);
        if (EXIT_SUCCESS == result) {
          result = libxstream_mem_copy_h2d(h_expa, d_expa_g, expa_size, stream_a);
        }
      }
      else result = EXIT_FAILURE;
    }
    else if (EXIT_SUCCESS == result) {
      const libxstream_opencl_stream_t* str_a = stream_a;
      size_t global_a[2], local_a[2];
      const int nblk_m_pre = (M + BM_PRE - 1) / BM_PRE;
      local_a[0] = BM_PRE; local_a[1] = BK_PRE;
      global_a[0] = (size_t)nblk_m_pre * BM_PRE;
      global_a[1] = BK_PRE; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_ag));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &lda));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &ta));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_as));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_expa_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &K_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &M_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_preprocess_a, 2,
        NULL, global_a, local_a, 0, NULL,
        (NULL != evt_prof && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof + n_profiled) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof
        && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
    }

    /* Phase 1: Preprocess B (host callback or GPU kernel on stream_b) */
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
      h_bs = calloc(bs_size, 1);
      h_expb = calloc(expb_size, 1);
      if (NULL != h_bs && NULL != h_expb) {
        ctx->host_preprocess_b(b, ldb, tb, N, K, K_pad, N_pad,
          nslices_g, ctx->use_xmx, h_bs, h_expb);
        result = libxstream_mem_copy_h2d(h_bs, d_bs, bs_size, stream_b);
        if (EXIT_SUCCESS == result) {
          result = libxstream_mem_copy_h2d(h_expb, d_expb_g, expb_size, stream_b);
        }
      }
      else result = EXIT_FAILURE;
    }
    else if (EXIT_SUCCESS == result) {
      const libxstream_opencl_stream_t* str_b = stream_b;
      size_t global_b[2], local_b[2];
      const int nblk_n_pre = (N + BN_PRE - 1) / BN_PRE;
      local_b[0] = BN_PRE; local_b[1] = BK_PRE;
      global_b[0] = (size_t)nblk_n_pre * BN_PRE;
      global_b[1] = BK_PRE; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_bg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &ldb));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &tb));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_bs));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_expb_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &K_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &N_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_b->queue, ctx->kern_preprocess_b, 2,
        NULL, global_b, local_b, 0, NULL,
        (NULL != evt_prof && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof + n_profiled) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof
        && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
    }

    /* Wait for both preprocessing to complete */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Scale C by beta */
    if (EXIT_SUCCESS == result && 1.0 != beta) {
      size_t global_s[2], local_s[2];
      local_s[0] = (size_t)BM_PRE; local_s[1] = 1;
      global_s[0] = (size_t)((M + BM_PRE - 1) / BM_PRE) * BM_PRE;
      global_s[1] = (size_t)N;
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_scale_beta, i++, d_cg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_scale_beta, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_scale_beta, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_scale_beta, i++, sizeof(int), &ldc));
        if (ctx->use_double) {
          double dbeta = beta;
          CL_CHECK(result, clSetKernelArg(ctx->kern_scale_beta, i++, sizeof(double), &dbeta));
        }
        else {
          float fbeta = (float)beta;
          CL_CHECK(result, clSetKernelArg(ctx->kern_scale_beta, i++, sizeof(float), &fbeta));
        }
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_scale_beta, 2,
        NULL, global_s, local_s, 0, NULL, NULL));
    }

    /* Tiled GEMM per slice pair */
    first_pair = (0.0 == beta) ? 1 : 0;
    for (sa = 0; sa < nslices_g && sa <= cutoff && EXIT_SUCCESS == result; ++sa) {
      const int sb_start = triangular ? sa : 0;
      const int sb_end_raw = cutoff + 1 - sa;
      const int sb_end = (sb_end_raw < nslices_g) ? sb_end_raw : nslices_g;
      for (sb = sb_start; sb < sb_end && EXIT_SUCCESS == result; ++sb) {
        /* Pointers to this slice pair's dense matrices */
        const size_t as_offset_sa = (size_t)sa * M_pad * K_pad;
        const size_t bs_offset_sb = (size_t)sb * K_pad * N_pad;

        size_t global_g[2], local_g[2];
        local_g[0] = 16; /* GEMM tile decomposition requires SG=16 */
        local_g[1] = (size_t)(ntm * ntn);
        global_g[0] = (size_t)nblk_gm * local_g[0];
        global_g[1] = (size_t)nblk_gn * local_g[1];

        if (symmetrize && sa != sb) {
          /* Symmetric path: compute (sa,sb) and (sb,sa) in one launch */
          const size_t as_offset_sb = (size_t)sb * M_pad * K_pad;
          const size_t bs_offset_sa = (size_t)sa * K_pad * N_pad;
          cl_int i = 0;
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++,
            (char*)d_as + as_offset_sa));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++,
            (char*)d_bs + bs_offset_sb));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++,
            (char*)d_as + as_offset_sb));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++,
            (char*)d_bs + bs_offset_sa));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++, d_expa_g));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++, d_expb_g));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused_sym, i++, d_cg));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &M));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &N));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &K_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &N_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &ldc));
          if (ctx->use_double) {
            double dalpha = alpha;
            CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(double), &dalpha));
          }
          else {
            float falpha = (float)alpha;
            CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(float), &falpha));
          }
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &sa));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &sb));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &first_pair));
          CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_fused_sym, 2,
              NULL, global_g, local_g, 0, NULL,
              (NULL != evt_prof && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile))
                ? (evt_prof + n_profiled) : NULL));
          if (EXIT_SUCCESS == result && NULL != evt_prof
              && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
        }
        else {
          /* Single pair (sa,sb) — diagonal or non-symmetric */
          cl_int i = 0;
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused, i++,
            (char*)d_as + as_offset_sa));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused, i++,
            (char*)d_bs + bs_offset_sb));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused, i++, d_expa_g));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused, i++, d_expb_g));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_fused, i++, d_cg));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &M));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &N));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &K_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &N_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &ldc));
          if (ctx->use_double) {
            double dalpha = alpha;
            CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(double), &dalpha));
          }
          else {
            float falpha = (float)alpha;
            CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(float), &falpha));
          }
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &sa));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &sb));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &first_pair));
          CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_fused, 2,
              NULL, global_g, local_g, 0, NULL,
              (NULL != evt_prof && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile))
                ? (evt_prof + n_profiled) : NULL));
          if (EXIT_SUCCESS == result && NULL != evt_prof
              && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
        }
        first_pair = 0;
      }
    }


    /* Collect profiling data */
    if (NULL != evt_prof) {
      int resprof = clWaitForEvents((cl_uint)n_profiled, evt_prof);
      double total = 0;
      { int pi;
        for (pi = 0; pi < n_profiled && EXIT_SUCCESS == resprof; ++pi) {
          total += libxstream_opencl_duration(evt_prof[pi], &resprof);
        }
        for (pi = 0; pi < n_profiled; ++pi) {
          if (NULL != evt_prof[pi]) clReleaseEvent(evt_prof[pi]);
        }
      }
      if (EXIT_SUCCESS == resprof && 0 < total) {
        const double gflops = (2.0 * M * N * K) / (total * 1E9);
        libxs_hist_push(NULL, ctx->hist, &gflops);
      }
      free(evt_prof);
    }
    /* D2H result and cleanup */
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);

    OZAKI_DEV_FREE(d_ag); OZAKI_DEV_FREE(d_bg); OZAKI_DEV_FREE(d_cg);
    OZAKI_DEV_FREE(d_as); OZAKI_DEV_FREE(d_bs);
    OZAKI_DEV_FREE(d_expa_g); OZAKI_DEV_FREE(d_expb_g);

    return result;
  }
  /* End Scheme-1 GEMM path */

  /* CRT GEMM path (Scheme 2): full-split-then-single-fused-GEMM.
   * Preprocesses entire K into dense per-prime CRT residue matrices,
   * then runs a single kernel per tile that loops over all primes
   * internally (full-K DPAS + Garner + Horner in one launch). */
  if (NULL != ctx->kern_crt_fused && 0 < K) {
    const int nprimes_g = ctx->nslices;
    const int BK_PRE = ctx->bk_pre;
    const int BM_PRE = ctx->bm_pre;
    const int BN_PRE = ctx->bn_pre;
    const int GBM = ctx->gbm, GBN = ctx->gbn;
    int K_pad = ((K + BK_PRE - 1) / BK_PRE) * BK_PRE;
    const int M_pad = ((M + BM_PRE - 1) / BM_PRE) * BM_PRE;
    int N_pad = ((N + BN_PRE - 1) / BN_PRE) * BN_PRE;
    const int nblk_gm = (M + GBM - 1) / GBM;
    const int nblk_gn = (N + GBN - 1) / GBN;
    const int ntm = GBM / 8, ntn = GBN / 16;
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    void *h_as = NULL, *h_expa = NULL, *h_bs = NULL, *h_expb = NULL;
    int first_tile;
    int n_profiled_c = 0;
    cl_event *evt_prof_c = NULL;

    (void)ntm; (void)ntn;
    if (K_pad < 64) K_pad = 64;
    if (N_pad < 64) N_pad = 64;

    as_size   = (size_t)nprimes_g * M_pad * K_pad;
    bs_size   = (size_t)nprimes_g * K_pad * N_pad;
    expa_size = (size_t)M * sizeof(cl_int);
    expb_size = (size_t)N * sizeof(cl_int);
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_as, as_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);

    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_zero(d_expb_g, 0, expb_size, stream_b);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs, 0, bs_size, stream_b);
    }

    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

    /* Profiling: allocate event array (pre_a + pre_b + crt_gemm) */
    LIBXS_ASSERT(0 == ctx->profile || NULL != ctx->hist);
    if (NULL != ctx->hist) {
      evt_prof_c = (cl_event*)calloc(3, sizeof(cl_event));
      if (EXIT_SUCCESS == result) result = libxstream_stream_set_profiling(stream);
    }

    /* Preprocess A (host callback or CRT GPU kernel on stream_a) */
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
      h_as = calloc(as_size, 1);
      h_expa = calloc(expa_size, 1);
      if (NULL != h_as && NULL != h_expa) {
        ctx->host_preprocess_a(a, lda, ta, M, K, K_pad, M_pad,
          nprimes_g, ctx->use_xmx, h_as, h_expa);
        result = libxstream_mem_copy_h2d(h_as, d_as, as_size, stream_a);
        if (EXIT_SUCCESS == result) {
          result = libxstream_mem_copy_h2d(h_expa, d_expa_g, expa_size, stream_a);
        }
      }
      else result = EXIT_FAILURE;
    }
    else if (EXIT_SUCCESS == result) {
      const libxstream_opencl_stream_t* str_a = stream_a;
      size_t global_a[2], local_a[2];
      const int nblk_m_pre = (M + BM_PRE - 1) / BM_PRE;
      local_a[0] = BM_PRE; local_a[1] = BK_PRE;
      global_a[0] = (size_t)nblk_m_pre * BM_PRE;
      global_a[1] = BK_PRE; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_ag));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &lda));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &ta));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_as));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_expa_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &K_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &M_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_crt_preprocess_a, 2,
        NULL, global_a, local_a, 0, NULL,
        (NULL != evt_prof_c && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof_c + n_profiled_c) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof_c
        && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile)) ++n_profiled_c;
    }

    /* Preprocess B (host callback or CRT GPU kernel on stream_b) */
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
      h_bs = calloc(bs_size, 1);
      h_expb = calloc(expb_size, 1);
      if (NULL != h_bs && NULL != h_expb) {
        ctx->host_preprocess_b(b, ldb, tb, N, K, K_pad, N_pad,
          nprimes_g, ctx->use_xmx, h_bs, h_expb);
        result = libxstream_mem_copy_h2d(h_bs, d_bs, bs_size, stream_b);
        if (EXIT_SUCCESS == result) {
          result = libxstream_mem_copy_h2d(h_expb, d_expb_g, expb_size, stream_b);
        }
      }
      else result = EXIT_FAILURE;
    }
    else if (EXIT_SUCCESS == result) {
      const libxstream_opencl_stream_t* str_b = stream_b;
      size_t global_b[2], local_b[2];
      const int nblk_n_pre = (N + BN_PRE - 1) / BN_PRE;
      local_b[0] = BN_PRE; local_b[1] = BK_PRE;
      global_b[0] = (size_t)nblk_n_pre * BN_PRE;
      global_b[1] = BK_PRE; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_bg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &ldb));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &tb));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_bs));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_expb_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &K_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &N_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_b->queue, ctx->kern_crt_preprocess_b, 2,
        NULL, global_b, local_b, 0, NULL,
        (NULL != evt_prof_c && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof_c + n_profiled_c) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof_c
        && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile)) ++n_profiled_c;
    }

    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Scale C by beta */
    if (EXIT_SUCCESS == result && 1.0 != beta) {
      size_t global_s[2], local_s[2];
      local_s[0] = (size_t)BM_PRE; local_s[1] = 1;
      global_s[0] = (size_t)((M + BM_PRE - 1) / BM_PRE) * BM_PRE;
      global_s[1] = (size_t)N;
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_scale_beta, i++, d_cg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_scale_beta, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_scale_beta, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_scale_beta, i++, sizeof(int), &ldc));
        if (ctx->use_double) {
          double dbeta = beta;
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_scale_beta, i++, sizeof(double), &dbeta));
        }
        else {
          float fbeta = (float)beta;
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_scale_beta, i++, sizeof(float), &fbeta));
        }
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_crt_scale_beta, 2,
        NULL, global_s, local_s, 0, NULL, NULL));
    }

    /* Single fused CRT GEMM: one launch per output tile covers all primes */
    first_tile = (0.0 == beta) ? 1 : 0;
    if (EXIT_SUCCESS == result) {
      size_t global_g[2], local_g[2];
      local_g[0] = 16; /* GEMM tile decomposition requires SG=16 */
      local_g[1] = (size_t)(GBM / 8) * (size_t)(GBN / 16);
      global_g[0] = (size_t)nblk_gm * local_g[0];
      global_g[1] = (size_t)nblk_gn * local_g[1];
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_as));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_bs));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_expa_g));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_expb_g));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_cg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &K_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &N_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &ldc));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &M_pad));
        if (ctx->use_double) {
          double dalpha = alpha;
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(double), &dalpha));
        }
        else {
          float falpha = (float)alpha;
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(float), &falpha));
        }
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &first_tile));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_crt_fused, 2,
        NULL, global_g, local_g, 0, NULL,
        (NULL != evt_prof_c && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof_c + n_profiled_c) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof_c
        && (1 == ctx->profile || 2 == ctx->profile || 0 > ctx->profile)) ++n_profiled_c;
    }


    /* Collect profiling data */
    if (NULL != evt_prof_c) {
      int resprof = clWaitForEvents((cl_uint)n_profiled_c, evt_prof_c);
      double total = 0;
      { int pi;
        for (pi = 0; pi < n_profiled_c && EXIT_SUCCESS == resprof; ++pi) {
          total += libxstream_opencl_duration(evt_prof_c[pi], &resprof);
        }
        for (pi = 0; pi < n_profiled_c; ++pi) {
          if (NULL != evt_prof_c[pi]) clReleaseEvent(evt_prof_c[pi]);
        }
      }
      if (EXIT_SUCCESS == resprof && 0 < total) {
        const double gflops = (2.0 * M * N * K) / (total * 1E9);
        libxs_hist_push(NULL, ctx->hist, &gflops);
      }
      free(evt_prof_c);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);

    OZAKI_DEV_FREE(d_ag); OZAKI_DEV_FREE(d_bg); OZAKI_DEV_FREE(d_cg);
    OZAKI_DEV_FREE(d_as); OZAKI_DEV_FREE(d_bs);
    OZAKI_DEV_FREE(d_expa_g); OZAKI_DEV_FREE(d_expb_g);

    return result;
  }

  return result;
}
