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


int ozaki_init(ozaki_context_t* ctx, int tm, int tn,
               int use_double, int kind, int verbosity,
               int ndecomp, int ozflags, int oztrim,
               int ozgroups)
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
    libxstream_opencl_device_name(
      device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char* const fp64_ext[] = { "cl_khr_fp64" };
    if (EXIT_SUCCESS != libxstream_opencl_device_ext(
          device, fp64_ext, 1))
    {
      if (0 > verbosity || 1 < verbosity) {
        fprintf(stderr,
          "WARN: device does not support cl_khr_fp64,"
          " falling back to float\n");
      }
      use_double = 0;
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

  if (0 >= ndecomp) ndecomp = (2 == kind ? (use_double ? 19 : 10) : 8);
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
  ctx->kern_fused_sym = NULL;
  ctx->kern_scale_beta = NULL;
  ctx->kern_crt_preprocess_a = NULL;
  ctx->kern_crt_preprocess_b = NULL;
  ctx->kern_crt_fused = NULL;
  ctx->kern_crt_scale_beta = NULL;
  { /* output tile sizes: fit SG * (tm/8) * (tn/16) <= max_wgs.
     * tm must be multiple of 8, tn must be multiple of 16.
     * Large GRF halves effective max work-group size. */
    const size_t max_wgs = (0 != devinfo->biggrf)
      ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0];
    const int bm_pre = 16, bn_pre = 16, bk_pre = 32;
    char build_params[1024];
    const char build_options[] =
      "-cl-fast-relaxed-math -cl-denorms-are-zero";
    const int mant_bits      = use_double ? 52 : 23;
    const int bias_plus_mant = use_double ? 1075 : 150;
    if (0 >= tm) tm = 256;
    if (0 >= tn) tn = 256;
    while ((size_t)tm * tn / 8 > max_wgs && (tm > 8 || tn > 16)) {
      if (tm >= tn) tm /= 2; else tn /= 2;
    }
    if (1 == kind) {
      size_t goff = 0;
      goff += (size_t)LIBXS_SNPRINTF(
        build_params + goff, sizeof(build_params) - goff,
        "-DBM=%d -DBN=%d -DBK=%d -DSG=16"
        " -DNSLICES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DCONSTANT=global",
        tm, tn, bk_pre,
        ndecomp, use_double,
        mant_bits, bias_plus_mant,
        bm_pre, bn_pre, bk_pre);
      if (use_xmx) {
        goff += (size_t)LIBXS_SNPRINTF(
          build_params + goff, sizeof(build_params) - goff,
          " -DUSE_XMX=1");
      }
      (void)goff;
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
            program, "gemm_fused_sym",
            &ctx->kern_fused_sym);
        }
        if (EXIT_SUCCESS == result) {
          result = libxstream_opencl_kernel_query(
            program, "scale_beta",
            &ctx->kern_scale_beta);
        }
        if (NULL != program) clReleaseProgram(program);
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
        if (NULL != ctx->kern_fused_sym) {
          clReleaseKernel(ctx->kern_fused_sym);
          ctx->kern_fused_sym = NULL;
        }
        if (NULL != ctx->kern_scale_beta) {
          clReleaseKernel(ctx->kern_scale_beta);
          ctx->kern_scale_beta = NULL;
        }
      }
    }
    /* CRT GEMM (kind==2): fused kernel per tile, all primes internal */
    else if (2 == kind) {
      size_t coff = 0;
      coff += (size_t)LIBXS_SNPRINTF(
        build_params + coff, sizeof(build_params) - coff,
        "-DBM=%d -DBN=%d -DBK=%d -DSG=16"
        " -DNPRIMES=%d -DUSE_DOUBLE=%d"
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
        " -DBM_PRE=%d -DBN_PRE=%d -DBK_PRE=%d"
        " -DKGROUPS=%d -DCONSTANT=global",
        tm, tn, bk_pre,
        ndecomp, use_double,
        mant_bits, bias_plus_mant,
        bm_pre, bn_pre, bk_pre,
        (2 == kind && 1 < ozgroups) ? ozgroups : 0);
      if (use_xmx) {
        coff += (size_t)LIBXS_SNPRINTF(
          build_params + coff, sizeof(build_params) - coff,
          " -DUSE_XMX=1");
      }
      (void)coff;
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
    else {
      fprintf(stderr, "ERROR OZAKI: unsupported kind=%d\n", kind);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      ctx->tm = tm;
      ctx->tn = tn;
      ctx->bm_pre = bm_pre;
      ctx->bn_pre = bn_pre;
      ctx->bk_pre = bk_pre;
    }
    else if (0 != verbosity) {
      fprintf(stderr, "ERROR OZAKI: kernel build failed\n");
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
    ozaki_print_opt(stderr, "ndecomp", ndecomp);
    if (1 == kind) ozaki_print_opt(stderr, "trim", oztrim);
    if (2 == kind) ozaki_print_opt(stderr, "kgroups", ozgroups);
    ozaki_print_opt(stderr, "cache", ctx->cache.flags);
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
  if (EXIT_SUCCESS != result) ozaki_destroy(ctx);

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
    if (NULL != ctx->kern_fused_sym) {
      clReleaseKernel(ctx->kern_fused_sym);
    }
    if (NULL != ctx->kern_scale_beta) {
      clReleaseKernel(ctx->kern_scale_beta);
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

    /* Free preprocessing cache (non-pool device memory) */
    if (NULL != ctx->cache.a.d_slices) libxstream_mem_deallocate(ctx->cache.a.d_slices);
    if (NULL != ctx->cache.a.d_exp) libxstream_mem_deallocate(ctx->cache.a.d_exp);
    if (NULL != ctx->cache.b.d_slices) libxstream_mem_deallocate(ctx->cache.b.d_slices);
    if (NULL != ctx->cache.b.d_exp) libxstream_mem_deallocate(ctx->cache.b.d_exp);

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
  libxstream_stream_t* const stream_a = ctx->stream_a;
  libxstream_stream_t* const stream_b = ctx->stream_b;
  libxstream_event_t* const evt_prep_a = ctx->evt_prep_a;
  libxstream_event_t* const evt_prep_b = ctx->evt_prep_b;
  size_t c_nbytes;
  const int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  const int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;

#if defined(OZAKI_DEVPOOL)
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
  ctx->stream = stream; /* expose to deallocate wrapper */
#endif

  /* GEMM path (Scheme 1): full-split-then-tiled-GEMM.
   * Preprocesses entire K dimension up front into dense per-slice
   * int8 matrices, then runs a proper tiled GEMM per slice pair. */
  if (NULL != ctx->kern_fused && 0 < K) {
    const int nslices_g = ctx->ndecomp;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const int tm = ctx->tm, tn = ctx->tn;
    /* Pad K to multiple of bk_pre (=32) and ensure >= 64 for 2D block I/O */
    int k_pad = ((K + bk_pre - 1) / bk_pre) * bk_pre;
    const int m_pad = ((M + bm_pre - 1) / bm_pre) * bm_pre;
    int n_pad = ((N + bn_pre - 1) / bn_pre) * bn_pre;
    const int nblk_gm = (M + tm - 1) / tm;
    const int nblk_gn = (N + tn - 1) / tn;
    const int ntm = tm / 8, ntn = tn / 16;
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
    int cache_hit_a = 0, cache_hit_b = 0;

    if (k_pad < 64) k_pad = 64;
    if (n_pad < 64) n_pad = 64;

    as_size   = (size_t)nslices_g * m_pad * k_pad;
    bs_size   = (size_t)nslices_g * k_pad * n_pad;
    expa_size = (size_t)nblk_gm * tm * sizeof(cl_int); /* pad to tile boundary */
    expb_size = (size_t)nblk_gn * tn * sizeof(cl_int);
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    /* Preprocessing cache: reuse slices+exponents when matrix unchanged */
    if (0 != (ctx->cache.flags & 1) && a == ctx->cache.a.ptr
        && M == ctx->cache.a.dim && K == ctx->cache.a.K
        && lda == ctx->cache.a.ld && ta == ctx->cache.a.trans
        && as_size == ctx->cache.a.slices_size && expa_size == ctx->cache.a.exp_size
        && NULL != ctx->cache.a.d_slices && NULL != ctx->cache.a.d_exp)
    {
      d_as = ctx->cache.a.d_slices; d_expa_g = ctx->cache.a.d_exp;
      cache_hit_a = 1;
    }
    if (0 != (ctx->cache.flags & 2) && b == ctx->cache.b.ptr
        && N == ctx->cache.b.dim && K == ctx->cache.b.K
        && ldb == ctx->cache.b.ld && tb == ctx->cache.b.trans
        && bs_size == ctx->cache.b.slices_size && expb_size == ctx->cache.b.exp_size
        && NULL != ctx->cache.b.d_slices && NULL != ctx->cache.b.d_exp)
    {
      d_bs = ctx->cache.b.d_slices; d_expb_g = ctx->cache.b.d_exp;
      cache_hit_b = 1;
    }

    /* Allocate device memory (skip cached sides and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_as, as_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);
    }

    /* H2D transfers (skip cached and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    /* Zero exponent arrays and slice buffers (skip cached and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
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

    /* Phase 1: Preprocess A (skip entirely on cache hit) */
    if (0 == cache_hit_a) {
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
      h_as = calloc(as_size, 1);
      h_expa = calloc(expa_size, 1);
      if (NULL != h_as && NULL != h_expa) {
        ctx->host_preprocess_a(a, lda, ta, M, K, k_pad, m_pad,
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
      const int nblk_m_pre = (M + bm_pre - 1) / bm_pre;
      local_a[0] = bm_pre; local_a[1] = bk_pre;
      global_a[0] = (size_t)nblk_m_pre * bm_pre;
      global_a[1] = bk_pre; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_ag));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &lda));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &ta));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_as));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, i++, d_expa_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &k_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, i++, sizeof(int), &m_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_preprocess_a, 2,
        NULL, global_a, local_a, 0, NULL,
        (NULL != evt_prof && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof + n_profiled) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof
        && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
    }
    } /* cache_hit_a */

    /* Phase 1: Preprocess B (skip entirely on cache hit) */
    if (0 == cache_hit_b) {
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
      h_bs = calloc(bs_size, 1);
      h_expb = calloc(expb_size, 1);
      if (NULL != h_bs && NULL != h_expb) {
        ctx->host_preprocess_b(b, ldb, tb, N, K, k_pad, n_pad,
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
      const int nblk_n_pre = (N + bn_pre - 1) / bn_pre;
      local_b[0] = bn_pre; local_b[1] = bk_pre;
      global_b[0] = (size_t)nblk_n_pre * bn_pre;
      global_b[1] = bk_pre; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_bg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &ldb));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &tb));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_bs));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, i++, d_expb_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &k_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, i++, sizeof(int), &n_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_b->queue, ctx->kern_preprocess_b, 2,
        NULL, global_b, local_b, 0, NULL,
        (NULL != evt_prof && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof + n_profiled) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof
        && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile)) ++n_profiled;
    }
    } /* cache_hit_b */

    /* Wait for both preprocessing to complete */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Save preprocessed buffers to cache on miss (transfers ownership) */
    if (0 == cache_hit_a && 0 != (ctx->cache.flags & 1) && EXIT_SUCCESS == result) {
      if (NULL != ctx->cache.a.d_slices && as_size != ctx->cache.a.slices_size) {
        libxstream_mem_deallocate(ctx->cache.a.d_slices); ctx->cache.a.d_slices = NULL;
      }
      if (NULL != ctx->cache.a.d_exp && expa_size != ctx->cache.a.exp_size) {
        libxstream_mem_deallocate(ctx->cache.a.d_exp); ctx->cache.a.d_exp = NULL;
      }
      ctx->cache.a.ptr = a; ctx->cache.a.dim = M; ctx->cache.a.K = K;
      ctx->cache.a.ld = lda; ctx->cache.a.trans = ta;
      ctx->cache.a.d_slices = d_as; ctx->cache.a.d_exp = d_expa_g;
      ctx->cache.a.slices_size = as_size; ctx->cache.a.exp_size = expa_size;
    }
    if (0 == cache_hit_b && 0 != (ctx->cache.flags & 2) && EXIT_SUCCESS == result) {
      if (NULL != ctx->cache.b.d_slices && bs_size != ctx->cache.b.slices_size) {
        libxstream_mem_deallocate(ctx->cache.b.d_slices); ctx->cache.b.d_slices = NULL;
      }
      if (NULL != ctx->cache.b.d_exp && expb_size != ctx->cache.b.exp_size) {
        libxstream_mem_deallocate(ctx->cache.b.d_exp); ctx->cache.b.d_exp = NULL;
      }
      ctx->cache.b.ptr = b; ctx->cache.b.dim = N; ctx->cache.b.K = K;
      ctx->cache.b.ld = ldb; ctx->cache.b.trans = tb;
      ctx->cache.b.d_slices = d_bs; ctx->cache.b.d_exp = d_expb_g;
      ctx->cache.b.slices_size = bs_size; ctx->cache.b.exp_size = expb_size;
    }

    /* Scale C by beta (skip for beta==0: first_pair overwrite handles it;
     * skip for beta==1: multiplying by 1 is a no-op) */
    if (EXIT_SUCCESS == result && 1.0 != beta && 0.0 != beta) {
      size_t global_s[2], local_s[2];
      local_s[0] = (size_t)bm_pre; local_s[1] = 1;
      global_s[0] = (size_t)((M + bm_pre - 1) / bm_pre) * bm_pre;
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
        const size_t as_offset_sa = (size_t)sa * m_pad * k_pad;
        const size_t bs_offset_sb = (size_t)sb * k_pad * n_pad;

        size_t global_g[2], local_g[2];
        local_g[0] = 16; /* GEMM tile decomposition requires SG=16 */
        local_g[1] = (size_t)(ntm * ntn);
        global_g[0] = (size_t)nblk_gm * local_g[0];
        global_g[1] = (size_t)nblk_gn * local_g[1];

        if (symmetrize && sa != sb) {
          /* Symmetric path: compute (sa,sb) and (sb,sa) in one launch */
          const size_t as_offset_sb = (size_t)sb * m_pad * k_pad;
          const size_t bs_offset_sa = (size_t)sa * k_pad * n_pad;
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
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &k_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused_sym, i++, sizeof(int), &n_pad));
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
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &k_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_fused, i++, sizeof(int), &n_pad));
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
    if (d_as != ctx->cache.a.d_slices) OZAKI_DEV_FREE(d_as);
    if (d_bs != ctx->cache.b.d_slices) OZAKI_DEV_FREE(d_bs);
    if (d_expa_g != ctx->cache.a.d_exp) OZAKI_DEV_FREE(d_expa_g);
    if (d_expb_g != ctx->cache.b.d_exp) OZAKI_DEV_FREE(d_expb_g);

  }
  /* CRT GEMM path (Scheme 2): full-split-then-single-fused-GEMM.
   * Preprocesses entire K into dense per-prime CRT residue matrices,
   * then runs a single kernel per tile that loops over all primes
   * internally (full-K DPAS + Garner + Horner in one launch). */
  else if (NULL != ctx->kern_crt_fused && 0 < K) {
    const int nprimes_g = ctx->ndecomp;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const int tm = ctx->tm, tn = ctx->tn;
    int k_pad = ((K + bk_pre - 1) / bk_pre) * bk_pre;
    const int m_pad = ((M + bm_pre - 1) / bm_pre) * bm_pre;
    int n_pad = ((N + bn_pre - 1) / bn_pre) * bn_pre;
    const int nblk_gm = (M + tm - 1) / tm;
    const int nblk_gn = (N + tn - 1) / tn;
    const int ntm = tm / 8, ntn = tn / 16;
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    void *h_as = NULL, *h_expa = NULL, *h_bs = NULL, *h_expb = NULL;
    int first_tile;
    int n_profiled_c = 0;
    cl_event *evt_prof_c = NULL;
    int cache_hit_a = 0, cache_hit_b = 0;

    (void)ntm; (void)ntn;
    if (k_pad < 64) k_pad = 64;
    if (n_pad < 64) n_pad = 64;

    as_size   = (size_t)nprimes_g * m_pad * k_pad;
    bs_size   = (size_t)nprimes_g * k_pad * n_pad;
    expa_size = (size_t)nblk_gm * tm * sizeof(cl_int); /* pad to tile boundary */
    expb_size = (size_t)nblk_gn * tn * sizeof(cl_int);
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    /* Preprocessing cache: reuse slices+exponents when matrix unchanged */
    if (0 != (ctx->cache.flags & 1) && a == ctx->cache.a.ptr
        && M == ctx->cache.a.dim && K == ctx->cache.a.K
        && lda == ctx->cache.a.ld && ta == ctx->cache.a.trans
        && as_size == ctx->cache.a.slices_size && expa_size == ctx->cache.a.exp_size
        && NULL != ctx->cache.a.d_slices && NULL != ctx->cache.a.d_exp)
    {
      d_as = ctx->cache.a.d_slices; d_expa_g = ctx->cache.a.d_exp;
      cache_hit_a = 1;
    }
    if (0 != (ctx->cache.flags & 2) && b == ctx->cache.b.ptr
        && N == ctx->cache.b.dim && K == ctx->cache.b.K
        && ldb == ctx->cache.b.ld && tb == ctx->cache.b.trans
        && bs_size == ctx->cache.b.slices_size && expb_size == ctx->cache.b.exp_size
        && NULL != ctx->cache.b.d_slices && NULL != ctx->cache.b.d_exp)
    {
      d_bs = ctx->cache.b.d_slices; d_expb_g = ctx->cache.b.d_exp;
      cache_hit_b = 1;
    }

    /* Allocate device memory (skip cached sides and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_as, as_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);
    }

    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
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

    /* Preprocess A (skip entirely on cache hit) */
    if (0 == cache_hit_a) {
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
      h_as = calloc(as_size, 1);
      h_expa = calloc(expa_size, 1);
      if (NULL != h_as && NULL != h_expa) {
        ctx->host_preprocess_a(a, lda, ta, M, K, k_pad, m_pad,
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
      const int nblk_m_pre = (M + bm_pre - 1) / bm_pre;
      local_a[0] = bm_pre; local_a[1] = bk_pre;
      global_a[0] = (size_t)nblk_m_pre * bm_pre;
      global_a[1] = bk_pre; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_ag));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &M));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &lda));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &ta));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_as));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_expa_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &k_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &m_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_crt_preprocess_a, 2,
        NULL, global_a, local_a, 0, NULL,
        (NULL != evt_prof_c && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof_c + n_profiled_c) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof_c
        && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile)) ++n_profiled_c;
    }
    } /* cache_hit_a */

    /* Preprocess B (skip entirely on cache hit) */
    if (0 == cache_hit_b) {
    if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
      h_bs = calloc(bs_size, 1);
      h_expb = calloc(expb_size, 1);
      if (NULL != h_bs && NULL != h_expb) {
        ctx->host_preprocess_b(b, ldb, tb, N, K, k_pad, n_pad,
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
      const int nblk_n_pre = (N + bn_pre - 1) / bn_pre;
      local_b[0] = bn_pre; local_b[1] = bk_pre;
      global_b[0] = (size_t)nblk_n_pre * bn_pre;
      global_b[1] = bk_pre; /* single WG in K: kernel loops internally */
      { cl_int i = 0;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_bg));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &N));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &K));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &ldb));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &tb));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_bs));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_b, i++, d_expb_g));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &k_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_b, i++, sizeof(int), &n_pad));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_b->queue, ctx->kern_crt_preprocess_b, 2,
        NULL, global_b, local_b, 0, NULL,
        (NULL != evt_prof_c && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile))
          ? (evt_prof_c + n_profiled_c) : NULL));
      if (EXIT_SUCCESS == result && NULL != evt_prof_c
        && (1 == ctx->profile || 4 == ctx->profile || 0 > ctx->profile)) ++n_profiled_c;
    }
    } /* cache_hit_b */

    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Save preprocessed buffers to cache on miss (transfers ownership) */
    if (0 == cache_hit_a && 0 != (ctx->cache.flags & 1) && EXIT_SUCCESS == result) {
      if (NULL != ctx->cache.a.d_slices && as_size != ctx->cache.a.slices_size) {
        libxstream_mem_deallocate(ctx->cache.a.d_slices); ctx->cache.a.d_slices = NULL;
      }
      if (NULL != ctx->cache.a.d_exp && expa_size != ctx->cache.a.exp_size) {
        libxstream_mem_deallocate(ctx->cache.a.d_exp); ctx->cache.a.d_exp = NULL;
      }
      ctx->cache.a.ptr = a; ctx->cache.a.dim = M; ctx->cache.a.K = K;
      ctx->cache.a.ld = lda; ctx->cache.a.trans = ta;
      ctx->cache.a.d_slices = d_as; ctx->cache.a.d_exp = d_expa_g;
      ctx->cache.a.slices_size = as_size; ctx->cache.a.exp_size = expa_size;
    }
    if (0 == cache_hit_b && 0 != (ctx->cache.flags & 2) && EXIT_SUCCESS == result) {
      if (NULL != ctx->cache.b.d_slices && bs_size != ctx->cache.b.slices_size) {
        libxstream_mem_deallocate(ctx->cache.b.d_slices); ctx->cache.b.d_slices = NULL;
      }
      if (NULL != ctx->cache.b.d_exp && expb_size != ctx->cache.b.exp_size) {
        libxstream_mem_deallocate(ctx->cache.b.d_exp); ctx->cache.b.d_exp = NULL;
      }
      ctx->cache.b.ptr = b; ctx->cache.b.dim = N; ctx->cache.b.K = K;
      ctx->cache.b.ld = ldb; ctx->cache.b.trans = tb;
      ctx->cache.b.d_slices = d_bs; ctx->cache.b.d_exp = d_expb_g;
      ctx->cache.b.slices_size = bs_size; ctx->cache.b.exp_size = expb_size;
    }

    /* Scale C by beta (skip for beta==0: first_tile overwrite handles it;
     * skip for beta==1: multiplying by 1 is a no-op) */
    if (EXIT_SUCCESS == result && 1.0 != beta && 0.0 != beta) {
      size_t global_s[2], local_s[2];
      local_s[0] = (size_t)bm_pre; local_s[1] = 1;
      global_s[0] = (size_t)((M + bm_pre - 1) / bm_pre) * bm_pre;
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
      local_g[1] = (size_t)(tm / 8) * (size_t)(tn / 16);
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
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &k_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &n_pad));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &ldc));
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &m_pad));
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
    if (d_as != ctx->cache.a.d_slices) OZAKI_DEV_FREE(d_as);
    if (d_bs != ctx->cache.b.d_slices) OZAKI_DEV_FREE(d_bs);
    if (d_expa_g != ctx->cache.a.d_exp) OZAKI_DEV_FREE(d_expa_g);
    if (d_expb_g != ctx->cache.b.d_exp) OZAKI_DEV_FREE(d_expb_g);

  }

  return result;
}
