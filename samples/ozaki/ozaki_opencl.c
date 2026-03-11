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
#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_BF16)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_BF16)"
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
  char build_options[256];
  char build_params[1024];
  size_t offset;
  const char* kernel_source;
  const char* env;
  int use_bf16, wg, sg, gpu, use_xmx, result;

  /* Derive bf16 flag from kind (3 = bf16, else int8) */
  use_bf16 = (3 == kind) ? 1 : 0;
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

  /* Choose smart defaults: XMX-friendly when hardware is available.
   * XMX requires BK==32 (int8) or BK==16 (bf16), BM/BN divisible by 8. */
  if (0 >= bm) bm = 16;
  if (0 >= bn) bn = 16;
  if (0 >= bk) bk = (use_xmx ? (use_bf16 ? 16 : 32) : 16);
  if (0 >= nslices) nslices = (2 == kind ? 17 : 8); /* CRT: 17 primes default */
  if (0 >= batch_k) batch_k = 4;
  if (0 > ozflags) ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;

  /* Validate XMX constraints against final block sizes.
   * DPAS SG=16: XMX_N=16, so BN must be divisible by 16. */
  if (use_xmx && ((use_bf16 ? 16 : 32) != bk || 0 != (bm % 8) || 0 != (bn % 16))) {
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
  ctx->use_bf16 = use_bf16;
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

  /* Assemble JIT build flags: compiler options and -D defines separately */
  env = getenv("OZAKI_GRF256");
  if (NULL != env && 0 != atoi(env) && 0 != devinfo->intel) {
    LIBXS_SNPRINTF(build_options, sizeof(build_options),
      "-cl-fast-relaxed-math -cl-denorms-are-zero"
      " -cl-intel-256-GRF-per-thread");
  }
  else {
    LIBXS_SNPRINTF(build_options, sizeof(build_options),
      "-cl-fast-relaxed-math -cl-denorms-are-zero");
  }

  { const char* constant_qual;
    env = getenv("OZAKI_CONSTANT");
    constant_qual = (NULL != env && 0 != atoi(env)) ? "constant" : "global";

    offset = 0;
    if (gpu) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        "-DGPU ");
    }

    offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
      "-DCONSTANT=%s -DBM=%d -DBN=%d -DBK=%d"
      " -DUSE_DOUBLE=%d",
      constant_qual, bm, bn, bk, use_double);

    if (2 == kind) {
      /* CRT: pass NPRIMES and mantissa parameters */
      const int mant_bits      = use_double ? 52 : 23;
      const int bias_plus_mant = use_double ? 1075 : 150;
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DNPRIMES=%d -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d -DKGROUP=%d",
        nslices, mant_bits, bias_plus_mant, ctx->kgroup);
    }
    else {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DNSLICES=%d -DTRIANGULAR=%d -DSYMMETRIZE=%d -DTRIM=%d",
        nslices,
        (ozflags & OZAKI_TRIANGULAR) ? 1 : 0,
        (ozflags & OZAKI_SYMMETRIZE) ? 1 : 0,
        oztrim);

      if (!use_bf16) {
        const int mant_bits      = use_double ? 52 : 23;
        const int bias_plus_mant = use_double ? 1075 : 150;
        offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
          " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d", mant_bits, bias_plus_mant);
      }
    }

    if (0 < wg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DWG=%d", wg);
    }
    if (0 < sg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DSG=%d", sg);
    }
    if (use_xmx) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DUSE_XMX=1");
    }
  }
  ctx->sg = sg;

  if (0 > verbosity || 2 < verbosity) {
    fprintf(stderr, "INFO OZAKI: build params: %s\n", build_params);
  }

  /* JIT compile kernels via ACC */
  if (2 == kind) {
    kernel_source = OPENCL_KERNELS_SOURCE_OZAKI2_INT8;
  }
  else {
    kernel_source = use_bf16
      ? OPENCL_KERNELS_SOURCE_OZAKI1_BF16
      : OPENCL_KERNELS_SOURCE_OZAKI1_INT8;
  }

  result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
    "preprocess_a", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_preprocess_a);
  if (EXIT_SUCCESS != result) {
    if (0 != verbosity) fprintf(stderr, "ERROR: failed to build preprocess_a kernel\n");
  }

  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
      "preprocess_b", build_params, build_options,
      NULL, NULL, NULL, 0, &ctx->kern_preprocess_b);
    if (EXIT_SUCCESS != result) {
      if (0 != verbosity) fprintf(stderr, "ERROR: failed to build preprocess_b kernel\n");
    }
  }

  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
      "dotprod", build_params, build_options,
      NULL, NULL, NULL, 0, &ctx->kern_dotprod);
    if (EXIT_SUCCESS != result) {
      if (0 != verbosity) fprintf(stderr, "ERROR: failed to build dotprod kernel\n");
    }
  }

  /* Report compiled kernel info */
  if (EXIT_SUCCESS == result && (0 > verbosity || 2 < verbosity)) {
    size_t wgs[3] = {0};
    if (CL_SUCCESS == clGetKernelWorkGroupInfo(ctx->kern_dotprod, device,
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgs), wgs, NULL))
    {
      fprintf(stderr, "INFO OZAKI: dotprod-%s compiled for WG=%ux%u\n",
        2 == kind ? "crt" : (use_bf16 ? "bf16" : "int8"),
        LIBXS_CAST_UINT(wgs[0]), LIBXS_CAST_UINT(wgs[1]));
    }
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "kind", kind);
    ozaki_print_opt(stderr, "bf16", use_bf16);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "xmx", use_xmx);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "nslices", nslices);
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

  /* OZAKI_PROF: kernel execution-time profiling */
  ctx->hist = NULL;
  { const char* env_prof = getenv("OZAKI_PROF");
    if (NULL != env_prof && '0' != *env_prof) {
      const libxs_hist_update_t update[] = { libxs_hist_update_avg };
      ctx->hist = libxs_hist_create(3, 1, update);
    }
  }

  /* Create persistent helper streams and synchronization events */
  { const int sflags = (NULL != ctx->hist
      ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT);
    if (EXIT_SUCCESS == result) result = libxstream_stream_create(&ctx->stream_a, "ozaki_a", sflags);
    if (EXIT_SUCCESS == result) result = libxstream_stream_create(&ctx->stream_b, "ozaki_b", sflags);
  }
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_prep_b);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_dotprod[0]);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&ctx->evt_dotprod[1]);
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
    if (NULL != ctx->kern_dotprod)      clReleaseKernel(ctx->kern_dotprod);

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
    if (NULL != ctx->evt_dotprod[0]) libxstream_event_destroy(ctx->evt_dotprod[0]);
    if (NULL != ctx->evt_dotprod[1]) libxstream_event_destroy(ctx->evt_dotprod[1]);
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
  const int BM = ctx->bm, BN = ctx->bn, BK = ctx->bk;
  /* Pad B column stride so surface width >= 64 bytes (2D block I/O).
   * bf16: 2 bytes/elem, min 32 elements.  int8: 1 byte/elem, min 64. */
  const int bn_min = ctx->use_bf16 ? 32 : 64;
  const int BN_PAD = ctx->use_xmx ? ((BN < bn_min) ? bn_min : BN) : BN;
  const int BATCH_K = ctx->batch_k;
  const int nslices = ctx->nslices;
  const int nblk_m = (M + BM - 1) / BM;
  const int nblk_n = (N + BN - 1) / BN;
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  const int nkb_total = (K + BK - 1) / BK;
  const int max_nkb = (BATCH_K < nkb_total) ? BATCH_K : nkb_total;
  const int kgroup = ctx->kgroup; /* K-grouping factor (1 = no grouping) */
  const int n_batches = (0 < K) ? ((nkb_total + BATCH_K - 1) / BATCH_K) : 0;
  /* Hoisted buffer sizes (used by both device and host staging allocation) */
  const size_t slice_elem = ctx->use_bf16 ? 2 : 1; /* ushort vs char */
  const size_t ak_max_size = (size_t)max_nkb * nblk_m * BM * nslices * BK * slice_elem;
  const size_t bk_max_size = (size_t)max_nkb * nblk_n * BN_PAD * nslices * BK * slice_elem;
  const int max_ngroups = (max_nkb + kgroup - 1) / kgroup;
  const size_t expa_max_size = ctx->use_bf16 ? 0 : (size_t)max_ngroups * nblk_m * BM * sizeof(cl_short);
  const size_t expb_max_size = ctx->use_bf16 ? 0 : (size_t)max_ngroups * nblk_n * BN * sizeof(cl_short);
  /* Host-side preprocessing flags */
  const int host_a = (NULL != ctx->host_preprocess_a) ? 1 : 0;
  const int host_b = (NULL != ctx->host_preprocess_b) ? 1 : 0;
  /* Device buffers for input matrices */
  void *d_a = NULL, *d_b = NULL, *d_c = NULL;
  /* Double-buffered preprocessing buffers (2 slots) */
  void *d_ak[2] = {NULL, NULL}, *d_expa[2] = {NULL, NULL};
  void *d_bk[2] = {NULL, NULL}, *d_expb[2] = {NULL, NULL};
  /* Host staging buffers for host-side preprocessing (double-buffered) */
  void *h_ak[2] = {NULL, NULL}, *h_expa[2] = {NULL, NULL};
  void *h_bk[2] = {NULL, NULL}, *h_expb[2] = {NULL, NULL};

  /* Persistent helper streams and events from context */
  libxstream_stream_t *stream_a = ctx->stream_a;
  libxstream_stream_t *stream_b = ctx->stream_b;
  libxstream_event_t *evt_prep_a = ctx->evt_prep_a;
  libxstream_event_t *evt_prep_b = ctx->evt_prep_b;
  libxstream_event_t *evt_dotprod[2];
  size_t c_nbytes;
  int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;
  int batch, n_profiled = 0;
  cl_event *evt_prof = NULL;

#if defined(OZAKI_DEVPOOL)
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
  ctx->stream = stream; /* expose to deallocate wrapper */
#endif

  evt_dotprod[0] = ctx->evt_dotprod[0];
  evt_dotprod[1] = ctx->evt_dotprod[1];

  /* Allocate device memory for A, B, C.
   * When host preprocessing is active for a side, the full matrix is not
   * needed on device — only the preprocessed slice buffers are transferred.
   * C is always needed on device for beta-scaling and result accumulation. */
  { c_nbytes = (size_t)ldc * (size_t)N * elem_size;
    if (0 == host_a) {
      const size_t a_cols = ta ? (size_t)M : (size_t)K;
      const size_t a_nbytes = (size_t)lda * a_cols * elem_size;
      if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_a, a_nbytes);
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(a, d_a, a_nbytes, stream_a);
    }
    if (0 == host_b) {
      const size_t b_cols = tb ? (size_t)K : (size_t)N;
      const size_t b_nbytes = (size_t)ldb * b_cols * elem_size;
      if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_b, b_nbytes);
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(b, d_b, b_nbytes, stream_b);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_c, c_nbytes);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_c, c_nbytes, stream);
  }

  /* Pre-allocate double-buffered preprocessing buffers (max batch size) */
  if (EXIT_SUCCESS == result) {
    int s;
    for (s = 0; s < 2 && s < n_batches; ++s) {
      if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_ak[s], ak_max_size);
      if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bk[s], bk_max_size);
      if (0 < expa_max_size) {
        if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expa[s], expa_max_size);
        if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_expb[s], expb_max_size);
      }
    }
  }

  /* Allocate host staging buffers for host-side preprocessing */
  if (EXIT_SUCCESS == result && (0 != host_a || 0 != host_b)) {
    int s;
    for (s = 0; s < 2 && s < n_batches; ++s) {
      if (0 != host_a) {
        h_ak[s] = calloc(1, ak_max_size);
        if (0 < expa_max_size) h_expa[s] = calloc(1, expa_max_size);
        if (NULL == h_ak[s]) result = EXIT_FAILURE;
      }
      if (0 != host_b) {
        h_bk[s] = calloc(1, bk_max_size);
        if (0 < expb_max_size) h_expb[s] = calloc(1, expb_max_size);
        if (NULL == h_bk[s]) result = EXIT_FAILURE;
      }
    }
  }

  /* Profiling: allocate event array (3 events per batch) */
  if (NULL != ctx->hist && 0 < n_batches) {
    evt_prof = (cl_event*)calloc(3 * (size_t)n_batches, sizeof(cl_event));
  }

  /* Double-buffered K-batch pipeline:
   *   stream_a  : preprocess_a  (parallel with preprocess_b)
   *   stream_b  : preprocess_b  (parallel with preprocess_a)
   *   stream    : dotprod + C transfers
   * Batch N dotprod overlaps with batch N+1 preprocessing. */
  for (batch = 0; batch < n_batches && EXIT_SUCCESS == result; ++batch) {
    const int cur = batch & 1;
    const int kb_batch = batch * BATCH_K * BK;
    const int batch_end = (kb_batch + BATCH_K * BK < K) ? (kb_batch + BATCH_K * BK) : K;
    const int nkb = (batch_end - kb_batch + BK - 1) / BK;
    const int first_batch = (0 == batch) ? 1 : 0;

    /* Ensure the dotprod that last used this buffer slot is done */
    if (2 <= batch) {
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_a, evt_dotprod[cur]);
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_b, evt_dotprod[cur]);
    }

    /* Launch preprocess_a on stream_a (or host callback + H2D) */
    if (0 != host_a) {
      const size_t ak_cur = (size_t)nkb * nblk_m * BM * nslices * BK * slice_elem;
      const int nkb_groups_a = (nkb + kgroup - 1) / kgroup;
      const size_t expa_cur = (size_t)nkb_groups_a * nblk_m * BM * sizeof(cl_short);
      ctx->host_preprocess_a(a, lda, ta, M, K, kb_batch,
        nkb, nblk_m, BM, BK, nslices, kgroup, ctx->use_xmx,
        h_ak[cur], h_expa[cur]);
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(
        h_ak[cur], d_ak[cur], ak_cur, stream_a);
      if (0 < expa_cur && NULL != h_expa[cur]) {
        if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(
          h_expa[cur], d_expa[cur], expa_cur, stream_a);
      }
    }
    else {
    /* GPU preprocess_a on stream_a */
    { const libxstream_opencl_stream_t* str_a = stream_a;
      const int nkb_groups_a = (nkb + kgroup - 1) / kgroup;
      size_t global_a[2], local_a[2];
      global_a[0] = (size_t)nblk_m * BM; global_a[1] = (size_t)nkb_groups_a * BK;
      local_a[0] = BM; local_a[1] = BK;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 0, d_a));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 1, sizeof(int), &M));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 2, sizeof(int), &K));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 3, sizeof(int), &lda));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 4, sizeof(int), &ta));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 5, sizeof(int), &kb_batch));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 6, d_ak[cur]));
      if (ctx->use_bf16) {
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 7, sizeof(int), &nblk_m));
      }
      else {
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 7, d_expa[cur]));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_a, 8, sizeof(int), &nblk_m));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_preprocess_a, 2,
                 NULL, global_a, local_a, 0, NULL,
                 NULL != evt_prof ? &evt_prof[3 * batch] : NULL));
    }
    } /* end host_a else */

    /* Launch preprocess_b on stream_b (or host callback + H2D) */
    if (0 != host_b) {
      const size_t bk_cur = (size_t)nkb * nblk_n * BN_PAD * nslices * BK * slice_elem;
      const int nkb_groups_b = (nkb + kgroup - 1) / kgroup;
      const size_t expb_cur = (size_t)nkb_groups_b * nblk_n * BN * sizeof(cl_short);
      ctx->host_preprocess_b(b, ldb, tb, N, K, kb_batch,
        nkb, nblk_n, BN, BK, nslices, kgroup, ctx->use_xmx,
        h_bk[cur], h_expb[cur]);
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(
        h_bk[cur], d_bk[cur], bk_cur, stream_b);
      if (0 < expb_cur && NULL != h_expb[cur]) {
        if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(
          h_expb[cur], d_expb[cur], expb_cur, stream_b);
      }
    }
    else {
    /* GPU preprocess_b on stream_b (parallel with preprocess_a) */
    { const libxstream_opencl_stream_t* str_b = stream_b;
      const int nkb_groups_b = (nkb + kgroup - 1) / kgroup;
      size_t global_b[2], local_b[2];
      global_b[0] = (size_t)nblk_n * BN; global_b[1] = (size_t)nkb_groups_b * BK;
      local_b[0] = BN; local_b[1] = BK;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 0, d_b));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 1, sizeof(int), &N));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 2, sizeof(int), &K));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 3, sizeof(int), &ldb));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 4, sizeof(int), &tb));
      CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 5, sizeof(int), &kb_batch));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 6, d_bk[cur]));
      if (ctx->use_bf16) {
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 7, sizeof(int), &nblk_n));
      }
      else {
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 7, d_expb[cur]));
        CL_CHECK(result, clSetKernelArg(ctx->kern_preprocess_b, 8, sizeof(int), &nblk_n));
      }
      CL_CHECK(result, clEnqueueNDRangeKernel(str_b->queue, ctx->kern_preprocess_b, 2,
                 NULL, global_b, local_b, 0, NULL,
                 NULL != evt_prof ? &evt_prof[3 * batch + 1] : NULL));
    }
    } /* end host_b else */

    /* Record preprocess completion events */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

    /* Main stream waits for both preprocess results */
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Launch dotprod on main stream */
    if (2 == ctx->kind && ctx->use_xmx) { 
      /* CRT + XMX: fused DPAS + Garner + Horner, writes directly to C */
      cl_int i = 0;
      size_t global_c[2], local_c[2];
      const int ntm = BM / 8, ntn = BN / 16;
      local_c[0] = (size_t)ctx->sg;
      local_c[1] = (size_t)(ntm * ntn);
      global_c[0] = (size_t)nblk_m * local_c[0];
      global_c[1] = (size_t)nblk_n * local_c[1];
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_ak[cur]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expa[cur]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_bk[cur]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expb[cur]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_c));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &M));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &N));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &ldc));
      if (ctx->use_double) {
        double dalpha = alpha, dbeta = beta;
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dalpha));
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dbeta));
      }
      else {
        float falpha = (float)alpha, fbeta = (float)beta;
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &falpha));
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &fbeta));
      }
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &first_batch));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nkb));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_m));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_n));
      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_dotprod, 2,
                 NULL, global_c, local_c, 0, NULL,
                 NULL != evt_prof ? &evt_prof[3 * batch + 2] : NULL));
    }
    else { /* Scheme 1 or scalar CRT path */
      cl_int i = 0;
      size_t global_c[2], local_c[2];
      if (ctx->use_xmx) {
        const int ntm = BM / 8, ntn = BN / 16;  /* XMX_M=8, XMX_N=16 */
        local_c[0] = (size_t)ctx->sg;  /* SG=16 */
        local_c[1] = (size_t)(ntm * ntn);
        global_c[0] = (size_t)nblk_m * local_c[0];
        global_c[1] = (size_t)nblk_n * local_c[1];
      }
      else {
        local_c[0] = BM; local_c[1] = BN;
        global_c[0] = (size_t)nblk_m * BM;
        global_c[1] = (size_t)nblk_n * BN;
      }
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_ak[cur]));
      if (!ctx->use_bf16) {
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expa[cur]));
      }
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_bk[cur]));
      if (!ctx->use_bf16) {
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expb[cur]));
      }
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_c));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &M));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &N));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &ldc));
      if (ctx->use_double) {
        double dalpha = alpha, dbeta = beta;
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dalpha));
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dbeta));
      }
      else {
        float falpha = (float)alpha, fbeta = (float)beta;
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &falpha));
        CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &fbeta));
      }
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &first_batch));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nkb));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_m));
      CL_CHECK(result, clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_n));
      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->kern_dotprod, 2,
                 NULL, global_c, local_c, 0, NULL,
                 NULL != evt_prof ? &evt_prof[3 * batch + 2] : NULL));
    } /* end else (non-CRT-XMX path) */

    /* Record dotprod completion for this buffer slot */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_dotprod[cur], stream);
    if (NULL != evt_prof && EXIT_SUCCESS == result) n_profiled = 3 * (batch + 1);
  }

  /* Collect profiling data (wait on captured events, not full queues) */
  if (NULL != evt_prof) {
    double total = 0;
    int i;
    if (0 < n_profiled) clWaitForEvents((cl_uint)n_profiled, evt_prof);
    for (i = 0; i < n_profiled; ++i) {
      total += libxstream_opencl_duration(evt_prof[i], NULL);
      clReleaseEvent(evt_prof[i]);
    }
    /* Release orphaned events from a partially-failed batch */
    for (; i < 3 * n_batches; ++i) {
      if (NULL != evt_prof[i]) clReleaseEvent(evt_prof[i]);
    }
    if (0 < total) {
      const double gflops = (2.0 * M * N * K) / (total * 1E9);
      libxs_hist_push(NULL, ctx->hist, &gflops);
    }
    free(evt_prof);
  }

  /* Read back result C; caller is responsible for syncing the stream */
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_c, c, c_nbytes, stream);

  /* Sync helper streams and free host staging buffers.
   * The sync ensures all H2D transfers from staging buffers have completed
   * before freeing the source memory. */
  if (0 != host_a) libxstream_stream_sync(stream_a);
  if (0 != host_b) libxstream_stream_sync(stream_b);
  { int s;
    for (s = 0; s < 2; ++s) {
      free(h_ak[s]); free(h_expa[s]);
      free(h_bk[s]); free(h_expb[s]);
    }
  }

  /* Return buffers to pool (no deallocation, no sync needed) or free directly */
  { int s;
    for (s = 0; s < 2; ++s) {
      OZAKI_DEV_FREE(d_ak[s]);   OZAKI_DEV_FREE(d_expa[s]);
      OZAKI_DEV_FREE(d_bk[s]);   OZAKI_DEV_FREE(d_expb[s]);
    }
  }
  /* Return input/output buffers to pool or free directly */
  OZAKI_DEV_FREE(d_a); OZAKI_DEV_FREE(d_b); OZAKI_DEV_FREE(d_c);

  return result;
}
