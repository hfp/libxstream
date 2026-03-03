/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/*
 * Ozaki Scheme 1 OpenCL host driver.
 *
 * Demonstrates high-precision GEMM emulation via mantissa slicing on GPU
 * matrix engines. The preprocessing (decompose FP64/FP32 into int8 slices)
 * and the dot-product accumulation are performed entirely on the device.
 *
 * Build:  make [USE_DOUBLE=1] [DBG=1]
 * Usage:  ./ozaki_bench M N K [nslices]
 */
#include "ozaki_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Embedded kernel source (generated at build time via acc_opencl.sh) */
#include "ozaki_kernels.h"

#define CL_CHECK(CALL) do { \
  cl_int _err = (CALL); \
  if (CL_SUCCESS != _err) { \
    fprintf(stderr, "ERROR %s (%d) at %s:%d\n", cl_strerror(_err), (int)_err, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI)"
#endif


/* Function prototypes */
static const char* cl_strerror(cl_int err);
static void ozaki_print_opt(FILE* stream, const char* name, int val);
static int ozaki_init(ozaki_context_t* ctx, int use_double, int nslices,
                      int ozflags, int oztrim);
static void ozaki_destroy(ozaki_context_t* ctx);
static int ozaki_gemm(ozaki_context_t* ctx, void* stream,
                      char transa, char transb,
                      int M, int N, int K,
                      double alpha, const void* a, int lda,
                                    const void* b, int ldb,
                      double beta,        void* c, int ldc);
static void ref_dgemm(char transa, char transb,
                      int M, int N, int K,
                      double alpha, const double* a, int lda,
                                    const double* b, int ldb,
                      double beta,        double* c, int ldc);


int main(int argc, char* argv[])
{
  ozaki_context_t ctx;
  int M = 64, N = 64, K = 64;
  int nslices = OZAKI_NSLICES;
  int ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;
  int oztrim = 0;
  double alpha = 1.0, beta = 0.0;
  void *a = NULL, *b = NULL, *c_oz = NULL, *c_ref = NULL;
  void* stream = NULL;
  libxs_matdiff_info_t diff;
  libxs_timer_tick_t t0, t1;
  int i, result;
  const char* env;

  if (argc > 1) M = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) K = atoi(argv[3]);
  if (argc > 4) nslices = atoi(argv[4]);
  if (M <= 0) M = 64;
  if (N <= 0) N = 64;
  if (K <= 0) K = 64;
  if (nslices < 1 || nslices > 16) nslices = OZAKI_NSLICES;

  env = getenv("OZAKI_FLAGS");
  if (NULL != env) ozflags = atoi(env);
  env = getenv("OZAKI_TRIM");
  if (NULL != env) oztrim = atoi(env);

  /* Initialize ACC (encompasses libxs initialization) */
  result = c_dbcsr_acc_init();
  if (EXIT_SUCCESS == result) {
    int ndevices = 0;
    result = c_dbcsr_acc_get_ndevices(&ndevices);
    if (EXIT_SUCCESS == result && 0 < ndevices) {
      result = c_dbcsr_acc_set_active_device(0);
    }
    else {
      fprintf(stderr, "ERROR: no ACC device found\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: ACC initialization failed\n");
    return EXIT_FAILURE;
  }

  printf("Ozaki Scheme 1 OpenCL benchmark\n");
  printf("M=%d N=%d K=%d nslices=%d flags=%d trim=%d\n", M, N, K, nslices, ozflags, oztrim);

  /* Initialize Ozaki context (kernels) */
  LIBXS_MEMZERO(&ctx);
  result = ozaki_init(&ctx, 1 /*use_double*/, nslices, ozflags, oztrim);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "Failed to initialize Ozaki OpenCL context\n");
    c_dbcsr_acc_finalize();
    return EXIT_FAILURE;
  }

  /* Create own ACC stream (enables double-buffered transfers) */
  result = c_dbcsr_acc_stream_create(&stream, "ozaki_main", -1 /*default priority*/);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to create ACC stream\n");
    ozaki_destroy(&ctx);
    c_dbcsr_acc_finalize();
    return EXIT_FAILURE;
  }

  /* Allocate and fill matrices (column-major, lda=M, ldb=K, ldc=M) */
  result = c_dbcsr_acc_host_mem_allocate(&a, (size_t)M * K * sizeof(double), stream);
  if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&b, (size_t)K * N * sizeof(double), stream);
  if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&c_oz, (size_t)M * N * sizeof(double), stream);
  if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&c_ref, (size_t)M * N * sizeof(double), stream);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: out of memory\n");
    c_dbcsr_acc_stream_destroy(stream);
    ozaki_destroy(&ctx);
    c_dbcsr_acc_finalize();
    return EXIT_FAILURE;
  }

  { double* da = (double*)a;
    double* db = (double*)b;
    double* dc = (double*)c_oz;
    srand(42);
    for (i = 0; i < M * K; ++i) da[i] = (double)rand() / RAND_MAX - 0.5;
    for (i = 0; i < K * N; ++i) db[i] = (double)rand() / RAND_MAX - 0.5;
    for (i = 0; i < M * N; ++i) dc[i] = 0.0;
    memcpy(c_ref, c_oz, (size_t)M * N * sizeof(double));
  }

  /* Run Ozaki OpenCL GEMM */
  t0 = libxs_timer_tick();
  result = ozaki_gemm(&ctx, stream, 'N', 'N', M, N, K, alpha, a, M, b, K, beta, c_oz, M);
  c_dbcsr_acc_stream_sync(stream);
  t1 = libxs_timer_tick();
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "Ozaki GEMM failed\n");
    c_dbcsr_acc_stream_destroy(stream);
    ozaki_destroy(&ctx);
    c_dbcsr_acc_finalize();
    return EXIT_FAILURE;
  }
  printf("Ozaki GEMM: %.3f ms\n", 1000.0 * libxs_timer_duration(t0, t1));

  /* Reference GEMM */
  t0 = libxs_timer_tick();
  ref_dgemm('N', 'N', M, N, K, alpha, (const double*)a, M, (const double*)b, K, beta, (double*)c_ref, M);
  t1 = libxs_timer_tick();
  printf("Ref   GEMM: %.3f ms\n", 1000.0 * libxs_timer_duration(t0, t1));

  /* Compare */
  result = libxs_matdiff(&diff, LIBXS_DATATYPE_F64, M, N, (const double*)c_ref, (const double*)c_oz, &M, &M);
  if (EXIT_SUCCESS == result) {
    printf("Max absolute diff: %.6e\n", diff.linf_abs);
    if (0.0 < diff.normi_abs) {
      printf("Max relative diff: %.6e\n", diff.linf_rel);
    }
  }

  c_dbcsr_acc_host_mem_deallocate(a, stream);
  c_dbcsr_acc_host_mem_deallocate(b, stream);
  c_dbcsr_acc_host_mem_deallocate(c_oz, stream);
  c_dbcsr_acc_host_mem_deallocate(c_ref, stream);
  c_dbcsr_acc_stream_destroy(stream);
  ozaki_destroy(&ctx);
  c_dbcsr_acc_finalize();
  return EXIT_SUCCESS;
}


static const char* cl_strerror(cl_int err) {
  switch (err) {
    case  0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -30: return "CL_INVALID_VALUE";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -48: return "CL_INVALID_KERNEL";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    default: return "CL_UNKNOWN_ERROR";
  }
}


static void ozaki_print_opt(FILE* stream, const char* name, int val) {
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}


static int ozaki_init(ozaki_context_t* ctx, int use_double, int nslices,
                      int ozflags, int oztrim)
{
  const c_dbcsr_acc_opencl_device_t* devinfo = &c_dbcsr_acc_opencl_config.device;
  cl_device_id device = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
  char build_options[256];
  char build_params[1024];
  size_t offset;
  const char* env;
  int verbosity, wg, sg, gpu, result;

  /* Verbosity from environment */
  env = getenv("OZAKI_VERBOSE");
  verbosity = (NULL != env ? atoi(env) : 0);

  gpu = (CL_DEVICE_TYPE_GPU == devinfo->type) ? 1 : 0;

  { char name[256] = "";
    c_dbcsr_acc_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char* const fp64_ext[] = {"cl_khr_fp64"};
    if (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_ext(device, fp64_ext, 1)) {
      fprintf(stderr, "WARN: device does not support cl_khr_fp64, falling back to float\n");
      use_double = 0;
    }
  }

  ctx->use_double = use_double;
  ctx->nslices = nslices;
  ctx->ozflags = ozflags;
  ctx->oztrim  = oztrim;
  ctx->verbosity = verbosity;



  /* Environment-driven tuning */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0);
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)devinfo->wgsize[2]);

  /* Assemble JIT build flags: compiler options and -D defines separately */
  LIBXS_SNPRINTF(build_options, sizeof(build_options),
    "-cl-fast-relaxed-math -cl-denorms-are-zero");

  { int mant_bits      = use_double ? 52 : 23;
    int bias_plus_mant = use_double ? 1075 : 150;
    const char* constant_qual;

    env = getenv("OZAKI_CONSTANT");
    constant_qual = (NULL != env && 0 != atoi(env)) ? "constant" : "global";

    offset = 0;
    if (gpu) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        "-DGPU ");
    }

    offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
      "-DCONSTANT=%s -DBM=%d -DBN=%d -DBK=%d -DNSLICES=%d"
      " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
      " -DTRIANGULAR=%d -DSYMMETRIZE=%d -DTRIM=%d"
      " -DUSE_DOUBLE=%d",
      constant_qual, OZAKI_BM, OZAKI_BN, OZAKI_BK, nslices,
      mant_bits, bias_plus_mant,
      (ozflags & OZAKI_TRIANGULAR) ? 1 : 0,
      (ozflags & OZAKI_SYMMETRIZE) ? 1 : 0,
      oztrim, use_double);

    if (0 < wg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DWG=%d", wg);
    }
    if (0 < sg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DSG=%d", sg);
    }
  }

  if (0 < verbosity) {
    fprintf(stderr, "INFO OZAKI: build params: %s\n", build_params);
    fprintf(stderr, "INFO OZAKI: build options: %s\n", build_options);
  }

  /* JIT compile kernels via ACC */
  result = c_dbcsr_acc_opencl_kernel(0 /*source*/, OPENCL_KERNELS_SOURCE_OZAKI,
    "preprocess_a", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_preprocess_a);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build preprocess_a kernel\n");
    return EXIT_FAILURE;
  }

  result = c_dbcsr_acc_opencl_kernel(0 /*source*/, OPENCL_KERNELS_SOURCE_OZAKI,
    "preprocess_b", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_preprocess_b);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build preprocess_b kernel\n");
    return EXIT_FAILURE;
  }

  result = c_dbcsr_acc_opencl_kernel(0 /*source*/, OPENCL_KERNELS_SOURCE_OZAKI,
    "dotprod", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_dotprod);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build dotprod kernel\n");
    return EXIT_FAILURE;
  }

  /* Report compiled kernel info */
  if (0 < verbosity) {
    size_t wgs[3] = {0};
    if (CL_SUCCESS == clGetKernelWorkGroupInfo(ctx->kern_dotprod, device,
          CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgs), wgs, NULL)) {
      fprintf(stderr, "INFO OZAKI: dotprod compiled WG=%lux%lux%lu\n",
              (unsigned long)wgs[0], (unsigned long)wgs[1], (unsigned long)wgs[2]);
    }
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "nslices", nslices);
    fprintf(stderr, "\n");
  }

  return EXIT_SUCCESS;
}


static void ozaki_destroy(ozaki_context_t* ctx)
{
  if (NULL != ctx->kern_preprocess_a) clReleaseKernel(ctx->kern_preprocess_a);
  if (NULL != ctx->kern_preprocess_b) clReleaseKernel(ctx->kern_preprocess_b);
  if (NULL != ctx->kern_dotprod)      clReleaseKernel(ctx->kern_dotprod);
  LIBXS_MEMZERO(ctx);
}


static int ozaki_gemm(ozaki_context_t* ctx, void* stream,
                      char transa, char transb,
                      int M, int N, int K,
                      double alpha, const void* a, int lda,
                                    const void* b, int ldb,
                      double beta,        void* c, int ldc)
{
  const c_dbcsr_acc_opencl_stream_t* str = ACC_OPENCL_STREAM(stream);
  const int BM = OZAKI_BM, BN = OZAKI_BN, BK = OZAKI_BK;
  const int BATCH_K = OZAKI_BATCH_K;
  const int nslices = ctx->nslices;
  const int nblk_m = (M + BM - 1) / BM;
  const int nblk_n = (N + BN - 1) / BN;
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  void *d_a = NULL, *d_b = NULL, *d_c = NULL;
  size_t c_nbytes;
  int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;

  /* Allocate device memory and transfer A, B, C */
  { size_t a_cols = ta ? (size_t)M : (size_t)K;
    size_t b_cols = tb ? (size_t)K : (size_t)N;
    size_t a_nbytes = (size_t)lda * a_cols * elem_size;
    size_t b_nbytes = (size_t)ldb * b_cols * elem_size;
    c_nbytes = (size_t)ldc * (size_t)N * elem_size;

    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_a, a_nbytes);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_b, b_nbytes);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_c, c_nbytes);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_memcpy_h2d(a, d_a, a_nbytes, stream);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_memcpy_h2d(b, d_b, b_nbytes, stream);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_memcpy_h2d(c, d_c, c_nbytes, stream);
    if (EXIT_SUCCESS != result) return EXIT_FAILURE;
  }

  /* Process K in batches of BATCH_K * BK */
  { int kb_batch;
    for (kb_batch = 0; kb_batch < K; kb_batch += BATCH_K * BK) {
      int batch_end = (kb_batch + BATCH_K * BK < K) ? (kb_batch + BATCH_K * BK) : K;
      int nkb = (batch_end - kb_batch + BK - 1) / BK;
      void *d_ak = NULL, *d_expa = NULL, *d_bk = NULL, *d_expb = NULL;

      /* Allocate preprocessing buffers for this batch */
      { size_t ak_size = (size_t)nkb * nblk_m * BM * nslices * BK;
        size_t expa_size = (size_t)nkb * nblk_m * BM * sizeof(cl_short);
        size_t bk_size = (size_t)nkb * nblk_n * BN * nslices * BK;
        size_t expb_size = (size_t)nkb * nblk_n * BN * sizeof(cl_short);

        if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_ak, ak_size);
        if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_expa, expa_size);
        if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_bk, bk_size);
        if (EXIT_SUCCESS == result) result = c_dbcsr_acc_dev_mem_allocate(&d_expb, expb_size);
        if (EXIT_SUCCESS != result) return EXIT_FAILURE;
      }

      /* Launch preprocess_a: one work-group per (row-block, k-sub) */
      { size_t global_a[2], local_a[2];
        global_a[0] = (size_t)nblk_m * BM; global_a[1] = (size_t)nkb * BK;
        local_a[0] = BM; local_a[1] = BK;
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 0, d_a));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 1, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 2, sizeof(int), &K));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 3, sizeof(int), &lda));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 4, sizeof(int), &ta));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 5, sizeof(int), &kb_batch));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 6, d_ak));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 7, d_expa));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 8, sizeof(int), &nblk_m));
        CL_CHECK(clEnqueueNDRangeKernel(str->queue, ctx->kern_preprocess_a, 2,
                   NULL, global_a, local_a, 0, NULL, NULL));
      }

      /* Launch preprocess_b: one work-group per (col-block, k-sub) */
      { size_t global_b[2], local_b[2];
        global_b[0] = (size_t)nblk_n * BN; global_b[1] = (size_t)nkb * BK;
        local_b[0] = BN; local_b[1] = BK;
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 0, d_b));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 1, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 2, sizeof(int), &K));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 3, sizeof(int), &ldb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 4, sizeof(int), &tb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 5, sizeof(int), &kb_batch));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 6, d_bk));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 7, d_expb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 8, sizeof(int), &nblk_n));
        CL_CHECK(clEnqueueNDRangeKernel(str->queue, ctx->kern_preprocess_b, 2,
                   NULL, global_b, local_b, 0, NULL, NULL));
      }

      /* Launch dotprod: one work-group per C tile */
      { int first_batch = (0 == kb_batch) ? 1 : 0;
        cl_int i = 0;
        size_t global_c[2], local_c[2];
        global_c[0] = (size_t)nblk_m * BM; global_c[1] = (size_t)nblk_n * BN;
        local_c[0] = BM; local_c[1] = BN;
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_ak));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expa));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_bk));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expb));
        CL_CHECK(c_dbcsr_acc_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_c));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &ldc));
        if (ctx->use_double) {
          double dalpha = alpha, dbeta = beta;
          CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dalpha));
          CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dbeta));
        }
        else {
          float falpha = (float)alpha, fbeta = (float)beta;
          CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &falpha));
          CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &fbeta));
        }
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &first_batch));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nkb));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_m));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_n));
        CL_CHECK(clEnqueueNDRangeKernel(str->queue, ctx->kern_dotprod, 2,
                   NULL, global_c, local_c, 0, NULL, NULL));
      }

      /* Sync and deallocate batch buffers */
      c_dbcsr_acc_stream_sync(stream);
      c_dbcsr_acc_dev_mem_deallocate(d_ak);
      c_dbcsr_acc_dev_mem_deallocate(d_expa);
      c_dbcsr_acc_dev_mem_deallocate(d_bk);
      c_dbcsr_acc_dev_mem_deallocate(d_expb);
    }
  }

  /* Read back result C */
  result = c_dbcsr_acc_memcpy_d2h(d_c, c, c_nbytes, stream);
  if (EXIT_SUCCESS == result) result = c_dbcsr_acc_stream_sync(stream);

  c_dbcsr_acc_dev_mem_deallocate(d_a);
  c_dbcsr_acc_dev_mem_deallocate(d_b);
  c_dbcsr_acc_dev_mem_deallocate(d_c);

  return result;
}


static void ref_dgemm(char transa, char transb,
                      int M, int N, int K,
                      double alpha, const double* a, int lda,
                                    const double* b, int ldb,
                      double beta,        double* c, int ldc)
{
  int i, j, p;
  int ta = (transa != 'N' && transa != 'n');
  int tb = (transb != 'N' && transb != 'n');
  for (j = 0; j < N; ++j) {
    for (i = 0; i < M; ++i) {
      double sum = 0.0;
      for (p = 0; p < K; ++p) {
        double aval = ta ? a[i * lda + p] : a[p * lda + i];
        double bval = tb ? b[p * ldb + j] : b[j * ldb + p];
        sum += aval * bval;
      }
      c[j * ldc + i] = alpha * sum + beta * c[j * ldc + i];
    }
  }
}
