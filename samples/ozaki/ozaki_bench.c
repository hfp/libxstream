/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/

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
#include <math.h>
#include <time.h>

/* Embedded kernel source (generated at build time via acc_opencl.sh) */
#include "ozaki_kernels.h"

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI)"
#endif


/* ---- helpers ---- */

static double seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
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

#define CL_CHECK(call) do { \
  cl_int _err = (call); \
  if (CL_SUCCESS != _err) { \
    fprintf(stderr, "ERROR %s (%d) at %s:%d\n", cl_strerror(_err), (int)_err, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


/* ---- context setup ---- */

static void ozaki_print_opt(FILE* stream, const char* name, int val) {
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}

static int ozaki_init(ozaki_context_t* ctx, int use_double, int nslices,
                      int ozflags, int oztrim)
{
  cl_platform_id platform;
  cl_device_type devtype = 0;
  cl_uint ndev = 0;
  cl_int err;
  char build_opts[1024];
  size_t offset;
  const char *env;
  int verbosity, wg, sg;

  /* Verbosity from environment (like DBM_MULTIPLY) */
  env = getenv("OZAKI_VERBOSE");
  verbosity = (NULL != env ? atoi(env) : 0);

  /* Pick first GPU (or first device if no GPU) */
  CL_CHECK(clGetPlatformIDs(1, &platform, NULL));
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, &ndev);
  if (CL_SUCCESS != err || 0 == ndev) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &ctx->device, &ndev);
    if (CL_SUCCESS != err || 0 == ndev) {
      fprintf(stderr, "ERROR: no OpenCL device found\n");
      return EXIT_FAILURE;
    }
  }

  /* Query device type and properties */
  clGetDeviceInfo(ctx->device, CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
  ctx->gpu = (CL_DEVICE_TYPE_GPU == devtype) ? 1 : 0;

  { char name[256] = "";
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Device: %s%s\n", name, ctx->gpu ? " (GPU)" : "");
  }

  /* Query max work-group size */
  { size_t max_wgsize = 0;
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_wgsize), &max_wgsize, NULL);
    ctx->wgsize = max_wgsize;
  }

  /* Query preferred sub-group size (Intel/OpenCL 2.1+) */
  ctx->sgsize = 0;
#if defined(CL_DEVICE_SUB_GROUP_SIZES_INTEL)
  { size_t sgsizes[8] = {0};
    size_t ret = 0;
    if (CL_SUCCESS == clGetDeviceInfo(ctx->device, CL_DEVICE_SUB_GROUP_SIZES_INTEL,
                                      sizeof(sgsizes), sgsizes, &ret) && ret > 0) {
      ctx->sgsize = sgsizes[0]; /* smallest available sub-group size */
    }
  }
#endif

  /* If double requested, verify fp64 support */
  if (use_double) {
    char exts[4096] = "";
    clGetDeviceInfo(ctx->device, CL_DEVICE_EXTENSIONS, sizeof(exts), exts, NULL);
    if (NULL == strstr(exts, "cl_khr_fp64")) {
      fprintf(stderr, "WARN: device does not support cl_khr_fp64, falling back to float\n");
      use_double = 0;
    }
  }

  ctx->use_double = use_double;
  ctx->nslices = nslices;
  ctx->ozflags = ozflags;
  ctx->oztrim  = oztrim;
  ctx->verbosity = verbosity;

  ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
  CL_CHECK(err);
  { cl_queue_properties props[] = { 0 };
    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
    CL_CHECK(err);
  }

  /* Environment-driven tuning (like dbm_multiply_opencl.c) */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0); /* 0: use defaults from kernel */
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)ctx->sgsize);

  /* Assemble JIT build flags with macro templating */
  { int mant_bits      = use_double ? 52 : 23;
    int bias_plus_mant = use_double ? 1075 : 150;
    const char *constant_qual;

    /* CONSTANT address-space qualifier: "constant" may be faster for small
     * read-only buffers on some devices, but "global" is the safe default.
     * Like dbm_multiply_opencl.c, allow override via environment. */
    env = getenv("OZAKI_CONSTANT");
    constant_qual = (NULL != env && 0 != atoi(env)) ? "constant" : "global";

    /* Fast-math flags (safe: ozaki kernels use integer slicing) */
    offset = (size_t)snprintf(build_opts, sizeof(build_opts),
      "-cl-fast-relaxed-math -cl-denorms-are-zero");

    /* GPU flag for sub-group broadcasts */
    if (ctx->gpu) {
      offset += (size_t)snprintf(build_opts + offset, sizeof(build_opts) - offset,
        " -DGPU");
    }

    /* Core parameters */
    offset += (size_t)snprintf(build_opts + offset, sizeof(build_opts) - offset,
      " -DCONSTANT=%s -DBM=%d -DBN=%d -DBK=%d -DNSLICES=%d"
      " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d"
      " -DTRIANGULAR=%d -DSYMMETRIZE=%d -DTRIM=%d"
      " -DUSE_DOUBLE=%d",
      constant_qual, OZAKI_BM, OZAKI_BN, OZAKI_BK, nslices,
      mant_bits, bias_plus_mant,
      (ozflags & OZAKI_TRIANGULAR) ? 1 : 0,
      (ozflags & OZAKI_SYMMETRIZE) ? 1 : 0,
      oztrim, use_double);

    /* Work-group / sub-group hints (like dbm_multiply_opencl.c -DWG= -DSG=) */
    if (0 < wg) {
      offset += (size_t)snprintf(build_opts + offset, sizeof(build_opts) - offset,
        " -DWG=%d", wg);
    }
    if (0 < sg) {
      offset += (size_t)snprintf(build_opts + offset, sizeof(build_opts) - offset,
        " -DSG=%d", sg);
    }
  }

  /* Create program from embedded source */
  { const char* src = OPENCL_KERNELS_SOURCE_OZAKI;
    size_t srclen = strlen(src);
    ctx->program = clCreateProgramWithSource(ctx->context, 1, &src, &srclen, &err);
    CL_CHECK(err);
  }

  /* JIT compile with assembled flags */
  if (0 < verbosity) {
    fprintf(stderr, "INFO OZAKI: build flags: %s\n", build_opts);
  }
  err = clBuildProgram(ctx->program, 1, &ctx->device, build_opts, NULL, NULL);
  if (CL_SUCCESS != err) {
    char log[8192];
    clGetProgramBuildInfo(ctx->program, ctx->device,
      CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "Build log:\n%s\n", log);
    return EXIT_FAILURE;
  }

  ctx->kern_preprocess_a = clCreateKernel(ctx->program, "preprocess_a", &err);
  CL_CHECK(err);
  ctx->kern_preprocess_b = clCreateKernel(ctx->program, "preprocess_b", &err);
  CL_CHECK(err);
  ctx->kern_dotprod = clCreateKernel(ctx->program, "dotprod", &err);
  CL_CHECK(err);

  /* Query compiled kernel for actual work-group size (like dbm_multiply_opencl.c) */
  if (0 < verbosity) {
    size_t wgs[3] = {0};
    if (CL_SUCCESS == clGetKernelWorkGroupInfo(ctx->kern_dotprod, ctx->device,
          CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgs), wgs, NULL)) {
      fprintf(stderr, "INFO OZAKI: dotprod compiled WG=%zux%zux%zu\n",
              wgs[0], wgs[1], wgs[2]);
    }
    fprintf(stderr, "INFO OZAKI: gpu=%d", ctx->gpu);
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
  if (NULL != ctx->program)           clReleaseProgram(ctx->program);
  if (NULL != ctx->queue)             clReleaseCommandQueue(ctx->queue);
  if (NULL != ctx->context)           clReleaseContext(ctx->context);
  memset(ctx, 0, sizeof(*ctx));
}


/* ---- GEMM driver ---- */

static int ozaki_gemm(ozaki_context_t* ctx,
                      char transa, char transb,
                      int M, int N, int K,
                      double alpha, const void* a, int lda,
                                    const void* b, int ldb,
                      double beta,        void* c, int ldc)
{
  const int BM = OZAKI_BM, BN = OZAKI_BN, BK = OZAKI_BK;
  const int BATCH_K = OZAKI_BATCH_K;
  const int nslices = ctx->nslices;
  const int nblk_m = (M + BM - 1) / BM;
  const int nblk_n = (N + BN - 1) / BN;
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);

  cl_mem d_a, d_b, d_c;
  cl_mem d_ak, d_expa, d_bk, d_expb;
  cl_int err;
  int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  int tb = (transb != 'N' && transb != 'n') ? 1 : 0;

  /* Transfer A, B, C to device */
  { size_t a_cols = ta ? (size_t)M : (size_t)K;
    size_t b_cols = tb ? (size_t)K : (size_t)N;
    d_a = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (size_t)lda * a_cols * elem_size, (void*)(uintptr_t)a, &err);
    CL_CHECK(err);

    d_b = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (size_t)ldb * b_cols * elem_size, (void*)(uintptr_t)b, &err);
    CL_CHECK(err);

    d_c = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            (size_t)ldc * (size_t)N * elem_size, c, &err);
    CL_CHECK(err);
  }

  /* Process K in batches of BATCH_K * BK */
  { int kb_batch;
    for (kb_batch = 0; kb_batch < K; kb_batch += BATCH_K * BK) {
      int batch_end = (kb_batch + BATCH_K * BK < K) ? (kb_batch + BATCH_K * BK) : K;
      int nkb = (batch_end - kb_batch + BK - 1) / BK;

      /* Allocate preprocessing buffers for this batch */
      { size_t ak_size = (size_t)nkb * nblk_m * BM * nslices * BK;
        size_t expa_size = (size_t)nkb * nblk_m * BM * sizeof(cl_short);
        size_t bk_size = (size_t)nkb * nblk_n * BN * nslices * BK;
        size_t expb_size = (size_t)nkb * nblk_n * BN * sizeof(cl_short);

        d_ak   = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, ak_size, NULL, &err);   CL_CHECK(err);
        d_expa = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, expa_size, NULL, &err);  CL_CHECK(err);
        d_bk   = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bk_size, NULL, &err);   CL_CHECK(err);
        d_expb = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, expb_size, NULL, &err);  CL_CHECK(err);
      }

      /* Launch preprocess_a: one work-group per (row-block, k-sub) */
      { size_t global_a[2] = { (size_t)nblk_m * BM, (size_t)nkb * BK };
        size_t local_a[2]  = { BM, BK };
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 0, sizeof(cl_mem), &d_a));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 1, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 2, sizeof(int), &K));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 3, sizeof(int), &lda));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 4, sizeof(int), &ta));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 5, sizeof(int), &kb_batch));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 6, sizeof(cl_mem), &d_ak));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 7, sizeof(cl_mem), &d_expa));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 8, sizeof(int), &nblk_m));
        CL_CHECK(clEnqueueNDRangeKernel(ctx->queue, ctx->kern_preprocess_a, 2,
                   NULL, global_a, local_a, 0, NULL, NULL));
      }

      /* Launch preprocess_b: one work-group per (col-block, k-sub) */
      { size_t global_b[2] = { (size_t)nblk_n * BN, (size_t)nkb * BK };
        size_t local_b[2]  = { BN, BK };
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 0, sizeof(cl_mem), &d_b));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 1, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 2, sizeof(int), &K));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 3, sizeof(int), &ldb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 4, sizeof(int), &tb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 5, sizeof(int), &kb_batch));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 6, sizeof(cl_mem), &d_bk));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 7, sizeof(cl_mem), &d_expb));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 8, sizeof(int), &nblk_n));
        CL_CHECK(clEnqueueNDRangeKernel(ctx->queue, ctx->kern_preprocess_b, 2,
                   NULL, global_b, local_b, 0, NULL, NULL));
      }

      /* Launch dotprod: one work-group per C tile */
      { size_t global_c[2] = { (size_t)nblk_m * BM, (size_t)nblk_n * BN };
        size_t local_c[2]  = { BM, BN };
        int first_batch = (0 == kb_batch) ? 1 : 0;
        cl_int i = 0;
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(cl_mem), &d_ak));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(cl_mem), &d_expa));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(cl_mem), &d_bk));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(cl_mem), &d_expb));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(cl_mem), &d_c));
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
        CL_CHECK(clEnqueueNDRangeKernel(ctx->queue, ctx->kern_dotprod, 2,
                   NULL, global_c, local_c, 0, NULL, NULL));
      }

      CL_CHECK(clReleaseMemObject(d_ak));
      CL_CHECK(clReleaseMemObject(d_expa));
      CL_CHECK(clReleaseMemObject(d_bk));
      CL_CHECK(clReleaseMemObject(d_expb));
    }
  }

  /* Read back result C */
  CL_CHECK(clEnqueueReadBuffer(ctx->queue, d_c, CL_TRUE, 0,
             (size_t)ldc * (size_t)N * elem_size, c, 0, NULL, NULL));

  CL_CHECK(clReleaseMemObject(d_a));
  CL_CHECK(clReleaseMemObject(d_b));
  CL_CHECK(clReleaseMemObject(d_c));

  return EXIT_SUCCESS;
}


/* ---- reference naive GEMM (for validation) ---- */

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


/* ---- main ---- */

int main(int argc, char* argv[])
{
  ozaki_context_t ctx;
  int M = 64, N = 64, K = 64;
  int nslices = OZAKI_NSLICES;
  int ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;
  int oztrim = 0;
  double alpha = 1.0, beta = 0.0;
  double *a, *b, *c_oz, *c_ref;
  double maxdiff, t0, t1;
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

  printf("Ozaki Scheme 1 OpenCL benchmark\n");
  printf("M=%d N=%d K=%d nslices=%d flags=%d trim=%d\n", M, N, K, nslices, ozflags, oztrim);

  /* Initialize OpenCL context and kernels */
  memset(&ctx, 0, sizeof(ctx));
  result = ozaki_init(&ctx, 1 /*use_double*/, nslices, ozflags, oztrim);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "Failed to initialize Ozaki OpenCL context\n");
    return EXIT_FAILURE;
  }

  /* Allocate and fill matrices (column-major, lda=M, ldb=K, ldc=M) */
  a     = (double*)malloc((size_t)M * K * sizeof(double));
  b     = (double*)malloc((size_t)K * N * sizeof(double));
  c_oz  = (double*)malloc((size_t)M * N * sizeof(double));
  c_ref = (double*)malloc((size_t)M * N * sizeof(double));
  if (NULL == a || NULL == b || NULL == c_oz || NULL == c_ref) {
    fprintf(stderr, "ERROR: out of memory\n");
    return EXIT_FAILURE;
  }

  srand(42);
  for (i = 0; i < M * K; ++i) a[i]     = (double)rand() / RAND_MAX - 0.5;
  for (i = 0; i < K * N; ++i) b[i]     = (double)rand() / RAND_MAX - 0.5;
  for (i = 0; i < M * N; ++i) c_oz[i]  = 0.0;
  memcpy(c_ref, c_oz, (size_t)M * N * sizeof(double));

  /* Run Ozaki OpenCL GEMM */
  t0 = seconds();
  result = ozaki_gemm(&ctx, 'N', 'N', M, N, K, alpha, a, M, b, K, beta, c_oz, M);
  clFinish(ctx.queue);
  t1 = seconds();
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "Ozaki GEMM failed\n");
    ozaki_destroy(&ctx);
    return EXIT_FAILURE;
  }
  printf("Ozaki GEMM: %.3f ms\n", 1000.0 * (t1 - t0));

  /* Reference GEMM */
  t0 = seconds();
  ref_dgemm('N', 'N', M, N, K, alpha, a, M, b, K, beta, c_ref, M);
  t1 = seconds();
  printf("Ref   GEMM: %.3f ms\n", 1000.0 * (t1 - t0));

  /* Compare */
  maxdiff = 0.0;
  for (i = 0; i < M * N; ++i) {
    double d = fabs(c_oz[i] - c_ref[i]);
    if (d > maxdiff) maxdiff = d;
  }
  { double norm = 0.0;
    for (i = 0; i < M * N; ++i) {
      double a2 = fabs(c_ref[i]);
      if (a2 > norm) norm = a2;
    }
    printf("Max absolute diff: %.6e\n", maxdiff);
    if (norm > 0.0) {
      printf("Max relative diff: %.6e\n", maxdiff / norm);
    }
  }

  free(a); free(b); free(c_oz); free(c_ref);
  ozaki_destroy(&ctx);
  return EXIT_SUCCESS;
}
