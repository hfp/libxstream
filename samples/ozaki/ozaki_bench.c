/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/*
 * Ozaki Scheme 1 OpenCL benchmark driver.
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
#include <libxs_rng.h>


/* Function prototypes */
static void ref_dgemm(char transa, char transb,
                      int M, int N, int K,
                      double alpha, const double* a, int lda,
                                    const double* b, int ldb,
                      double beta,        double* c, int ldc);
static void ref_sgemm(char transa, char transb,
                      int M, int N, int K,
                      float alpha, const float* a, int lda,
                                   const float* b, int ldb,
                      float beta,        float* c, int ldc);


int main(int argc, char* argv[])
{
  ozaki_context_t ctx;
  int M = (1 < argc ? atoi(argv[1]) : 257);
  int N = (2 < argc ? atoi(argv[2]) : M);
  int K = (3 < argc ? atoi(argv[3]) : M);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  double alpha = (6 < argc ? atof(argv[6]) : 1);
  double beta  = (7 < argc ? atof(argv[7]) : 1);
  int lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? M : K));
  int ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? K : N));
  int ldc = (10 < argc ? atoi(argv[10]) : M);
  const char transa = (0 == ta ? 'N' : 'T');
  const char transb = (0 == tb ? 'N' : 'T');
  void *a = NULL, *b = NULL, *c_oz = NULL, *c_ref = NULL;
  void* stream = NULL;
  libxs_matdiff_info_t diff;
  libxs_timer_tick_t t0, t1;
  size_t elem_size;
  int result;

  if (1 > M || 1 > N || 1 > K || lda < (0 == ta ? M : K)
      || ldb < (0 == tb ? K : N) || ldc < M) {
    fprintf(stderr, "Invalid dimensions: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n",
      M, N, K, lda, ldb, ldc);
    return EXIT_FAILURE;
  }

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
  printf("%c%c M=%d N=%d K=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
    transa, transb, M, N, K, lda, ldb, ldc, alpha, beta);

  /* Initialize Ozaki context (kernels) */
  LIBXS_MEMZERO(&ctx);
  result = ozaki_init(&ctx, 1 /*use_double*/, OZAKI_NSLICES,
    OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE, 0 /*oztrim*/);
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

  /* Element size matches actual precision (may fall back to fp32) */
  elem_size = ctx.use_double ? sizeof(double) : sizeof(float);

  /* Allocate and fill matrices (column-major) */
  { const int a_rows = (0 == ta ? M : K), a_cols = (0 == ta ? K : M);
    const int b_rows = (0 == tb ? K : N), b_cols = (0 == tb ? N : K);
    result = c_dbcsr_acc_host_mem_allocate(&a, (size_t)lda * a_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&b, (size_t)ldb * b_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&c_oz, (size_t)ldc * N * elem_size, stream);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_host_mem_allocate(&c_ref, (size_t)ldc * N * elem_size, stream);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: out of memory\n");
      c_dbcsr_acc_stream_destroy(stream);
      ozaki_destroy(&ctx);
      c_dbcsr_acc_finalize();
      return EXIT_FAILURE;
    }
    if (ctx.use_double) {
      LIBXS_MATRNG(int, double, 0, a, a_rows, a_cols, lda, 1.0);
      LIBXS_MATRNG(int, double, 0, b, b_rows, b_cols, ldb, 1.0);
      LIBXS_MATRNG(int, double, 0, c_oz, M, N, ldc, 1.0);
    }
    else {
      LIBXS_MATRNG(int, float, 0, a, a_rows, a_cols, lda, 1.0);
      LIBXS_MATRNG(int, float, 0, b, b_rows, b_cols, ldb, 1.0);
      LIBXS_MATRNG(int, float, 0, c_oz, M, N, ldc, 1.0);
    }
    memcpy(c_ref, c_oz, (size_t)ldc * N * elem_size);
  }

  /* Run Ozaki OpenCL GEMM */
  t0 = libxs_timer_tick();
  result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc);
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
  if (ctx.use_double) {
    ref_dgemm(transa, transb, M, N, K, alpha, (const double*)a, lda, (const double*)b, ldb, beta, (double*)c_ref, ldc);
  }
  else {
    ref_sgemm(transa, transb, M, N, K, (float)alpha, (const float*)a, lda, (const float*)b, ldb, (float)beta, (float*)c_ref, ldc);
  }
  t1 = libxs_timer_tick();
  printf("Ref   GEMM: %.3f ms\n", 1000.0 * libxs_timer_duration(t0, t1));

  /* Compare */
  { libxs_datatype dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    result = libxs_matdiff(&diff, dtype, M, N, c_ref, c_oz, &ldc, &ldc);
  }
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


static void ref_sgemm(char transa, char transb,
                      int M, int N, int K,
                      float alpha, const float* a, int lda,
                                   const float* b, int ldb,
                      float beta,        float* c, int ldc)
{
  int i, j, p;
  int ta = (transa != 'N' && transa != 'n');
  int tb = (transb != 'N' && transb != 'n');
  for (j = 0; j < N; ++j) {
    for (i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (p = 0; p < K; ++p) {
        float aval = ta ? a[i * lda + p] : a[p * lda + i];
        float bval = tb ? b[p * ldb + j] : b[j * ldb + p];
        sum += aval * bval;
      }
      c[j * ldc + i] = alpha * sum + beta * c[j * ldc + i];
    }
  }
}
