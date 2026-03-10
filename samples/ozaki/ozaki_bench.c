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
 */
#include "ozaki_opencl.h"
#include <libxs_rng.h>

/* BLAS GEMM symbols and prototypes (Fortran calling convention) */
#define DGEMM LIBXS_FSYMBOL(dgemm)
#define SGEMM LIBXS_FSYMBOL(sgemm)


void DGEMM(const char* transa, const char* transb,
           const int* m, const int* n, const int* k,
           const double* alpha, const double* a, const int* lda,
                                const double* b, const int* ldb,
           const double* beta,        double* c, const int* ldc);
void SGEMM(const char* transa, const char* transb,
           const int* m, const int* n, const int* k,
           const float* alpha, const float* a, const int* lda,
                               const float* b, const int* ldb,
           const float* beta,        float* c, const int* ldc);

/* Function prototypes */
static void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff);


int main(int argc, char* argv[])
{
  ozaki_context_t ctx;
  const char *const env_nrepeat = getenv("NREPEAT");
  const int nrepeat = (NULL != env_nrepeat ? LIBXS_MAX(atoi(env_nrepeat), 1) : 1);
  const int M = (1 < argc ? atoi(argv[1]) : 257);
  const int N = (2 < argc ? atoi(argv[2]) : M);
  const int K = (3 < argc ? atoi(argv[3]) : M);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  const double alpha = (6 < argc ? atof(argv[6]) : 1);
  const double beta  = (7 < argc ? atof(argv[7]) : 1);
  const int lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? M : K));
  const int ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? K : N));
  const int ldc = (10 < argc ? atoi(argv[10]) : M);
  const char transa = (0 == ta ? 'N' : 'T');
  const char transb = (0 == tb ? 'N' : 'T');
  void *a = NULL, *b = NULL, *c_oz = NULL, *c_ref = NULL;
  libxstream_stream_t* stream = NULL;
  libxs_matdiff_info_t diff;
  libxs_timer_tick_t t0, t1;
  size_t elem_size = 0;
  int result = EXIT_SUCCESS;
  int initialized = 0;

  LIBXS_MEMZERO(&ctx);

  if (1 > M || 1 > N || 1 > K || lda < (0 == ta ? M : K)
      || ldb < (0 == tb ? K : N) || ldc < M) {
    fprintf(stderr, "Invalid dimensions: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n",
      M, N, K, lda, ldb, ldc);
    result = EXIT_FAILURE;
  }

  /* Initialize ACC (encompasses libxs initialization) */
  if (EXIT_SUCCESS == result) {
    result = libxstream_init();
    if (EXIT_SUCCESS == result) {
      int ndevices = 0;
      initialized = 1;
      result = libxstream_device_count(&ndevices);
      if (EXIT_SUCCESS == result && 0 < ndevices) {
        result = libxstream_device_set_active(0);
      }
      else if (EXIT_SUCCESS == result) {
        fprintf(stderr, "ERROR: no ACC device found\n");
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: ACC initialization failed\n");
    }
  }

  if (EXIT_SUCCESS == result) {
    printf("OpenCL benchmark for Ozaki's methods\n");
    printf("GEMM: %c%c M=%d N=%d K=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
      transa, transb, M, N, K, lda, ldb, ldc, alpha, beta);
  }

  /* Initialize Ozaki context (kernels) */
  if (EXIT_SUCCESS == result) {
    const char* env;
    int ozflags = -1 /*auto*/, oztrim = 0, nslices = 0 /*auto*/;
    int kind = 1 /*int8*/, verbosity = 0;
    env = getenv("OZAKI_FLAGS");
    if (NULL != env) ozflags = atoi(env);
    env = getenv("OZAKI_TRIM");
    if (NULL != env) oztrim = atoi(env);
    env = getenv("OZAKI_N");
    if (NULL != env) nslices = atoi(env);
    env = getenv("OZAKI");
    if (NULL != env) kind = atoi(env);
    env = getenv("OZAKI_VERBOSE");
    if (NULL != env) verbosity = atoi(env);
    result = ozaki_init(&ctx, 0, 0, 0, 1 /*use_double*/, kind, verbosity,
      nslices, 0, ozflags, oztrim);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "Failed to initialize Ozaki OpenCL context\n");
    }
  }

  /* Create own ACC stream (enables double-buffered transfers) */
  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_create(&stream, "ozaki_main", -1 /*default priority*/);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: failed to create ACC stream\n");
    }
  }

  /* Element size matches actual precision (may fall back to fp32) */
  if (EXIT_SUCCESS == result) {
    elem_size = ctx.use_double ? sizeof(double) : sizeof(float);
  }

  /* Allocate and fill matrices (column-major) */
  if (EXIT_SUCCESS == result) {
    const int a_rows = (0 == ta ? M : K), a_cols = (0 == ta ? K : M);
    const int b_rows = (0 == tb ? K : N), b_cols = (0 == tb ? N : K);
    result = libxstream_mem_host_allocate((void**)&a, (size_t)lda * a_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&b, (size_t)ldb * b_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&c_oz, (size_t)ldc * N * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&c_ref, (size_t)ldc * N * elem_size, stream);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: out of memory\n");
      result = EXIT_FAILURE;
    }
    else {
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
  }

  /* Run Ozaki OpenCL GEMM */
  if (EXIT_SUCCESS == result) {
    int i;
    /* warmup (not timed) */
    result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc);
    libxstream_stream_sync(stream);
    /* restore C for the timed run (beta may be non-zero) */
    if (EXIT_SUCCESS == result) memcpy(c_oz, c_ref, (size_t)ldc * N * elem_size);
    t0 = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc);
      if (EXIT_SUCCESS != result) break;
    }
    libxstream_stream_sync(stream);
    t1 = libxs_timer_tick();
    if (EXIT_SUCCESS == result) {
      printf("Ozaki GEMM: %.1f ms\n", 1E3 * libxs_timer_duration(t0, t1) / nrepeat);
    }
    else fprintf(stderr, "Ozaki GEMM failed\n");
  }

  /* Reference BLAS GEMM */
  if (EXIT_SUCCESS == result) {
    int i;
    /* save original C (still in c_ref) into c_oz; the Ozaki result will
     * be recomputed below for comparison after BLAS timing is done */
    memcpy(c_oz, c_ref, (size_t)ldc * N * elem_size);
    t0 = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      if (ctx.use_double) {
        DGEMM(&transa, &transb, &M, &N, &K, &alpha, (const double*)a, &lda, (const double*)b, &ldb, &beta, (double*)c_ref, &ldc);
      }
      else {
        const float falpha = (float)alpha, fbeta = (float)beta;
        SGEMM(&transa, &transb, &M, &N, &K, &falpha, (const float*)a, &lda, (const float*)b, &ldb, &fbeta, (float*)c_ref, &ldc);
      }
      /* restore C before next iteration so beta does not accumulate */
      if (i < nrepeat - 1) memcpy(c_ref, c_oz, (size_t)ldc * N * elem_size);
    }
    t1 = libxs_timer_tick();
    printf("BLAS  GEMM: %.1f ms\n", 1E3 * libxs_timer_duration(t0, t1) / nrepeat);
  }

  /* Recompute Ozaki GEMM once for accuracy comparison (c_oz holds original C) */
  if (EXIT_SUCCESS == result) {
    result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc);
    libxstream_stream_sync(stream);
  }

  /* Compare */
  if (EXIT_SUCCESS == result) {
    const libxs_data_t dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    result = libxs_matdiff(&diff, dtype, M, N, c_ref, c_oz, &ldc, &ldc);
    if (EXIT_SUCCESS == result) {
      diff.r = nrepeat;
      print_diff(stdout, &diff);
    }
  }

  if (0 != initialized) {
    if (NULL != a) libxstream_mem_host_deallocate(a, stream);
    if (NULL != b) libxstream_mem_host_deallocate(b, stream);
    if (NULL != c_oz) libxstream_mem_host_deallocate(c_oz, stream);
    if (NULL != c_ref) libxstream_mem_host_deallocate(c_ref, stream);
    if (NULL != stream) libxstream_stream_destroy(stream);
    ozaki_destroy(&ctx);
    libxstream_finalize();
  }
  return result;
}


static void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff)
{
  const double epsilon = libxs_matdiff_epsilon(diff);
  if (1E-6 <= epsilon) {
    fprintf(ostream, "DIFF: ncalls=%i linf=%f linf_rel=%f l2_rel=%f eps=%f rsq=%f -> %g != %g\n",
      diff->r, diff->linf_abs, diff->linf_rel, diff->l2_rel, epsilon, diff->rsq,
      diff->v_ref, diff->v_tst);
  }
  else {
    fprintf(ostream, "DIFF: ncalls=%i linf=%f linf_rel=%f l2_rel=%f eps=%f rsq=%f\n",
      diff->r, diff->linf_abs, diff->linf_rel, diff->l2_rel, epsilon, diff->rsq);
  }
}
