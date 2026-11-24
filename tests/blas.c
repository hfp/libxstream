/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Device BLAS (libxstream_blas.h): every entry point against the host if a plugin serves the device,
 * with the copies and the work enqueued without synchronizing in between (the stream keeps the order),
 * a refusal of every call otherwise, and nothing without a device. */
#include <libxstream/libxstream_blas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 37
#define N 29
#define K 53
/* elements of C: SYRK writes M x M, the batch M x 2N */
#define NC (M * M > 2 * M * N ? M * M : 2 * M * N)


/* Reproducible values in [-1/3, 1/3), not dyadic: sums round as with real data. */
static double value(int which, int i);
static double value(int which, int i)
{
  unsigned int h = (unsigned int)which * 7919U + (unsigned int)i * 104729U + 1U;
  h ^= h >> 13;
  h *= 0x5bd1e995U;
  h ^= h >> 15;
  return ((double)(h & 0xFFFF) / 32768.0 - 1.0) / 3.0;
}


static void put(void* x, int f64, int i, double v);
static void put(void* x, int f64, int i, double v)
{
  if (0 != f64) ((double*)x)[i] = v;
  else ((float*)x)[i] = (float)v;
}


static double get(const void* x, int f64, int i);
static double get(const void* x, int f64, int i)
{
  return (0 != f64 ? ((const double*)x)[i] : (double)((const float*)x)[i]);
}


/**
 * GEMM (C := 0.5*A*B^T + 2*C), SYRK (lower, C := A*A^T, the upper triangle untouched), DGMM, and a
 * batch of two GEMMs into two C blocks (an M x 2N matrix). A precision the plugin declines at the
 * first call (e.g., FP64 on a device without it) is not checked (served is set to zero), but each
 * call after a served one must succeed.
 */
static int check(libxstream_stream_t* stream, int f64, int* served);
static int check(libxstream_stream_t* stream, int f64, int* served)
{
  const libxs_data_t datatype = (0 != f64 ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32);
  const size_t es = (0 != f64 ? sizeof(double) : sizeof(float));
  const double tol = (0 != f64 ? 1E-13 : 1E-5);
  void *const ha = malloc(es * M * K), *const hb = malloc(es * N * K), *const hc = malloc(es * NC);
  void *const hx = malloc(es * N), *da = NULL, *db = NULL, *dc = NULL, *dx = NULL;
  int result = EXIT_SUCCESS, i, j, p, op;
  if (NULL == ha || NULL == hb || NULL == hc || NULL == hx) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, es * M * K);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&db, es * N * K);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, es * NC);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dx, es * N);
  for (i = 0; i < M * K && EXIT_SUCCESS == result; ++i) put(ha, f64, i, value(1, i));
  for (i = 0; i < N * K && EXIT_SUCCESS == result; ++i) put(hb, f64, i, value(2, i));
  for (i = 0; i < N && EXIT_SUCCESS == result; ++i) put(hx, f64, i, 1.0 + value(3, i));
  *served = 1;
  for (op = 0; op < 4 && 0 != *served && EXIT_SUCCESS == result; ++op) {
    for (i = 0; i < NC; ++i) put(hc, f64, i, value(4, i));
    result = libxstream_mem_copy_h2d(ha, da, es * M * K, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hb, db, es * N * K, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hc, dc, es * NC, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hx, dx, es * N, stream);
    if (EXIT_SUCCESS == result) {
      if (0 == op) { /* declined: nothing to check, the copies are synchronized below */
        *served = (EXIT_SUCCESS == libxstream_blas_gemm(stream, datatype, 'N', 'T', M, N, K, 0.5, da, M, db, N, 2.0, dc, M));
      }
      else if (1 == op) result = libxstream_blas_syrk(stream, datatype, 'L', 'N', M, K, 1.0, da, M, NULL, 0, 0.0, dc, M);
      else if (2 == op) result = libxstream_blas_dgmm(stream, datatype, 'R', M, N, da, M, dx, 1, dc, M);
      else {
        const void *a[2], *b[2];
        void* c[2];
        a[0] = a[1] = da;
        b[0] = b[1] = db;
        c[0] = dc;
        c[1] = (char*)dc + es * M * N;
        result = libxstream_blas_gemm_batch(stream, datatype, 'N', 'T', M, N, K, 0.5, a, M, b, N, 2.0, c, M, 2);
      }
    }
    if (EXIT_SUCCESS == result && 0 != *served) result = libxstream_mem_copy_d2h(dc, hc, es * NC, stream);
    if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
    for (j = 0; j < (0 == *served ? 0 : (1 == op ? M : (3 == op ? 2 * N : N))) && EXIT_SUCCESS == result; ++j) {
      for (i = 0; i < M && EXIT_SUCCESS == result; ++i) {
        const int x = i + j * M, untouched = (1 == op && i < j);
        const double c0 = (0 != f64 ? value(4, x) : (double)(float)value(4, x)); /* C as it was copied */
        double ref = c0;
        if (0 == op || 3 == op) { /* the batch's second entry is columns N to 2N-1 */
          double sum = 0;
          for (p = 0; p < K; ++p) sum += get(ha, f64, i + p * M) * get(hb, f64, (j % N) + p * N);
          ref = 0.5 * sum + 2.0 * c0;
        }
        else if (1 == op && 0 == untouched) {
          ref = 0;
          for (p = 0; p < K; ++p) ref += get(ha, f64, i + p * M) * get(ha, f64, j + p * M);
        }
        else if (2 == op) ref = get(ha, f64, x) * get(hx, f64, j);
        if (0 != untouched ? (get(hc, f64, x) != ref) : !(fabs(get(hc, f64, x) - ref) <= tol * (1.0 + fabs(ref)))) {
          fprintf(stderr, "ERROR: op=%i f64=%i C(%i,%i) = %g instead of %g\n", op, f64, i, j, get(hc, f64, x), ref);
          result = EXIT_FAILURE;
        }
      }
    }
  }
  libxstream_mem_deallocate(da);
  libxstream_mem_deallocate(db);
  libxstream_mem_deallocate(dc);
  libxstream_mem_deallocate(dx);
  free(ha); free(hb); free(hc); free(hx);
  return result;
}


int main(void)
{
  libxstream_stream_t* stream = NULL;
  int result = libxstream_init(), ndevices = 0;
  if (EXIT_SUCCESS == result) result = libxstream_device_count(&ndevices);
  if (EXIT_SUCCESS == result && 0 < ndevices) {
    result = libxstream_stream_create(&stream, "blas", LIBXSTREAM_STREAM_DEFAULT);
  }
  else if (EXIT_SUCCESS == result) printf("device BLAS: no device\n");
  if (EXIT_SUCCESS == result && NULL != stream) {
    const char* const name = libxstream_blas_query(stream);
    if (NULL != name) { /* single precision is served by any plugin which serves the device */
      int f64 = 0, f32 = 0;
      result = check(stream, 1, &f64);
      if (EXIT_SUCCESS == result) result = check(stream, 0, &f32);
      if (EXIT_SUCCESS == result && 0 == f32) result = EXIT_FAILURE;
      printf("device BLAS %s: %s%s\n", name, EXIT_SUCCESS == result ? "OK" : "FAILED", 0 == f64 ? " (FP64 declined)" : "");
    }
    else { /* every call refuses, hence the caller computes otherwise */
      double c = 0;
      const void* const pc = &c;
      void* const qc = &c;
      if (EXIT_SUCCESS == libxstream_blas_gemm(stream, LIBXS_DATATYPE_F64, 'N', 'N', 1, 1, 1, 1, &c, 1, &c, 1, 0, &c, 1)
        || EXIT_SUCCESS == libxstream_blas_gemm_batch(stream, LIBXS_DATATYPE_F64, 'N', 'N', 1, 1, 1, 1, &pc, 1, &pc, 1, 0, &qc, 1, 1)
        || EXIT_SUCCESS == libxstream_blas_syrk(stream, LIBXS_DATATYPE_F64, 'L', 'N', 1, 1, 1, &c, 1, NULL, 0, 0, &c, 1)
        || EXIT_SUCCESS == libxstream_blas_dgmm(stream, LIBXS_DATATYPE_F64, 'L', 1, 1, &c, 1, &c, 1, &c, 1))
      {
        result = EXIT_FAILURE;
      }
      printf("device BLAS: none\n");
    }
  }
  if (NULL != stream) libxstream_stream_destroy(stream);
  libxstream_finalize();
  return result;
}
