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
 * and a refusal of every call otherwise. */
#include <libxstream/libxstream_blas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 37
#define N 29
#define K 53


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


/* GEMM (C := 0.5*A*B^T + 2*C), SYRK (lower, C := A*A^T, the upper triangle untouched), and DGMM. */
static int check(libxstream_stream_t* stream, int f64);
static int check(libxstream_stream_t* stream, int f64)
{
  const libxs_data_t datatype = (0 != f64 ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32);
  const size_t es = (0 != f64 ? sizeof(double) : sizeof(float));
  const double tol = (0 != f64 ? 1E-13 : 1E-5);
  void *const ha = malloc(es * M * K), *const hb = malloc(es * N * K), *const hc = malloc(es * M * M);
  void *const hx = malloc(es * N), *da = NULL, *db = NULL, *dc = NULL, *dx = NULL;
  int result = EXIT_SUCCESS, i, j, p, op;
  if (NULL == ha || NULL == hb || NULL == hc || NULL == hx) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, es * M * K);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&db, es * N * K);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, es * M * M);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dx, es * N);
  for (i = 0; i < M * K && EXIT_SUCCESS == result; ++i) put(ha, f64, i, value(1, i));
  for (i = 0; i < N * K && EXIT_SUCCESS == result; ++i) put(hb, f64, i, value(2, i));
  for (i = 0; i < N && EXIT_SUCCESS == result; ++i) put(hx, f64, i, 1.0 + value(3, i));
  for (op = 0; op < 3 && EXIT_SUCCESS == result; ++op) {
    for (i = 0; i < M * M; ++i) put(hc, f64, i, value(4, i));
    result = libxstream_mem_copy_h2d(ha, da, es * M * K, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hb, db, es * N * K, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hc, dc, es * M * M, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(hx, dx, es * N, stream);
    if (EXIT_SUCCESS == result) {
      if (0 == op) result = libxstream_blas_gemm(stream, datatype, 'N', 'T', M, N, K, 0.5, da, M, db, N, 2.0, dc, M);
      else if (1 == op) result = libxstream_blas_syrk(stream, datatype, 'L', 'N', M, K, 1.0, da, M, NULL, 0, 0.0, dc, M);
      else result = libxstream_blas_dgmm(stream, datatype, 'R', M, N, da, M, dx, 1, dc, M);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(dc, hc, es * M * M, stream);
    if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
    for (j = 0; j < (1 == op ? M : N) && EXIT_SUCCESS == result; ++j) {
      for (i = 0; i < M && EXIT_SUCCESS == result; ++i) {
        const int x = i + j * M, untouched = (1 == op && i < j);
        const double c0 = (0 != f64 ? value(4, x) : (double)(float)value(4, x)); /* C as it was copied */
        double ref = c0;
        if (0 == op) {
          double sum = 0;
          for (p = 0; p < K; ++p) sum += get(ha, f64, i + p * M) * get(hb, f64, j + p * N);
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
  int result = libxstream_init(), xmem = 0;
  if (EXIT_SUCCESS == result) result = libxstream_stream_create(&stream, "blas", LIBXSTREAM_STREAM_DEFAULT);
  if (EXIT_SUCCESS == result) {
    const char* const name = libxstream_blas_query(stream, &xmem);
    if (NULL != name && 0 != xmem) {
      result = check(stream, 1);
      if (EXIT_SUCCESS == result) result = check(stream, 0);
      printf("device BLAS %s: %s\n", name, EXIT_SUCCESS == result ? "OK" : "FAILED");
    }
    else { /* every call refuses, hence the caller computes otherwise */
      double c = 0;
      if (EXIT_SUCCESS == libxstream_blas_gemm(stream, LIBXS_DATATYPE_F64, 'N', 'N', 1, 1, 1, 1, &c, 1, &c, 1, 0, &c, 1)
        || EXIT_SUCCESS == libxstream_blas_syrk(stream, LIBXS_DATATYPE_F64, 'L', 'N', 1, 1, 1, &c, 1, NULL, 0, 0, &c, 1)
        || EXIT_SUCCESS == libxstream_blas_dgmm(stream, LIBXS_DATATYPE_F64, 'L', 1, 1, &c, 1, &c, 1, &c, 1))
      {
        result = EXIT_FAILURE;
      }
      printf("device BLAS: none%s\n", NULL != name ? " (no LIBXSTREAM memory)" : "");
    }
  }
  if (NULL != stream) libxstream_stream_destroy(stream);
  libxstream_finalize();
  return result;
}
