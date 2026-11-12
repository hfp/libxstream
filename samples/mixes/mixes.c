/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Named mixes: a STREAM triad, a tiled GEMM, and FP atomics, each a known
 * mixture of instructions. Counted (e.g., on a simulated GPU), another kernel's
 * counts are expressed in them; timed on a device, they say what the mixture
 * costs there. Inputs are small integers, hence every result is exact and is
 * checked for equality, whatever the order of the atomic updates. */
#include <mixes_kernels.h>
#include <libxs/libxs_timer.h>
#include <libxstream/libxstream_opencl.h>

#if !defined(OPENCL_KERNELS_SOURCE_MIXES)
# error "OpenCL kernel code not found (mixes_kernels.h)!"
#endif
#if !defined(ELEM_TYPE)
# define ELEM_TYPE double
#endif
#if !defined(MIXES_TILE)
# define MIXES_TILE 16
#endif
#define MIXES_ROUNDUP(N, M) ((((N) + (M) - 1) / (M)) * (M))


typedef ELEM_TYPE mixes_real_t;

typedef struct mixes_kernels_t {
  cl_kernel stream, gemm, atomics, gather, empty;
} mixes_kernels_t;


/* Builds the kernels with the FP-atomics flags of the active device. */
static int mixes_build(mixes_kernels_t* kernels);
static int mixes_build(mixes_kernels_t* kernels)
{
  const libxstream_opencl_atomic_fp_t kind = (sizeof(double) == sizeof(mixes_real_t) ? libxstream_opencl_atomic_fp_64
                                                                                     : libxstream_opencl_atomic_fp_32);
  const char* exts[] = {NULL, NULL};
  size_t nexts = sizeof(exts) / sizeof(*exts);
  char flags[LIBXSTREAM_BUFFERSIZE];
  cl_program program = NULL;
  int result = EXIT_SUCCESS, offset;
  offset = LIBXS_SNPRINTF(flags, sizeof(flags), "-DT=%s -DTILE=%i", LIBXS_STRINGIFY(ELEM_TYPE), MIXES_TILE);
  if (0 < offset && (int)sizeof(flags) > offset) {
    const int n = libxstream_opencl_flags_atomics(
      &libxstream_opencl_config.device, kind, exts, &nexts, flags + offset, sizeof(flags) - offset);
    if (0 >= n) result = EXIT_FAILURE; /* no FP atomics, e.g., fp64 without cl_khr_fp64 */
  }
  else result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_program(0 /*source_kind*/, OPENCL_KERNELS_SOURCE_MIXES, "mixes", flags, NULL /*options*/,
      NULL /*try*/, NULL /*try_ok*/, exts, nexts, &program);
  }
  if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "mixes_stream", &kernels->stream);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "mixes_gemm", &kernels->gemm);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "mixes_atomics", &kernels->atomics);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "mixes_gather", &kernels->gather);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "mixes_empty", &kernels->empty);
  if (NULL != program) clReleaseProgram(program); /* the kernels keep what they need */
  return result;
}


/* Launches a kernel nwarmup + nrepeat times and returns the seconds per timed launch (negative on failure). */
static double mixes_launch(libxstream_stream_t* stream, cl_kernel kernel, cl_uint ndims, const size_t global[],
  const size_t local[], int nwarmup, int nrepeat);
static double mixes_launch(libxstream_stream_t* stream, cl_kernel kernel, cl_uint ndims, const size_t global[],
  const size_t local[], int nwarmup, int nrepeat)
{
  libxs_timer_tick_t start = 0;
  int result = EXIT_SUCCESS, i = 0;
  double seconds = -1;
  for (; i < (nwarmup + nrepeat) && EXIT_SUCCESS == result; ++i) {
    if (nwarmup == i) {
      result = libxstream_stream_sync(stream);
      start = libxs_timer_tick();
    }
    if (EXIT_SUCCESS == result) result = libxstream_opencl_launch(stream, kernel, ndims, NULL, global, local, 0, NULL, NULL);
  }
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
  if (EXIT_SUCCESS == result && 0 < nrepeat) seconds = libxs_timer_duration(start, libxs_timer_tick()) / nrepeat;
  return seconds;
}


/* STREAM triad a = b + s * c over n elements. */
static int mixes_stream(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat);
static int mixes_stream(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat)
{
  const size_t size = sizeof(mixes_real_t) * n;
  const mixes_real_t s = 3;
  mixes_real_t *a = (mixes_real_t*)malloc(size), *b = (mixes_real_t*)malloc(size), *c = (mixes_real_t*)malloc(size);
  void *da = NULL, *db = NULL, *dc = NULL;
  int result = ((NULL != a && NULL != b && NULL != c) ? EXIT_SUCCESS : EXIT_FAILURE), i, nerrors = 0;
  double seconds = -1;
  size_t global[1];
  global[0] = (size_t)n;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    b[i] = (mixes_real_t)(i % 7 - 3);
    c[i] = (mixes_real_t)(i % 5 - 2);
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&db, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(b, db, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, dc, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 0, da);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 1, db);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 2, dc);
  if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 3, sizeof(mixes_real_t), &s);
  if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 4, sizeof(int), &n);
  if (EXIT_SUCCESS == result) {
    seconds = mixes_launch(stream, kernel, 1, global, NULL, nwarmup, nrepeat);
    if (0 > seconds) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(da, a, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    if (a[i] != b[i] + s * c[i]) ++nerrors;
  }
  if (EXIT_SUCCESS == result) {
    printf("mixes_stream: n=%i bytes=%.0f %.3f ms %.1f GB/s errors=%i\n", n, 3.0 * size, 1E3 * seconds,
      3E-9 * size / LIBXS_MAX(seconds, 1E-12), nerrors);
    if (0 != nerrors) result = EXIT_FAILURE;
  }
  if (NULL != da) libxstream_mem_deallocate(da);
  if (NULL != db) libxstream_mem_deallocate(db);
  if (NULL != dc) libxstream_mem_deallocate(dc);
  free(a);
  free(b);
  free(c);
  return result;
}


/* GEMM c = a * b, all n x n and row-major (n a multiple of the tile). */
static int mixes_gemm(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat);
static int mixes_gemm(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat)
{
  const size_t nn = (size_t)n * n, size = sizeof(mixes_real_t) * nn;
  const size_t local[] = {MIXES_TILE, MIXES_TILE};
  mixes_real_t *a = (mixes_real_t*)malloc(size), *b = (mixes_real_t*)malloc(size), *c = (mixes_real_t*)malloc(size);
  void *da = NULL, *db = NULL, *dc = NULL;
  int result = ((NULL != a && NULL != b && NULL != c) ? EXIT_SUCCESS : EXIT_FAILURE), i, j, nerrors = 0;
  double seconds = -1;
  size_t global[2];
  global[0] = global[1] = (size_t)n;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < n; ++j) {
      a[i * n + j] = (mixes_real_t)((i * 7 + j * 3) % 5 - 2);
      b[i * n + j] = (mixes_real_t)((i * 3 + j * 5) % 3 - 1);
    }
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&db, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(a, da, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(b, db, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 0, dc);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 1, da);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 2, db);
  if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 3, sizeof(int), &n);
  if (EXIT_SUCCESS == result) {
    seconds = mixes_launch(stream, kernel, 2, global, local, nwarmup, nrepeat);
    if (0 > seconds) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(dc, c, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < n; ++j) {
      mixes_real_t ref = 0;
      int k = 0;
      for (; k < n; ++k) ref += a[i * n + k] * b[k * n + j];
      if (c[i * n + j] != ref) ++nerrors;
    }
  }
  if (EXIT_SUCCESS == result) {
    const double flops = 2.0 * nn * n;
    printf("mixes_gemm: n=%i flops=%.0f %.3f ms %.1f GFLOPS/s errors=%i\n", n, flops, 1E3 * seconds,
      1E-9 * flops / LIBXS_MAX(seconds, 1E-12), nerrors);
    if (0 != nerrors) result = EXIT_FAILURE;
  }
  if (NULL != da) libxstream_mem_deallocate(da);
  if (NULL != db) libxstream_mem_deallocate(db);
  if (NULL != dc) libxstream_mem_deallocate(dc);
  free(a);
  free(b);
  free(c);
  return result;
}


/* FP atomics: n elements accumulate into m slots, over every launch. */
static int mixes_atomics(libxstream_stream_t* stream, cl_kernel kernel, int n, int m, int nwarmup, int nrepeat);
static int mixes_atomics(libxstream_stream_t* stream, cl_kernel kernel, int n, int m, int nwarmup, int nrepeat)
{
  const size_t size = sizeof(mixes_real_t) * n, csize = sizeof(mixes_real_t) * m;
  mixes_real_t *a = (mixes_real_t*)malloc(size), *c = (mixes_real_t*)malloc(csize);
  void *da = NULL, *dc = NULL;
  int result = ((NULL != a && NULL != c) ? EXIT_SUCCESS : EXIT_FAILURE), i, nerrors = 0;
  double seconds = -1;
  size_t global[1];
  global[0] = (size_t)n;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) a[i] = (mixes_real_t)(i % 7 - 3);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, csize);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(a, da, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_mem_zero(dc, 0, csize, stream);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 0, dc);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 1, da);
  if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 2, sizeof(int), &n);
  if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 3, sizeof(int), &m);
  if (EXIT_SUCCESS == result) {
    seconds = mixes_launch(stream, kernel, 1, global, NULL, nwarmup, nrepeat);
    if (0 > seconds) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(dc, c, csize, stream);
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
  for (i = 0; i < m && EXIT_SUCCESS == result; ++i) {
    mixes_real_t ref = 0;
    int j = i;
    for (; j < n; j += m) ref += a[j];
    if (c[i] != ref * (nwarmup + nrepeat)) ++nerrors;
  }
  if (EXIT_SUCCESS == result) {
    printf("mixes_atomics: n=%i m=%i updates=%i %.3f ms %.3f G/s errors=%i\n", n, m, n, 1E3 * seconds,
      1E-9 * n / LIBXS_MAX(seconds, 1E-12), nerrors);
    if (0 != nerrors) result = EXIT_FAILURE;
  }
  if (NULL != da) libxstream_mem_deallocate(da);
  if (NULL != dc) libxstream_mem_deallocate(dc);
  free(a);
  free(c);
  return result;
}


/* Gather: block g of c is the square of block idx[g] of a, over n blocks of a tile by a tile. */
static int mixes_gather(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat);
static int mixes_gather(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat)
{
  const int bs = MIXES_TILE * MIXES_TILE, nsrc = LIBXS_MAX(n / 3, 1);
  const size_t size = sizeof(mixes_real_t) * bs * n, asize = sizeof(mixes_real_t) * bs * nsrc, isize = sizeof(int) * n;
  const size_t local[] = {MIXES_TILE * MIXES_TILE};
  mixes_real_t *a = (mixes_real_t*)malloc(asize), *c = (mixes_real_t*)malloc(size);
  int* idx = (int*)malloc(isize);
  void *da = NULL, *dc = NULL, *di = NULL;
  int result = ((NULL != a && NULL != c && NULL != idx) ? EXIT_SUCCESS : EXIT_FAILURE), i, nerrors = 0;
  double seconds = -1;
  size_t global[1];
  global[0] = (size_t)bs * n;
  for (i = 0; i < bs * nsrc && EXIT_SUCCESS == result; ++i) a[i] = (mixes_real_t)(i % 5 - 2);
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) idx[i] = (int)(((size_t)i * 7) % nsrc);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&da, asize);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&dc, size);
  if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&di, isize);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(a, da, asize, stream);
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(idx, di, isize, stream);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 0, dc);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 1, da);
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 2, di);
  if (EXIT_SUCCESS == result) {
    seconds = mixes_launch(stream, kernel, 1, global, local, nwarmup, nrepeat);
    if (0 > seconds) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(dc, c, size, stream);
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const mixes_real_t* const src = a + (size_t)idx[i] * bs;
    int r = 0;
    for (; r < MIXES_TILE; ++r) {
      int s = 0;
      for (; s < MIXES_TILE; ++s) {
        mixes_real_t ref = 0;
        int k = 0;
        for (; k < MIXES_TILE; ++k) ref += src[r * MIXES_TILE + k] * src[k * MIXES_TILE + s];
        if (c[(size_t)i * bs + r * MIXES_TILE + s] != ref) ++nerrors;
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    const double bytes = (double)size + (double)isize + (double)sizeof(mixes_real_t) * bs * n;
    printf("mixes_gather: n=%i bytes=%.0f %.3f ms %.1f GB/s errors=%i\n", n, bytes, 1E3 * seconds,
      1E-9 * bytes / LIBXS_MAX(seconds, 1E-12), nerrors);
    if (0 != nerrors) result = EXIT_FAILURE;
  }
  if (NULL != da) libxstream_mem_deallocate(da);
  if (NULL != dc) libxstream_mem_deallocate(dc);
  if (NULL != di) libxstream_mem_deallocate(di);
  free(a);
  free(c);
  free(idx);
  return result;
}


/* Empty: a kernel without work over n work-items, i.e., what dispatching them costs. */
static int mixes_empty(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat);
static int mixes_empty(libxstream_stream_t* stream, cl_kernel kernel, int n, int nwarmup, int nrepeat)
{
  void* da = NULL;
  int result = libxstream_mem_allocate(&da, sizeof(mixes_real_t));
  double seconds = -1;
  size_t global[1];
  global[0] = (size_t)n;
  if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(kernel, 0, da);
  if (EXIT_SUCCESS == result) {
    seconds = mixes_launch(stream, kernel, 1, global, NULL, nwarmup, nrepeat);
    if (0 > seconds) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) printf("mixes_empty: n=%i items=%i %.3f ms\n", n, n, 1E3 * seconds);
  if (NULL != da) libxstream_mem_deallocate(da);
  return result;
}


int main(int argc, char* argv[])
{
  /* arguments and their defaults: stream elements, GEMM order, atomic slots, timed launches */
  const int stream_n = LIBXS_MAX(1 < argc ? atoi(argv[1]) : (1 << 20), 1);
  const int gemm_n = MIXES_ROUNDUP(LIBXS_MAX(2 < argc ? atoi(argv[2]) : 256, 1), MIXES_TILE);
  const int atomics_m = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 1024, 1);
  const int nrepeat = LIBXS_MAX(4 < argc ? atoi(argv[4]) : 3, 1), nwarmup = 1;
  libxstream_stream_t* stream = NULL;
  mixes_kernels_t kernels;
  int result = libxstream_init();
  LIBXS_MEMZERO(&kernels);
  if (EXIT_SUCCESS == result) result = libxstream_stream_create(&stream, "mixes", LIBXSTREAM_STREAM_DEFAULT);
  if (EXIT_SUCCESS == result) result = mixes_build(&kernels);
  if (EXIT_SUCCESS == result) {
    printf("%s %i %i %i %i (%s)\n", argv[0], stream_n, gemm_n, atomics_m, nrepeat, LIBXS_STRINGIFY(ELEM_TYPE));
    result = mixes_stream(stream, kernels.stream, stream_n, nwarmup, nrepeat);
  }
  if (EXIT_SUCCESS == result) result = mixes_gemm(stream, kernels.gemm, gemm_n, nwarmup, nrepeat);
  if (EXIT_SUCCESS == result) result = mixes_atomics(stream, kernels.atomics, stream_n, atomics_m, nwarmup, nrepeat);
  if (EXIT_SUCCESS == result) {
    result = mixes_gather(stream, kernels.gather, LIBXS_MAX(stream_n / (MIXES_TILE * MIXES_TILE), 1), nwarmup, nrepeat);
  }
  if (EXIT_SUCCESS == result) result = mixes_empty(stream, kernels.empty, stream_n, nwarmup, nrepeat);
  if (NULL != kernels.stream) clReleaseKernel(kernels.stream);
  if (NULL != kernels.gemm) clReleaseKernel(kernels.gemm);
  if (NULL != kernels.atomics) clReleaseKernel(kernels.atomics);
  if (NULL != kernels.gather) clReleaseKernel(kernels.gather);
  if (NULL != kernels.empty) clReleaseKernel(kernels.empty);
  if (NULL != stream) libxstream_stream_destroy(stream);
  libxstream_finalize();
  if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: mixes failed (%i)\n", result);
  return (EXIT_SUCCESS == result ? EXIT_SUCCESS : EXIT_FAILURE);
}
