/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* SYRK and SYR2K of LIBXS with the panels on an OpenCL device (syrk_opencl), checked against LIBXS
 * on the host, which also gives the reference of the weighted form op(A)*diag(d)*op(A)^T. With MPI
 * (-D__MPI), every rank holds a share of k, and C is block-cyclic on a grid of the ranks. */
#include "syrk_opencl.h"
#include <libxs/libxs_timer.h>
#include <libxstream/libxstream.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__MPI)
# include <mpi.h>
#endif
#include <math.h>

/* blocks of C if distributed */
#if !defined(SYRK_BLOCK)
# define SYRK_BLOCK 64
#endif


/* Reproducible values in [-1/3, 1/3), not dyadic: sums round as with real data. */
static double syrk_value(int which, int i, int j);
static double syrk_value(int which, int i, int j)
{
  unsigned int h = (unsigned int)which * 7919U + (unsigned int)i * 104729U + (unsigned int)j * 1299709U + 1U;
  h ^= h >> 13;
  h *= 0x5bd1e995U;
  h ^= h >> 15;
  return ((double)(h & 0xFFFF) / 32768.0 - 1.0) / 3.0;
}


static void syrk_put(void* x, int f64, size_t i, double v);
static void syrk_put(void* x, int f64, size_t i, double v)
{
  if (0 != f64) ((double*)x)[i] = v;
  else ((float*)x)[i] = (float)v;
}


static double syrk_get(const void* x, int f64, size_t i);
static double syrk_get(const void* x, int f64, size_t i)
{
  return (0 != f64 ? ((const double*)x)[i] : (double)((const float*)x)[i]);
}


/* Rows (or columns) of a block-cyclic distribution held by grid row iproc (as ScaLAPACK's NUMROC). */
static int syrk_numroc(int n, int nb, int iproc, int nprocs);
static int syrk_numroc(int n, int nb, int iproc, int nprocs)
{
  const int nblocks = n / nb, extra = nblocks % nprocs;
  int result = (nblocks / nprocs) * nb;
  if (iproc < extra) result += nb;
  else if (iproc == extra) result += n % nb;
  return result;
}


/* Global index of a local one (as ScaLAPACK's INDXL2G). */
static int syrk_l2g(int il, int nb, int iproc, int nprocs);
static int syrk_l2g(int il, int nb, int iproc, int nprocs)
{
  return ((il / nb) * nprocs + iproc) * nb + il % nb;
}


/* Largest of the ranks' durations (seconds), or negative if one failed. */
static double syrk_max(double seconds);
static double syrk_max(double seconds)
{
#if defined(__MPI)
  double result = seconds, smallest = seconds;
  MPI_Allreduce(&seconds, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&seconds, &smallest, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return (0 > smallest ? smallest : result);
#else
  return seconds;
#endif
}


/* One update with the tasks a plan asks for (one with a device), returns the seconds. */
static double syrk_run(const libxs_syrk_t* plan, double alpha, const void* a, const void* b, double beta, void* c);
static double syrk_run(const libxs_syrk_t* plan, double alpha, const void* a, const void* b, double beta, void* c)
{
#if defined(_OPENMP)
  const int ntasks = libxs_syrk_ntasks(plan, omp_get_max_threads());
#else
  const int ntasks = libxs_syrk_ntasks(plan, 1);
#endif
  libxs_timer_tick_t start;
  int result = EXIT_SUCCESS, t;
#if defined(__MPI)
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  start = libxs_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel for schedule(dynamic)
#endif
  for (t = 0; t < ntasks; ++t) {
    if (EXIT_SUCCESS != libxs_syrk_task(plan, alpha, a, b, beta, c, t, ntasks)) {
#if defined(_OPENMP)
#     pragma omp critical
#endif
      result = EXIT_FAILURE;
    }
  }
  return (EXIT_SUCCESS == result ? libxs_timer_duration(start, libxs_timer_tick()) : -1.0);
}


int main(int argc, char* argv[])
{
  /* arguments: n, k, timed repetitions, transposed, upper, SYR2K, weights, single precision */
  const int n = LIBXS_MAX(1 < argc ? atoi(argv[1]) : 1000, 1), k = LIBXS_MAX(2 < argc ? atoi(argv[2]) : n, 1);
  const int nrepeat = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 3, 1);
  const int trans = (4 < argc ? atoi(argv[4]) : 0), upper = (5 < argc ? atoi(argv[5]) : 0);
  const int twok = (6 < argc ? atoi(argv[6]) : 0), weighted = (7 < argc ? atoi(argv[7]) : 0);
  const int f64 = (8 < argc ? (0 == atoi(argv[8])) : 1);
  const size_t es = (0 != f64 ? sizeof(double) : sizeof(float));
  const double alpha = 1.0, beta = 0.5, nflops = (1.0 + twok) * n * (n + 1.0) * k;
  int rank = 0, nranks = 1, nprow = 1, npcol = 1, myprow = 0, mypcol = 0, kl, k0, rows, cols, lda, lrows, lcols, ldc;
  void *a = NULL, *as = NULL, *b = NULL, *d = NULL, *cref = NULL, *cdev = NULL;
  int *ranks = NULL, result, i, j, r;
  size_t na, nc;
  libxs_syrk_shape_t shape;
  libxs_syrk_device_t device;
  libxs_pdist_t dist;
  libxs_syrk_t host, plan;
  syrk_opencl_t* ctx = NULL;
  double thost = 0, tdev = 0, tres = 0, maxref = 0, maxerr = 0;
#if defined(__MPI)
  { int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  }
  for (nprow = 1; (nprow + 1) * (nprow + 1) <= nranks; ++nprow);
  for (; 0 != nranks % nprow; --nprow);
  npcol = nranks / nprow;
  myprow = rank % nprow; /* column-major grid in rank order */
  mypcol = rank / nprow;
#endif
  kl = k / nranks + (rank < k % nranks ? 1 : 0); /* this rank's share of k */
  k0 = rank * (k / nranks) + LIBXS_MIN(rank, k % nranks);
  rows = (0 == trans ? n : kl);
  cols = (0 == trans ? kl : n);
  lda = LIBXS_MAX(rows, 1);
  lrows = syrk_numroc(n, SYRK_BLOCK, myprow, nprow); /* all of n without MPI (a 1x1 grid) */
  lcols = syrk_numroc(n, SYRK_BLOCK, mypcol, npcol);
  ldc = LIBXS_MAX(lrows, 1);
  na = (size_t)lda * LIBXS_MAX(cols, 1);
  nc = (size_t)ldc * LIBXS_MAX(lcols, 1);
  result = libxstream_init();
  LIBXS_MEMZERO(&host);
  LIBXS_MEMZERO(&plan);
  LIBXS_MEMZERO(&dist);
  a = malloc(na * es); as = malloc(na * es); b = malloc(na * es);
  d = malloc((size_t)LIBXS_MAX(kl, 1) * es); cref = malloc(nc * es); cdev = malloc(nc * es);
  ranks = (int*)malloc(sizeof(int) * nranks);
  if (EXIT_SUCCESS == result && (NULL == a || NULL == as || NULL == b || NULL == d || NULL == cref || NULL == cdev
    || NULL == ranks))
  {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (j = 0; j < kl; ++j) { /* positive weights, of which SYRK's reference takes the square root */
      syrk_put(d, f64, j, 0 != weighted ? (1.0 + 0.5 * syrk_value(3, k0 + j, 0)) : 1.0);
    }
    for (j = 0; j < cols; ++j) { /* element (i, p) of op(A) is the same whatever the ranks */
      for (i = 0; i < rows; ++i) {
        const size_t x = i + (size_t)j * lda;
        const int ni = (0 == trans ? i : j), p = (0 == trans ? j : i);
        const double w = syrk_get(d, f64, p);
        syrk_put(a, f64, x, syrk_value(1, ni, k0 + p));
        syrk_put(b, f64, x, syrk_value(2, ni, k0 + p));
        syrk_put(as, f64, x, syrk_value(1, ni, k0 + p) * (0 != twok ? w : sqrt(w)));
      }
    }
    for (j = 0; j < lcols; ++j) {
      for (i = 0; i < lrows; ++i) {
        const size_t x = i + (size_t)j * ldc;
        const double v = syrk_value(4, syrk_l2g(i, SYRK_BLOCK, myprow, nprow), syrk_l2g(j, SYRK_BLOCK, mypcol, npcol));
        syrk_put(cref, f64, x, v);
        syrk_put(cdev, f64, x, v);
      }
    }
    for (r = 0; r < nranks; ++r) ranks[r] = r;
    LIBXS_MEMZERO(&shape);
    shape.datatype = (0 != f64 ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32);
    shape.uplo = (char)(0 != upper ? 'U' : 'L');
    shape.trans = (char)(0 != trans ? 'T' : 'N');
    shape.n = n; shape.k = kl;
    shape.lda = lda; shape.ldb = (0 != twok ? lda : 0); shape.ldc = ldc;
#if defined(__MPI)
    dist.fcomm = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
    dist.mb = dist.nb = SYRK_BLOCK;
    dist.nprow = nprow; dist.npcol = npcol;
    dist.ranks = ranks;
    shape.dist = &dist;
#endif
    if (0 == libxs_syrk_dispatch(&host, &shape)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = syrk_opencl_create(&ctx);
  if (EXIT_SUCCESS == result) {
    syrk_opencl_device(ctx, &device);
    syrk_opencl_weights(ctx, 0 != weighted ? d : NULL);
    shape.device = &device;
    if (0 == libxs_syrk_dispatch(&plan, &shape)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* one update each, then the comparison */
    const double t0 = syrk_max(syrk_run(&host, alpha, as, 0 != twok ? b : NULL, beta, cref));
    const double t1 = syrk_max(syrk_run(&plan, alpha, a, 0 != twok ? b : NULL, beta, cdev));
    if (0 > t0 || 0 > t1) result = EXIT_FAILURE;
    for (j = 0; j < lcols && EXIT_SUCCESS == result; ++j) {
      const int gj = syrk_l2g(j, SYRK_BLOCK, mypcol, npcol);
      for (i = 0; i < lrows; ++i) {
        const int gi = syrk_l2g(i, SYRK_BLOCK, myprow, nprow);
        if (0 != upper ? (gi <= gj) : (gi >= gj)) {
          const size_t x = i + (size_t)j * ldc;
          const double ref = syrk_get(cref, f64, x), diff = fabs(syrk_get(cdev, f64, x) - ref);
          if (maxref < fabs(ref)) maxref = fabs(ref);
          if (!(diff <= maxerr)) maxerr = diff; /* also NaN */
        }
      }
    }
#if defined(__MPI)
    { double local[2], global[2];
      local[0] = maxerr; local[1] = maxref;
      MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      maxerr = global[0]; maxref = global[1];
    }
#endif
  }
  for (r = 0; r < nrepeat && EXIT_SUCCESS == result; ++r) { /* timed: host, device, device with A resident */
    const double t0 = syrk_max(syrk_run(&host, alpha, as, 0 != twok ? b : NULL, beta, cref));
    const double t1 = syrk_max(syrk_run(&plan, alpha, a, 0 != twok ? b : NULL, beta, cdev));
    double t2;
    syrk_opencl_resident(ctx, 1);
    syrk_run(&plan, alpha, a, 0 != twok ? b : NULL, beta, cdev); /* copies A, then it stays */
    t2 = syrk_max(syrk_run(&plan, alpha, a, 0 != twok ? b : NULL, beta, cdev));
    syrk_opencl_resident(ctx, 0);
    if (0 > t0 || 0 > t1 || 0 > t2) result = EXIT_FAILURE;
    if (0 == r || t0 < thost) thost = t0;
    if (0 == r || t1 < tdev) tdev = t1;
    if (0 == r || t2 < tres) tres = t2;
  }
  if (EXIT_SUCCESS == result) {
    const double tol = (0 != f64 ? 1E-12 : 1E-4) * sqrt((double)k);
    const double rel = maxerr / LIBXS_MAX(maxref, 1.0);
    if (0 == rank) {
      printf("%s: n=%i k=%i %s%s%s%s", 0 != twok ? "SYR2K" : "SYRK", n, k, 0 != upper ? "upper" : "lower",
        0 != trans ? " transposed" : "", 0 != weighted ? " weighted" : "", 0 != f64 ? "" : " (single)");
      if (1 < nranks) printf(" ranks=%i grid=%ix%i block=%i", nranks, nprow, npcol, SYRK_BLOCK);
      printf("\n  error (relative):  %.2e\n", rel);
      printf("  LIBXS (host):      %.3f ms  %.1f GFLOPS/s\n", 1E3 * thost, 1E-9 * nflops / thost);
      printf("  LIBXSTREAM (%s): %.3f ms  %.1f GFLOPS/s\n", syrk_opencl_backend(ctx), 1E3 * tdev, 1E-9 * nflops / tdev);
      printf("  LIBXSTREAM (%s, A on device): %.3f ms  %.1f GFLOPS/s\n", syrk_opencl_backend(ctx), 1E3 * tres,
        1E-9 * nflops / tres);
    }
    if (!(rel <= tol)) {
      if (0 == rank) fprintf(stderr, "ERROR: relative error %.2e exceeds %.2e\n", rel, tol);
      result = EXIT_FAILURE;
    }
  }
  libxs_syrk_release(&plan);
  libxs_syrk_release(&host);
  syrk_opencl_destroy(ctx);
  libxstream_finalize();
  free(a); free(as); free(b); free(d); free(cref); free(cdev); free(ranks);
  if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: syrk failed (rank %i)\n", rank);
#if defined(__MPI)
  MPI_Finalize();
#endif
  return (EXIT_SUCCESS == result ? EXIT_SUCCESS : EXIT_FAILURE);
}
