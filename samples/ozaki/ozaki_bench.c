/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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
#include <libxs/libxs_timer.h>
#include <libxs/libxs_rng.h>
#if defined(__CUBLAS)
# include <cuda_runtime_api.h>
# include <cublas_v2.h>
#endif

/* BLAS GEMM symbols and prototypes (Fortran calling convention) */
#define DGEMM LIBXS_FSYMBOL(dgemm)
#define SGEMM LIBXS_FSYMBOL(sgemm)

#if defined(__CUBLAS)
/**
 * Workspace handed to cuBLAS (bytes, 0: do not call cublasSetWorkspace).
 * The FP64 emulation takes its workspace from cudaMallocAsync, which is
 * released on every stream synchronization, i.e. once per timed iteration.
 * A failing allocation is a soft error but silently disables emulation.
 */
# if !defined(OZAKI_CUBLAS_WORKSPACE)
#   define OZAKI_CUBLAS_WORKSPACE (2048UL * 1024 * 1024)
# endif
#endif


LIBXS_EXTERN void DGEMM(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha, const double* a,
  const int* lda, const double* b, const int* ldb, const double* beta, double* c, const int* ldc);
LIBXS_EXTERN void SGEMM(const char* transa, const char* transb, const int* m, const int* n, const int* k, const float* alpha, const float* a,
  const int* lda, const float* b, const int* ldb, const float* beta, float* c, const int* ldc);

/* Function prototypes */
static void print_diff(FILE* ostream, const char* label, const libxs_matdiff_t* diff);
static void gemm_bound(int use_double, char transa, char transb, int M, int N, int K, double alpha,
  void* a, int lda, void* b, int ldb, double beta, void* c_bnd, int ldc);
static double ozaki_duration(double* times, int nrepeat, double total);
static int xref_row(int s, int xrows, int m);
static void xref_gather(void* dst, const void* src, int xrows, int M, int N, int ldc, size_t elem_size);
static void xref_print(const char* key, const libxs_matdiff_t* diff, int graded);
#if defined(__CUBLAS)
static void cublas_putenv(int use_double, int nslices);
static const char* cublas_mode(int use_double);
static int cublas_gemm(libxstream_stream_t* stream, int use_double, char transa, char transb, int M, int N, int K, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, const void* c_in, void* c_out, int ldc, int nrepeat, int nslices,
  double* duration, double* devtime);
#endif


int main(int argc, char* argv[])
{
  ozaki_context_t ctx;
  const char* const env_nrepeat = getenv("NREPEAT");
  const int nrepeat = (NULL != env_nrepeat ? LIBXS_MAX(atoi(env_nrepeat), 1) : 1);
  const int M = (1 < argc ? atoi(argv[1]) : 257);
  const int N = (2 < argc ? atoi(argv[2]) : M);
  const int K = (3 < argc ? atoi(argv[3]) : M);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  const double alpha = (6 < argc ? atof(argv[6]) : 1);
  const double beta = (7 < argc ? atof(argv[7]) : 1);
  const int lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? M : K));
  const int ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? K : N));
  const int ldc = (10 < argc ? atoi(argv[10]) : M);
  const char transa = (0 == ta ? 'N' : 'T');
  const char transb = (0 == tb ? 'N' : 'T');
  void *a = NULL, *b = NULL, *c_oz = NULL, *c_ref = NULL;
  /* Componentwise bound |alpha||A||B| + |beta||C| (GRADE), host-only. */
  void* c_bnd = NULL;
  void* scratch = NULL; /* caller-owned device scratch (OZAKI_SCRATCH) */
  /**
   * XREF=r: compensated reference (libxs_dot2_gemm) on r rows spread over C, all rows if r
   * is negative or at least M. The host BLAS rounds as much as the results it grades, so an
   * error read off it is partly its own; this one grades Ozaki, BLAS and cuBLAS on one scale.
   */
  const char* const env_xref = getenv("XREF");
  const int xref = (NULL != env_xref && 0 != *env_xref) ? atoi(env_xref) : 0;
  /**
   * EVIL=N (N>0) is the per-column exponent span of the libxs harness (samples/ozaki/gemm.c):
   * column j of A scaled by a ramp over 2N binades, B by its inverse. The per-element mode
   * (N<0) is not implemented here and is refused rather than ignored: a row labelled with a
   * span it never had is indistinguishable from a measurement.
   */
  const char* const env_evil = getenv("EVIL");
  const int evil = (NULL != env_evil && 0 != *env_evil) ? atoi(env_evil) : 0;
  const char* const env_tame = getenv("TAME");
  const int tame = (NULL != env_tame && 0 != *env_tame) ? atoi(env_tame) : 0;
  void* x_ref = NULL;
  int xrows = 0;
#if defined(__CUBLAS)
  void* c_cu = NULL;
  int cublas_result = EXIT_FAILURE;
#endif
  libxstream_stream_t* stream = NULL;
  libxs_matdiff_t diff;
  libxs_timer_tick_t t0, t1;
  /* Per-call durations, one buffer for every timed loop; see ozaki_duration. */
  double* const times = (double*)malloc((size_t)nrepeat * sizeof(double));
  size_t elem_size = 0;
  int result = EXIT_SUCCESS;
  int initialized = 0;

  memset(&ctx, 0, sizeof(ctx)); /* exceeds LIBXS_MEMZERO scope */
  if (1 > M || 1 > N || 1 > K || lda < (0 == ta ? M : K) || ldb < (0 == tb ? K : N) || ldc < M) {
    fprintf(stderr, "Invalid dimensions: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
    result = EXIT_FAILURE;
  }
  if (0 > evil) {
    fprintf(stderr, "ERROR: EVIL=%i (per-element) is not supported by this driver\n", evil);
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
    printf("GEMM: %c%c M=%d N=%d K=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n", transa, transb, M, N, K, lda, ldb, ldc, alpha, beta);
  }

  /* Initialize Ozaki context (kernels) */
  if (EXIT_SUCCESS == result) {
    const char* env;
    int ozflags = -1 /*auto*/, oztrim = 0, ndecomp = 0 /*auto*/;
    int ozgroups = 0, kind = 0 /*auto: ozaki_init decides*/, verbosity = 0;
    /* tm/tn stay 0: ozaki_init reads OZAKI_TM/OZAKI_TN and selects per call. */
    const int tm = 0, tn = 0;
    int use_double = 1;
    env = getenv("OZAKI_FLAGS");
    if (NULL != env) ozflags = atoi(env);
    env = getenv("OZAKI_TRIM");
    if (NULL != env) oztrim = atoi(env);
    env = getenv("OZAKI");
    if (NULL != env) kind = atoi(env);
    env = getenv("OZAKI_GROUPS");
    if (NULL != env) ozgroups = atoi(env);
    env = getenv("OZAKI_VERBOSE");
    if (NULL != env) verbosity = atoi(env);
    env = getenv("OZAKI_FP");
    if (NULL != env) use_double = (32 != atoi(env));
    /* Zero leaves OZAKI_N and OZAKI_MAXK to ozaki_init, which is where they are resolved. */
    result = ozaki_init(&ctx, tm, tn, use_double, kind, verbosity, ndecomp, ozflags, oztrim, ozgroups, 0 /*maxk*/);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "Failed to initialize Ozaki OpenCL context\n");
    }
  }

  /* Create own ACC stream (enables double-buffered transfers) */
  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_create(&stream, "ozaki_main", LIBXSTREAM_STREAM_DEFAULT);
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
    const size_t maxsize = SIZE_MAX / elem_size;
    if (maxsize / a_cols < (size_t)lda || maxsize / b_cols < (size_t)ldb || maxsize / N < (size_t)ldc) {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&a, (size_t)lda * a_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&b, (size_t)ldb * b_cols * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&c_oz, (size_t)ldc * N * elem_size, stream);
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&c_ref, (size_t)ldc * N * elem_size, stream);
#if defined(__CUBLAS)
    if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate((void**)&c_cu, (size_t)ldc * N * elem_size, stream);
#endif
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: out of memory\n");
      result = EXIT_FAILURE;
    }
    else {
      if (ctx.use_double) {
        LIBXS_MATRNG(int, double, evil, a, a_rows, a_cols, lda, 1.0);
        LIBXS_MATRNG(int, double, -evil, b, b_rows, b_cols, ldb, 1.0);
        LIBXS_MATRNG(int, double, 0, c_oz, M, N, ldc, 1.0);
      }
      else {
        LIBXS_MATRNG(int, float, evil, a, a_rows, a_cols, lda, 1.0);
        LIBXS_MATRNG(int, float, -evil, b, b_rows, b_cols, ldb, 1.0);
        LIBXS_MATRNG(int, float, 0, c_oz, M, N, ldc, 1.0);
      }
      /* stamped by the driver, so a row records the data it was given and not what was asked */
      fprintf(stderr, "DATA: matrng=%i evil=%i tame=%i scale=0\n", LIBXS_MATRNG_VERSION, evil, tame);
      /**
       * TAME keeps that many mantissa bits and clears the rest, matching the harness knob
       * of the same name (libxs samples/ozaki/gemm.c, where it is EVIL's sibling on the
       * significand axis). It exercises the precision detection against a known answer: the
       * trimmable low bits reported must come out at 53 - TAME.
       */
      { const int mbits = tame;
        if (0 < mbits) {
          const int mant = ctx.use_double ? 53 : 24;
          const int drop = (mbits < mant) ? (mant - mbits) : 0;
          if (0 < drop) {
            const size_t na = (size_t)lda * a_cols, nb = (size_t)ldb * b_cols;
            size_t i;
            if (ctx.use_double) {
              const unsigned long long mask = ~((1ULL << drop) - 1ULL);
              union { double d; unsigned long long u; } v;
              for (i = 0; i < na; ++i) { v.d = ((double*)a)[i]; v.u &= mask; ((double*)a)[i] = v.d; }
              for (i = 0; i < nb; ++i) { v.d = ((double*)b)[i]; v.u &= mask; ((double*)b)[i] = v.d; }
            }
            else {
              const unsigned int mask = ~((1U << drop) - 1U);
              union { float f; unsigned int u; } v;
              for (i = 0; i < na; ++i) { v.f = ((float*)a)[i]; v.u &= mask; ((float*)a)[i] = v.f; }
              for (i = 0; i < nb; ++i) { v.f = ((float*)b)[i]; v.u &= mask; ((float*)b)[i] = v.f; }
            }
          }
        }
      }
      memcpy(c_ref, c_oz, (size_t)ldc * N * elem_size);
      /**
       * GRADE reports the componentwise criterion of the graded BLAS accuracy tests,
       * |fl(AB) - AB| <= f(K) * u * (|alpha||A||B| + |beta||C|), which is the figure
       * that compares across implementations - unlike l2_rel, whose per-element
       * quotient is dominated by the near-zero entries of C. |C| has to be captured
       * here because every reference below overwrites C.
       */
      if (0 != xref) { /* rows of the original C, where the single reference call starts */
        xrows = (0 < xref && xref < M) ? xref : M;
        x_ref = malloc((size_t)xrows * N * elem_size);
        if (NULL != x_ref) xref_gather(x_ref, c_oz, xrows, M, N, ldc, elem_size);
        else result = EXIT_FAILURE;
      }
      if (NULL != getenv("GRADE") && 0 != atoi(getenv("GRADE"))) {
        c_bnd = malloc((size_t)ldc * N * elem_size);
        if (NULL != c_bnd) {
          const size_t nc = (size_t)ldc * N;
          size_t i;
          if (ctx.use_double) {
            for (i = 0; i < nc; ++i) ((double*)c_bnd)[i] = fabs(((const double*)c_oz)[i]);
          }
          else {
            for (i = 0; i < nc; ++i) ((float*)c_bnd)[i] = (float)fabs(((const float*)c_oz)[i]);
          }
        }
      }
    }
  }

  /**
   * Caller-owned device scratch (OZAKI_SCRATCH=1), the counterpart of
   * OZAKI_CUBLAS_WORKSPACE on the reference side: the context would otherwise
   * grow its own arena, which is what a BLAS interceptor relies on, so this only
   * exercises the path where the application owns the memory. Failure is soft -
   * the internal arena remains.
   */
  if (EXIT_SUCCESS == result && NULL != getenv("OZAKI_SCRATCH") && 0 != atoi(getenv("OZAKI_SCRATCH"))) {
    const size_t nbytes = ozaki_scratch_size(&ctx, transa, transb, M, N, K, lda, ldb, ldc);
    if (EXIT_SUCCESS == libxstream_mem_dev_allocate_hint(&scratch, nbytes, libxstream_opencl_mem_hint_atomics)) {
      if (EXIT_SUCCESS != ozaki_scratch_set(&ctx, scratch, nbytes)) {
        libxstream_mem_dev_deallocate_hint(scratch, libxstream_opencl_mem_hint_atomics);
        scratch = NULL;
      }
      else if (0 != ctx.verbosity) fprintf(stderr, "INFO OZAKI: %u MB of caller-owned scratch\n", (unsigned int)(nbytes >> 20));
    }
  }

  /* Run Ozaki OpenCL GEMM */
  if (EXIT_SUCCESS == result) {
    int i;
    /* warmup (not timed) */
    result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc, 0);
    libxstream_stream_sync(stream);
    /**
     * Switch the modulus count between calls, which is what the precision detection
     * does once it has seen the data. Here it is explicit so that the switch can be
     * compared against starting at the same count, which must give the same result.
     */
    if (EXIT_SUCCESS == result && NULL != getenv("OZAKI_SWITCH")) {
      result = ozaki_crt_select(&ctx, atoi(getenv("OZAKI_SWITCH")));
    }
    /* restore C for the timed run (beta may be non-zero) */
    if (EXIT_SUCCESS == result) memcpy(c_oz, c_ref, (size_t)ldc * N * elem_size);
    /**
     * Synchronized per call, not once for the whole loop: both references below
     * are synchronous per call (the host BLAS by nature, cuBLAS through its
     * blocking D2H), so leaving consecutive Ozaki calls to overlap would credit
     * this side with a pipelining the comparison does not offer the other two.
     * The per-kernel figures are unaffected either way.
     */
    t0 = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      const libxs_timer_tick_t tick = libxs_timer_tick();
      result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc, 0);
      if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
      if (EXIT_SUCCESS != result) break;
      if (NULL != times) times[i] = libxs_timer_duration(tick, libxs_timer_tick());
    }
    t1 = libxs_timer_tick();
    if (EXIT_SUCCESS == result) {
      const double duration = ozaki_duration(times, nrepeat, libxs_timer_duration(t0, t1));
      printf("Ozaki GEMM: %.3f ms", 1E3 * duration);
      if (NULL != times && 1 < nrepeat) {
        printf(" [%i calls %.3f-%.3f ms]", nrepeat, 1E3 * times[0], 1E3 * times[nrepeat-1]);
      }
      printf("\n");
    }
    else fprintf(stderr, "Ozaki GEMM failed (%s)\n", libxstream_opencl_strerror(result));
  }

  /* Reference BLAS GEMM */
  if (EXIT_SUCCESS == result) {
    int i;
    /**
     * save original C (still in c_ref) into c_oz; the Ozaki result will
     * be recomputed below for comparison after BLAS timing is done
     */
    memcpy(c_oz, c_ref, (size_t)ldc * N * elem_size);
    t0 = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      const libxs_timer_tick_t tick = libxs_timer_tick();
      if (ctx.use_double) {
        DGEMM(&transa, &transb, &M, &N, &K, &alpha, (const double*)a, &lda, (const double*)b, &ldb, &beta, (double*)c_ref, &ldc);
      }
      else {
        const float falpha = (float)alpha, fbeta = (float)beta;
        SGEMM(&transa, &transb, &M, &N, &K, &falpha, (const float*)a, &lda, (const float*)b, &ldb, &fbeta, (float*)c_ref, &ldc);
      }
      if (NULL != times) times[i] = libxs_timer_duration(tick, libxs_timer_tick());
      /* restore C before next iteration so beta does not accumulate - after the
       * sample is taken, since it is bookkeeping rather than part of the GEMM */
      if (i < nrepeat - 1) memcpy(c_ref, c_oz, (size_t)ldc * N * elem_size);
    }
    t1 = libxs_timer_tick();
    { const double duration = ozaki_duration(times, nrepeat, libxs_timer_duration(t0, t1));
      printf("BLAS  GEMM: %.3f ms", 1E3 * duration);
      if (NULL != times && 1 < nrepeat) {
        printf(" [%i calls %.3f-%.3f ms]", nrepeat, 1E3 * times[0], 1E3 * times[nrepeat-1]);
      }
      printf("\n");
    }
  }

#if defined(__CUBLAS)
  /**
   * Reference cuBLAS GEMM (device-side). Failure is soft: the host BLAS
   * result above remains the accuracy reference. c_oz holds the original
   * C at this point, hence it is the input of every timed iteration.
   */
  if (EXIT_SUCCESS == result) {
    double devtime[3] = {0, 0, 0}, duration = 0;
    cublas_result = cublas_gemm(stream, ctx.use_double, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, c_cu, ldc,
      nrepeat, ctx.ndecomp, &duration, devtime);
    if (EXIT_SUCCESS == cublas_result) {
      printf("cuBLAS GEMM: %.3f ms (%s)\n", 1E3 * duration, cublas_mode(ctx.use_double));
      if (0 < devtime[0]) { /* device-side split as measured with CUDA events */
        printf("cuBLAS: median gemm %.3f ms, h2d %.3f ms, d2h %.3f ms\n", devtime[0], devtime[1], devtime[2]);
      }
    }
    else fprintf(stderr, "cuBLAS GEMM failed\n");
  }
#endif

  /* Recompute Ozaki GEMM once for accuracy comparison (c_oz holds original C) */
  if (EXIT_SUCCESS == result) {
    result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc, 0);
    libxstream_stream_sync(stream);
  }

  /* Every result compared below is one call from the original C, so is the reference. */
  if (EXIT_SUCCESS == result && NULL != x_ref) {
    const libxs_data_t dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    const float falpha = (float)alpha, fbeta = (float)beta;
    const void* const pa = ctx.use_double ? (const void*)&alpha : (const void*)&falpha;
    const void* const pb = ctx.use_double ? (const void*)&beta : (const void*)&fbeta;
    int s, nfail = 0;
#if defined(_OPENMP)
#   pragma omp parallel for reduction(+:nfail) schedule(dynamic)
#endif
    for (s = 0; s < xrows; ++s) {
      const int row = xref_row(s, xrows, M);
      const char* const arow = (const char*)a + elem_size * (0 == ta ? (size_t)row : ((size_t)row * lda));
      if (EXIT_SUCCESS != libxs_dot2_gemm(dtype, transa, transb, 1, N, K, pa, arow, lda, b, ldb, pb,
        (char*)x_ref + elem_size * s, xrows))
      {
        ++nfail;
      }
    }
    if (0 != nfail) {
      fprintf(stderr, "ERROR: XREF reference failed on %i row(s): arguments, or no compensation in this build\n", nfail);
      result = EXIT_FAILURE;
    }
  }

  /**
   * A and B are dead once every result is computed, so |A| and |B| are formed
   * in place to bound the exact product. One driver or the other, never both:
   * libxs_matdiff_grade fills what libxs_matdiff fills.
   */
  if (EXIT_SUCCESS == result && NULL != c_bnd) {
    gemm_bound(ctx.use_double, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_bnd, ldc);
  }

  /* Compare */
  if (EXIT_SUCCESS == result) {
    const libxs_data_t dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    if (NULL != c_bnd) {
      result = libxs_matdiff_grade(&diff, dtype, M, N, c_ref, c_oz, c_bnd, &ldc, &ldc, &ldc);
    }
    else result = libxs_matdiff(&diff, dtype, M, N, c_ref, c_oz, &ldc, &ldc);
    if (EXIT_SUCCESS == result) {
      diff.r = nrepeat;
      print_diff(stdout, "", &diff);
    }
#if defined(__CUBLAS)
    if (EXIT_SUCCESS == result && EXIT_SUCCESS == cublas_result) {
      libxs_matdiff_t diff_cu;
      int diff_result;
      if (NULL != c_bnd) {
        diff_result = libxs_matdiff_grade(&diff_cu, dtype, M, N, c_ref, c_cu, c_bnd, &ldc, &ldc, &ldc);
      }
      else diff_result = libxs_matdiff(&diff_cu, dtype, M, N, c_ref, c_cu, &ldc, &ldc);
      if (EXIT_SUCCESS == diff_result) {
        diff_cu.r = nrepeat;
        print_diff(stdout, "cuBLAS ", &diff_cu);
      }
      /**
       * The same grade against the device reference instead of the host BLAS.
       * Emulators that publish an accuracy figure are measured this way, and the
       * difference to the line above is what the choice of reference is worth.
       */
      if (NULL != c_bnd) {
        libxs_matdiff_t diff_dev;
        if (EXIT_SUCCESS == libxs_matdiff_grade(&diff_dev, dtype, M, N, c_cu, c_oz, c_bnd, &ldc, &ldc, &ldc)) {
          diff_dev.r = nrepeat;
          print_diff(stdout, "vs-cuBLAS ", &diff_dev);
        }
      }
    }
#endif
  }

  /* The keys carry a prefix: a bare linf_rel= would be read as the BLAS-relative error. */
  if (EXIT_SUCCESS == result && NULL != x_ref) {
    const libxs_data_t dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    const size_t nx = (size_t)xrows * N * elem_size;
    char* const x_tst = (char*)malloc(nx * (NULL != c_bnd ? 2 : 1));
    if (NULL != x_tst) {
      void* const x_bnd = (NULL != c_bnd ? (x_tst + nx) : NULL);
      libxs_matdiff_t dx;
      int i;
      if (NULL != x_bnd) xref_gather(x_bnd, c_bnd, xrows, M, N, ldc, elem_size);
      printf("XREF: rows=%i", xrows);
      for (i = 0; i < 3 && EXIT_SUCCESS == result; ++i) {
        const void* const src = (0 == i ? c_oz : (1 == i ? c_ref : NULL));
        const char* const key = (0 == i ? "ozaki" : (1 == i ? "blas" : "cublas"));
        const void* tst = src;
#if defined(__CUBLAS)
        if (2 == i && EXIT_SUCCESS == cublas_result) tst = c_cu;
#endif
        if (NULL != tst) {
          xref_gather(x_tst, tst, xrows, M, N, ldc, elem_size);
          libxs_matdiff_clear(&dx);
          if (NULL != x_bnd) result = libxs_matdiff_grade(&dx, dtype, xrows, N, x_ref, x_tst, x_bnd, &xrows, &xrows, &xrows);
          else result = libxs_matdiff(&dx, dtype, xrows, N, x_ref, x_tst, &xrows, &xrows);
          if (EXIT_SUCCESS == result) xref_print(key, &dx, NULL != x_bnd);
        }
      }
      printf("\n");
      free(x_tst);
    }
    else result = EXIT_FAILURE;
  }

  if (0 != initialized) {
    /**
     * The host buffers are allocated against the stream, so they can only be
     * released while it exists. An early failure (no device, kernel build)
     * leaves stream NULL with nothing allocated yet.
     */
    if (NULL != stream) {
      if (NULL != a) libxstream_mem_host_deallocate(a, stream);
      if (NULL != b) libxstream_mem_host_deallocate(b, stream);
      if (NULL != c_oz) libxstream_mem_host_deallocate(c_oz, stream);
      if (NULL != c_ref) libxstream_mem_host_deallocate(c_ref, stream);
#if defined(__CUBLAS)
      if (NULL != c_cu) libxstream_mem_host_deallocate(c_cu, stream);
#endif
      libxstream_stream_destroy(stream);
    }
    ozaki_destroy(&ctx);
    /* after the context, which never frees caller-owned scratch */
    if (NULL != scratch) libxstream_mem_dev_deallocate_hint(scratch, libxstream_opencl_mem_hint_atomics);
    libxstream_finalize();
  }
  free(x_ref);
  free(c_bnd);
  free(times);
  return result;
}


static void print_diff(FILE* ostream, const char* label, const libxs_matdiff_t* diff)
{
  const double epsilon = libxs_matdiff_epsilon(diff);
  if (1E-6 <= epsilon) {
    fprintf(ostream, "%sDIFF: ncalls=%i linf=%.17g linf_rel=%.17g l2_rel=%.17g eps=%f rsq=%f -> %g != %g\n", label, diff->r,
      diff->linf_abs, diff->linf_rel, diff->l2_rel, epsilon, diff->rsq, diff->v_ref, diff->v_tst);
  }
  else {
    fprintf(ostream, "%sDIFF: ncalls=%i linf=%.17g linf_rel=%.17g l2_rel=%.17g eps=%f rsq=%f\n", label, diff->r, diff->linf_abs,
      diff->linf_rel, diff->l2_rel, epsilon, diff->rsq);
  }
  if (0 != diff->grade) fprintf(ostream, "%sGRADE: %g\n", label, diff->grade);
}


/* Row of C that the s-th of xrows samples grades: evenly spread, first row included. */
static int xref_row(int s, int xrows, int m)
{
  return (int)(((size_t)s * (size_t)m) / (size_t)(0 < xrows ? xrows : 1));
}


/* Sampled rows of an M-by-N column-major matrix into a compact xrows-by-N one. */
static void xref_gather(void* dst, const void* src, int xrows, int M, int N, int ldc, size_t elem_size)
{
  int j, s;
  for (j = 0; j < N; ++j) {
    for (s = 0; s < xrows; ++s) {
      memcpy((char*)dst + elem_size * ((size_t)j * xrows + s),
        (const char*)src + elem_size * ((size_t)j * ldc + xref_row(s, xrows, M)), elem_size);
    }
  }
}


static void xref_print(const char* key, const libxs_matdiff_t* diff, int graded)
{
  printf(" %s_linf_rel=%.17g %s_l2_rel=%.17g", key, diff->linf_rel, key, diff->l2_rel);
  if (0 != graded) printf(" %s_grade=%g", key, diff->grade);
}


/* Bound of the exact result, |alpha||A||B| + |beta||C|, with |C| already in c_bnd. */
static void gemm_bound(int use_double, char transa, char transb, int M, int N, int K, double alpha,
  void* a, int lda, void* b, int ldb, double beta, void* c_bnd, int ldc)
{
  const int a_cols = ('N' == transa ? K : M), b_cols = ('N' == transb ? N : K);
  const size_t na = (size_t)lda * a_cols, nb = (size_t)ldb * b_cols;
  size_t i;
  if (0 != use_double) {
    const double absa = fabs(alpha), absb = fabs(beta);
    for (i = 0; i < na; ++i) ((double*)a)[i] = fabs(((const double*)a)[i]);
    for (i = 0; i < nb; ++i) ((double*)b)[i] = fabs(((const double*)b)[i]);
    DGEMM(&transa, &transb, &M, &N, &K, &absa, (const double*)a, &lda, (const double*)b, &ldb, &absb, (double*)c_bnd, &ldc);
  }
  else {
    const float absa = (float)fabs(alpha), absb = (float)fabs(beta);
    for (i = 0; i < na; ++i) ((float*)a)[i] = (float)fabs(((const float*)a)[i]);
    for (i = 0; i < nb; ++i) ((float*)b)[i] = (float)fabs(((const float*)b)[i]);
    SGEMM(&transa, &transb, &M, &N, &K, &absa, (const float*)a, &lda, (const float*)b, &ldb, &absb, (float*)c_bnd, &ldc);
  }
}


/**
 * Time of one call out of nrepeat: the median of the per-call durations rather
 * than the mean of their sum, so that one interrupted call cannot move a figure
 * that is read as a rate, and so that the wall-clock line is the same kind of
 * statistic as the per-kernel medians of LIBXSTREAM_PROFILE.  Sorts times in
 * place - times[0] and times[nrepeat-1] are the extremes afterwards.  A NULL
 * times (allocation failed) leaves the mean of total as the only figure.
 */
static double ozaki_duration(double* times, int nrepeat, double total)
{
  double result;
  if (NULL != times && 0 < nrepeat) {
    int i;
    for (i = 1; i < nrepeat; ++i) { /* insertion sort: nrepeat is a repetition count, i.e. tens */
      const double t = times[i];
      int j = i;
      for (; 0 < j && times[j-1] > t; --j) times[j] = times[j-1];
      times[j] = t;
    }
    result = (0 == (nrepeat % 2)) ? (0.5 * (times[nrepeat/2-1] + times[nrepeat/2])) : times[nrepeat/2];
  }
  else result = total / (0 < nrepeat ? nrepeat : 1);
  return result;
}


#if defined(__CUBLAS)
/**
 * Request pure emulation, i.e. without the built-in fallback to native
 * FP64: "eager" emulates whenever possible rather than only when it is
 * profitable, and an explicit mantissa bit count switches the library
 * from dynamic control (which dispatches to native FP64 as soon as the
 * required precision exceeds the maximum) to fixed control. A variable
 * already present in the environment always wins. The values are read
 * when the cuBLAS runtime is entered, hence populated before that.
 */
static void cublas_putenv(int use_double, int nslices)
{
  static char emu_double[] = "CUBLAS_EMULATE_DOUBLE_PRECISION=1";
  static char emu_single[] = "CUBLAS_EMULATE_SINGLE_PRECISION=1";
  static char strategy[] = "CUBLAS_EMULATION_STRATEGY=eager";
  static char special[] = "CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0";
  static char mantissa[64] = "";
  const char* const env_bits = getenv("OZAKI_CUBLAS_BITS");
  const int bits = (NULL != env_bits ? atoi(env_bits) : 0);
  const char* const key = (0 != use_double ? "CUBLAS_EMULATE_DOUBLE_PRECISION" : "CUBLAS_EMULATE_SINGLE_PRECISION");
  if (NULL == getenv(key)) {
    LIBXS_EXPECT(0 == LIBXS_PUTENV(0 != use_double ? emu_double : emu_single));
  }
  if (NULL == getenv("CUBLAS_EMULATION_STRATEGY")) {
    LIBXS_EXPECT(0 == LIBXS_PUTENV(strategy));
  }
  if (NULL == getenv("CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK")) {
    LIBXS_EXPECT(0 == LIBXS_PUTENV(special));
  }
  /* negative: match the slice count of this sample (int8 slices carry 8 bits including the sign) */
  if (0 != bits && NULL == getenv("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT")) {
    const int n = (0 < bits ? bits : (8 * nslices - 1));
    if (0 < n && 0 < LIBXS_SNPRINTF(mantissa, sizeof(mantissa), "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=%i", n)) {
      LIBXS_EXPECT(0 == LIBXS_PUTENV(mantissa));
    }
  }
}


/**
 * Reports the emulation that was requested, not the one that was taken:
 * cuBLAS offers no way to query whether a particular call was emulated.
 */
static const char* cublas_mode(int use_double)
{
  const char* const env = getenv(0 != use_double ? "CUBLAS_EMULATE_DOUBLE_PRECISION" : "CUBLAS_EMULATE_SINGLE_PRECISION");
  const char* result = "native";
# if defined(CUBLAS_VER_MAJOR) && (13 <= CUBLAS_VER_MAJOR)
  if (NULL != env && 0 != atoi(env)) result = (0 != use_double ? "fixed-point emulation" : "bf16x9 emulation");
# else
  LIBXS_UNUSED(env);
# endif
  return result;
}


static int cublas_gemm(libxstream_stream_t* stream, int use_double, char transa, char transb, int M, int N, int K, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, const void* c_in, void* c_out, int ldc, int nrepeat, int nslices,
  double* duration, double* devtime)
{
  const char* const env_xptr = getenv("OZAKI_CUBLAS_XPTR");
  const int xptr = (NULL != env_xptr ? atoi(env_xptr) : 0);
  const size_t elem_size = (0 != use_double ? sizeof(double) : sizeof(float));
  const size_t size_a = (size_t)lda * ('N' == transa ? K : M) * elem_size;
  const size_t size_b = (size_t)ldb * ('N' == transb ? N : K) * elem_size;
  const size_t size_c = (size_t)ldc * N * elem_size;
  const cublasOperation_t opa = ('N' == transa ? CUBLAS_OP_N : CUBLAS_OP_T);
  const cublasOperation_t opb = ('N' == transb ? CUBLAS_OP_N : CUBLAS_OP_T);
  const float falpha = (float)alpha, fbeta = (float)beta;
  void *da = NULL, *db = NULL, *dc = NULL;
  cudaEvent_t event[4] = {NULL, NULL, NULL, NULL};
  cublasHandle_t handle = NULL;
  libxs_timer_tick_t t0 = 0, t1 = 0;
  double* const times = (double*)malloc((size_t)nrepeat * sizeof(double));
  /* per-call device phases, reported as medians like the call itself: means next to a median total broke their sum */
  double* const phases = (double*)malloc((size_t)3 * nrepeat * sizeof(double));
  int nevents = 0, nphases = 0;
# if (0 != OZAKI_CUBLAS_WORKSPACE)
  void* ws = NULL;
# endif
  int result = EXIT_SUCCESS, i;
  cublas_putenv(use_double, nslices);
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(&handle)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    struct cudaDeviceProp prop;
    int device = 0;
    if (cudaSuccess == cudaGetDevice(&device) && cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
      printf("cuBLAS: device %i \"%s\" (compute capability %i.%i)\n", device, prop.name, prop.major, prop.minor);
    }
  }
# if (0 != OZAKI_CUBLAS_WORKSPACE)
  if (EXIT_SUCCESS == result && cudaSuccess == cudaMalloc(&ws, OZAKI_CUBLAS_WORKSPACE)) {
    LIBXS_EXPECT(CUBLAS_STATUS_SUCCESS == cublasSetWorkspace(handle, ws, OZAKI_CUBLAS_WORKSPACE));
  }
# endif
  /**
   * Experiment (xptr): LIBXSTREAM hands out device-side pointers even for
   * LIBXSTREAM_USM=0, where they are tokens resolved to a cl_mem plus an
   * offset rather than addresses. Passing them to cuBLAS is expected to
   * fault; a USM-backed allocation may be an actual address, but it still
   * belongs to the address space of the OpenCL context.
   */
  if (EXIT_SUCCESS == result) {
    if (0 == xptr) {
      if (cudaSuccess != cudaMalloc(&da, size_a) || cudaSuccess != cudaMalloc(&db, size_b) ||
          cudaSuccess != cudaMalloc(&dc, size_c))
      {
        result = EXIT_FAILURE;
      }
    }
    else if (EXIT_SUCCESS != libxstream_mem_allocate(&da, size_a) || EXIT_SUCCESS != libxstream_mem_allocate(&db, size_b) ||
             EXIT_SUCCESS != libxstream_mem_allocate(&dc, size_c))
    {
      result = EXIT_FAILURE;
    }
  }
  /**
   * CUDA events are the counterpart of LIBXSTREAM_PROFILE: they separate the
   * GEMM from the transfers on the device timeline. Insight per kernel (and
   * thereby evidence of emulation) requires an external profiler.
   */
  if (EXIT_SUCCESS == result) {
    for (nevents = 0; nevents < 4 && cudaSuccess == cudaEventCreate(event + nevents); ++nevents);
  }
  /* the first iteration is the warmup (kernel setup, emulation workspace) and stays untimed */
  for (i = -1; i < nrepeat && EXIT_SUCCESS == result; ++i) {
    libxs_timer_tick_t tick;
    if (0 == i) t0 = libxs_timer_tick();
    tick = libxs_timer_tick();
    if (4 == nevents) LIBXS_ELIDE_RESULT(int, cudaEventRecord(event[0], 0));
    if (0 == xptr) {
      if (cudaSuccess != cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice) ||
          cudaSuccess != cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice) ||
          cudaSuccess != cudaMemcpy(dc, c_in, size_c, cudaMemcpyHostToDevice))
      {
        result = EXIT_FAILURE;
      }
    }
    else if (EXIT_SUCCESS != libxstream_mem_copy_h2d(a, da, size_a, stream) ||
             EXIT_SUCCESS != libxstream_mem_copy_h2d(b, db, size_b, stream) ||
             EXIT_SUCCESS != libxstream_mem_copy_h2d(c_in, dc, size_c, stream) ||
             EXIT_SUCCESS != libxstream_stream_sync(stream))
    { /* the two runtimes are unordered, hence the synchronization */
      result = EXIT_FAILURE;
    }
    if (4 == nevents) LIBXS_ELIDE_RESULT(int, cudaEventRecord(event[1], 0));
    if (EXIT_SUCCESS == result) {
      const cublasStatus_t status =
        (0 != use_double ? cublasDgemm(handle, opa, opb, M, N, K, &alpha, (const double*)da, lda, (const double*)db, ldb, &beta,
                             (double*)dc, ldc)
                         : cublasSgemm(handle, opa, opb, M, N, K, &falpha, (const float*)da, lda, (const float*)db, ldb, &fbeta,
                             (float*)dc, ldc));
      if (CUBLAS_STATUS_SUCCESS != status) result = EXIT_FAILURE;
    }
    if (4 == nevents) LIBXS_ELIDE_RESULT(int, cudaEventRecord(event[2], 0));
    if (EXIT_SUCCESS == result) {
      if (0 == xptr) { /* ordered against the GEMM by the default stream */
        if (cudaSuccess != cudaMemcpy(c_out, dc, size_c, cudaMemcpyDeviceToHost)) result = EXIT_FAILURE;
      }
      else if (cudaSuccess != cudaDeviceSynchronize() || EXIT_SUCCESS != libxstream_mem_copy_d2h(dc, c_out, size_c, stream) ||
               EXIT_SUCCESS != libxstream_stream_sync(stream))
      {
        result = EXIT_FAILURE;
      }
    }
    if (4 == nevents) LIBXS_ELIDE_RESULT(int, cudaEventRecord(event[3], 0));
    /* the transfers of the xptr experiment are not on the CUDA timeline, hence they read as zero */
    if (0 <= i && 4 == nevents && EXIT_SUCCESS == result && cudaSuccess == cudaEventSynchronize(event[3])) {
      float ms[3] = {0, 0, 0};
      if (cudaSuccess == cudaEventElapsedTime(ms + 0, event[1], event[2]) &&
          cudaSuccess == cudaEventElapsedTime(ms + 1, event[0], event[1]) &&
          cudaSuccess == cudaEventElapsedTime(ms + 2, event[2], event[3]))
      {
        int j;
        for (j = 0; j < 3; ++j) {
          if (NULL != phases) phases[j * nrepeat + nphases] = ms[j];
          devtime[j] += ms[j];
        }
        ++nphases;
      }
    }
    if (0 <= i && NULL != times) times[i] = libxs_timer_duration(tick, libxs_timer_tick());
  }
  if (EXIT_SUCCESS == result && cudaSuccess != cudaDeviceSynchronize()) result = EXIT_FAILURE;
  t1 = libxs_timer_tick();
  /* an error is not reported: a faulting GEMM makes the CUDA runtime fail persistently */
  for (i = 0; i < nevents; ++i) LIBXS_ELIDE_RESULT(int, cudaEventDestroy(event[i]));
  if (0 == xptr) {
    if (NULL != da) LIBXS_ELIDE_RESULT(int, cudaFree(da));
    if (NULL != db) LIBXS_ELIDE_RESULT(int, cudaFree(db));
    if (NULL != dc) LIBXS_ELIDE_RESULT(int, cudaFree(dc));
  }
  else {
    if (NULL != da) LIBXS_ELIDE_RESULT(int, libxstream_mem_deallocate(da));
    if (NULL != db) LIBXS_ELIDE_RESULT(int, libxstream_mem_deallocate(db));
    if (NULL != dc) LIBXS_ELIDE_RESULT(int, libxstream_mem_deallocate(dc));
  }
  if (NULL != handle) LIBXS_ELIDE_RESULT(int, cublasDestroy(handle));
# if (0 != OZAKI_CUBLAS_WORKSPACE)
  if (NULL != ws) LIBXS_ELIDE_RESULT(int, cudaFree(ws));
# endif
  /* the median of one call, matching the Ozaki and host-BLAS figures */
  if (EXIT_SUCCESS == result) *duration = ozaki_duration(times, nrepeat, libxs_timer_duration(t0, t1));
  if (0 < nphases) {
    int j;
    for (j = 0; j < 3; ++j) { /* the mean of the sum where the per-call record could not be kept */
      devtime[j] = (NULL != phases) ? ozaki_duration(phases + j * nrepeat, nphases, 0) : (devtime[j] / nphases);
    }
  }
  free(phases);
  free(times);
  return result;
}
#endif
