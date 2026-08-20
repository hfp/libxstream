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
  void* scratch = NULL; /* caller-owned device scratch (OZAKI_SCRATCH) */
#if defined(__CUBLAS)
  void* c_cu = NULL;
  int cublas_result = EXIT_FAILURE;
#endif
  libxstream_stream_t* stream = NULL;
  libxs_matdiff_t diff;
  libxs_timer_tick_t t0, t1;
  size_t elem_size = 0;
  int result = EXIT_SUCCESS;
  int initialized = 0;

  LIBXS_MEMZERO(&ctx);

  if (1 > M || 1 > N || 1 > K || lda < (0 == ta ? M : K) || ldb < (0 == tb ? K : N) || ldc < M) {
    fprintf(stderr, "Invalid dimensions: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
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
    env = getenv("OZAKI_N");
    if (NULL != env) ndecomp = atoi(env);
    env = getenv("OZAKI");
    if (NULL != env) kind = atoi(env);
    env = getenv("OZAKI_GROUPS");
    if (NULL != env) ozgroups = atoi(env);
    env = getenv("OZAKI_VERBOSE");
    if (NULL != env) verbosity = atoi(env);
    env = getenv("OZAKI_FP");
    if (NULL != env) use_double = (32 != atoi(env));
    result = ozaki_init(&ctx, tm, tn, use_double, kind, verbosity, ndecomp, ozflags, oztrim, ozgroups, 0 /*maxk: no grouping*/);
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
        libxstream_mem_dev_deallocate_hint(scratch);
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
    /* restore C for the timed run (beta may be non-zero) */
    if (EXIT_SUCCESS == result) memcpy(c_oz, c_ref, (size_t)ldc * N * elem_size);
    t0 = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      result = ozaki_gemm(&ctx, stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c_oz, ldc, 0);
      if (EXIT_SUCCESS != result) break;
    }
    libxstream_stream_sync(stream);
    t1 = libxs_timer_tick();
    if (EXIT_SUCCESS == result) {
      printf("Ozaki GEMM: %.1f ms\n", 1E3 * libxs_timer_duration(t0, t1) / nrepeat);
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
      printf("cuBLAS GEMM: %.1f ms (%s)\n", 1E3 * duration / nrepeat, cublas_mode(ctx.use_double));
      if (0 < devtime[0]) { /* device-side split as measured with CUDA events */
        printf("cuBLAS: gemm %.1f ms, h2d %.1f ms, d2h %.1f ms\n", devtime[0] / nrepeat, devtime[1] / nrepeat,
          devtime[2] / nrepeat);
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

  /* Compare */
  if (EXIT_SUCCESS == result) {
    const libxs_data_t dtype = ctx.use_double ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32;
    result = libxs_matdiff(&diff, dtype, M, N, c_ref, c_oz, &ldc, &ldc);
    if (EXIT_SUCCESS == result) {
      diff.r = nrepeat;
      print_diff(stdout, "", &diff);
    }
#if defined(__CUBLAS)
    if (EXIT_SUCCESS == result && EXIT_SUCCESS == cublas_result) {
      libxs_matdiff_t diff_cu;
      if (EXIT_SUCCESS == libxs_matdiff(&diff_cu, dtype, M, N, c_ref, c_cu, &ldc, &ldc)) {
        diff_cu.r = nrepeat;
        print_diff(stdout, "cuBLAS ", &diff_cu);
      }
    }
#endif
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
    if (NULL != scratch) libxstream_mem_dev_deallocate_hint(scratch);
    libxstream_finalize();
  }
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
  int nevents = 0;
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
    if (0 == i) t0 = libxs_timer_tick();
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
      float ms = 0;
      if (cudaSuccess == cudaEventElapsedTime(&ms, event[1], event[2])) devtime[0] += ms;
      if (cudaSuccess == cudaEventElapsedTime(&ms, event[0], event[1])) devtime[1] += ms;
      if (cudaSuccess == cudaEventElapsedTime(&ms, event[2], event[3])) devtime[2] += ms;
    }
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
  if (EXIT_SUCCESS == result) *duration = libxs_timer_duration(t0, t1);
  return result;
}
#endif
