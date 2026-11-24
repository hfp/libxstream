/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Plugin of the device BLAS (libxstream_blas.h): oneMKL's SYCL interface on the OpenCL context and
 * queue of a stream, compiled by the host's C++ compiler (no SYCL compiler, no oneMKL headers). */
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include "libxstream_blas_plugin.h"
#include <sycl/sycl.hpp>
#include <sycl/backend/opencl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

/**
 * oneMKL's SYCL interface declared here rather than included. A declaration that no longer matches
 * libmkl_sycl_blas links and computes wrong results (the enumerations are chars, which the mangled
 * names do not say), hence the self-test before first use. The group interfaces take alpha and beta
 * as arrays, which keeps oneMKL's value_or_pointer out of the declarations.
 */
namespace oneapi {
namespace mkl {
enum class transpose : char { nontrans = 0, trans = 1, conjtrans = 3 };
enum class uplo : char { upper = 0, lower = 1 };
enum class side : char { left = 0, right = 1 };
namespace blas {
namespace column_major {
sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb, const std::int64_t* m,
  const std::int64_t* n, const std::int64_t* k, const double* alpha, const double** a, const std::int64_t* lda, const double** b,
  const std::int64_t* ldb, const double* beta, double** c, const std::int64_t* ldc, std::int64_t group_count,
  const std::int64_t* group_size, const std::vector<sycl::event>& dependencies);
sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb, const std::int64_t* m,
  const std::int64_t* n, const std::int64_t* k, const float* alpha, const float** a, const std::int64_t* lda, const float** b,
  const std::int64_t* ldb, const float* beta, float** c, const std::int64_t* ldc, std::int64_t group_count,
  const std::int64_t* group_size, const std::vector<sycl::event>& dependencies);
sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans, const std::int64_t* n,
  const std::int64_t* k, const double* alpha, const double** a, const std::int64_t* lda, const double* beta, double** c,
  const std::int64_t* ldc, std::int64_t group_count, const std::int64_t* group_size, const std::vector<sycl::event>& dependencies);
sycl::event syrk_batch(sycl::queue& queue, const uplo* upper_lower, const transpose* trans, const std::int64_t* n,
  const std::int64_t* k, const float* alpha, const float** a, const std::int64_t* lda, const float* beta, float** c,
  const std::int64_t* ldc, std::int64_t group_count, const std::int64_t* group_size, const std::vector<sycl::event>& dependencies);
sycl::event dgmm(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n, const double* a, std::int64_t lda,
  const double* x, std::int64_t incx, double* c, std::int64_t ldc, const std::vector<sycl::event>& dependencies);
sycl::event dgmm(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n, const float* a, std::int64_t lda,
  const float* x, std::int64_t incx, float* c, std::int64_t ldc, const std::vector<sycl::event>& dependencies);
} /* namespace column_major */
} /* namespace blas */
} /* namespace mkl */
} /* namespace oneapi */

namespace {
namespace blas = oneapi::mkl::blas::column_major;

/* arguments of a batch (libxstream_mkl_args_t), freed when its work is done */
struct libxstream_mkl_pending_t {
  sycl::event event;
  void* args;
};

/**
 * The SYCL context made of the OpenCL context, whether oneMKL passed its self-test per precision
 * (single, double), and the arguments still in flight. An interop object neither retains nor releases its OpenCL handle, hence a queue is
 * made per call (a cached one could outlive the stream). Never destroyed: the SYCL runtime may be
 * gone by the time the process exits.
 */
struct libxstream_mkl_state_t {
  std::mutex lock;
  cl_context clctx = nullptr;
  sycl::context* context = nullptr;
  int status[2] = {0, 0}; /* 0: untested, 1: usable, -1: disabled */
  std::vector<libxstream_mkl_pending_t> pending;
};

libxstream_mkl_state_t& libxstream_mkl_state()
{
  static libxstream_mkl_state_t* const state = new libxstream_mkl_state_t;
  return *state;
}

oneapi::mkl::transpose libxstream_mkl_trans(char t)
{
  return ('N' == t || 'n' == t) ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
}

/* Frees the arguments whose work completed (all of them after waiting if wait is true). */
void libxstream_mkl_reap(libxstream_mkl_state_t& state, bool wait)
{
  std::vector<libxstream_mkl_pending_t> keep;
  for (libxstream_mkl_pending_t& p : state.pending) {
    if (wait) p.event.wait();
    if (sycl::info::event_command_status::complete ==
        p.event.get_info<sycl::info::event::command_execution_status>())
    {
      sycl::free(p.args, *state.context);
    }
    else keep.push_back(p);
  }
  state.pending.swap(keep);
}

/**
 * Arguments of a batch, which oneMKL reads when the work runs (a host task on a CPU device), hence
 * in USM until the work is done rather than on the stack of the caller. The matrix pointers follow
 * (see libxstream_mkl_pointers and libxstream_mkl_nargs).
 */
template <typename T>
struct libxstream_mkl_args_t {
  std::int64_t m, n, k, lda, ldb, ldc, size;
  T alpha, beta;
  oneapi::mkl::transpose transa, transb;
  oneapi::mkl::uplo uplo;
};

/* The matrix pointers of a batch, which follow its arguments (the size of the struct keeps their alignment). */
template <typename T>
const T** libxstream_mkl_pointers(libxstream_mkl_args_t<T>* args)
{
  return reinterpret_cast<const T**>(args + 1);
}

/* A batch of size entries C := alpha*op(A)*op(B) + beta*C, which may run concurrently. */
template <typename T>
sycl::event libxstream_mkl_gemm(sycl::queue& queue, void* args, char transa, char transb, int m, int n, int k,
  double alpha, const void* const* a, int lda, const void* const* b, int ldb, double beta, void* const* c, int ldc,
  int size)
{
  libxstream_mkl_args_t<T>* const x = static_cast<libxstream_mkl_args_t<T>*>(args);
  const T** const p = libxstream_mkl_pointers(x);
  int i;
  for (i = 0; i < size; ++i) {
    p[i] = static_cast<const T*>(a[i]);
    p[size + i] = static_cast<const T*>(b[i]);
    p[2 * size + i] = static_cast<const T*>(c[i]);
  }
  x->m = m; x->n = n; x->k = k; x->lda = lda; x->ldb = ldb; x->ldc = ldc; x->size = size;
  x->alpha = static_cast<T>(alpha);
  x->beta = static_cast<T>(beta);
  x->transa = libxstream_mkl_trans(transa);
  x->transb = libxstream_mkl_trans(transb);
  return blas::gemm_batch(queue, &x->transa, &x->transb, &x->m, &x->n, &x->k, &x->alpha, p, &x->lda, p + size,
    &x->ldb, &x->beta, const_cast<T**>(p + 2 * size), &x->ldc, 1 /*group_count*/, &x->size, std::vector<sycl::event>());
}

template <typename T>
sycl::event libxstream_mkl_syrk(sycl::queue& queue, void* args, char uplo, char trans, int n, int k, double alpha,
  const void* a, int lda, double beta, void* c, int ldc)
{
  libxstream_mkl_args_t<T>* const x = static_cast<libxstream_mkl_args_t<T>*>(args);
  const T** const p = libxstream_mkl_pointers(x);
  p[0] = static_cast<const T*>(a);
  p[1] = static_cast<const T*>(c);
  x->n = n; x->k = k; x->lda = lda; x->ldc = ldc; x->size = 1;
  x->alpha = static_cast<T>(alpha);
  x->beta = static_cast<T>(beta);
  x->transa = libxstream_mkl_trans(trans);
  x->uplo = (('U' == uplo || 'u' == uplo) ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower);
  return blas::syrk_batch(queue, &x->uplo, &x->transa, &x->n, &x->k, &x->alpha, p, &x->lda, &x->beta,
    const_cast<T**>(p + 1), &x->ldc, 1 /*group_count*/, &x->size, std::vector<sycl::event>());
}

template <typename T>
sycl::event libxstream_mkl_dgmm(sycl::queue& queue, char side, int m, int n, const void* a, int lda, const void* x,
  int incx, void* c, int ldc)
{
  const oneapi::mkl::side s = (('L' == side || 'l' == side) ? oneapi::mkl::side::left : oneapi::mkl::side::right);
  return blas::dgmm(queue, s, m, n, static_cast<const T*>(a), lda, static_cast<const T*>(x), incx, static_cast<T*>(c), ldc,
    std::vector<sycl::event>());
}

/* Each entry point in precision T against the host, synchronously: small integers make it exact. */
template <typename T>
bool libxstream_mkl_selftest_type(sycl::queue& queue)
{
  const int m = 3, n = 2, k = 4;
  std::vector<T> ha(m * k), hb(k * n), hc(m * m), ref(m * m), hx(m);
  bool result = true;
  int i, j, p;
  T* const d = sycl::malloc_device<T>(m * k + k * n + m * m + m, queue);
  void* const args = sycl::malloc_shared(sizeof(libxstream_mkl_args_t<T>) + 3 * sizeof(void*), queue);
  if (nullptr == d || nullptr == args) throw std::bad_alloc();
  for (i = 0; i < m * k; ++i) ha[i] = static_cast<T>(i % 5 - 2);
  for (i = 0; i < k * n; ++i) hb[i] = static_cast<T>(i % 3 - 1);
  for (i = 0; i < m; ++i) hx[i] = static_cast<T>(i + 2);
  queue.memcpy(d, ha.data(), sizeof(T) * m * k).wait_and_throw();
  queue.memcpy(d + m * k, hb.data(), sizeof(T) * k * n).wait_and_throw();
  queue.memcpy(d + m * k + k * n + m * m, hx.data(), sizeof(T) * m).wait_and_throw();
  { /* GEMM (a batch of one), C (m x n) := 2 * A * B + 1 * C */
    const void* const pa = d;
    const void* const pb = d + m * k;
    void* const pc = d + m * k + k * n;
    for (i = 0; i < m * n; ++i) ref[i] = hc[i] = static_cast<T>(i + 1);
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        for (p = 0; p < k; ++p) ref[i + j * m] += 2 * ha[i + p * m] * hb[p + j * k];
      }
    }
    queue.memcpy(d + m * k + k * n, hc.data(), sizeof(T) * m * n).wait_and_throw();
    libxstream_mkl_gemm<T>(queue, args, 'N', 'N', m, n, k, 2.0, &pa, m, &pb, k, 1.0, &pc, m, 1).wait_and_throw();
    queue.memcpy(hc.data(), d + m * k + k * n, sizeof(T) * m * n).wait_and_throw();
    for (i = 0; i < m * n; ++i) result = (result && ref[i] == hc[i]);
  }
  { /* SYRK, lower triangle of C (m x m) := A * A^T, the upper triangle untouched */
    for (i = 0; i < m * m; ++i) ref[i] = hc[i] = static_cast<T>(-7);
    for (j = 0; j < m; ++j) {
      for (i = j; i < m; ++i) {
        ref[i + j * m] = 0;
        for (p = 0; p < k; ++p) ref[i + j * m] += ha[i + p * m] * ha[j + p * m];
      }
    }
    queue.memcpy(d + m * k + k * n, hc.data(), sizeof(T) * m * m).wait_and_throw();
    libxstream_mkl_syrk<T>(queue, args, 'L', 'N', m, k, 1.0, d, m, 0.0, d + m * k + k * n, m).wait_and_throw();
    queue.memcpy(hc.data(), d + m * k + k * n, sizeof(T) * m * m).wait_and_throw();
    for (i = 0; i < m * m; ++i) result = (result && ref[i] == hc[i]);
  }
  { /* DGMM, C (m x n) := diag(x) * B[0:m, 0:n] (lda k) */
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) ref[i + j * m] = hx[i] * hb[i + j * k];
    }
    libxstream_mkl_dgmm<T>(queue, 'L', m, n, d + m * k, k, d + m * k + k * n + m * m, 1, d + m * k + k * n, m)
      .wait_and_throw();
    queue.memcpy(hc.data(), d + m * k + k * n, sizeof(T) * m * n).wait_and_throw();
    for (i = 0; i < m * n; ++i) result = (result && ref[i] == hc[i]);
  }
  sycl::free(args, queue);
  sycl::free(d, queue);
  return result;
}

/**
 * Self-test of each precision the device has, once per context. A failure or an exception disables
 * that precision only, e.g., double precision on a device without it.
 */
void libxstream_mkl_selftest(libxstream_mkl_state_t& state, sycl::queue& queue)
{
  int f64;
  for (f64 = 0; f64 < 2; ++f64) {
    if (0 == state.status[f64]) {
      const char* const precision = (0 != f64 ? "double" : "single");
      state.status[f64] = -1;
      try {
        if (0 == f64 || queue.get_device().has(sycl::aspect::fp64)) {
          if (0 != f64 ? libxstream_mkl_selftest_type<double>(queue) : libxstream_mkl_selftest_type<float>(queue)) {
            state.status[f64] = 1;
          }
          else std::fprintf(stderr, "WARNING LIBXSTREAM: oneMKL (SYCL) failed its self-test in %s precision\n", precision);
        }
      }
      catch (const std::exception& e) {
        std::fprintf(stderr, "WARNING LIBXSTREAM: oneMKL (SYCL) failed its self-test in %s precision: %s\n", precision,
          e.what());
      }
    }
  }
}

/**
 * The work of one call on a queue made of the stream's OpenCL queue (hence in order with the stream's
 * other work), after the context is set up and oneMKL passed its self-test in the precision f64 (any
 * precision if negative). Returns zero on success.
 */
template <typename F>
int libxstream_mkl_run(void* context, void* queue, int f64, std::size_t nargs, F work)
{
  libxstream_mkl_state_t& state = libxstream_mkl_state();
  int result = -1;
  try {
    const cl_context clctx = static_cast<cl_context>(context);
    const std::lock_guard<std::mutex> guard(state.lock);
    if (clctx != state.clctx) {
      if (nullptr != state.context) libxstream_mkl_reap(state, true);
      delete state.context;
      state.context = new sycl::context(sycl::make_context<sycl::backend::opencl>(clctx));
      state.clctx = clctx;
      state.status[0] = state.status[1] = 0;
    }
    {
      sycl::queue q = sycl::make_queue<sycl::backend::opencl>(static_cast<cl_command_queue>(queue), *state.context);
      libxstream_mkl_selftest(state, q);
      if (0 > f64 ? (0 < state.status[0] || 0 < state.status[1]) : (0 < state.status[0 != f64 ? 1 : 0])) {
        void* const args = (0 < nargs ? sycl::malloc_shared(nargs, q) : nullptr);
        if (0 < nargs && nullptr == args) throw std::bad_alloc();
        /* on a CPU, oneMKL runs host tasks which the OpenCL queue does not order: synchronous */
        const bool host = q.get_device().is_cpu();
        libxstream_mkl_reap(state, false);
        if (host && CL_SUCCESS != clFinish(static_cast<cl_command_queue>(queue))) throw std::runtime_error("clFinish");
        sycl::event event = work(q, args);
        if (host) event.wait_and_throw();
        if (nullptr != args) state.pending.push_back(libxstream_mkl_pending_t{event, args});
        result = 0;
      }
    }
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "WARNING LIBXSTREAM: oneMKL (SYCL) failed: %s\n", e.what());
  }
  return result;
}

/* Bytes of the arguments of a batch with the given number of matrix pointers. */
std::size_t libxstream_mkl_nargs(int f64, int npointers)
{
  return (0 != f64 ? sizeof(libxstream_mkl_args_t<double>) : sizeof(libxstream_mkl_args_t<float>))
       + sizeof(void*) * static_cast<std::size_t>(npointers);
}

/**
 * An Intel device, memory oneMKL addresses, and a passed self-test (set up once per context, nothing
 * enqueued). Intel USM is taken as is. A buffer's device address is taken where memory is unified
 * (the driver keeps such a buffer resident); elsewhere a buffer oneMKL does not reference may not be.
 * SVM is taken on a CPU only: on an integrated GPU (Gen12LP), oneMKL produced wrong results with SVM.
 */
int libxstream_mkl_probe_entry(void* context, void* queue, int kind)
{
  const cl_command_queue clqueue = static_cast<cl_command_queue>(queue);
  cl_device_id device = nullptr;
  cl_device_type type = 0;
  cl_uint vendor = 0;
  cl_bool unified = CL_FALSE;
  int result = -1;
  if (CL_SUCCESS == clGetCommandQueueInfo(clqueue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr)
    && CL_SUCCESS == clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, nullptr)
    && CL_SUCCESS == clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr)
    && CL_SUCCESS == clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr)
    && 0x8086 == vendor
    && (LIBXSTREAM_BLAS_MEM_USM == kind || (LIBXSTREAM_BLAS_MEM_SVM == kind && 0 != (CL_DEVICE_TYPE_CPU & type))
      || (LIBXSTREAM_BLAS_MEM_BUFFER == kind && CL_FALSE != unified)))
  {
    result = libxstream_mkl_run(context, queue, -1, 0, [](sycl::queue&, void*) { return sycl::event(); });
  }
  return result;
}

int libxstream_mkl_gemm_batch_entry(void* context, void* queue, int f64, char transa, char transb, int m, int n, int k,
  double alpha, const void* const* a, int lda, const void* const* b, int ldb, double beta, void* const* c, int ldc,
  int size)
{
  /* the pointers are copied into the arguments before this returns (work runs within libxstream_mkl_run) */
  return (0 < size ? libxstream_mkl_run(context, queue, f64, libxstream_mkl_nargs(f64, 3 * size), [=](sycl::queue& q, void* args) {
    return (0 != f64
      ? libxstream_mkl_gemm<double>(q, args, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, size)
      : libxstream_mkl_gemm<float>(q, args, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, size));
  }) : -1);
}

int libxstream_mkl_syrk_entry(void* context, void* queue, int f64, char uplo, char trans, int n, int k, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  (void)ldb;
  return (nullptr == b ? libxstream_mkl_run(context, queue, f64, libxstream_mkl_nargs(f64, 2), [=](sycl::queue& q, void* args) {
    return (0 != f64 ? libxstream_mkl_syrk<double>(q, args, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
                     : libxstream_mkl_syrk<float>(q, args, uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
  })
                       : -1); /* SYR2K is declined: there is no group interface of it */
}

int libxstream_mkl_dgmm_entry(void* context, void* queue, int f64, char side, int m, int n, const void* a, int lda,
  const void* x, int incx, void* c, int ldc)
{
  return libxstream_mkl_run(context, queue, f64, 0, [=](sycl::queue& q, void*) {
    return (0 != f64 ? libxstream_mkl_dgmm<double>(q, side, m, n, a, lda, x, incx, c, ldc)
                     : libxstream_mkl_dgmm<float>(q, side, m, n, a, lda, x, incx, c, ldc));
  });
}
} /* namespace */


extern "C" __attribute__((visibility("default"))) int libxstream_blas_plugin(libxstream_blas_plugin_t* table)
{
  int result = -1;
  if (nullptr != table && LIBXSTREAM_BLAS_PLUGIN_VERSION == table->version) {
    table->name = "mkl";
    table->probe = libxstream_mkl_probe_entry;
    table->gemm_batch = libxstream_mkl_gemm_batch_entry;
    table->syrk = libxstream_mkl_syrk_entry;
    table->dgmm = libxstream_mkl_dgmm_entry;
    result = 0;
  }
  return result;
}
