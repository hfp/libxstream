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
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
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

/* pointers a batch reads on the device, hence USM until its work is done */
struct libxstream_mkl_pending_t {
  sycl::event event;
  void* pointers;
};

/**
 * The SYCL context made of the OpenCL context, whether oneMKL passed its self-test, and the pointers
 * still in flight. An interop object neither retains nor releases its OpenCL handle, hence a queue is
 * made per call (a cached one could outlive the stream). Never destroyed: the SYCL runtime may be
 * gone by the time the process exits.
 */
struct libxstream_mkl_state_t {
  std::mutex lock;
  cl_context clctx = nullptr;
  sycl::context* context = nullptr;
  int status = 0; /* 0: untested, 1: usable, -1: disabled */
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

/* Frees the pointers whose work completed (all of them after waiting if wait is true). */
void libxstream_mkl_reap(libxstream_mkl_state_t& state, bool wait)
{
  std::vector<libxstream_mkl_pending_t> keep;
  for (libxstream_mkl_pending_t& p : state.pending) {
    if (wait) p.event.wait();
    if (sycl::info::event_command_status::complete ==
        p.event.get_info<sycl::info::event::command_execution_status>())
    {
      sycl::free(p.pointers, *state.context);
    }
    else keep.push_back(p);
  }
  state.pending.swap(keep);
}

template <typename T>
sycl::event libxstream_mkl_gemm(sycl::queue& queue, void* pointers, char transa, char transb, int m, int n, int k,
  double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  const oneapi::mkl::transpose ta = libxstream_mkl_trans(transa), tb = libxstream_mkl_trans(transb);
  const std::int64_t mm = m, nn = n, kk = k, la = lda, lb = ldb, lc = ldc, size = 1;
  const T fa = static_cast<T>(alpha), fb = static_cast<T>(beta);
  const T** const p = static_cast<const T**>(pointers);
  p[0] = static_cast<const T*>(a);
  p[1] = static_cast<const T*>(b);
  p[2] = static_cast<const T*>(c);
  return blas::gemm_batch(queue, &ta, &tb, &mm, &nn, &kk, &fa, p, &la, p + 1, &lb, &fb, const_cast<T**>(p + 2), &lc,
    1 /*group_count*/, &size, std::vector<sycl::event>());
}

template <typename T>
sycl::event libxstream_mkl_syrk(sycl::queue& queue, void* pointers, char uplo, char trans, int n, int k, double alpha,
  const void* a, int lda, double beta, void* c, int ldc)
{
  const oneapi::mkl::uplo ul = (('U' == uplo || 'u' == uplo) ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower);
  const oneapi::mkl::transpose t = libxstream_mkl_trans(trans);
  const std::int64_t nn = n, kk = k, la = lda, lc = ldc, size = 1;
  const T fa = static_cast<T>(alpha), fb = static_cast<T>(beta);
  const T** const p = static_cast<const T**>(pointers);
  p[0] = static_cast<const T*>(a);
  p[1] = static_cast<const T*>(c);
  return blas::syrk_batch(queue, &ul, &t, &nn, &kk, &fa, p, &la, &fb, const_cast<T**>(p + 1), &lc, 1 /*group_count*/,
    &size, std::vector<sycl::event>());
}

template <typename T>
sycl::event libxstream_mkl_dgmm(sycl::queue& queue, char side, int m, int n, const void* a, int lda, const void* x,
  int incx, void* c, int ldc)
{
  const oneapi::mkl::side s = (('L' == side || 'l' == side) ? oneapi::mkl::side::left : oneapi::mkl::side::right);
  return blas::dgmm(queue, s, m, n, static_cast<const T*>(a), lda, static_cast<const T*>(x), incx, static_cast<T*>(c), ldc,
    std::vector<sycl::event>());
}

/* Each entry point in both precisions against the host, synchronously: small integers make it exact. */
template <typename T>
bool libxstream_mkl_selftest_type(sycl::queue& queue)
{
  const int m = 3, n = 2, k = 4;
  std::vector<T> ha(m * k), hb(k * n), hc(m * m), ref(m * m), hx(m);
  bool result = true;
  int i, j, p;
  T* const d = sycl::malloc_device<T>(m * k + k * n + m * m + m, queue);
  void* const pointers = sycl::malloc_shared(3 * sizeof(void*), queue);
  if (nullptr == d || nullptr == pointers) throw std::bad_alloc();
  for (i = 0; i < m * k; ++i) ha[i] = static_cast<T>(i % 5 - 2);
  for (i = 0; i < k * n; ++i) hb[i] = static_cast<T>(i % 3 - 1);
  for (i = 0; i < m; ++i) hx[i] = static_cast<T>(i + 2);
  queue.memcpy(d, ha.data(), sizeof(T) * m * k).wait_and_throw();
  queue.memcpy(d + m * k, hb.data(), sizeof(T) * k * n).wait_and_throw();
  queue.memcpy(d + m * k + k * n + m * m, hx.data(), sizeof(T) * m).wait_and_throw();
  { /* GEMM, C (m x n) := 2 * A * B + 1 * C */
    for (i = 0; i < m * n; ++i) ref[i] = hc[i] = static_cast<T>(i + 1);
    for (j = 0; j < n; ++j) {
      for (i = 0; i < m; ++i) {
        for (p = 0; p < k; ++p) ref[i + j * m] += 2 * ha[i + p * m] * hb[p + j * k];
      }
    }
    queue.memcpy(d + m * k + k * n, hc.data(), sizeof(T) * m * n).wait_and_throw();
    libxstream_mkl_gemm<T>(queue, pointers, 'N', 'N', m, n, k, 2.0, d, m, d + m * k, k, 1.0, d + m * k + k * n, m)
      .wait_and_throw();
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
    libxstream_mkl_syrk<T>(queue, pointers, 'L', 'N', m, k, 1.0, d, m, 0.0, d + m * k + k * n, m).wait_and_throw();
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
  sycl::free(pointers, queue);
  sycl::free(d, queue);
  return result;
}

/**
 * The work of one call on a queue made of the stream's OpenCL queue (hence in order with the stream's
 * other work), after the context is set up and oneMKL passed its self-test. Returns zero on success.
 */
template <typename F>
int libxstream_mkl_run(void* context, void* queue, int npointers, F work)
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
      state.status = 0;
    }
    {
      sycl::queue q = sycl::make_queue<sycl::backend::opencl>(static_cast<cl_command_queue>(queue), *state.context);
      if (0 == state.status) {
        state.status = ((libxstream_mkl_selftest_type<double>(q) && libxstream_mkl_selftest_type<float>(q)) ? 1 : -1);
        if (0 > state.status) {
          std::fprintf(stderr, "WARNING LIBXSTREAM: oneMKL (SYCL) failed its self-test, no device BLAS\n");
        }
      }
      if (0 < state.status) {
        void* const pointers = (0 < npointers ? sycl::malloc_shared(npointers * sizeof(void*), q) : nullptr);
        if (0 < npointers && nullptr == pointers) throw std::bad_alloc();
        libxstream_mkl_reap(state, false);
        const sycl::event event = work(q, pointers);
        if (nullptr != pointers) state.pending.push_back(libxstream_mkl_pending_t{event, pointers});
        result = 0;
      }
    }
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "WARNING LIBXSTREAM: oneMKL (SYCL) failed: %s\n", e.what());
  }
  return result;
}

int libxstream_mkl_gemm_entry(void* context, void* queue, int f64, char transa, char transb, int m, int n, int k,
  double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  return libxstream_mkl_run(context, queue, 3, [=](sycl::queue& q, void* pointers) {
    return (0 != f64 ? libxstream_mkl_gemm<double>(q, pointers, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                     : libxstream_mkl_gemm<float>(q, pointers, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
  });
}

int libxstream_mkl_syrk_entry(void* context, void* queue, int f64, char uplo, char trans, int n, int k, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  (void)ldb;
  return (nullptr == b ? libxstream_mkl_run(context, queue, 2, [=](sycl::queue& q, void* pointers) {
    return (0 != f64 ? libxstream_mkl_syrk<double>(q, pointers, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
                     : libxstream_mkl_syrk<float>(q, pointers, uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
  })
                       : -1); /* SYR2K is declined: there is no group interface of it */
}

int libxstream_mkl_dgmm_entry(void* context, void* queue, int f64, char side, int m, int n, const void* a, int lda,
  const void* x, int incx, void* c, int ldc)
{
  return libxstream_mkl_run(context, queue, 0, [=](sycl::queue& q, void*) {
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
    table->xmem = 1; /* Intel USM of libxstream_mem_allocate */
    table->gemm = libxstream_mkl_gemm_entry;
    table->syrk = libxstream_mkl_syrk_entry;
    table->dgmm = libxstream_mkl_dgmm_entry;
    result = 0;
  }
  return result;
}
