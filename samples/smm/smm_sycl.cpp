/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include "smm_sycl.h"
#include <sycl/sycl.hpp>
#include <sycl/backend/opencl.hpp>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>


/**
 * oneMKL's SYCL interface declared here rather than included: its headers and a
 * SYCL compiler are not required, the host compiler and the runtime libraries are.
 * A declaration that no longer matches libmkl_sycl_blas links and computes wrong
 * results (transpose is a char enumeration, which the mangled name does not say),
 * hence the self-test before first use. The group interface takes alpha and beta
 * as arrays, which keeps oneMKL's value_or_pointer out of the declaration.
 */
namespace oneapi {
namespace mkl {
enum class transpose : char { nontrans = 0, trans = 1, conjtrans = 3 };
namespace blas {
namespace column_major {
sycl::event gemm_batch(sycl::queue& queue, const transpose* transa, const transpose* transb, const std::int64_t* m,
  const std::int64_t* n, const std::int64_t* k, const double* alpha, const double** a, const std::int64_t* lda, const double** b,
  const std::int64_t* ldb, const double* beta, double** c, const std::int64_t* ldc, std::int64_t group_count,
  const std::int64_t* group_size, const std::vector<sycl::event>& dependencies);
} /* namespace column_major */
} /* namespace blas */
} /* namespace mkl */
} /* namespace oneapi */


namespace {

/**
 * The SYCL context made of LIBXSTREAM's OpenCL context, and whether oneMKL passed
 * its self-test. An interop object neither retains nor releases its OpenCL handle,
 * hence a queue is made per call: a cached one would outlive a stream destroyed.
 */
struct smm_sycl_state_t {
  std::mutex lock;
  cl_context clctx = nullptr;
  sycl::context* context = nullptr;
  int status = 0; /* 0: untested, 1: usable, -1: disabled */
};

/* the batch's matrix pointers, which the device reads, hence USM until the work is done */
struct smm_sycl_pointers_t {
  sycl::queue& queue;
  const double** ab;
  double** c;
  smm_sycl_pointers_t(sycl::queue& q, int n)
    : queue(q), ab(sycl::malloc_shared<const double*>(2 * static_cast<size_t>(n), q)),
      c(sycl::malloc_shared<double*>(static_cast<size_t>(n), q))
  {
    if (nullptr == ab || nullptr == c) throw std::bad_alloc();
  }
  ~smm_sycl_pointers_t()
  {
    if (nullptr != ab) sycl::free(ab, queue);
    if (nullptr != c) sycl::free(c, queue);
  }
  smm_sycl_pointers_t(const smm_sycl_pointers_t&) = delete;
  smm_sycl_pointers_t& operator=(const smm_sycl_pointers_t&) = delete;
};


/* never destroyed at exit: the SYCL runtime may be gone by then (see opencl_libsmm_sycl_finalize) */
smm_sycl_state_t& smm_sycl_state()
{
  static smm_sycl_state_t* const state = new smm_sycl_state_t;
  return *state;
}


void smm_sycl_reset(smm_sycl_state_t& state)
{
  delete state.context;
  state.context = nullptr;
  state.clctx = nullptr;
  state.status = 0;
}


/**
 * The stack in waves: the j-th occurrence of a C block goes into wave j, hence no
 * wave updates a block twice, which a batch running its entries concurrently would
 * race on. The waves follow each other, and the call returns when all are done.
 */
void smm_sycl_gemm(sycl::queue& queue, const int* stack, int stack_size, int m, int n, int k, int transb, const double* a,
  const double* b, double* c)
{
  const oneapi::mkl::transpose ta = oneapi::mkl::transpose::nontrans;
  const oneapi::mkl::transpose tb = (0 != transb ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans);
  const std::int64_t mm = m, nn = n, kk = k, lda = m, ldb = (0 != transb ? n : k), ldc = m;
  const double alpha = 1.0, beta = 1.0;
  std::unordered_map<int, int> seen;
  std::vector<int> wave(static_cast<size_t>(stack_size)), count, start;
  std::vector<sycl::event> dependencies;
  smm_sycl_pointers_t pointers(queue, stack_size);
  int i;
  for (i = 0; i < stack_size; ++i) {
    int& occurrence = seen[stack[3 * i + 2]];
    wave[i] = occurrence++;
    if (static_cast<int>(count.size()) < occurrence) count.push_back(0);
    ++count[wave[i]];
  }
  start.assign(count.size() + 1, 0);
  for (i = 0; i < static_cast<int>(count.size()); ++i) start[i + 1] = start[i] + count[i];
  {
    std::vector<int> next(start.begin(), start.end() - 1);
    for (i = 0; i < stack_size; ++i) {
      const int j = next[wave[i]]++;
      pointers.ab[j] = a + (stack[3 * i] - 1);
      pointers.ab[stack_size + j] = b + (stack[3 * i + 1] - 1);
      pointers.c[j] = c + (stack[3 * i + 2] - 1);
    }
  }
  for (i = 0; i < static_cast<int>(count.size()); ++i) {
    const std::int64_t size = count[i];
    const sycl::event event = oneapi::mkl::blas::column_major::gemm_batch(queue, &ta, &tb, &mm, &nn, &kk, &alpha,
      pointers.ab + start[i], &lda, pointers.ab + stack_size + start[i], &ldb, &beta, pointers.c + start[i], &ldc,
      1 /*group_count*/, &size, dependencies);
    dependencies.assign(1, event);
  }
  if (!dependencies.empty()) dependencies.front().wait_and_throw();
}


/* two entries on the same C block, op(B) either way, against the host: small integers make it exact */
int smm_sycl_selftest(sycl::queue& queue)
{
  const int m = 3, n = 2, k = 4, stack[] = {1, 1, 1, 1 + m * k, 1 + k * n, 1};
  const int na = 2 * m * k, nb = 2 * k * n, nc = m * n;
  std::vector<double> ha(na), hb(nb), hc(nc), ref(nc);
  int result = 0, transb, i;
  double* const da = sycl::malloc_device<double>(na + nb + nc, queue);
  if (nullptr == da) throw std::bad_alloc();
  for (i = 0; i < na; ++i) ha[i] = static_cast<double>(i % 5 - 2);
  for (i = 0; i < nb; ++i) hb[i] = static_cast<double>(i % 3 - 1);
  for (transb = 0; transb <= 1 && 0 == result; ++transb) {
    int s, x, y, l;
    for (i = 0; i < nc; ++i) ref[i] = hc[i] = static_cast<double>(i + 1);
    for (s = 0; s < 2; ++s) {
      for (y = 0; y < n; ++y) {
        for (x = 0; x < m; ++x) {
          for (l = 0; l < k; ++l) {
            const double bv = (0 != transb ? hb[s * k * n + l * n + y] : hb[s * k * n + y * k + l]);
            ref[y * m + x] += ha[s * m * k + l * m + x] * bv;
          }
        }
      }
    }
    queue.memcpy(da, ha.data(), sizeof(double) * na);
    queue.memcpy(da + na, hb.data(), sizeof(double) * nb);
    queue.memcpy(da + na + nb, hc.data(), sizeof(double) * nc).wait_and_throw();
    smm_sycl_gemm(queue, stack, 2, m, n, k, transb, da, da + na, da + na + nb);
    queue.memcpy(hc.data(), da + na + nb, sizeof(double) * nc).wait_and_throw();
    for (i = 0; i < nc && 0 == result; ++i) {
      if (ref[i] != hc[i]) result = -1;
    }
  }
  sycl::free(da, queue);
  return result;
}

} /* namespace */


extern "C" int opencl_libsmm_sycl_dgemm(void* cl_context_handle, void* cl_command_queue_handle, const int* stack, int stack_size,
  int m, int n, int k, int transb, const double* a, const double* b, double* c)
{
  int result = -1;
  smm_sycl_state_t& state = smm_sycl_state();
  try {
    const cl_context clctx = static_cast<cl_context>(cl_context_handle);
    const cl_command_queue clqueue = static_cast<cl_command_queue>(cl_command_queue_handle);
    std::optional<sycl::queue> queue;
    int status;
    {
      const std::lock_guard<std::mutex> guard(state.lock);
      if (clctx != state.clctx) {
        smm_sycl_reset(state);
        state.context = new sycl::context(sycl::make_context<sycl::backend::opencl>(clctx));
        state.clctx = clctx;
      }
      queue.emplace(sycl::make_queue<sycl::backend::opencl>(clqueue, *state.context));
      if (0 == state.status) {
        state.status = (0 == smm_sycl_selftest(*queue) ? 1 : -1);
        if (0 > state.status) {
          std::fprintf(stderr, "WARNING ACC/LIBSMM: oneMKL (SYCL) failed its self-test, large stacks stay on the host\n");
        }
      }
      status = state.status;
    }
    if (0 < status && 0 < stack_size) {
      smm_sycl_gemm(*queue, stack, stack_size, m, n, k, transb, a, b, c);
      result = 0;
    }
  }
  catch (const std::exception& e) {
    /* C may be partially updated if oneMKL failed only after it began */
    std::fprintf(stderr, "ERROR ACC/LIBSMM: oneMKL (SYCL) %s\n", e.what());
  }
  catch (...) {
    std::fprintf(stderr, "ERROR ACC/LIBSMM: oneMKL (SYCL) failed\n");
  }
  return result;
}


extern "C" void opencl_libsmm_sycl_finalize(void)
{
  smm_sycl_state_t& state = smm_sycl_state();
  const std::lock_guard<std::mutex> guard(state.lock);
  try {
    smm_sycl_reset(state);
  }
  catch (...) {
    std::fprintf(stderr, "ERROR ACC/LIBSMM: oneMKL (SYCL) failed to release its context\n");
  }
}
