/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"


/**
 * GPU-native complex GEMM via 3M (Karatsuba) method.
 * All intermediate buffers remain on device to minimize PCIe transfers.
 *
 * Algorithm:
 *   1. H2D: upload interleaved complex A, B, C
 *   2. GPU: deinterleave A → Ar, Ai; B → Br, Bi
 *   3. GPU: matadd Ta = Ar + Ai, Tb = Br + Bi
 *   4. GPU: ozaki_gemm(Ar, Br) → P1
 *   5. GPU: ozaki_gemm(Ai, Bi) → P2
 *   6. GPU: ozaki_gemm(Ta, Tb) → P3
 *   7. GPU: finalize(P1, P2, P3, alpha, beta) → C (interleaved)
 *   8. D2H: download result C
 *
 * This reduces 6 PCIe transfers (3 in + 3 out) to 2 (1 in + 1 out).
 */
int ozaki_gemm3m(ozaki_context_t* ctx, libxstream_stream_t* stream, char transa, char transb, int M, int N, int K,
  const double* alpha, const void* a, int lda, const void* b, int ldb, const double* beta, void* c, int ldc)
{
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  const int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  const int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  const int a_rows = ta ? K : M;
  const int a_cols = ta ? M : K;
  const int b_rows = tb ? N : K;
  const int b_cols = tb ? K : N;
  const double ar_d = alpha[0], ai_d = alpha[1];
  const double br_d = beta[0], bi_d = beta[1];
  const libxstream_opencl_stream_t* str = stream;
  int result = EXIT_SUCCESS;
#if defined(OZAKI_DEVPOOL)
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
#endif
  void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
  void *d_ar = NULL, *d_ai = NULL, *d_br = NULL, *d_bi = NULL;
  void *d_ta = NULL, *d_tb = NULL;
  void *d_p1 = NULL, *d_p2 = NULL, *d_p3 = NULL;
  size_t sz_a_complex, sz_b_complex, sz_c_complex;
  size_t sz_a_real, sz_b_real, sz_c_real;

  /* Check if 3M kernels are available */
  if (NULL == ctx->kern_zgemm3m_deinterleave || NULL == ctx->kern_zgemm3m_matadd || NULL == ctx->kern_zgemm3m_finalize) {
    return EXIT_FAILURE;
  }

#if defined(OZAKI_DEVPOOL)
  ctx->stream = stream; /* expose to deallocate wrapper */
#endif

  /* Buffer sizes */
  sz_a_complex = (size_t)lda * (size_t)a_cols * 2 * elem_size;
  sz_b_complex = (size_t)ldb * (size_t)b_cols * 2 * elem_size;
  sz_c_complex = (size_t)ldc * (size_t)N * 2 * elem_size;
  sz_a_real = (size_t)a_rows * (size_t)a_cols * elem_size;
  sz_b_real = (size_t)b_rows * (size_t)b_cols * elem_size;
  sz_c_real = (size_t)M * (size_t)N * elem_size;

  /* Allocate device memory */
  result = OZAKI_DEV_ALLOC(&d_ag, sz_a_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bg, sz_b_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, sz_c_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_ar, sz_a_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_ai, sz_a_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_br, sz_b_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bi, sz_b_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_ta, sz_a_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_tb, sz_b_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_p1, sz_c_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_p2, sz_c_real);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_p3, sz_c_real);

  /* H2D: upload interleaved complex A, B, C */
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_h2d(a, d_ag, sz_a_complex, stream);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_h2d(b, d_bg, sz_b_complex, stream);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_h2d(c, d_cg, sz_c_complex, stream);
  }

  /* Phase 1: Deinterleave A → Ar, Ai */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm3m_deinterleave;
    cl_int iarg = 0;
    global[0] = (size_t)a_rows;
    global[1] = (size_t)a_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ag));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ar));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ai));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &lda));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 1: Deinterleave B → Br, Bi */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm3m_deinterleave;
    cl_int iarg = 0;
    global[0] = (size_t)b_rows;
    global[1] = (size_t)b_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_bg));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_br));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_bi));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ldb));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 2: Matrix add Ta = Ar + Ai */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm3m_matadd;
    cl_int iarg = 0;
    global[0] = (size_t)a_rows;
    global[1] = (size_t)a_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ta));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ar));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ai));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 2: Matrix add Tb = Br + Bi */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm3m_matadd;
    cl_int iarg = 0;
    global[0] = (size_t)b_rows;
    global[1] = (size_t)b_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_tb));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_br));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_bi));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 3: Three real GEMM calls (via ozaki_gemm)
   * P1 = Ar * Br, P2 = Ai * Bi, P3 = Ta * Tb
   * All use alpha=1, beta=0 (accumulation done in finalize) */
  if (EXIT_SUCCESS == result) {
    const double one = 1.0, zero = 0.0;
    const int ld_real = ta ? K : M; /* leading dimension of real A matrices */
    const int ldb_real = tb ? N : K; /* leading dimension of real B matrices */
    result = ozaki_gemm(ctx, stream, transa, transb, M, N, K, one, d_ar, ld_real, d_br, ldb_real, zero, d_p1, M, NULL, 0);
  }
  if (EXIT_SUCCESS == result) {
    const double one = 1.0, zero = 0.0;
    const int ld_real = ta ? K : M;
    const int ldb_real = tb ? N : K;
    result = ozaki_gemm(ctx, stream, transa, transb, M, N, K, one, d_ai, ld_real, d_bi, ldb_real, zero, d_p2, M, NULL, 0);
  }
  if (EXIT_SUCCESS == result) {
    const double one = 1.0, zero = 0.0;
    const int ld_real = ta ? K : M;
    const int ldb_real = tb ? N : K;
    result = ozaki_gemm(ctx, stream, transa, transb, M, N, K, one, d_ta, ld_real, d_tb, ldb_real, zero, d_p3, M, NULL, 0);
  }

  /* Phase 4: Finalize - compute complex result and apply alpha/beta */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm3m_finalize;
    const int ld_prod = M;
    cl_int iarg = 0;
    global[0] = (size_t)M;
    global[1] = (size_t)N;

    /* Set kernel arguments (precision-dependent alpha/beta) */
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_cg));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_p1));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_p2));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_p3));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &N));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ldc));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ld_prod));
    if (ctx->use_double) {
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(double), &ar_d));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(double), &ai_d));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(double), &br_d));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(double), &bi_d));
    }
    else {
      const float ar_f = (float)ar_d, ai_f = (float)ai_d;
      const float br_f = (float)br_d, bi_f = (float)bi_d;
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(float), &ar_f));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(float), &ai_f));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(float), &br_f));
      CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(float), &bi_f));
    }
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* D2H: download result C */
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_d2h(d_cg, c, sz_c_complex, stream);
  }

  /* Sync stream before freeing device buffers to ensure finalize kernel
   * and D2H transfer have completed. Without this, freed buffers can be
   * recycled by the pool while DMA is still reading them. */
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);

  /* Cleanup device buffers */
  OZAKI_DEV_FREE(d_ag);
  OZAKI_DEV_FREE(d_bg);
  OZAKI_DEV_FREE(d_cg);
  OZAKI_DEV_FREE(d_ar);
  OZAKI_DEV_FREE(d_ai);
  OZAKI_DEV_FREE(d_br);
  OZAKI_DEV_FREE(d_bi);
  OZAKI_DEV_FREE(d_ta);
  OZAKI_DEV_FREE(d_tb);
  OZAKI_DEV_FREE(d_p1);
  OZAKI_DEV_FREE(d_p2);
  OZAKI_DEV_FREE(d_p3);

  return result;
}
