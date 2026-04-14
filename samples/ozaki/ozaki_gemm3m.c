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
 * GPU-native complex GEMM via block embedding.
 * All intermediate buffers remain on device to minimize PCIe transfers.
 *
 * Algorithm:
 *   1. H2D: upload interleaved complex A, B, C
 *   2. GPU: construct A_hat (2*a_rows x 2*a_cols) from interleaved A
 *   3. GPU: construct B_hat from interleaved B
 *   4. GPU: ozaki_gemm(A_hat, B_hat) -> C_hat (2M x N)
 *   5. GPU: finalize(C_hat, alpha, beta) -> C (interleaved)
 *   6. D2H: download result C
 *
 * The block structure guarantees that Re and Im contributions from the
 * same complex row/column share a common Ozaki exponent base, eliminating
 * the catastrophic cancellation that plagued the 3M (Karatsuba) method.
 */
int ozaki_gemm3m(ozaki_context_t* ctx, libxstream_stream_t* stream, char transa, char transb, int M, int N, int K,
  const double* alpha, const void* a, int lda, const void* b, int ldb, const double* beta, void* c, int ldc)
{
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  const int ca = ('C' == transa || 'c' == transa);
  const int cb = ('C' == transb || 'c' == transb);
  const int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  const int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  const int ta_sign = (ta && !ca); /* 'C' uses 'N' sign pattern */
  const int a_rows = ta ? K : M;
  const int a_cols = ta ? M : K;
  const int b_rows = tb ? N : K;
  const int b_cols = tb ? K : N;
  const double ar_d = alpha[0], ai_d = alpha[1];
  const double br_d = beta[0], bi_d = beta[1];
  const libxstream_opencl_stream_t* str = stream;
  int result = EXIT_SUCCESS;
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
  void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
  void *d_a_hat = NULL, *d_b_hat = NULL, *d_c_hat = NULL;
  size_t sz_a_complex, sz_b_complex, sz_c_complex;
  size_t sz_a_hat, sz_b_hat, sz_c_hat;
  int m_hat, k_hat, lda_hat, ldb_hat, ldc_hat;

  /* Check if block-embedding kernels are available */
  if (NULL == ctx->kern_zgemm_block_construct_a || NULL == ctx->kern_zgemm_block_finalize ||
      NULL == ctx->kern_zgemm_block_construct_b_n || NULL == ctx->kern_zgemm_block_construct_b_t)
  {
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) {
    ctx->stream = stream; /* expose to deallocate wrapper */
  }

  /* Buffer sizes */
  sz_a_complex = (size_t)lda * (size_t)a_cols * 2 * elem_size;
  sz_b_complex = (size_t)ldb * (size_t)b_cols * 2 * elem_size;
  sz_c_complex = (size_t)ldc * (size_t)N * 2 * elem_size;

  /* Augmented real matrix sizes */
  m_hat = 2 * M;
  k_hat = 2 * K;
  lda_hat = 2 * a_rows;
  ldb_hat = tb ? b_rows : 2 * b_rows;
  ldc_hat = 2 * M;
  sz_a_hat = (size_t)(2 * a_rows) * (size_t)(2 * a_cols) * elem_size;
  sz_b_hat = (tb ? (size_t)b_rows * (size_t)(2 * b_cols) : (size_t)(2 * b_rows) * (size_t)b_cols) * elem_size;
  sz_c_hat = (size_t)(2 * M) * (size_t)N * elem_size;

  /* Allocate device memory */
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_ag, sz_a_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_bg, sz_b_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, sz_c_complex);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_a_hat, sz_a_hat);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_b_hat, sz_b_hat);
  if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_c_hat, sz_c_hat);

  /* H2D: upload interleaved complex A, B.
   * Skip C when beta == 0: finalize kernel will not read C_old. */
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_h2d(a, d_ag, sz_a_complex, stream);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_mem_copy_h2d(b, d_bg, sz_b_complex, stream);
  }
  if (EXIT_SUCCESS == result && (0.0 != br_d || 0.0 != bi_d)) {
    result = libxstream_mem_copy_h2d(c, d_cg, sz_c_complex, stream);
  }

  /* Phase 1: Construct A_hat from interleaved A */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm_block_construct_a;
    cl_int iarg = 0;
    global[0] = (size_t)a_rows;
    global[1] = (size_t)a_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_ag));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_a_hat));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &a_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &lda));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ta_sign));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 1: Construct B_hat from interleaved B */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = tb ? ctx->kern_zgemm_block_construct_b_t : ctx->kern_zgemm_block_construct_b_n;
    cl_int iarg = 0;
    global[0] = (size_t)b_rows;
    global[1] = (size_t)b_cols;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_bg));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_b_hat));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_rows));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &b_cols));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ldb));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &cb));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2, NULL, global, NULL, 0, NULL, NULL));
  }

  /* Phase 2: Single real GEMM via Ozaki: C_hat = op(A_hat) * op(B_hat) */
  if (EXIT_SUCCESS == result) {
    const double one = 1.0, zero = 0.0;
    result = ozaki_gemm(ctx, stream, transa, transb, m_hat, N, k_hat, one,
      d_a_hat, lda_hat, d_b_hat, ldb_hat, zero, d_c_hat, ldc_hat, NULL, 0, 1);
  }

  /* Phase 3: Finalize - extract Re/Im from C_hat, apply alpha/beta */
  if (EXIT_SUCCESS == result) {
    size_t global[2];
    cl_kernel kern = ctx->kern_zgemm_block_finalize;
    cl_int iarg = 0;
    global[0] = (size_t)M;
    global[1] = (size_t)N;

    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_cg));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, iarg++, d_c_hat));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &N));
    CL_CHECK(result, clSetKernelArg(kern, iarg++, sizeof(int), &ldc));
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
  OZAKI_DEV_FREE(d_a_hat);
  OZAKI_DEV_FREE(d_b_hat);
  OZAKI_DEV_FREE(d_c_hat);

  return result;
}
