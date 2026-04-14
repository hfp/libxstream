/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* OpenCL kernels for complex GEMM via block embedding.
 * These kernels handle the pre/post-processing phases, keeping
 * all intermediate buffers on device to avoid unnecessary PCIe transfers.
 *
 * Block embedding constructs augmented real matrices:
 *   A_hat = [Ar, -Ai; Ai, Ar]  (2M x 2K)   B_hat = [Br; Bi]  (2K x N)
 * so that A_hat * B_hat = [Re(A*B); Im(A*B)]  (2M x N).
 *
 * A single real GEMM of size (2M) x N x (2K) produces both real and
 * imaginary parts with naturally shared Ozaki exponent bases.
 *
 * Compile-time parameters (-D):
 *   USE_DOUBLE - if 1, fp64 (double-complex); otherwise fp32 (single-complex)
 */
#include "../../../include/opencl/libxstream_common.h"

/**
 * Construct block-augmented A_hat from interleaved complex A.
 *
 * For transa=0 ('N'): A_hat = [Ar, -Ai; Ai, Ar]  (2*a_rows x 2*a_cols)
 * For transa=1 ('T'): A_hat = [Ar, Ai; -Ai, Ar]   (2*a_rows x 2*a_cols)
 *
 * Input:  z[i,j] stored as z[2*(i + j*ldz)], z[2*(i + j*ldz) + 1]
 * Output: a_hat[2*a_rows x 2*a_cols], column-major, ld = 2*a_rows
 */
kernel void zgemm_block_construct_a(
  global const real_t* restrict z, global real_t* restrict a_hat,
  int a_rows, int a_cols, int ldz, int transa)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < a_rows && j < a_cols) {
    const int lda_hat = 2 * a_rows;
    const size_t z_base = 2 * (i + (size_t)j * ldz);
    const real_t re = z[z_base];
    const real_t im = z[z_base + 1];

    /* Left half: column j */
    const size_t left = i + (size_t)j * lda_hat;
    a_hat[left] = re;                                       /* Q1: Ar */
    a_hat[left + a_rows] = transa ? -im : im;               /* Q3 */

    /* Right half: column a_cols + j */
    const size_t right = i + (size_t)(a_cols + j) * lda_hat;
    a_hat[right] = transa ? im : -im;                       /* Q2 */
    a_hat[right + a_rows] = re;                             /* Q4: Ar */
  }
}


/**
 * Construct block-augmented B_hat from interleaved complex B (transb=0, 'N').
 *
 * B_hat = [Br; Bi]  (2*b_rows x b_cols), stacked vertically.
 *
 * Input:  z[k,j] stored as z[2*(k + j*ldz)], z[2*(k + j*ldz) + 1]
 * Output: b_hat[2*b_rows x b_cols], column-major, ld = 2*b_rows
 */
kernel void zgemm_block_construct_b_n(
  global const real_t* restrict z, global real_t* restrict b_hat,
  int b_rows, int b_cols, int ldz, int conj)
{
  const int k = get_global_id(0);
  const int j = get_global_id(1);

  if (k < b_rows && j < b_cols) {
    const int ldb_hat = 2 * b_rows;
    const size_t z_base = 2 * (k + (size_t)j * ldz);
    const size_t out_base = k + (size_t)j * ldb_hat;
    const real_t im = z[z_base + 1];
    b_hat[out_base] = z[z_base];                       /* top: Br */
    b_hat[out_base + b_rows] = conj ? -im : im;        /* bottom: Bi (negated for 'C') */
  }
}


/**
 * Construct block-augmented B_hat from interleaved complex B (transb=1, 'T').
 *
 * B_hat = [Br, Bi]  (b_rows x 2*b_cols), placed side by side.
 *
 * Input:  z[j,k] stored as z[2*(j + k*ldz)], z[2*(j + k*ldz) + 1]
 * Output: b_hat[b_rows x 2*b_cols], column-major, ld = b_rows
 */
kernel void zgemm_block_construct_b_t(
  global const real_t* restrict z, global real_t* restrict b_hat,
  int b_rows, int b_cols, int ldz, int conj)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < b_rows && j < b_cols) {
    const size_t z_base = 2 * (i + (size_t)j * ldz);
    const size_t left = i + (size_t)j * b_rows;
    const size_t right = i + (size_t)(b_cols + j) * b_rows;
    const real_t im = z[z_base + 1];
    b_hat[left] = z[z_base];              /* Br (left half) */
    b_hat[right] = conj ? -im : im;       /* Bi (right half, negated for 'C') */
  }
}


/**
 * Finalize: extract Re/Im from block result C_hat (2M x N),
 * apply complex alpha/beta, write to interleaved output C.
 *
 * C_hat layout (2M x N, column-major, ld = 2*M):
 *   Rows 0..M-1   = Re(A*B)
 *   Rows M..2M-1  = Im(A*B)
 */
kernel void zgemm_block_finalize(global real_t* restrict c,
  global const real_t* restrict c_hat,
  int M, int N, int ldc, real_t ar, real_t ai, real_t br, real_t bi)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < M && j < N) {
    const int ldc_hat = 2 * M;
    const size_t hat_base = i + (size_t)j * ldc_hat;
    const size_t c_base = 2 * (i + (size_t)j * ldc);
    const real_t re_ab = c_hat[hat_base];
    const real_t im_ab = c_hat[hat_base + M];

    const real_t alpha_ab_re = MAD(ar, re_ab, -ai * im_ab);
    const real_t alpha_ab_im = MAD(ar, im_ab, ai * re_ab);
    if (ZERO != br || ZERO != bi) {
      const real_t c_re = c[c_base];
      const real_t c_im = c[c_base + 1];
      c[c_base] = alpha_ab_re + MAD(br, c_re, -bi * c_im);
      c[c_base + 1] = alpha_ab_im + MAD(br, c_im, bi * c_re);
    }
    else {
      c[c_base] = alpha_ab_re;
      c[c_base + 1] = alpha_ab_im;
    }
  }
}
