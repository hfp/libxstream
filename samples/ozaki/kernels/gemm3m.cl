/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* OpenCL kernels for complex GEMM via 3M (Karatsuba) method.
 * These kernels handle the pre/post-processing phases, keeping
 * all intermediate buffers on device to avoid unnecessary PCIe transfers.
 *
 * Compile-time parameters (-D):
 *   USE_DOUBLE - if 1, fp64 (double-complex); otherwise fp32 (single-complex)
 */
#include "../../../include/opencl/libxstream_common.h"

/**
 * Deinterleave complex column-major matrix into separate real and imaginary parts.
 * Input:  z[i,j] stored as z[2*(i + j*ldz)], z[2*(i + j*ldz) + 1]
 * Output: re[i + j*ld_out], im[i + j*ld_out]
 */
kernel void zgemm3m_deinterleave(
  global const real_t* restrict z, global real_t* restrict re, global real_t* restrict im, int rows, int cols, int ldz, int ld_out)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < rows && j < cols) {
    const size_t z_base = 2 * (i + (size_t)j * ldz);
    const size_t out_idx = i + (size_t)j * ld_out;
    re[out_idx] = z[z_base];
    im[out_idx] = z[z_base + 1];
  }
}


/**
 * Matrix addition: dst = a + b
 * All matrices are column-major real matrices.
 */
kernel void zgemm3m_matadd(global real_t* restrict dst, global const real_t* restrict a, global const real_t* restrict b, int rows,
  int cols, int ld_dst, int ld_a, int ld_b)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < rows && j < cols) {
    const size_t idx_dst = i + (size_t)j * ld_dst;
    const size_t idx_a = i + (size_t)j * ld_a;
    const size_t idx_b = i + (size_t)j * ld_b;
    dst[idx_dst] = a[idx_a] + b[idx_b];
  }
}


/**
 * Finalize: compute complex result from 3M products and write to interleaved output.
 *
 * For each element C[i,j]:
 *   C_new = alpha * (re_ab + i*im_ab) + beta * C_old
 * where:
 *   re_ab = Re(A*B) = P1 - P2
 *   im_ab = Im(A*B) = P3 - P1 - P2
 * and alpha = ar + i*ai, beta = br + i*bi
 *
 * Inputs:
 *   p1 = Re(A)*Re(B)
 *   p2 = Im(A)*Im(B)
 *   p3 = (Re(A)+Im(A)) * (Re(B)+Im(B))
 *   c  = original C (interleaved complex)
 * Output:
 *   c  = result (interleaved complex, in-place)
 */
kernel void zgemm3m_finalize(global real_t* restrict c, global const real_t* restrict p1, global const real_t* restrict p2,
  global const real_t* restrict p3, int M, int N, int ldc, int ld_prod, real_t ar, real_t ai, real_t br, real_t bi)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  if (i < M && j < N) {
    const size_t prod_idx = i + (size_t)j * ld_prod;
    const size_t c_base = 2 * (i + (size_t)j * ldc);

    /* Recover real and imaginary parts of A*B:
     *   Re(A*B) = P1 - P2
     *   Im(A*B) = P3 - P1 - P2 */
    const real_t v1 = p1[prod_idx];
    const real_t v2 = p2[prod_idx];
    const real_t v3 = p3[prod_idx];
    const real_t re_ab = v1 - v2;
    const real_t im_ab = v3 - v1 - v2;

    /* Compute C_new = alpha * (A*B) + beta * C_old
     * where alpha = ar + i*ai, beta = br + i*bi
     * Complex multiplication: (ar + i*ai) * (re_ab + i*im_ab)
     *   = ar*re_ab - ai*im_ab + i*(ar*im_ab + ai*re_ab) */
    const real_t alpha_ab_re = MAD(ar, re_ab, -ai * im_ab);
    const real_t alpha_ab_im = MAD(ar, im_ab, ai * re_ab);
    if (ZERO != br || ZERO != bi) {
      /* Read original C (beta != 0) */
      const real_t c_re = c[c_base];
      const real_t c_im = c[c_base + 1];
      c[c_base] = alpha_ab_re + MAD(br, c_re, -bi * c_im);
      c[c_base + 1] = alpha_ab_im + MAD(br, c_im, bi * c_re);
    }
    else { /* beta == 0: do not read C (may contain NaN/Inf) */
      c[c_base] = alpha_ab_re;
      c[c_base + 1] = alpha_ab_im;
    }
  }
}
