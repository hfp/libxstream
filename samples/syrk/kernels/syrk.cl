/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Panel of a symmetric update: T is given by the host (double or float). */
#if !defined(T)
# define T double
#endif
/* work-items per dimension of a work-group, and elements per work-item and dimension */
#if !defined(BS)
# define BS 16
#endif
#if !defined(RB)
# define RB 4
#endif
/* depth of a step through k */
#if !defined(BK)
# define BK 16
#endif
#define BT (BS * RB)


/* Element (i, p) of op(X): X is n x k, or k x n if transposed. */
#define SYRK_OP(X, LD, TRANS, I, P) ((0 == (TRANS)) ? (X)[(I) + (size_t)(P) * (LD)] : (X)[(P) + (size_t)(I) * (LD)])


/**
 * out (m x w, compact) := alpha * (op(A)[r0..] * D * op(B)[c0..]^T), plus op(B) * D * op(A)^T for
 * SYR2K (b != a). D is diag(d) if hasd is non-zero. The host keeps the triangle of the result.
 */
__attribute__((reqd_work_group_size(BS, BS, 1))) kernel void syrk_panel(global T* restrict out,
  global const T* restrict a, global const T* restrict b, global const T* restrict d, int m, int w, int r0, int c0,
  int k, int lda, int ldb, int trans, int twok, int hasd, T alpha)
{
  local T lx[BK][BT + 1], ly[BK][BT + 1];
  const int tx = (int)get_local_id(0), ty = (int)get_local_id(1);
  const int i0 = (int)get_group_id(0) * BT, j0 = (int)get_group_id(1) * BT;
  const int tid = ty * BS + tx;
  T acc[RB][RB];
  int term = 0, u, v;
  for (u = 0; u < RB; ++u) {
    for (v = 0; v < RB; ++v) acc[u][v] = 0;
  }
  for (; term <= twok; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? lda : ldb), ldy = (0 == term ? ldb : lda);
    int p0 = 0;
    for (; p0 < k; p0 += BK) {
      int e = tid;
      for (; e < BK * BT; e += BS * BS) { /* stage both operands, the weights on one side */
        const int pp = e / BT, q = e % BT, p = p0 + pp;
        const int gi = i0 + q, gj = j0 + q;
        T xv = 0, yv = 0;
        if (p < k) {
          if (gi < m) {
            xv = SYRK_OP(x, ldx, trans, r0 + gi, p);
            if (0 != hasd) xv *= d[p];
          }
          if (gj < w) yv = SYRK_OP(y, ldy, trans, c0 + gj, p);
        }
        lx[pp][q] = xv;
        ly[pp][q] = yv;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      { int pp = 0;
        for (; pp < BK; ++pp) {
          T xr[RB], yr[RB];
          for (u = 0; u < RB; ++u) xr[u] = lx[pp][tx + u * BS];
          for (v = 0; v < RB; ++v) yr[v] = ly[pp][ty + v * BS];
          for (u = 0; u < RB; ++u) {
            for (v = 0; v < RB; ++v) acc[u][v] = fma(xr[u], yr[v], acc[u][v]);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (v = 0; v < RB; ++v) {
    const int j = j0 + ty + v * BS;
    if (j < w) {
      for (u = 0; u < RB; ++u) {
        const int i = i0 + tx + u * BS;
        if (i < m) out[i + (size_t)j * m] = alpha * acc[u][v];
      }
    }
  }
}
