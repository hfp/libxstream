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
/* work-items of a work-group along the rows (i) and columns (j), and elements per work-item */
#if !defined(BSX)
# define BSX 16
#endif
#if !defined(BSY)
# define BSY 16
#endif
#if !defined(RX)
# define RX 4
#endif
#if !defined(RY)
# define RY 4
#endif
/* depth of a step through k */
#if !defined(BK)
# define BK 16
#endif
/* block of a work-group: BM rows and BN columns */
#define BM (BSX * RX)
#define BN (BSY * RY)


/* Element (i, p) of op(X): X is n x k, or k x n if transposed. */
#define SYRK_OP(X, LD, TRANS, I, P) ((0 == (TRANS)) ? (X)[(I) + (size_t)(P) * (LD)] : (X)[(P) + (size_t)(I) * (LD)])


/**
 * out (m x w, compact) := alpha * (op(A)[r0..] * D * op(B)[c0..]^T), plus op(B) * D * op(A)^T for
 * SYR2K (b != a). D is diag(d) if hasd is non-zero. The host keeps the triangle (upper or lower) of
 * the result, hence a block of work-groups wholly outside of it is skipped. The third dimension of
 * the grid splits k into slices of kslice: slice 0 writes out, slice s > 0 writes part[s-1] (m x w),
 * and syrk_sum adds the parts.
 */
__attribute__((reqd_work_group_size(BSX, BSY, 1))) kernel void syrk_panel(global T* restrict out,
  global const T* restrict a, global const T* restrict b, global const T* restrict d, int m, int w, int r0, int c0,
  int k, int lda, int ldb, int trans, int twok, int hasd, T alpha, int upper, global T* restrict part, int kslice)
{
  local T lx[BK][BM + 1], ly[BK][BN + 1];
  const int tx = (int)get_local_id(0), ty = (int)get_local_id(1);
  const int i0 = (int)get_group_id(0) * BM, j0 = (int)get_group_id(1) * BN;
  const int tid = ty * BSX + tx, slice = (int)get_group_id(2);
  const int kb = slice * kslice, ke = min(k, kb + kslice);
  global T* const dst = (0 == slice ? out : (part + (size_t)(slice - 1) * m * w));
  T acc[RX][RY];
  int term = 0, u, v;
  if (0 != upper ? (r0 + i0 > c0 + j0 + BN - 1) : (r0 + i0 + BM - 1 < c0 + j0)) return; /* the whole work-group */
  for (u = 0; u < RX; ++u) {
    for (v = 0; v < RY; ++v) acc[u][v] = 0;
  }
  for (; term <= twok; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? lda : ldb), ldy = (0 == term ? ldb : lda);
    int p0 = kb;
    for (; p0 < ke; p0 += BK) {
      int e = tid;
      for (; e < BK * BM; e += BSX * BSY) { /* stage both operands, the weights on one side */
        const int pp = e / BM, q = e % BM, p = p0 + pp, gi = i0 + q;
        T xv = 0;
        if (p < ke && gi < m) {
          xv = SYRK_OP(x, ldx, trans, r0 + gi, p);
          if (0 != hasd) xv *= d[p];
        }
        lx[pp][q] = xv;
      }
      for (e = tid; e < BK * BN; e += BSX * BSY) {
        const int pp = e / BN, q = e % BN, p = p0 + pp, gj = j0 + q;
        ly[pp][q] = ((p < ke && gj < w) ? SYRK_OP(y, ldy, trans, c0 + gj, p) : 0);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      { int pp = 0;
        for (; pp < BK; ++pp) {
          T xr[RX], yr[RY];
          for (u = 0; u < RX; ++u) xr[u] = lx[pp][tx + u * BSX];
          for (v = 0; v < RY; ++v) yr[v] = ly[pp][ty + v * BSY];
          for (u = 0; u < RX; ++u) {
            for (v = 0; v < RY; ++v) acc[u][v] = fma(xr[u], yr[v], acc[u][v]);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (v = 0; v < RY; ++v) {
    const int j = j0 + ty + v * BSY;
    if (j < w) {
      for (u = 0; u < RX; ++u) {
        const int i = i0 + tx + u * BSX;
        if (i < m) dst[i + (size_t)j * m] = alpha * acc[u][v];
      }
    }
  }
}


/* out[e] += part[s][e] for s = 0, 1, ... in this order (the result does not depend on a schedule). */
kernel void syrk_sum(global T* restrict out, global const T* restrict part, int n, int nparts)
{
  const size_t e = get_global_id(0);
  if (e < (size_t)n) {
    T sum = out[e];
    int s = 0;
    for (; s < nparts; ++s) sum += part[e + (size_t)s * n];
    out[e] = sum;
  }
}
