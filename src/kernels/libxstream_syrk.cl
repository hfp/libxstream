/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <opencl/libxstream_common.h>

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
#define NT (BSX * BSY)

/* hard-coded by the host per plan (SYRK_*), else the kernel's arguments */
#if defined(SYRK_TRANS)
# define TRANS_ SYRK_TRANS
#else
# define TRANS_ trans
#endif
#if defined(SYRK_TWOK)
# define TWOK_ SYRK_TWOK
#else
# define TWOK_ twok
#endif
#if defined(SYRK_HASD)
# define HASD_ SYRK_HASD
#else
# define HASD_ hasd
#endif
#if defined(SYRK_UPPER)
# define UPPER_ SYRK_UPPER
#else
# define UPPER_ upper
#endif
#if defined(SYRK_LDA)
# define LDA_ SYRK_LDA
#else
# define LDA_ lda
#endif
#if defined(SYRK_LDB)
# define LDB_ SYRK_LDB
#else
# define LDB_ ldb
#endif

/* Element (i, p) of op(X): X is n x k, or k x n if transposed. */
#define SYRK_OP(X, LD, TRANS, I, P) ((0 == (TRANS)) ? (X)[(I) + (size_t)(P) * (LD)] : (X)[(P) + (size_t)(I) * (LD)])


/**
 * out (m x w, compact) := alpha * (op(A)[r0..] * D * op(B)[c0..]^T), plus op(B) * D * op(A)^T for
 * SYR2K (b != a). D is diag(d) if hasd is non-zero. The host keeps the triangle (upper or lower) of
 * the result, hence a block wholly outside of it is skipped. The third dimension of the grid splits
 * k into slices of kslice: slice 0 writes out, slice s > 0 writes part[s-1] (m x w), and syrk_sum
 * adds the parts. Code versions: through local memory in blocks of BM x BN per work-group, or
 * SYRK_SG (sub-groups of SG lanes): a lane holds RX rows, and a column's value is broadcast from the
 * lane that loaded it (RY <= SG columns), i.e., neither local memory nor a barrier.
 */
#if defined(SYRK_SG) && (0 != SYRK_SG)
# if !defined(BCST_SG)
#   error "SYRK_SG requires sub-groups of exactly SG lanes (SG_EXACT)"
# endif
REQD_SG kernel void syrk_panel(global T* restrict out, global const T* restrict a, global const T* restrict b,
  global const T* restrict d, int m, int w, int r0, int c0, int k, int lda, int ldb, int trans, int twok, int hasd, T alpha,
  int upper, global T* restrict part, int kslice)
{
  const int lane = (int)SGLID(), slice = (int)get_global_id(2);
  const int i0 = (int)(get_global_id(0) / SG) * (SG * RX), j0 = (int)get_global_id(1) * RY;
  const int kb = slice * kslice, ke = min(k, kb + kslice);
  global T* const dst = (0 == slice ? out : (part + (size_t)(slice - 1) * m * w));
  T acc[RX][RY];
  int term = 0, u, v;
  if (0 != UPPER_ ? (r0 + i0 > c0 + j0 + RY - 1) : (r0 + i0 + SG * RX - 1 < c0 + j0)) return; /* the whole sub-group */
  for (u = 0; u < RX; ++u) {
    for (v = 0; v < RY; ++v) acc[u][v] = 0;
  }
  for (; term <= TWOK_; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? LDA_ : LDB_), ldy = (0 == term ? LDB_ : LDA_);
    const int gj = j0 + lane;
    int p = kb;
    UNROLL(4)
    for (; p < ke; ++p) {
      const T yl = ((lane < RY && gj < w) ? SYRK_OP(y, ldy, TRANS_, c0 + gj, p) : 0);
      const T dp = (0 != HASD_ ? d[p] : 1);
      for (u = 0; u < RX; ++u) { /* a row at a time: no array of rows, and the broadcast is an operand */
        const int gi = i0 + lane + u * SG;
        T xu = (gi < m ? SYRK_OP(x, ldx, TRANS_, r0 + gi, p) : 0);
        if (0 != HASD_) xu *= dp;
        for (v = 0; v < RY; ++v) acc[u][v] = fma(xu, BCST_SG(yl, v), acc[u][v]);
      }
    }
  }
  for (u = 0; u < RX; ++u) {
    const int i = i0 + lane + u * SG;
    if (i < m) {
      for (v = 0; v < RY; ++v) {
        const int j = j0 + v;
        if (j < w) dst[i + (size_t)j * m] = alpha * acc[u][v];
      }
    }
  }
}
#else
/**
 * Stages BK steps of k from p0 (but not beyond ke) into local memory: BM rows of op(x) from r0+i0,
 * scaled by the weights d if hasd is non-zero, and BN rows of op(y) from c0+j0.
 */
inline void syrk_stage(local T (*restrict lx)[BM + 1], local T (*restrict ly)[BN + 1], global const T* restrict x,
  global const T* restrict y, global const T* restrict d, int ldx, int ldy, int trans, int hasd, int r0, int c0, int i0,
  int j0, int m, int w, int p0, int ke, int tid)
{
  int e = tid;
  UNROLL(BK * BM / NT)
  for (; e < BK * BM; e += NT) {
    const int pp = e / BM, q = e % BM, p = p0 + pp, gi = i0 + q;
    T xv = 0;
    if (p < ke && gi < m) {
      xv = SYRK_OP(x, ldx, trans, r0 + gi, p);
      if (0 != hasd) xv *= d[p];
    }
    lx[pp][q] = xv;
  }
  UNROLL(BK * BN / NT)
  for (e = tid; e < BK * BN; e += NT) {
    const int pp = e / BN, q = e % BN, p = p0 + pp, gj = j0 + q;
    ly[pp][q] = ((p < ke && gj < w) ? SYRK_OP(y, ldy, trans, c0 + gj, p) : 0);
  }
}


/* Accumulates BK staged steps into acc: the rows tx + u*BSX and the columns ty + v*BSY of the block. */
inline void syrk_accumulate(private T (*restrict acc)[RY], local T (*restrict lx)[BM + 1], local T (*restrict ly)[BN + 1],
  int tx, int ty)
{
  int pp = 0;
  UNROLL(BK)
  for (; pp < BK; ++pp) {
    T yr[RY];
    int u, v;
    UNROLL(RY)
    for (v = 0; v < RY; ++v) yr[v] = ly[pp][ty + v * BSY];
    UNROLL(RX)
    for (u = 0; u < RX; ++u) { /* a row at a time, fused with its products */
      const T xu = lx[pp][tx + u * BSX];
      UNROLL(RY)
      for (v = 0; v < RY; ++v) acc[u][v] = fma(xu, yr[v], acc[u][v]);
    }
  }
}


# if defined(SYRK_SIMD) && (0 < SYRK_SIMD) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SYRK_SIMD)))
# endif
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
  if (0 != UPPER_ ? (r0 + i0 > c0 + j0 + BN - 1) : (r0 + i0 + BM - 1 < c0 + j0)) return; /* the whole work-group */
  UNROLL(RX)
  for (u = 0; u < RX; ++u) {
    UNROLL(RY)
    for (v = 0; v < RY; ++v) acc[u][v] = 0;
  }
  for (; term <= TWOK_; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? LDA_ : LDB_), ldy = (0 == term ? LDB_ : LDA_);
    int p0 = kb;
    for (; p0 < ke; p0 += BK) {
      syrk_stage(lx, ly, x, y, d, ldx, ldy, TRANS_, HASD_, r0, c0, i0, j0, m, w, p0, ke, tid);
      barrier(CLK_LOCAL_MEM_FENCE);
      syrk_accumulate(acc, lx, ly, tx, ty);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  UNROLL(RX)
  for (u = 0; u < RX; ++u) {
    const int i = i0 + tx + u * BSX;
    if (i < m) {
      UNROLL(RY)
      for (v = 0; v < RY; ++v) {
        const int j = j0 + ty + v * BSY;
        if (j < w) dst[i + (size_t)j * m] = alpha * acc[u][v];
      }
    }
  }
}
#endif


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
