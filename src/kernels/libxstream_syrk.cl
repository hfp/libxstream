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
/**
 * Local memory: VT=0 has the strided register tile of a work-item (rows tx + u*BSX, a read per element),
 * VT=1 the contiguous one (rows tx*RX + u, a vector read per step). Staging loads SV elements per lane.
 */
#if !defined(VT)
# define VT 0
#endif
#if !defined(SV)
# define SV 1
#endif
#if (0 != VT)
# define PAD 0
# define XROW(TX, U) ((TX) * RX + (U))
# define YCOL(TY, V) ((TY) * RY + (V))
#else
# define PAD 1 /* rows of the strided tile apart by one bank */
# define XROW(TX, U) ((TX) + (U) * BSX)
# define YCOL(TY, V) ((TY) + (V) * BSY)
#endif
#if (0 != (BM % SV)) || (0 != (BN % SV))
# error "SV must divide the block of a work-group"
#endif
/* N contiguous elements at P into the private array A, as one vector access where N is a vector width */
#define SYRK_VCOPY(N, P, A) CAT(vstore, N)(CAT(vload, N)(0, P), 0, A)
#define SYRK_VWIDTH(N) (2 == (N) || 4 == (N) || 8 == (N) || 16 == (N))

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
 * lane that loaded it (RY <= SG columns), i.e., neither local memory nor a barrier. SYRK_SG=2 reads
 * BK steps of 16 rows (or columns) at once (Intel's 2D block reads): plain reads if trans='N' (single
 * precision), transposing reads of 32-bit words if trans='T' (SYRK_ES of 4 or 8 bytes). Double precision
 * with trans='N' has no such read (no 64-bit block reads).
 */
#if defined(SYRK_SG) && (0 != SYRK_SG)
# if !defined(BCST_SG)
#   error "SYRK_SG requires sub-groups of exactly SG lanes (SG_EXACT)"
# endif
# if (2 == SYRK_SG)
#   if !defined(cl_intel_subgroup_2d_block_io) || (16 != SG) || (16 != RY) || !defined(SYRK_ES)
#     error "SYRK_SG=2 requires cl_intel_subgroup_2d_block_io, SG=16, RY=16, and SYRK_ES"
#   endif
/* words (32-bit) per element, and element R of the words a lane received (W) */
#   if (8 == SYRK_ES)
#     define SYRK_EW 2
#     define SYRK_ELEM(W, R) as_double((uint2)((W)[2 * (R)], (W)[2 * (R) + 1]))
#   elif (4 == SYRK_ES)
#     define SYRK_EW 1
#     define SYRK_ELEM(W, R) as_float((W)[R])
#   else
#     error "SYRK_ES must be 4 or 8 (bytes)"
#   endif
#   if defined(SYRK_TRANS) && (0 != SYRK_TRANS)
#     if (0 != ((BK * SYRK_EW) % 8))
#       error "transposing 2D block reads take 8 words of a row at once (BK * SYRK_EW a multiple of 8)"
#     endif
/**
 * Rows y to y+15 (rows of op(X)) and words x to x+7 of the surface X (rows of LD elements apart,
 * HEIGHT_T rows), transposed and repeated until BK steps of k: lane l receives row y+l, as 32-bit words
 * (there are neither 64-bit block reads nor wider transposing ones). Rows beyond the panel only feed
 * elements that are never stored, and HEIGHT_T keeps a read within the operand.
 */
#     define SYRK_READ2D(X, LD, HEIGHT_N, HEIGHT_T, I, P, DST) \
        do { \
          int q_ = 0; \
          UNROLL_FORCE(BK * SYRK_EW / 8) \
          for (; q_ < (BK * SYRK_EW / 8); ++q_) { \
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void*)(X), (LD) * SYRK_ES, HEIGHT_T, \
              (LD) * SYRK_ES, (int2)((P) * SYRK_EW + 8 * q_, I), (DST) + 8 * q_); \
          } \
        } while (0)
#   elif (1 == SYRK_EW)
/**
 * Rows y to y+BK-1 (steps of k) and columns x to x+15 (rows of op(X)) of the surface X (columns of
 * LD elements apart, HEIGHT_N rows): lane l receives column x+l of every row, i.e., one row of op(X)
 * for BK steps. Rows and columns beyond the panel only feed elements that are never stored.
 */
#     define SYRK_READ2D(X, LD, HEIGHT_N, HEIGHT_T, I, P, DST) \
        CAT(CAT(intel_sub_group_2d_block_read_32b_, BK), r16x1c)( \
          (global void*)(X), (LD) * 4, HEIGHT_N, (LD) * 4, (int2)(I, P), DST)
#   else
#     error "SYRK_SG=2 with trans='N' requires single precision (SYRK_ES=4)"
#   endif
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
  UNROLL_FORCE(RX)
  for (u = 0; u < RX; ++u) {
    UNROLL_FORCE(RY)
    for (v = 0; v < RY; ++v) acc[u][v] = 0;
  }
  UNROLL_OUTER(2)
  for (; term <= TWOK_; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? LDA_ : LDB_), ldy = (0 == term ? LDB_ : LDA_);
    const int gj = j0 + lane;
    int p = kb;
# if (2 == SYRK_SG)
    UNROLL_OUTER(1)
    for (; p + BK <= ke; p += BK) { /* whole steps of BK by block reads, the rest below */
      uint yb[BK * SYRK_EW];
      int r;
      SYRK_READ2D(y, ldy, k, c0 + w, c0 + j0, p, yb);
      /* unrolled whatever LU says (fixed bounds): the reads take their destination by address */
      UNROLL_FORCE(RX)
      for (u = 0; u < RX; ++u) {
        uint xb[BK * SYRK_EW];
        SYRK_READ2D(x, ldx, k, r0 + m, r0 + i0 + u * SG, p, xb);
        UNROLL_FORCE(BK)
        for (r = 0; r < BK; ++r) {
          const T xu = (0 != HASD_ ? (SYRK_ELEM(xb, r) * d[p + r]) : SYRK_ELEM(xb, r));
          const T yr = SYRK_ELEM(yb, r);
          UNROLL_FORCE(RY)
          for (v = 0; v < RY; ++v) acc[u][v] = fma(xu, BCST_SG(yr, v), acc[u][v]);
        }
      }
    }
# endif
    UNROLL_AUTO
    for (; p < ke; ++p) {
      const T yl = ((lane < RY && gj < w) ? SYRK_OP(y, ldy, TRANS_, c0 + gj, p) : 0);
      const T dp = (0 != HASD_ ? d[p] : 1);
      UNROLL_FORCE(RX)
      for (u = 0; u < RX; ++u) { /* a row at a time: no array of rows, and the broadcast is an operand */
        const int gi = i0 + lane + u * SG;
        T xu = (gi < m ? SYRK_OP(x, ldx, TRANS_, r0 + gi, p) : 0);
        if (0 != HASD_) xu *= dp;
        UNROLL_FORCE(RY)
        for (v = 0; v < RY; ++v) acc[u][v] = fma(xu, BCST_SG(yl, v), acc[u][v]);
      }
    }
  }
  UNROLL_FORCE(RX)
  for (u = 0; u < RX; ++u) {
    const int i = i0 + lane + u * SG;
    if (i < m) {
      UNROLL_FORCE(RY)
      for (v = 0; v < RY; ++v) {
        const int j = j0 + v;
        if (j < w) dst[i + (size_t)j * m] = alpha * acc[u][v];
      }
    }
  }
}
#else
/**
 * Stages SV rows of op(x), from row on and nvalid of them valid, at step p (zero if p is not before ke)
 * into local memory at dst, scaled by the weight d[p] if hasd is non-zero.
 */
inline void syrk_stage_run(local T* restrict dst, global const T* restrict x, global const T* restrict d, int ldx, int trans,
  int hasd, int row, int nvalid, int p, int ke)
{
  T v[SV];
  int s;
  if (p < ke) {
#if (1 < SV) && SYRK_VWIDTH(SV)
    if (0 == trans && SV == nvalid) SYRK_VCOPY(SV, x + row + (size_t)p * ldx, v); /* contiguous rows */
    else
#endif
    {
      UNROLL_FORCE(SV)
      for (s = 0; s < SV; ++s) v[s] = (s < nvalid ? SYRK_OP(x, ldx, trans, row + s, p) : 0);
    }
    if (0 != hasd) {
      const T dp = d[p];
      UNROLL_FORCE(SV)
      for (s = 0; s < SV; ++s) v[s] *= dp;
    }
  }
  else {
    UNROLL_FORCE(SV)
    for (s = 0; s < SV; ++s) v[s] = 0;
  }
  UNROLL_FORCE(SV)
  for (s = 0; s < SV; ++s) dst[s] = v[s];
}


/**
 * Stages BK steps of k from p0 (but not beyond ke) into local memory: BM rows of op(x) from r0+i0,
 * scaled by the weights d if hasd is non-zero, and BN rows of op(y) from c0+j0.
 */
inline void syrk_stage(local T (*restrict lx)[BM + PAD], local T (*restrict ly)[BN + PAD], global const T* restrict x,
  global const T* restrict y, global const T* restrict d, int ldx, int ldy, int trans, int hasd, int r0, int c0, int i0,
  int j0, int m, int w, int p0, int ke, int tid)
{
  int e = tid;
  UNROLL(BK * BM / (NT * SV))
  for (; e < BK * BM / SV; e += NT) {
    const int pp = e / (BM / SV), q = (e % (BM / SV)) * SV, gi = i0 + q;
    syrk_stage_run(&lx[pp][q], x, d, ldx, trans, hasd, r0 + gi, gi < m ? min(SV, m - gi) : 0, p0 + pp, ke);
  }
  UNROLL(BK * BN / (NT * SV))
  for (e = tid; e < BK * BN / SV; e += NT) {
    const int pp = e / (BN / SV), q = (e % (BN / SV)) * SV, gj = j0 + q;
    syrk_stage_run(&ly[pp][q], y, d, ldy, trans, 0, c0 + gj, gj < w ? min(SV, w - gj) : 0, p0 + pp, ke);
  }
}


/* Accumulates BK staged steps into acc: the rows XROW(tx, u) and the columns YCOL(ty, v) of the block. */
inline void syrk_accumulate(private T (*restrict acc)[RY], local T (*restrict lx)[BM + PAD], local T (*restrict ly)[BN + PAD],
  int tx, int ty)
{
  int pp = 0;
  UNROLL(BK)
  for (; pp < BK; ++pp) {
    T xr[RX], yr[RY];
    int u, v;
#if (0 != VT) && SYRK_VWIDTH(RY)
    SYRK_VCOPY(RY, &ly[pp][YCOL(ty, 0)], yr); /* uniform within a sub-group along tx */
#else
    UNROLL_FORCE(RY)
    for (v = 0; v < RY; ++v) yr[v] = ly[pp][YCOL(ty, v)];
#endif
#if (0 != VT) && SYRK_VWIDTH(RX)
    SYRK_VCOPY(RX, &lx[pp][XROW(tx, 0)], xr);
#else
    UNROLL_FORCE(RX)
    for (u = 0; u < RX; ++u) xr[u] = lx[pp][XROW(tx, u)];
#endif
    UNROLL_FORCE(RX)
    for (u = 0; u < RX; ++u) {
      UNROLL_FORCE(RY)
      for (v = 0; v < RY; ++v) acc[u][v] = fma(xr[u], yr[v], acc[u][v]);
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
  local T lx[BK][BM + PAD], ly[BK][BN + PAD];
  const int tx = (int)get_local_id(0), ty = (int)get_local_id(1);
  const int i0 = (int)get_group_id(0) * BM, j0 = (int)get_group_id(1) * BN;
  const int tid = ty * BSX + tx, slice = (int)get_group_id(2);
  const int kb = slice * kslice, ke = min(k, kb + kslice);
  global T* const dst = (0 == slice ? out : (part + (size_t)(slice - 1) * m * w));
  T acc[RX][RY];
  int term = 0, u, v;
  if (0 != UPPER_ ? (r0 + i0 > c0 + j0 + BN - 1) : (r0 + i0 + BM - 1 < c0 + j0)) return; /* the whole work-group */
  UNROLL_FORCE(RX)
  for (u = 0; u < RX; ++u) {
    UNROLL_FORCE(RY)
    for (v = 0; v < RY; ++v) acc[u][v] = 0;
  }
  UNROLL_OUTER(2)
  for (; term <= TWOK_; ++term) {
    global const T* const x = (0 == term ? a : b);
    global const T* const y = (0 == term ? b : a);
    const int ldx = (0 == term ? LDA_ : LDB_), ldy = (0 == term ? LDB_ : LDA_);
    int p0 = kb;
    UNROLL_OUTER(1)
    for (; p0 < ke; p0 += BK) {
      syrk_stage(lx, ly, x, y, d, ldx, ldy, TRANS_, HASD_, r0, c0, i0, j0, m, w, p0, ke, tid);
      barrier(CLK_LOCAL_MEM_FENCE);
      syrk_accumulate(acc, lx, ly, tx, ty);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  UNROLL_FORCE(RX)
  for (u = 0; u < RX; ++u) {
    const int i = i0 + XROW(tx, u);
    if (i < m) {
      UNROLL_FORCE(RY)
      for (v = 0; v < RY; ++v) {
        const int j = j0 + YCOL(ty, v);
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
    UNROLL_AUTO
    for (; s < nparts; ++s) sum += part[e + (size_t)s * n];
    out[e] = sum;
  }
}
