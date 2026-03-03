/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../include/opencl/libxstream_common.h"

/* Ozaki Scheme 1: mantissa-slicing low-precision GEMM via OpenCL.
 *
 * Three kernels:
 *   preprocess_a  - decompose rows of A into int8 slices
 *   preprocess_b  - decompose cols of B into int8 slices
 *   dotprod       - int8 dot products over slice pairs, accumulate into C
 *
 * Compile-time parameters (supplied via -D):
 *   BM, BN, BK     - block dimensions (default 16)
 *   WG              - work-group size hint (0 to disable)
 *   SG              - sub-group size hint (0 to disable)
 *   NSLICES         - number of mantissa slices
 *   MANT_BITS       - mantissa bits  (52 for double, 23 for float)
 *   BIAS_PLUS_MANT  - exponent bias + mantissa bits
 *   TRIANGULAR      - if 1, iterate upper triangle of slice pairs
 *   SYMMETRIZE      - if 1, compute mirror D(sb,sa) for off-diagonal pairs
 *   TRIM            - number of least-significant diagonals to skip
 *   USE_DOUBLE      - if 1, accumulate in double; otherwise float
 *   CONSTANT        - address-space qualifier for read-only buffers
 *   USE_XMX         - if 1, use hardware matrix multiply-accumulate
 *                     (requires BK == 32, BM/BN divisible by 8)
 */

#if !defined(BM)
# define BM 16
#endif
#if !defined(BN)
# define BN 16
#endif
#if !defined(BK)
# define BK 16
#endif
#if !defined(NSLICES)
# define NSLICES 8
#endif
#if !defined(MANT_BITS)
# define MANT_BITS 52
#endif
#if !defined(BIAS_PLUS_MANT)
# define BIAS_PLUS_MANT 1075
#endif
#if !defined(TRIANGULAR)
# define TRIANGULAR 1
#endif
#if !defined(SYMMETRIZE)
# define SYMMETRIZE 1
#endif
#if !defined(TRIM)
# define TRIM 0
#endif
#if !defined(CONSTANT)
# define CONSTANT global
#endif

/* Small integer type for loop counters (states value range) */
#if !defined(SINT)
# define SINT signed char
#endif

/* Reinterpret a floating-point value as its unsigned integer representation.
 * real_t, uint_repr_t, EXP_MASK, and AS_UINT are defined in libxstream_common.h. */

#if defined(USE_XMX) && (0 < USE_XMX)
# pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
# pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
/* Sub-tile dimensions for the XMX dot-product path */
# define XMX_M 8
# define XMX_N 8
# define NTM (BM / XMX_M)
# define NTN (BN / XMX_N)
# if (32 != BK)
#   error "USE_XMX requires BK == 32"
# endif
# if (0 != (BM % XMX_M) || 0 != (BN % XMX_N))
#   error "USE_XMX requires BM/BN divisible by XMX_M/XMX_N (8)"
# endif
/* B layout for XMX: bk[panel][s][kk][nj]  (K-major for VNNI load).
 * Default layout: bk[panel][nj][s][kk] (N-major for scalar access). */
# define BK_IDX(panel, nj, s, kk) \
    ((((long)(panel) * NSLICES + (s)) * BK + (kk)) * BN + (nj))
#else
# define BK_IDX(panel, nj, s, kk) \
    ((((long)(panel) * BN + (nj)) * NSLICES + (s)) * BK + (kk))
#endif


/**
 * preprocess_a: decompose rows of A into int8 mantissa slices.
 *
 * One work-group per (row-block, k-sub-block).
 * Work-group size: (BM, BK).
 *
 * Inputs:
 *   a       - source matrix A (row-major or col-major depending on transa)
 *   M, K    - matrix dimensions
 *   lda     - leading dimension of A
 *   transa  - 0: not transposed, 1: transposed
 *   kb      - k-offset for this sub-block
 *
 * Outputs:
 *   ak      - int8 slices [nblk_m * nkb][BM][NSLICES][BK]
 *   expa    - per-row max exponent [nblk_m * nkb][BM]
 */
__attribute__((reqd_work_group_size(BM, BK, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_a(
  CONSTANT const real_t* restrict a,
  int M, int K, int lda, int transa,
  int kb_offset,
  global char* restrict ak,    /* int8: [panels][BM][NSLICES][BK] */
  global short* restrict expa, /* int16: [panels][BM] */
  int nblk_m)
{
  const int ib_idx = (int)get_group_id(0); /* row-block index */
  const int ki     = (int)get_group_id(1); /* sub-panel index within batch */
  const int mi     = (int)get_local_id(0); /* row within block */
  const int kk     = (int)get_local_id(1); /* k within block */

  const int ib = ib_idx * BM;
  const int kb = kb_offset + ki * BK;
  const int row = ib + mi;
  const int col = kb + kk;

  /* Panel index for output arrays */
  const int panel = ki * nblk_m + ib_idx;

  /* Local memory for per-row max exponent reduction */
  local int row_max_exp[BM];

  /* Extract element */
  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;
  SINT s;

  if (row < M && col < K) {
    const int idx = transa ? (row * lda + col) : (col * lda + row);
    const real_t val = a[idx];
    const uint_repr_t bits = AS_UINT(val);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    elem_sign = (int)(bits >> 63);
    elem_exp  = (short)((bits >> 52) & EXP_MASK);
    elem_mant = (bits & 0x000FFFFFFFFFFFFFUL) | 0x0010000000000000UL;
#else
    elem_sign = (int)(bits >> 31);
    elem_exp  = (short)((bits >> 23) & EXP_MASK);
    elem_mant = (bits & 0x007FFFFFU) | 0x00800000U;
#endif
    if (0 == elem_exp) { /* zero or subnormal: treat as zero */
      elem_mant = 0;
      elem_exp  = 0;
    }
  }

  /* Find per-row max exponent (reduction across K dimension within work-group) */
  /* Use atomic max on local memory since BK is small */
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row < M && col < K && elem_exp > 0) {
    atomic_max(&row_max_exp[mi], (int)elem_exp);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write per-row max exponent (one thread per row) */
  if (0 == kk && mi < MIN(BM, M - ib)) {
    expa[panel * BM + mi] = (short)row_max_exp[mi];
  }

  /* Compute slices: align mantissa relative to row max, then extract 7-bit digits */
  if (row < M && col < K && elem_mant != 0) {
    const short max_exp = (short)row_max_exp[mi];
    const int shift = (int)(max_exp - elem_exp); /* always >= 0 */
    const uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;

    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      const int high = MANT_BITS - (7 * s);
      const int low = MAX(0, high - 6);
      const int width = high - low + 1;
      char digit;
      if (width > 0 && high >= 0) {
        const uint mask = (1U << width) - 1U;
        digit = (char)((aligned >> low) & mask);
      }
      else {
        digit = 0;
      }
      /* Apply sign */
      if (elem_sign) digit = -digit;
      ak[(((long)panel * BM + mi) * NSLICES + s) * BK + kk] = digit;
    }
  }
  else if (row < M && col < K) {
    /* Zero element: write zero slices */
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      ak[(((long)panel * BM + mi) * NSLICES + s) * BK + kk] = 0;
    }
  }
}


/**
 * preprocess_b: decompose columns of B into int8 mantissa slices.
 *
 * One work-group per (col-block, k-sub-block).
 * Work-group size: (BN, BK).
 */
__attribute__((reqd_work_group_size(BN, BK, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_b(
  CONSTANT const real_t* restrict b,
  int N, int K, int ldb, int transb,
  int kb_offset,
#if defined(USE_XMX) && (0 < USE_XMX)
  global char* restrict bk,    /* int8: [panels][NSLICES][BK][BN] */
#else
  global char* restrict bk,    /* int8: [panels][BN][NSLICES][BK] */
#endif
  global short* restrict expb, /* int16: [panels][BN] */
  int nblk_n)
{
  const int jb_idx = (int)get_group_id(0);
  const int ki     = (int)get_group_id(1);
  const int nj     = (int)get_local_id(0);
  const int kk     = (int)get_local_id(1);

  const int jb  = jb_idx * BN;
  const int kb  = kb_offset + ki * BK;
  const int col = jb + nj;
  const int row = kb + kk;

  const int panel = ki * nblk_n + jb_idx;

  local int col_max_exp[BN];

  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;
  SINT s;

  if (row < K && col < N) {
    const int idx = transb ? (row * ldb + col) : (col * ldb + row);
    const real_t val = b[idx];
    const uint_repr_t bits = AS_UINT(val);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    elem_sign = (int)(bits >> 63);
    elem_exp  = (short)((bits >> 52) & EXP_MASK);
    elem_mant = (bits & 0x000FFFFFFFFFFFFFUL) | 0x0010000000000000UL;
#else
    elem_sign = (int)(bits >> 31);
    elem_exp  = (short)((bits >> 23) & EXP_MASK);
    elem_mant = (bits & 0x007FFFFFU) | 0x00800000U;
#endif
    if (0 == elem_exp) {
      elem_mant = 0;
      elem_exp  = 0;
    }
  }

  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row < K && col < N && elem_exp > 0) {
    atomic_max(&col_max_exp[nj], (int)elem_exp);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && nj < MIN(BN, N - jb)) {
    expb[panel * BN + nj] = (short)col_max_exp[nj];
  }

  if (row < K && col < N && elem_mant != 0) {
    const short max_exp = (short)col_max_exp[nj];
    const int shift = (int)(max_exp - elem_exp);
    const uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;

    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      const int high = MANT_BITS - (7 * s);
      const int low = MAX(0, high - 6);
      const int width = high - low + 1;
      char digit;
      if (width > 0 && high >= 0) {
        const uint mask = (1U << width) - 1U;
        digit = (char)((aligned >> low) & mask);
      }
      else {
        digit = 0;
      }
      if (elem_sign) digit = -digit;
      bk[BK_IDX(panel, nj, s, kk)] = digit;
    }
  }
  else if (row < K && col < N) {
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      bk[BK_IDX(panel, nj, s, kk)] = 0;
    }
  }
}


#if defined(USE_XMX) && (0 < USE_XMX)
/**
 * dotprod (XMX path): hardware matrix multiply-accumulate for int8 slices.
 *
 * Work-group: (SG, NTM * NTN, 1).
 * Each sub-group computes one XMX_M x XMX_N (8x8) sub-tile of the
 * BM x BN output block.  BK must equal 32 for the k32 built-in.
 *
 * A layout: ak[panel][mi][s][kk]  (row of BK contiguous — same as scalar).
 * B layout: bk[panel][s][kk][nj]  (K-major, loaded with VNNI transform).
 *
 * Driver must dispatch with:
 *   local  = { SG, NTM * NTN, 1 }
 *   global = { nblk_m * SG, nblk_n * NTM * NTN, 1 }
 */
__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void dotprod(
  CONSTANT const char* restrict ak,
  CONSTANT const short* restrict expa,
  CONSTANT const char* restrict bk,
  CONSTANT const short* restrict expb,
  global real_t* restrict c,
  int M, int N, int ldc,
  real_t alpha, real_t beta,
  int first_batch, int nkb,
  int nblk_m, int nblk_n)
{
  const int ib_idx  = (int)get_group_id(0);
  const int jb_idx  = (int)get_group_id(1);
  const int sg_lid  = (int)get_sub_group_local_id();
  const int sg_id   = (int)get_sub_group_id();
  const int tile_m  = sg_id / NTN;
  const int tile_n  = sg_id % NTN;
  const int mi_base = tile_m * XMX_M;
  const int nj_base = tile_n * XMX_N;
  const int row     = ib_idx * BM + mi_base + sg_lid;
  const int cutoff  = MAX(0, 2 * (NSLICES - 1) - TRIM);
  const int a_stride = NSLICES * BK;           /* row stride in ak panel */
  int slice_low_bit[NSLICES];
  real_t cval[XMX_N];
  union { int8 v; int a[XMX_N]; } dot_u;       /* MAD result accessor */
  int ki, j;
  SINT s, sa, sb;

  /* Precompute slice low-bit positions */
  UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
    const int high = MANT_BITS - (7 * s);
    slice_low_bit[s] = MAX(0, high - 6);
  }

  /* Guard: skip out-of-bounds rows (all lanes in sub-group are uniform) */
  if (row >= M) return;

  /* Beta scaling at first batch */
  UNROLL_FORCE(XMX_N) for (j = 0; j < XMX_N; ++j) {
    const int col_j = jb_idx * BN + nj_base + j;
    if (col_j < N) {
      cval[j] = c[col_j * ldc + row];
      if (first_batch) cval[j] *= beta;
    }
    else {
      cval[j] = (real_t)0;
    }
  }

  /* Loop over k-sub-panels */
  for (ki = 0; ki < nkb; ++ki) {
    const int a_panel = ki * nblk_m + ib_idx;
    const int b_panel = ki * nblk_n + jb_idx;
    const short ea = expa[a_panel * BM + mi_base + sg_lid];
    /* Per-column B exponents for the XMX_N output columns */
    short eb[XMX_N];
    UNROLL_FORCE(XMX_N) for (j = 0; j < XMX_N; ++j) {
      eb[j] = expb[b_panel * BN + nj_base + j];
    }

    /* Slice-pair loop */
    for (sa = 0; sa < NSLICES && sa <= cutoff; ++sa) {
#if (1 == TRIANGULAR)
      const SINT sb_start = sa;
#else
      const SINT sb_start = 0;
#endif
      const SINT sb_end = MIN(NSLICES, cutoff + 1 - sa);

      for (sb = sb_start; sb < sb_end; ++sb) {
        int8 a_tile, b_tile;

        /* Load A tile [XMX_M x BK]: WI sg_lid loads its row from ak.
         * Source: ak[a_panel][mi_base+sg_lid][sa][0..BK-1]
         * 2D surface: width = a_stride, height = BM, pitch = a_stride.
         * Coord: (sa * BK, mi_base). Reads XMX_M rows x BK(=32) cols. */
        a_tile = as_int8(intel_sub_group_2d_block_read_8b_8r32c(
            (long)(ak + (long)a_panel * BM * a_stride),
            a_stride - 1, BM - 1, a_stride - 1,
            (int2)(sa * BK, mi_base)));

        /* Load B tile [BK x XMX_N] with VNNI transform from bk.
         * B layout: bk[b_panel][sb][kk][nj] — a BK x BN plane.
         * 2D surface: width = BN, height = BK, pitch = BN.
         * Coord: (nj_base, 0). Reads BK(=32) rows x XMX_N(=8) cols. */
        b_tile = as_int8(intel_sub_group_2d_block_read_transform_8b_32r8c(
            (long)(bk + ((long)b_panel * NSLICES + sb) * BK * BN),
            BN - 1, BK - 1, BN - 1,
            (int2)(nj_base, 0)));

        /* D[8x8] = A[8x32] * B[32x8] */
        dot_u.v = intel_sub_group_i8_i8_matrix_mad_k32(
                      a_tile, b_tile, (int8)(0));

#if (1 == SYMMETRIZE)
        if (sa != sb) {
          int8 a_mir, b_mir;
          /* Mirror: swap slice indices (sb for A, sa for B) */
          a_mir = as_int8(intel_sub_group_2d_block_read_8b_8r32c(
              (long)(ak + (long)a_panel * BM * a_stride),
              a_stride - 1, BM - 1, a_stride - 1,
              (int2)(sb * BK, mi_base)));
          b_mir = as_int8(intel_sub_group_2d_block_read_transform_8b_32r8c(
              (long)(bk + ((long)b_panel * NSLICES + sa) * BK * BN),
              BN - 1, BK - 1, BN - 1,
              (int2)(nj_base, 0)));
          dot_u.v = intel_sub_group_i8_i8_matrix_mad_k32(
                        a_mir, b_mir, dot_u.v);
        }
#endif

        /* Scale int32 dot products and accumulate into cval */
        UNROLL_FORCE(XMX_N) for (j = 0; j < XMX_N; ++j) {
          if (0 != dot_u.a[j]) {
            const int base_sh = (int)ea + (int)eb[j] - (2 * BIAS_PLUS_MANT);
            const int shift = base_sh + slice_low_bit[sa] + slice_low_bit[sb];
            const real_t scale = alpha * pown((real_t)2.0, shift);
            cval[j] += (real_t)dot_u.a[j] * scale;
          }
        }
      }
    }
  }

  /* Write results to C */
  UNROLL_FORCE(XMX_N) for (j = 0; j < XMX_N; ++j) {
    const int col_j = jb_idx * BN + nj_base + j;
    if (col_j < N) {
      c[col_j * ldc + row] = cval[j];
    }
  }
}

#else /* !USE_XMX: scalar dotprod */

/**
 * dotprod: int8 dot products over slice pairs, accumulate into C.
 *
 * One work-group per (row-block, col-block) tile of C.
 * Work-group size: (BM, BN).
 *
 * Each work-item handles one element C[ib+mi, jb+nj] and iterates over
 * all k-sub-panels and slice pairs within the current K-batch.
 *
 * Inputs:
 *   ak, expa  - preprocessed A panels for this K-batch
 *   bk, expb  - preprocessed B panels for this K-batch
 *   alpha     - scalar multiplier
 *   beta      - scalar multiplier for C (applied only when first_batch != 0)
 *   nkb       - number of k-sub-panels in this batch
 *   nblk_m/n  - number of M/N blocks
 *
 * In/Out:
 *   c         - output matrix C (column-major, ldc stride)
 */
__attribute__((reqd_work_group_size(BM, BN, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void dotprod(
  CONSTANT const char* restrict ak,
  CONSTANT const short* restrict expa,
  CONSTANT const char* restrict bk,
  CONSTANT const short* restrict expb,
  global real_t* restrict c,
  int M, int N, int ldc,
  real_t alpha, real_t beta,
  int first_batch, int nkb,
  int nblk_m, int nblk_n)
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int mi     = (int)get_local_id(0);
  const int nj     = (int)get_local_id(1);

  const int row = ib_idx * BM + mi;
  const int col = jb_idx * BN + nj;
  const int cutoff = MAX(0, 2 * (NSLICES - 1) - TRIM);
  int slice_low_bit[NSLICES];
  real_t cval;
  int ki;
  SINT s, sa, sb, kk;

  if (row >= M || col >= N) return;

  /* Beta scaling at first batch */
  cval = c[col * ldc + row];
  if (first_batch) {
    cval *= beta;
  }

  /* Precompute slice low-bit positions */
  UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
    const int high = MANT_BITS - (7 * s);
    slice_low_bit[s] = MAX(0, high - 6);
  }

  /* Loop over k-sub-panels */
  UNROLL_OUTER(1) for (ki = 0; ki < nkb; ++ki) {
    const int a_panel = ki * nblk_m + ib_idx;
    const int b_panel = ki * nblk_n + jb_idx;
    const short ea = expa[a_panel * BM + mi];
    const short eb = expb[b_panel * BN + nj];
    const int base_sh = (int)ea + (int)eb - (2 * BIAS_PLUS_MANT);
    const long a_base = ((long)a_panel * BM + mi) * NSLICES;
    const long b_base = ((long)b_panel * BN + nj) * NSLICES;

    /* Slice-pair loop with optional triangular + symmetrize */
    UNROLL_AUTO for (sa = 0; sa < NSLICES && sa <= cutoff; ++sa) {
#if (1 == TRIANGULAR)
      const SINT sb_start = sa;
#else
      const SINT sb_start = 0;
#endif
      const SINT sb_end = MIN(NSLICES, cutoff + 1 - sa);

      UNROLL_AUTO for (sb = sb_start; sb < sb_end; ++sb) {
        int dot = 0;

        UNROLL(BK) for (kk = 0; kk < BK; ++kk) {
          dot += (int)ak[(a_base + sa) * BK + kk]
               * (int)bk[(b_base + sb) * BK + kk];
        }

#if (1 == SYMMETRIZE)
        if (sa != sb) {
          UNROLL(BK) for (kk = 0; kk < BK; ++kk) {
            dot += (int)ak[(a_base + sb) * BK + kk]
                 * (int)bk[(b_base + sa) * BK + kk];
          }
        }
#endif

        if (0 != dot) {
          const int shift = base_sh + slice_low_bit[sa] + slice_low_bit[sb];
          const real_t scale = alpha * pown((real_t)2.0, shift);
          cval += (real_t)dot * scale;
        }
      }
    }
  }

  c[col * ldc + row] = cval;
}

#endif /* USE_XMX */
