/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../include/opencl/libxsmm_common.h"

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

/* Short integer type for small loop counters (saves registers) */
#if !defined(SINT)
# define SINT short
#endif

#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
# pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
  typedef ulong  uint_repr_t;
# define EXP_MASK 2047U
#else
  typedef float  real_t;
  typedef uint   uint_repr_t;
# define EXP_MASK 255U
#endif

/* Reinterpret a floating-point value as its unsigned integer representation. */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
# define AS_UINT(x) as_ulong(x)
#else
# define AS_UINT(x) as_uint(x)
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
  local short row_max_exp[BM];

  /* Extract element */
  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;

  if (row < M && col < K) {
    int idx = transa ? (row * lda + col) : (col * lda + row);
    real_t val = a[idx];
    uint_repr_t bits = AS_UINT(val);
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
    expa[panel * BM + mi] = row_max_exp[mi];
  }

  /* Compute slices: align mantissa relative to row max, then extract 7-bit digits */
  if (row < M && col < K && elem_mant != 0) {
    short max_exp = row_max_exp[mi];
    int shift = (int)(max_exp - elem_exp); /* always >= 0 */
    uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;

    UNROLL_FORCE(NSLICES) for (SINT s = 0; s < NSLICES; ++s) {
      int high = MANT_BITS - (7 * s);
      int low  = (high >= 0) ? (high - 6) : 0;
      if (low < 0) low = 0;
      int bits_above_low = high - low + 1;
      char digit;
      if (bits_above_low > 0 && high >= 0) {
        digit = (char)((aligned >> low) & 0x7F);
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
    UNROLL_FORCE(NSLICES) for (SINT s = 0; s < NSLICES; ++s) {
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
  global char* restrict bk,    /* int8: [panels][BN][NSLICES][BK] */
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

  local short col_max_exp[BN];

  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;

  if (row < K && col < N) {
    int idx = transb ? (row * ldb + col) : (col * ldb + row);
    real_t val = b[idx];
    uint_repr_t bits = AS_UINT(val);
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
    expb[panel * BN + nj] = col_max_exp[nj];
  }

  if (row < K && col < N && elem_mant != 0) {
    short max_exp = col_max_exp[nj];
    int shift = (int)(max_exp - elem_exp);
    uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;

    UNROLL_FORCE(NSLICES) for (SINT s = 0; s < NSLICES; ++s) {
      int high = MANT_BITS - (7 * s);
      int low  = (high >= 0) ? (high - 6) : 0;
      if (low < 0) low = 0;
      int bits_above_low = high - low + 1;
      char digit;
      if (bits_above_low > 0 && high >= 0) {
        digit = (char)((aligned >> low) & 0x7F);
      }
      else {
        digit = 0;
      }
      if (elem_sign) digit = -digit;
      bk[(((long)panel * BN + nj) * NSLICES + s) * BK + kk] = digit;
    }
  }
  else if (row < K && col < N) {
    UNROLL_FORCE(NSLICES) for (SINT s = 0; s < NSLICES; ++s) {
      bk[(((long)panel * BN + nj) * NSLICES + s) * BK + kk] = 0;
    }
  }
}


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

  if (row >= M || col >= N) return;

  /* Beta scaling at first batch */
  real_t cval = c[col * ldc + row];
  if (first_batch) {
    cval *= beta;
  }

  /* Precompute slice low-bit positions */
  int slice_low_bit[NSLICES];
  UNROLL_FORCE(NSLICES) for (SINT s = 0; s < NSLICES; ++s) {
    int high = MANT_BITS - (7 * s);
    int low  = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[s] = (low > 0) ? low : 0;
  }

  /* Cutoff for diagonal trim */
  int cutoff = MAX(0, 2 * (NSLICES - 1) - TRIM);

  /* Loop over k-sub-panels */
  UNROLL_OUTER(1) for (int ki = 0; ki < nkb; ++ki) {
    int a_panel = ki * nblk_m + ib_idx;
    int b_panel = ki * nblk_n + jb_idx;

    short ea = expa[a_panel * BM + mi];
    short eb = expb[b_panel * BN + nj];
    int base_sh = (int)ea + (int)eb - (2 * BIAS_PLUS_MANT);

    /* Slice-pair loop with optional triangular + symmetrize */
    UNROLL_AUTO for (SINT sa = 0; sa < NSLICES && sa <= cutoff; ++sa) {
#if (1 == TRIANGULAR)
      SINT sb_start = sa;
#else
      SINT sb_start = 0;
#endif
      SINT sb_end = MIN(NSLICES, cutoff + 1 - sa);

      UNROLL_AUTO for (SINT sb = sb_start; sb < sb_end; ++sb) {
        /* int8 dot product over K dimension */
        int dot = 0;
        long a_base = ((long)a_panel * BM + mi) * NSLICES;
        long b_base = ((long)b_panel * BN + nj) * NSLICES;

        UNROLL(BK) for (SINT kk = 0; kk < BK; ++kk) {
          dot += (int)ak[(a_base + sa) * BK + kk]
               * (int)bk[(b_base + sb) * BK + kk];
        }

#if (1 == SYMMETRIZE)
        if (sa != sb) {
          /* Mirror pair: D(sb, sa) */
          UNROLL(BK) for (SINT kk = 0; kk < BK; ++kk) {
            dot += (int)ak[(a_base + sb) * BK + kk]
                 * (int)bk[(b_base + sa) * BK + kk];
          }
        }
#endif

        if (0 != dot) {
          int shift = base_sh + slice_low_bit[sa] + slice_low_bit[sb];
          /* ldexp via native: alpha * dot * 2^shift */
          real_t scale = alpha * pown((real_t)2.0, shift);
          cval += (real_t)dot * scale;
        }
      }
    }
  }

  c[col * ldc + row] = cval;
}
