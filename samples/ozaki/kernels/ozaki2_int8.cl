/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../include/opencl/libxstream_common.h"

/* Ozaki Scheme 2: CRT-based int8 GEMM via OpenCL.
 *
 * Three kernels:
 *   preprocess_a  - reduce mantissas of A mod coprime moduli -> int8 residues
 *   preprocess_b  - reduce mantissas of B mod coprime moduli -> int8 residues
 *   dotprod       - int8 dot products per modulus, CRT (Garner) reconstruction
 *
 * Unlike Scheme 1 (mantissa-slicing with S*(S+1)/2 slice-pair iterations),
 * Scheme 2 performs one int8 dot product per modulus channel and reconstructs
 * the exact dot product via Chinese Remainder Theorem.  The number of
 * modular dot products equals NPRIMES (constant), eliminating the O(S^2)
 * slice-pair loop.
 *
 * CRT moduli are pairwise coprime integers <= 128 whose product P exceeds
 * 2 * BK * (2^(MANT+1))^2 (signed dot-product range).  Residues (0..m_i-1)
 * are stored as signed int8 with the element sign folded in, enabling
 * VNNI int8 dot products (VPDPBUSD treats first operand as unsigned,
 * second as signed; XOR+bias correction maps naturally).
 *
 * Compile-time parameters (supplied via -D):
 *   BM, BN, BK     - block dimensions (default 16)
 *   NPRIMES         - number of CRT moduli (max 18 for fp64, 10 for fp32)
 *   MANT_BITS       - mantissa bits (52 for double, 23 for float)
 *   BIAS_PLUS_MANT  - exponent bias + mantissa bits
 *   USE_DOUBLE      - if 1, accumulate in double; otherwise float
 *   CONSTANT        - address-space qualifier for read-only buffers
 *   WG              - work-group size hint (0 to disable)
 *   SG              - sub-group size hint (0 to disable)
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
#if !defined(NPRIMES)
# define NPRIMES 17
#endif
#if !defined(MANT_BITS)
# define MANT_BITS 52
#endif
#if !defined(BIAS_PLUS_MANT)
# define BIAS_PLUS_MANT 1075
#endif
#if !defined(CONSTANT)
# define CONSTANT global
#endif

/* Small integer type for loop counters (states value range) */
#if !defined(SINT)
# define SINT signed char
#endif

/* CRT moduli table: 18 pairwise coprime moduli <= 128.
 * Product of 17 moduli ~ 2^112 > 2^111 (sufficient for fp64).
 * Includes prime powers (128=2^7, 125=5^3, 121=11^2, 81=3^4)
 * alongside primes. Residues 0..127 fit in int8 with sign folded in. */
constant ushort oz2_moduli[18] = {
  128, 127, 125, 121, 113, 109, 107, 103,
  101,  97,  89,  83,  81,  79,  73,  71,
   67,  61
};

/* Barrett reciprocals: floor(2^16 / m_i) for OpenCL 32-bit reduction.
 * Using 16-bit shift to avoid 64-bit arithmetic in OpenCL kernels. */
constant ushort oz2_rcp16[18] = {
  512,  515,  524,  541,  579,  601,  611,  635,
  648,  674,  735,  788,  808,  828,  896,  921,
  976,  1073
};

/* Modular reduction: x mod oz2_moduli[pidx] via Barrett.
 * x must be < moduli[pidx]^2 (always true for mantissa residues and
 * dot products in the range [-BK*127^2, +BK*127^2]). */
inline uint oz2_mod(uint x, SINT pidx)
{
  return x % oz2_moduli[pidx];
}

/* 64-bit modular reduction: x mod oz2_moduli[pidx].
 * Used for mantissa values (up to 2^53). */
inline uint oz2_mod64(ulong x, SINT pidx)
{
  return (uint)(x % (ulong)oz2_moduli[pidx]);
}


/**
 * preprocess_a: reduce rows of A into CRT int8 residues.
 *
 * One work-group per (row-block, k-sub-block).
 * Work-group size: (BM, BK).
 *
 * Each work-item handles one element A[row,col]:
 *   1. Extract sign, exponent, mantissa from IEEE representation.
 *   2. Find per-row max exponent via local memory reduction.
 *   3. Align mantissa (right-shift by max_exp - elem_exp).
 *   4. Reduce aligned mantissa modulo each modulus.
 *   5. Fold sign into residues (negate for negative elements).
 *   6. Write int8 residues to ak[panel][mi][pidx][kk].
 *
 * Outputs:
 *   ak   - int8 residues: [panels][BM][NPRIMES][BK]
 *   expa - per-row max exponent: [panels][BM]
 */
__attribute__((reqd_work_group_size(BM, BK, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_a(
  CONSTANT const real_t* restrict a,
  int M, int K, int lda, int transa,
  int kb_offset,
  global char* restrict ak,     /* int8: [panels][BM][NPRIMES][BK] */
  global short* restrict expa,  /* int16: [panels][BM] */
  int nblk_m)
{
  const int ib_idx = (int)get_group_id(0);
  const int ki     = (int)get_group_id(1);
  const int mi     = (int)get_local_id(0);
  const int kk     = (int)get_local_id(1);

  const int ib  = ib_idx * BM;
  const int kb  = kb_offset + ki * BK;
  const int row = ib + mi;
  const int col = kb + kk;

  const int panel = ki * nblk_m + ib_idx;

  /* Local memory for per-row max exponent reduction */
  local int row_max_exp[BM];

  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;
  SINT pidx;

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

  /* Compute CRT residues: align mantissa, reduce mod each modulus, fold sign */
  if (row < M && col < K && elem_mant != 0) {
    const short max_exp = (short)row_max_exp[mi];
    const int shift = (int)(max_exp - elem_exp); /* always >= 0 */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    const ulong aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0UL;
#else
    const uint aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0U;
#endif

    UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      char residue = (char)oz2_mod64(aligned, pidx);
#else
      char residue = (char)oz2_mod((uint)aligned, pidx);
#endif
      /* Fold sign into residue */
      if (elem_sign) residue = -residue;
      ak[(((long)panel * BM + mi) * NPRIMES + pidx) * BK + kk] = residue;
    }
  }
  else {
    /* Zero or out-of-bounds: write zero residues */
    UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
      ak[(((long)panel * BM + mi) * NPRIMES + pidx) * BK + kk] = 0;
    }
  }
}


/**
 * preprocess_b: reduce columns of B into CRT int8 residues.
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
  global char* restrict bk,     /* int8: [panels][BN][NPRIMES][BK] */
  global short* restrict expb,  /* int16: [panels][BN] */
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
  SINT pidx;

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
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    const ulong aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0UL;
#else
    const uint aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0U;
#endif

    UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      char residue = (char)oz2_mod64(aligned, pidx);
#else
      char residue = (char)oz2_mod((uint)aligned, pidx);
#endif
      if (elem_sign) residue = -residue;
      bk[(((long)panel * BN + nj) * NPRIMES + pidx) * BK + kk] = residue;
    }
  }
  else {
    UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
      bk[(((long)panel * BN + nj) * NPRIMES + pidx) * BK + kk] = 0;
    }
  }
}


/* Garner modular inverse table: garner_inv[i][j] = m_i^{-1} mod m_j.
 * Precomputed for all 18 moduli.  Only entries with i < j are used.
 * Stored in constant memory (read-only, cached). */
constant uint garner_inv[18][18] = {
  /* m_0=128 */ {0,   1,  42,  52,  98,  23,  51,  33,  15,  72,  16,  24,  50,  50,   4,   5,  11,  51},
  /* m_1=127 */ {0,   0,  63, 101, 105, 103,  91,  73,  35,  55,  82,  17,  37,  28,  23,  52,  19,  49},
  /* m_2=125 */ {0,   0,   0,  91,  66,  75,   6,  89,  80,  52,  47,   2,  35,  67,  66,  25,  52,  41},
  /* m_3=121 */ {0,   0,   0,   0,  99, 100,  23,  63,  96,  93,  64,  59,  79,  32,  35,  27,  36,  60},
  /* m_4=113 */ {0,   0,   0,   0,   0,  82,  18,  31,  59,  91,  26,  36,  38,   7,  42,  22,  51,  27},
  /* m_5=109 */ {0,   0,   0,   0,   0,   0,  54,  86,  38,  89,  49,  16,  55,  29,  71,  43,   8,  14},
  /* m_6=107 */ {0,   0,   0,   0,   0,   0,   0,  26,  17,  68,   5,  45,  53,  48,  58,   2,  62,   4},
  /* m_7=103 */ {0,   0,   0,   0,   0,   0,   0,   0,  51,  81,  70,  54,  70,  56,  56,  20,  54,  16},
  /* m_8=101 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,  73,  52,  60,  77,  18,  60,  45,   2,  29},
  /* m_9= 97 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  78,   6,  76,  22,  70,  41,  38,  39},
  /* m10= 89 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  14,  71,   8,  32,   4,  64,  24},
  /* m11= 83 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  41,  20,  22,   6,  21,  25},
  /* m12= 81 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  40,  64,  64,  24,  58},
  /* m13= 79 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61,   9,  28,  17},
  /* m14= 73 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,  56,  56},
  /* m15= 71 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  55},
  /* m16= 67 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  51},
  /* m17= 61 */ {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
};


/**
 * dotprod: CRT-based int8 dot products, Garner reconstruction, accumulate into C.
 *
 * One work-group per (row-block, col-block) tile of C.
 * Work-group size: (BM, BN).
 *
 * Each work-item handles one element C[ib+mi, jb+nj]:
 *   1. For each modulus, compute int8 dot product A_p · B_p (BK terms).
 *   2. Reduce dot product into unsigned residue mod modulus.
 *   3. Reconstruct signed dot product via Garner's algorithm (CRT).
 *   4. Scale by exponent and alpha, accumulate into C.
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
  real_t cval;
  int ki;
  SINT pidx, kk;

  if (row >= M || col >= N) return;

  /* Beta scaling at first batch */
  cval = c[col * ldc + row];
  if (first_batch) {
    cval *= beta;
  }

  /* Loop over k-sub-panels */
  UNROLL_OUTER(1) for (ki = 0; ki < nkb; ++ki) {
    const int a_panel = ki * nblk_m + ib_idx;
    const int b_panel = ki * nblk_n + jb_idx;
    const short ea = expa[a_panel * BM + mi];
    const short eb = expb[b_panel * BN + nj];
    const int base_sh = (int)ea + (int)eb - (2 * BIAS_PLUS_MANT);
    const long a_base = ((long)a_panel * BM + mi) * NPRIMES;
    const long b_base = ((long)b_panel * BN + nj) * NPRIMES;

    /* Step 1: Compute int8 dot product for each modulus channel */
    uint dot_residues[18]; /* sized for max NPRIMES */
    UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
      int dot = 0;
      UNROLL(BK) for (kk = 0; kk < BK; ++kk) {
        dot += (int)ak[(a_base + pidx) * BK + kk]
             * (int)bk[(b_base + pidx) * BK + kk];
      }
      /* Reduce signed dot product to unsigned residue in [0, m_i) */
      if (dot >= 0) {
        dot_residues[pidx] = oz2_mod((uint)dot, pidx);
      }
      else {
        const uint r = oz2_mod((uint)(-dot), pidx);
        dot_residues[pidx] = (0 != r)
          ? (oz2_moduli[pidx] - r) : 0;
      }
    }

    /* Step 2: Garner's algorithm (CRT reconstruction) */
    { uint v[18]; /* mixed-radix digits */
      SINT i, j;
      int is_negative;

      UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i) {
        uint u = dot_residues[i];
        const uint pi = oz2_moduli[i];
        for (j = 0; j < i; ++j) {
          /* Bounded subtract: v[j] < m_j <= 128, two subtracts suffice */
          uint vj = v[j];
          if (vj >= pi) vj -= pi;
          if (vj >= pi) vj -= pi;
          { const uint diff = (u >= vj) ? (u - vj) : (pi + u - vj);
            u = oz2_mod(diff * garner_inv[j][i], i);
          }
        }
        v[i] = u;
      }

      /* Step 3: Sign detection from MSB digit (centered representation) */
      is_negative = (v[NPRIMES - 1]
        >= (uint)(oz2_moduli[NPRIMES - 1] + 1) / 2) ? 1 : 0;

      /* Complement digits for negative values */
      if (0 != is_negative) {
        UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i) {
          v[i] = oz2_moduli[i] - 1 - v[i];
        }
      }

      /* Step 4: Horner's method to evaluate mixed-radix number.
       * Evaluate MSB to LSB: result = v[N-1]*m[N-2]*...*m[0] + ... + v[0].
       * fp64: use double (product of 17 moduli ~ 2^112 exceeds int64).
       * fp32: use long   (product of 10 moduli ~ 2^59  fits in int64). */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      { double r = (double)v[NPRIMES - 1];
        double result;
        for (i = NPRIMES - 2; i >= 0; --i) {
          r = r * (double)oz2_moduli[i] + (double)v[i];
        }
        result = (0 != is_negative) ? -(r + 1.0) : r;
        if (0.0 != result && (real_t)0 != alpha) {
          const real_t scale = alpha * pown((real_t)2.0, base_sh);
          cval += (real_t)(result * (double)scale);
        }
      }
#else
      { long r = (long)v[NPRIMES - 1];
        for (i = NPRIMES - 2; i >= 0; --i) {
          r = r * (long)oz2_moduli[i] + (long)v[i];
        }
        { const long result = (0 != is_negative) ? -(r + 1) : r;
          if (0 != result && (real_t)0 != alpha) {
            const real_t scale = alpha * pown((real_t)2.0f, base_sh);
            cval += (real_t)result * scale;
          }
        }
      }
#endif
    }
  }

  c[col * ldc + row] = cval;
}
