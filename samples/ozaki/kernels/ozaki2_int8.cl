/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../include/opencl/libxstream_common.h"
#include "ozaki_common.cl"

/* Ozaki Scheme 2: CRT-based int8 GEMM via OpenCL.
 *
 * Kernels (scalar path):
 *   preprocess_a  - reduce mantissas of A mod coprime moduli -> int8 residues
 *   preprocess_b  - reduce mantissas of B mod coprime moduli -> int8 residues
 *   dotprod       - int8 dot products per modulus, CRT (Garner) reconstruction
 *
 * Kernels (XMX path, USE_XMX=1):
 *   preprocess_a  - same as scalar
 *   preprocess_b  - same (K-major B layout for VNNI/2D block I/O)
 *   dotprod       - DPAS int8x8 matmul per modulus, store residues to global
 *   postprocess   - Garner reconstruction + Horner evaluation, accumulate C
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

/* KGROUP: number of consecutive K sub-panels whose exponents are grouped.
 * Preprocessing uses the max exponent over KGROUP*BK elements instead of BK.
 * Dotprod accumulates int32 DPAS results across KGROUP panels before
 * mod reduction + Garner reconstruction.  Reduces Garner calls by KGROUP×
 * at the cost of wider exponent alignment (more mantissa bits lost for
 * elements far from the group max).  Set via GEMM_OZTRIM: kgroup = 2^oztrim.
 * Default 1 = no grouping (current behaviour). */
#if !defined(KGROUP)
# define KGROUP 1
#endif

#if defined(USE_XMX) && (0 < USE_XMX)
# define XMX_M 8
# define XMX_N 16
# define NTM (BM / XMX_M)
# define NTN (BN / XMX_N)
# define BN_PAD ((BN) < 64 ? 64 : (BN))
# if (32 != BK)
#   error "USE_XMX requires BK == 32"
# endif
# if (0 != (BM % XMX_M) || 0 != (BN % XMX_N))
#   error "USE_XMX requires BM divisible by XMX_M (8) and BN by XMX_N (16)"
# endif
/* B layout for XMX: bk[panel][pidx][kk][nj] (K-major for VNNI load).
 * Rows are padded to BN_PAD >= 64 for 2D block I/O surface constraints. */
# define BK_IDX(panel, nj, pidx, kk) \
    ((((long)(panel) * NPRIMES + (pidx)) * BK + (kk)) * BN_PAD + (nj))
#else
/* Default B layout: bk[panel][nj][pidx][kk] (N-major for scalar access). */
# define BK_IDX(panel, nj, pidx, kk) \
    ((((long)(panel) * BN + (nj)) * NPRIMES + (pidx)) * BK + (kk))
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
 *   expa - per-row max exponent: [groups][BM]  (one per KGROUP panels)
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
  global short* restrict expa,  /* int16: [groups][BM] */
  int nblk_m)
{
  const int ib_idx = (int)get_group_id(0);
  const int ki_group = (int)get_group_id(1); /* group index: covers KGROUP panels */
  const int mi     = (int)get_local_id(0);
  const int kk     = (int)get_local_id(1);

  const int ib  = ib_idx * BM;
  const int row = ib + mi;

  local int row_max_exp[BM];
  SINT pidx;
  int s;

  /* Phase 1: find per-row max exponent across all KGROUP sub-panels */
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = 0; s < KGROUP; ++s) {
    const int ki  = ki_group * KGROUP + s;
    const int col = kb_offset + ki * BK + kk;
    if (row < M && col < K) {
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      short elem_exp;
      uint_repr_t elem_mant;
      int elem_sign;
      ieee_decompose(a[idx], &elem_sign, &elem_exp, &elem_mant);
      if (elem_exp > 0) {
        atomic_max(&row_max_exp[mi], (int)elem_exp);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write per-row max exponent (one entry per group) */
  if (0 == kk && mi < MIN(BM, M - ib)) {
    const int group_idx = ki_group * nblk_m + ib_idx;
    expa[group_idx * BM + mi] = (short)row_max_exp[mi];
  }

  /* Phase 2: compute CRT residues for each sub-panel using group max_exp */
  for (s = 0; s < KGROUP; ++s) {
    const int ki    = ki_group * KGROUP + s;
    const int col   = kb_offset + ki * BK + kk;
    const int panel = ki * nblk_m + ib_idx;
    short elem_exp = 0;
    uint_repr_t elem_mant = 0;
    int elem_sign = 0;

    if (row < M && col < K) {
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      ieee_decompose(a[idx], &elem_sign, &elem_exp, &elem_mant);
    }

    if (row < M && col < K && elem_mant != 0) {
      const short max_exp = (short)row_max_exp[mi];
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
        ak[(((long)panel * BM + mi) * NPRIMES + pidx) * BK + kk] = residue;
      }
    }
    else {
      UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
        ak[(((long)panel * BM + mi) * NPRIMES + pidx) * BK + kk] = 0;
      }
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
#if defined(USE_XMX) && (0 < USE_XMX)
  global char* restrict bk,     /* int8: [panels][NPRIMES][BK][BN_PAD] */
#else
  global char* restrict bk,     /* int8: [panels][BN][NPRIMES][BK] */
#endif
  global short* restrict expb,  /* int16: [groups][BN] */
  int nblk_n)
{
  const int jb_idx  = (int)get_group_id(0);
  const int ki_group = (int)get_group_id(1);
  const int nj      = (int)get_local_id(0);
  const int kk      = (int)get_local_id(1);

  const int jb  = jb_idx * BN;
  const int col = jb + nj;

  local int col_max_exp[BN];
  SINT pidx;
  int s;

  /* Phase 1: find per-col max exponent across KGROUP sub-panels */
  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (s = 0; s < KGROUP; ++s) {
    const int ki  = ki_group * KGROUP + s;
    const int row = kb_offset + ki * BK + kk;
    if (row < K && col < N) {
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      short elem_exp;
      uint_repr_t elem_mant;
      int elem_sign;
      ieee_decompose(b[idx], &elem_sign, &elem_exp, &elem_mant);
      if (elem_exp > 0) {
        atomic_max(&col_max_exp[nj], (int)elem_exp);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && nj < MIN(BN, N - jb)) {
    const int group_idx = ki_group * nblk_n + jb_idx;
    expb[group_idx * BN + nj] = (short)col_max_exp[nj];
  }

  /* Phase 2: compute CRT residues for each sub-panel using group max_exp */
  for (s = 0; s < KGROUP; ++s) {
    const int ki    = ki_group * KGROUP + s;
    const int row   = kb_offset + ki * BK + kk;
    const int panel = ki * nblk_n + jb_idx;
    short elem_exp = 0;
    uint_repr_t elem_mant = 0;
    int elem_sign = 0;

    if (row < K && col < N) {
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &elem_sign, &elem_exp, &elem_mant);
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
        bk[BK_IDX(panel, nj, pidx, kk)] = residue;
      }
    }
    else {
      UNROLL_FORCE(NPRIMES) for (pidx = 0; pidx < NPRIMES; ++pidx) {
        bk[BK_IDX(panel, nj, pidx, kk)] = 0;
      }
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
 * oz2_garner_reconstruct: Garner's algorithm for CRT reconstruction.
 *
 * Converts NPRIMES unsigned residues into mixed-radix digits,
 * detects sign via the MSB digit (centered representation),
 * and returns the signed flag.
 *
 * On output v[] holds complemented mixed-radix digits when negative.
 */
inline int oz2_garner_reconstruct(
  const uint* restrict dot_residues,
  uint* restrict v)
{
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

  /* Sign detection from MSB digit (centered representation) */
  is_negative = (v[NPRIMES - 1]
    >= (uint)(oz2_moduli[NPRIMES - 1] + 1) / 2) ? 1 : 0;

  /* Complement digits for negative values */
  if (0 != is_negative) {
    UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i) {
      v[i] = oz2_moduli[i] - 1 - v[i];
    }
  }
  return is_negative;
}


/**
 * oz2_horner_accumulate: Horner evaluation of mixed-radix digits,
 * scale by exponent, and accumulate into *cval.
 *
 * fp64: grouped uint64 Horner — partitions digits into groups of up to
 *       OZ2_HORNER_GROUP (9) digits, evaluates each group exactly in
 *       ulong (product of 9 largest moduli ~ 2^62.9 fits uint64),
 *       then combines groups with a single double mul-add per group
 *       boundary.  For 17 primes: 2 groups, 1 double mul-add.
 * fp32: use long   (product of 10 moduli ~ 2^59  fits in int64).
 */

#if !defined(OZ2_HORNER_GROUP)
# define OZ2_HORNER_GROUP 9
#endif

inline void oz2_horner_accumulate(
  const uint* restrict v, int is_negative,
  real_t alpha, int base_sh, real_t* cval)
{
  SINT i;
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  { /* Grouped Horner: exact uint64 within groups, double only between */
    const int ngroups = (NPRIMES + OZ2_HORNER_GROUP - 1) / OZ2_HORNER_GROUP;
    double result;
    int g;

    /* MSB group: digits [(ngroups-1)*OZ2_HORNER_GROUP .. NPRIMES-1] */
    { const int lo = (ngroups - 1) * OZ2_HORNER_GROUP;
      ulong r = (ulong)v[NPRIMES - 1];
      for (i = NPRIMES - 2; i >= lo; --i) {
        r = r * (ulong)oz2_moduli[i] + (ulong)v[i];
      }
      result = (double)r;
    }

    /* Remaining groups, MSB to LSB */
    for (g = ngroups - 2; g >= 0; --g) {
      const int lo = g * OZ2_HORNER_GROUP;
      const int hi = lo + OZ2_HORNER_GROUP - 1;
      ulong gval, gprod = 1;
      for (i = lo; i <= hi; ++i) gprod *= (ulong)oz2_moduli[i];
      gval = (ulong)v[hi];
      for (i = hi - 1; i >= lo; --i) {
        gval = gval * (ulong)oz2_moduli[i] + (ulong)v[i];
      }
      result = result * (double)gprod + (double)gval;
    }

    result = (0 != is_negative) ? -(result + 1.0) : result;
    if (0.0 != result && (real_t)0 != alpha) {
      const real_t scale = alpha * EXP2I(base_sh);
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  { long r = (long)v[NPRIMES - 1];
    for (i = NPRIMES - 2; i >= 0; --i) {
      r = r * (long)oz2_moduli[i] + (long)v[i];
    }
    { const long result = (0 != is_negative) ? -(r + 1) : r;
      if (0 != result && (real_t)0 != alpha) {
        const real_t scale = alpha * EXP2I(base_sh);
        *cval += (real_t)result * scale;
      }
    }
  }
#endif
}


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

#if defined(USE_XMX) && (0 < USE_XMX)

/**
 * dotprod (XMX path): DPAS-based int8 dot products with fused Garner
 * reconstruction and Horner evaluation — writes directly to C.
 *
 * Each DPAS computes an XMX_M(8) x XMX_N(16) sub-tile for one modulus.
 * After all NPRIMES DPASes for a given k-sub-panel, the signed dot
 * products are reduced modulo each prime in registers, reconstructed
 * via Garner's algorithm, and accumulated into C via Horner evaluation.
 * No intermediate global buffer is needed.
 *
 * Work-group: (SG=16, NTM * NTN, 1).
 * Each sub-group handles one sub-tile; WI sg_lid owns column sg_lid.
 *
 * A layout: ak[panel][mi][pidx][kk]  (row of BK contiguous).
 * B layout: bk[panel][pidx][kk][nj]  (K-major, padded to BN_PAD >= 64).
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
  const int col     = jb_idx * BN + nj_base + sg_lid;
  const int a_stride = NPRIMES * BK;
  int ki, m;
  SINT pidx;

  /* C accumulators — one per row of this sub-group's 8x16 tile */
  real_t c_acc[XMX_M];
  UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
    const int row_m = ib_idx * BM + mi_base + m;
    c_acc[m] = (row_m < M && col < N)
      ? (first_batch ? beta * c[col * ldc + row_m] : c[col * ldc + row_m])
      : (real_t)0;
  }

  /* Loop over k-groups (KGROUP consecutive sub-panels share one exponent) */
  { const int nkb_groups = (nkb + KGROUP - 1) / KGROUP;
    int gi;
    for (gi = 0; gi < nkb_groups; ++gi) {
      const int ki_start = gi * KGROUP;
      const int ki_end = ((ki_start + KGROUP) < nkb)
                       ? (ki_start + KGROUP) : nkb;
      const int group_idx_a = gi * nblk_m + ib_idx;
      const int group_idx_b = gi * nblk_n + jb_idx;

#if KGROUP == 1
      /* KGROUP=1: direct per-panel processing, no accumulation */
      { uint row_res[XMX_M * NPRIMES];
        const int a_panel = ki_start * nblk_m + ib_idx;
        const int b_panel = ki_start * nblk_n + jb_idx;

        UNROLL_OUTER(1) for (pidx = 0; pidx < NPRIMES; ++pidx) {
          ushort8 a_raw; uint8 b_raw;
          int8 dot;
          union { int8 v; int a[8]; } dot_u;

          intel_sub_group_2d_block_read_8b_8r32x1c(
              (global void*)(ak + (long)a_panel * BM * a_stride),
              a_stride, BM, a_stride,
              (int2)(pidx * BK, mi_base), (private ushort*)&a_raw);

          intel_sub_group_2d_block_read_transform_8b_32r16x1c(
              (global void*)(bk + ((long)b_panel * NPRIMES + pidx) * BK * BN_PAD),
              BN_PAD, BK, BN_PAD,
              (int2)(nj_base, 0), (private uint*)&b_raw);

          dot = intel_sub_group_i8_i8_matrix_mad_k32(
                    as_short8(a_raw), as_int8(b_raw), (int8)(0));
          dot_u.v = dot;
          UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
            uint r;
            if (dot_u.a[m] >= 0) {
              r = oz2_mod((uint)dot_u.a[m], pidx);
            }
            else {
              const uint neg_r = oz2_mod((uint)(-dot_u.a[m]), pidx);
              r = (0 != neg_r) ? (oz2_moduli[pidx] - neg_r) : 0;
            }
            row_res[m * NPRIMES + pidx] = r;
          }
        }

        UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
          const int row_m = ib_idx * BM + mi_base + m;
          if (row_m < M && col < N) {
            const short ea = expa[group_idx_a * BM + mi_base + m];
            const short eb = expb[group_idx_b * BN + nj_base + sg_lid];
            const int base_sh = (int)ea + (int)eb - (2 * BIAS_PLUS_MANT);
            uint v[NPRIMES];
            const int is_negative = oz2_garner_reconstruct(
              row_res + m * NPRIMES, v);
            oz2_horner_accumulate(v, is_negative, alpha, base_sh, &c_acc[m]);
          }
        }
      }
#else
      /* KGROUP>1: accumulate int32 dot products across panels, one Garner
       * per group.  Requires NPRIMES=18 (M_crt ~ 2^118) for headroom. */
      { uint row_res[XMX_M * NPRIMES];

        UNROLL_OUTER(1) for (pidx = 0; pidx < NPRIMES; ++pidx) {
          int8 dot = (int8)(0);

          for (ki = ki_start; ki < ki_end; ++ki) {
            ushort8 a_raw; uint8 b_raw;
            int8 panel_dot;
            const int a_panel = ki * nblk_m + ib_idx;
            const int b_panel = ki * nblk_n + jb_idx;

            intel_sub_group_2d_block_read_8b_8r32x1c(
                (global void*)(ak + (long)a_panel * BM * a_stride),
                a_stride, BM, a_stride,
                (int2)(pidx * BK, mi_base), (private ushort*)&a_raw);

            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                (global void*)(bk + ((long)b_panel * NPRIMES + pidx) * BK * BN_PAD),
                BN_PAD, BK, BN_PAD,
                (int2)(nj_base, 0), (private uint*)&b_raw);

            panel_dot = intel_sub_group_i8_i8_matrix_mad_k32(
                      as_short8(a_raw), as_int8(b_raw), (int8)(0));
            dot += panel_dot;
          }

          { union { int8 v; int a[8]; } dot_u;
            dot_u.v = dot;
            UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
              if (dot_u.a[m] >= 0) {
                row_res[m * NPRIMES + pidx] = oz2_mod(
                  (uint)dot_u.a[m], pidx);
              }
              else {
                const uint neg_r = oz2_mod(
                  (uint)(-dot_u.a[m]), pidx);
                row_res[m * NPRIMES + pidx] = (0 != neg_r)
                  ? (oz2_moduli[pidx] - neg_r) : 0;
              }
            }
          }
        }

        UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
          const int row_m = ib_idx * BM + mi_base + m;
          if (row_m < M && col < N) {
            const short ea = expa[group_idx_a * BM + mi_base + m];
            const short eb = expb[group_idx_b * BN + nj_base + sg_lid];
            const int base_sh = (int)ea + (int)eb - (2 * BIAS_PLUS_MANT);
            uint v[NPRIMES];
            const int is_negative = oz2_garner_reconstruct(
              row_res + m * NPRIMES, v);
            oz2_horner_accumulate(v, is_negative, alpha, base_sh, &c_acc[m]);
          }
        }
      }
#endif
    }
  }

  /* Write C */
  UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
    const int row_m = ib_idx * BM + mi_base + m;
    if (row_m < M && col < N) {
      c[col * ldc + row_m] = c_acc[m];
    }
  }
}


#endif /* USE_XMX */
