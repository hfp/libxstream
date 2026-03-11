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

/* Ozaki Scheme 1 — GEMM-based XMX path.
 *
 * Unlike the panel-batched dotprod path, this approach:
 *   1. Preprocesses the FULL K dimension of A and B into dense int8 slices
 *      (one M x K_pad or K_pad x N matrix per slice)
 *   2. For each slice pair (sa, sb), runs a tiled int8 GEMM with full K-loop,
 *      cooperative matrix accumulation in i32 registers, and 2D block I/O
 *   3. Fuses the i32→fp scaling + exponent accumulation into the same kernel
 *
 * Compile-time parameters (-D):
 *   BM, BN          - output tile dimensions per work-group (256x256 default)
 *   TM, TN          - sub-tile per sub-group via DPAS (8x16 * NTM x NTN)
 *   XMX_M, XMX_N    - DPAS result shape (8, 16)
 *   BK              - K-unroll factor for DPAS (32 for int8)
 *   KU              - K-loop unroll depth (2)
 *   NSLICES         - number of mantissa slices
 *   MANT_BITS       - mantissa bit count (52 for fp64, 23 for fp32)
 *   BIAS_PLUS_MANT  - exponent bias + mantissa bits
 *   TRIANGULAR      - if 1, iterate upper triangle of slice pairs
 *   USE_DOUBLE      - if 1, fp64 accumulation; otherwise fp32
 *   SG              - sub-group size (16)
 *   BN_A_PAD        - padded K stride for A slices (>= 64 for 2D block I/O)
 *   BN_B_PAD        - padded N stride for B slices (>= 64 for 2D block I/O)
 */

#if !defined(BM)
# define BM 256
#endif
#if !defined(BN)
# define BN 256
#endif
#if !defined(BK)
# define BK 32
#endif
#if !defined(KU)
# define KU 2
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
#if !defined(SG)
# define SG 16
#endif

/* DPAS tile dimensions */
#define XMX_M 8
#define XMX_N 16

/* Sub-tiles per work-group dimension */
#define NTM (BM / XMX_M)
#define NTN (BN / XMX_N)

/* Minimum strides for 2D block I/O (64 bytes for int8) */
#if !defined(BN_A_PAD)
# define BN_A_PAD 64
#endif
#if !defined(BN_B_PAD)
# define BN_B_PAD 64
#endif

#if !defined(SINT)
# define SINT signed char
#endif


/* Composable macros (DBM-style factoring).
 * Each is a do{...}while(0) block for use in kernel functions. */

/* Extract NSLICES int8 digits from an aligned mantissa into DST buffer.
 * DST[s * SS + ROW * RS + COL] = digit(s). */
#define OZAKI_EXTRACT_SLICES(ALIGNED, SIGN, DST, SS, RS, ROW, COL) \
  do { \
    SINT s_; \
    UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) { \
      (DST)[(long)(s_) * (SS) + (long)(ROW) * (RS) + (COL)] = \
          ozaki_slice_digit((ALIGNED), (SIGN), (int)(s_)); \
    } \
  } while (0)

/* Zero NSLICES entries at the given position. */
#define OZAKI_ZERO_SLICES(DST, SS, RS, ROW, COL) \
  do { \
    SINT s_; \
    UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) { \
      (DST)[(long)(s_) * (SS) + (long)(ROW) * (RS) + (COL)] = 0; \
    } \
  } while (0)

/* Alias the shared DPAS primitive from ozaki_common.cl */
#define OZAKI_GEMM_DPAS OZAKI_DPAS

/* Scale i32 accumulator and write/accumulate into fp C.
 *   shift = ea[m] + eb - 2*BIAS_PLUS_MANT + LOW_SA + LOW_SB
 *   C[col*ldc+m] =/+= (real_t)dot[m] * alpha * native_exp2(shift) */
#define OZAKI_GEMM_STORE(DOT, EXPA, EXPB, C_PTR, M, N, MI, COL, \
                         LDC, ALPHA, LOW_SA, LOW_SB, FIRST) \
  do { \
    short ea_s_[XMX_M]; \
    const short eb_s_ = ((COL) < (N)) ? (EXPB)[(COL)] : 0; \
    union { int8 v_; int a_[8]; } du_s_; \
    int m_s_; \
    UNROLL_FORCE(XMX_M) for (m_s_ = 0; m_s_ < XMX_M; ++m_s_) { \
      const int rm_ = (MI) + m_s_; \
      ea_s_[m_s_] = (rm_ < (M)) ? (EXPA)[rm_] : 0; \
    } \
    du_s_.v_ = (DOT); \
    UNROLL_FORCE(XMX_M) for (m_s_ = 0; m_s_ < XMX_M; ++m_s_) { \
      const int rm_ = (MI) + m_s_; \
      if (rm_ < (M) && (COL) < (N)) { \
        const int sh_ = (int)ea_s_[m_s_] + (int)eb_s_ \
                        - (2 * BIAS_PLUS_MANT) + (LOW_SA) + (LOW_SB); \
        const real_t sc_ = (ALPHA) * native_exp2((real_t)sh_); \
        const real_t ct_ = (real_t)du_s_.a_[m_s_] * sc_; \
        if (FIRST) { (C_PTR)[(COL) * (LDC) + rm_] = ct_; } \
        else       { (C_PTR)[(COL) * (LDC) + rm_] += ct_; } \
      } \
    } \
  } while (0)


/**
 * preprocess_a_dense: decompose A into dense per-slice int8 matrices.
 *
 * Output layout: As[s * M_pad * K_pad + row * K_pad + col].
 * For 2D block read of A[8x32]: surface width = K_pad, height = M_pad.
 * K_pad must be >= 64 for 2D block I/O alignment.
 *
 * Work-group: (BM_PRE, BK_PRE, 1).
 */
__attribute__((reqd_work_group_size(BM_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_a_dense(
  CONSTANT const real_t* restrict a,
  int M, int K, int lda, int transa,
  global char* restrict as,       /* [NSLICES * M_pad * K_pad] */
  global short* restrict expa,    /* [M] per-row max exponent */
  int K_pad,                      /* padded K stride (>= 64) */
  int M_pad)                      /* padded M = nblk_m * BM_PRE */
{
  const int mi   = (int)get_local_id(0);
  const int kk   = (int)get_local_id(1);
  const int ib   = (int)get_group_id(0) * BM_PRE;
  const int kb   = (int)get_group_id(1) * BK_PRE;
  const int row  = ib + mi;
  const int col  = kb + kk;

  local int row_max_exp[BM_PRE];

  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;

  if (row < M && col < K) {
    const int idx = transa ? (row * lda + col) : (col * lda + row);
    ieee_decompose(a[idx], &elem_sign, &elem_exp, &elem_mant);
  }

  /* Per-row max exponent reduction across K within work-group */
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row < M && col < K && elem_exp > 0) {
    atomic_max(&row_max_exp[mi], (int)elem_exp);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write row max exponent (global atomic for cross-WG reduction) */
  if (0 == kk && row < M) {
    atomic_max((global volatile int*)(expa + row), (int)(short)row_max_exp[mi]);
  }

  /* Compute and store int8 slices into dense layout */
  if (row < M && col < K && elem_mant != 0) {
    const short max_exp = (short)row_max_exp[mi];
    const int shift = (int)(max_exp - elem_exp);
    const uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;
    OZAKI_EXTRACT_SLICES(aligned, elem_sign, as, M_pad * K_pad, K_pad, row, col);
  }
  else if (col < K) {
    OZAKI_ZERO_SLICES(as, M_pad * K_pad, K_pad, row, col);
  }
}


/**
 * preprocess_b_dense: decompose B into dense per-slice int8 matrices.
 *
 * Output layout: Bs[s][K_pad][N_pad] — K_pad rows, N_pad columns per slice.
 *   N_pad must be >= 64 for 2D block I/O.
 *   Stored row-major (K-major): Bs_s[k * N_pad + n].
 *   Full: Bs[s * K_pad * N_pad + k * N_pad + n].
 *
 * For DPAS B-side (VNNI transform read): surface width = N_pad bytes,
 * height = K_pad rows.  N_pad >= 64 required.
 *
 * Work-group: (BN_PRE, BK_PRE, 1).
 */
__attribute__((reqd_work_group_size(BN_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_b_dense(
  CONSTANT const real_t* restrict b,
  int N, int K, int ldb, int transb,
  global char* restrict bs,       /* [NSLICES * K_pad * N_pad] */
  global short* restrict expb,    /* [N] per-column max exponent */
  int K_pad,
  int N_pad)
{
  const int nj   = (int)get_local_id(0);
  const int kk   = (int)get_local_id(1);
  const int jb   = (int)get_group_id(0) * BN_PRE;
  const int kb   = (int)get_group_id(1) * BK_PRE;
  const int col  = jb + nj;
  const int row  = kb + kk;

  local int col_max_exp[BN_PRE];

  short elem_exp = 0;
  uint_repr_t elem_mant = 0;
  int elem_sign = 0;

  if (row < K && col < N) {
    const int idx = transb ? (row * ldb + col) : (col * ldb + row);
    ieee_decompose(b[idx], &elem_sign, &elem_exp, &elem_mant);
  }

  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row < K && col < N && elem_exp > 0) {
    atomic_max(&col_max_exp[nj], (int)elem_exp);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write column max exponent (global atomic for cross-WG reduction) */
  if (0 == kk && col < N) {
    atomic_max((global volatile int*)(expb + col), (int)(short)col_max_exp[nj]);
  }

  /* Compute and store int8 slices into dense K-major layout */
  if (row < K && col < N && elem_mant != 0) {
    const short max_exp = (short)col_max_exp[nj];
    const int shift = (int)(max_exp - elem_exp);
    const uint_repr_t aligned = (shift < MANT_BITS) ? (elem_mant >> shift) : 0;
    OZAKI_EXTRACT_SLICES(aligned, elem_sign, bs, K_pad * N_pad, N_pad, row, col);
  }
  else if (row < K) {
    OZAKI_ZERO_SLICES(bs, K_pad * N_pad, N_pad, row, col);
  }
}


/**
 * gemm_fused: Tiled int8 GEMM with K-loop + fused i32->fp accumulation.
 *
 * For one slice pair (sa, sb), computes:
 *   C_i32[tile] = As[sa][m_tile, :] * Bs[sb][:, n_tile]  (full K sum)
 *   C_fp[tile] += C_i32[tile] * scale * eA[m] * eB[n]
 *
 * where scale = alpha * exp2(base_shift + low_bit[sa] + low_bit[sb]).
 *
 * Layout assumptions:
 *   As[sa]: M_pad x K_pad, row-major (each row is K_pad bytes)
 *   Bs[sb]: K_pad x N_pad, row-major (each row is N_pad bytes)
 *   C:      M x N, column-major with stride ldc
 *
 * Work-group: (SG, NTM * NTN, 1) — each sub-group owns one XMX_M x XMX_N tile.
 * Dispatch: global = (nblk_m * SG, nblk_n * NTM * NTN, 1).
 */
__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void gemm_fused(
  CONSTANT const char* restrict as_base,    /* As[sa]: M_pad x K_pad */
  CONSTANT const char* restrict bs_base,    /* Bs[sb]: K_pad x N_pad */
  CONSTANT const short* restrict expa,      /* [M] per-row max exponent */
  CONSTANT const short* restrict expb,      /* [N] per-col max exponent */
  global real_t* restrict c,                /* [M x N] column-major, ldc */
  int M, int N, int K_pad, int N_pad, int ldc,
  real_t alpha,
  int sa, int sb,                           /* slice indices */
  int first_pair)                           /* 1 if this is the first (sa,sb) */
{
  const int ib_idx  = (int)get_group_id(0);
  const int jb_idx  = (int)get_group_id(1);
  const int sg_lid  = (int)get_sub_group_local_id();
  const int sg_id   = (int)get_sub_group_id();
  const int tile_m  = sg_id / NTN;
  const int tile_n  = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M;
  const int nj_base = jb_idx * BN + tile_n * XMX_N;
  const int col     = nj_base + sg_lid;

  /* Precompute slice low-bit positions */
  const int high_sa = MANT_BITS - (7 * sa);
  const int low_bit_sa = MAX(0, high_sa - 6);
  const int high_sb = MANT_BITS - (7 * sb);
  const int low_bit_sb = MAX(0, high_sb - 6);

  /* K-loop: full DPAS accumulation */
  int8 c_i32 = (int8)(0);
  for (int k = 0; k < K_pad; k += BK) {
    OZAKI_GEMM_DPAS(as_base, bs_base, K_pad, N_pad, mi_base, nj_base, k, M, c_i32);
  }

  /* Scale and accumulate into fp C */
  OZAKI_GEMM_STORE(c_i32, expa, expb, c, M, N, mi_base, col,
                   ldc, alpha, low_bit_sa, low_bit_sb, first_pair);
}


/**
 * gemm_fused_sym: Same as gemm_fused but computes BOTH (sa,sb) AND (sb,sa)
 * in one kernel launch (SYMMETRIZE optimization for off-diagonal pairs).
 *
 * This halves the number of kernel launches for off-diagonal slice pairs.
 */
__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void gemm_fused_sym(
  CONSTANT const char* restrict as_sa,      /* As[sa]: M_pad x K_pad */
  CONSTANT const char* restrict bs_sb,      /* Bs[sb]: K_pad x N_pad */
  CONSTANT const char* restrict as_sb,      /* As[sb]: M_pad x K_pad */
  CONSTANT const char* restrict bs_sa,      /* Bs[sa]: K_pad x N_pad */
  CONSTANT const short* restrict expa,
  CONSTANT const short* restrict expb,
  global real_t* restrict c,
  int M, int N, int K_pad, int N_pad, int ldc,
  real_t alpha,
  int sa, int sb,
  int first_pair)
{
  const int ib_idx  = (int)get_group_id(0);
  const int jb_idx  = (int)get_group_id(1);
  const int sg_lid  = (int)get_sub_group_local_id();
  const int sg_id   = (int)get_sub_group_id();
  const int tile_m  = sg_id / NTN;
  const int tile_n  = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M;
  const int nj_base = jb_idx * BN + tile_n * XMX_N;
  const int col     = nj_base + sg_lid;

  const int high_sa = MANT_BITS - (7 * sa);
  const int low_bit_sa = MAX(0, high_sa - 6);
  const int high_sb = MANT_BITS - (7 * sb);
  const int low_bit_sb = MAX(0, high_sb - 6);

  /* Two concurrent i32 accumulators: (sa,sb) and (sb,sa) */
  int8 c_fwd = (int8)(0);
  int8 c_mir = (int8)(0);

  for (int k = 0; k < K_pad; k += BK) {
    OZAKI_GEMM_DPAS(as_sa, bs_sb, K_pad, N_pad, mi_base, nj_base, k, M, c_fwd);
    OZAKI_GEMM_DPAS(as_sb, bs_sa, K_pad, N_pad, mi_base, nj_base, k, M, c_mir);
  }

  /* Both pairs share the same scale: low_bit[sa]+low_bit[sb] is symmetric */
  {
    int8 c_sum = c_fwd + c_mir;
    OZAKI_GEMM_STORE(c_sum, expa, expb, c, M, N, mi_base, col,
                     ldc, alpha, low_bit_sa, low_bit_sb, first_pair);
  }
}


/**
 * scale_beta: Prescale C by beta before accumulation.
 *
 * Work-group: (BM_PRE, 1, 1).
 * Dispatch: global = (ceil(M, BM_PRE) * BM_PRE, N, 1).
 */
__attribute__((reqd_work_group_size(BM_PRE, 1, 1)))
kernel void scale_beta(
  global real_t* restrict c,
  int M, int N, int ldc,
  real_t beta)
{
  const int row = (int)get_global_id(0);
  const int col = (int)get_global_id(1);
  if (row < M && col < N) {
    c[col * ldc + row] *= beta;
  }
}
