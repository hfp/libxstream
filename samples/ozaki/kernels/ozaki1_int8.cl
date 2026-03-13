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
 *   USE_DOUBLE      - if 1, fp64 accumulation; otherwise fp32
 *   SG              - sub-group size (16)
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
# define KU 4
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
#if !defined(SG)
# define SG 16
#endif

/* DPAS tile dimensions are in ozaki_common.cl (XMX_M=8, XMX_N=16) */

/* Sub-tiles per work-group dimension, accounting for register tiling */
#define NTM (BM / (XMX_M * RTM))
#define NTN (BN / (XMX_N * RTN))

/* Minimum strides for 2D block I/O (64 bytes for int8) */
#if !defined(BN_A_PAD)
# define BN_A_PAD 64
#endif
#if !defined(BN_B_PAD)
# define BN_B_PAD 64
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

/* Scale i32 accumulator and accumulate into register-resident fp C.
 *   shift = ea[m] + eb - 2*BIAS_PLUS_MANT + LOW_SA + LOW_SB
 *   C_REG[m] += (real_t)dot[m] * alpha * EXP2I(shift)
 * C_REG is a real_t array of XMX_M elements owned by this lane. */
#define OZAKI_GEMM_ACCUM(DOT, EXPA, EXPB, C_REG, M, N, MI, COL, \
                         ALPHA, LOW_SA, LOW_SB) \
  do { \
    const short eb_a_ = ((COL) < (N)) ? (EXPB)[(COL)] : 0; \
    union { int8 v_; int a_[8]; } du_a_; \
    int m_a_; \
    du_a_.v_ = (DOT); \
    UNROLL_FORCE(XMX_M) for (m_a_ = 0; m_a_ < XMX_M; ++m_a_) { \
      const int rm_a_ = (MI) + m_a_; \
      if (rm_a_ < (M) && (COL) < (N)) { \
        const int sh_a_ = (int)(EXPA)[(MI) + m_a_] + (int)eb_a_ \
                         - (2 * BIAS_PLUS_MANT) + (LOW_SA) + (LOW_SB); \
        const real_t sc_a_ = (ALPHA) * EXP2I(sh_a_); \
        (C_REG)[m_a_] += (real_t)du_a_.a_[m_a_] * sc_a_; \
      } \
    } \
  } while (0)

/* Scale i32 accumulator and write/accumulate into fp C (global memory).
 *   shift = ea[m] + eb - 2*BIAS_PLUS_MANT + LOW_SA + LOW_SB
 *   C[col*ldc+m] =/+= (real_t)dot[m] * alpha * EXP2I(shift) */
#define OZAKI_GEMM_STORE(DOT, EXPA, EXPB, C_PTR, M, N, MI, COL, \
                         LDC, ALPHA, LOW_SA, LOW_SB, FIRST) \
  do { \
    short ea_s_[XMX_M]; \
    const short eb_s_ = ((COL) < (N)) ? (EXPB)[(COL)] : 0; \
    union { int8 v_; int a_[8]; } du_s_; \
    int m_s_; \
    UNROLL_FORCE(XMX_M) for (m_s_ = 0; m_s_ < XMX_M; ++m_s_) { \
      ea_s_[m_s_] = (EXPA)[(MI) + m_s_]; \
    } \
    du_s_.v_ = (DOT); \
    UNROLL_FORCE(XMX_M) for (m_s_ = 0; m_s_ < XMX_M; ++m_s_) { \
      const int rm_ = (MI) + m_s_; \
      if (rm_ < (M) && (COL) < (N)) { \
        const int sh_ = (int)ea_s_[m_s_] + (int)eb_s_ \
                        - (2 * BIAS_PLUS_MANT) + (LOW_SA) + (LOW_SB); \
        const real_t sc_ = (ALPHA) * EXP2I(sh_); \
        const real_t ct_ = (real_t)du_s_.a_[m_s_] * sc_; \
        const real_t old_ = (FIRST) ? ZERO : (C_PTR)[(COL) * (LDC) + rm_]; \
        (C_PTR)[(COL) * (LDC) + rm_] = old_ + ct_; \
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
 * Dispatch: global_a[1] = BK_PRE (single WG in K) — the kernel loops
 * over K internally so that the local max exponent IS the global max.
 */
__attribute__((reqd_work_group_size(BM_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_a_dense(
  CONSTANT const real_t* restrict a,
  int M, int K, int lda, int transa,
  global char* restrict as,       /* [NSLICES * M_pad * K_pad] */
  global int* restrict expa,      /* [M] per-row max exponent (int for atomic_max) */
  int K_pad,                      /* padded K stride (>= 64) */
  int M_pad)                      /* padded M = nblk_m * BM_PRE */
{
  const int mi  = (int)get_local_id(0);
  const int kk  = (int)get_local_id(1);
  const int row = (int)get_group_id(0) * BM_PRE + mi;
  int col;

  local int row_max_exp[BM_PRE];
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Pass 1: find max exponent across ALL of K for this row */
  for (col = kk; col < K; col += BK_PRE) {
    if (row < M) {
      int s0; short e0; uint_repr_t m0;
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      ieee_decompose(a[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&row_max_exp[mi], (int)e0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write global max exponent (single WG in K, so local == global) */
  if (0 == kk && row < M) expa[row] = row_max_exp[mi];

  /* Pass 2: compute and store int8 slices using the true max exponent */
  if (row < M) {
    const short max_exp = (short)row_max_exp[mi];
    for (col = kk; col < K; col += BK_PRE) {
      int s1; short e1; uint_repr_t m1;
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      ieee_decompose(a[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift < MANT_BITS) ? (m1 >> shift) : 0;
        OZAKI_EXTRACT_SLICES(aligned, s1, as, M_pad * K_pad, K_pad, row, col);
      }
    }
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
 * Dispatch: global_b[1] = BK_PRE (single WG in K) — loops internally.
 */
__attribute__((reqd_work_group_size(BN_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_b_dense(
  CONSTANT const real_t* restrict b,
  int N, int K, int ldb, int transb,
  global char* restrict bs,       /* [NSLICES * K_pad * N_pad] */
  global int* restrict expb,      /* [N] per-column max exponent (int for atomic_max) */
  int K_pad,
  int N_pad)
{
  const int nj  = (int)get_local_id(0);
  const int kk  = (int)get_local_id(1);
  const int col = (int)get_group_id(0) * BN_PRE + nj;
  int row;

  local int col_max_exp[BN_PRE];
  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Pass 1: find max exponent across ALL of K for this column */
  for (row = kk; row < K; row += BK_PRE) {
    if (col < N) {
      int s0; short e0; uint_repr_t m0;
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&col_max_exp[nj], (int)e0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write global max exponent */
  if (0 == kk && col < N) expb[col] = col_max_exp[nj];

  /* Pass 2: compute and store int8 slices using the true max exponent */
  if (col < N) {
    const short max_exp = (short)col_max_exp[nj];
    for (row = kk; row < K; row += BK_PRE) {
      int s1; short e1; uint_repr_t m1;
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift < MANT_BITS) ? (m1 >> shift) : 0;
        OZAKI_EXTRACT_SLICES(aligned, s1, bs, K_pad * N_pad, N_pad, row, col);
      }
    }
  }
}


/**
 * gemm_fused: Single-launch GEMM over ALL slice pairs.
 *
 * C is kept in fp registers across all pairs — only one global C read
 * (or zero) at the start, and one global C write at the end.
 *
 * Triangular iteration (default): sa in [0..nslices), sb in [sa..nslices)
 * subject to sa + sb <= cutoff.  Off-diagonal pairs compute both (sa,sb)
 * and (sb,sa) sequentially, summing i32 accumulators before scaling.
 *
 * Square iteration (sq=1): sa in [0..nslices), sb in [0..nslices)
 * subject to sa + sb <= cutoff.  Each pair computed individually.
 */
__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void gemm_fused(
  CONSTANT const char* restrict as_base,    /* all slices: [nslices][M_pad][K_pad] */
  CONSTANT const char* restrict bs_base,    /* all slices: [nslices][K_pad][N_pad] */
  CONSTANT const int* restrict expa,        /* [M] per-row max exponent */
  CONSTANT const int* restrict expb,        /* [N] per-col max exponent */
  global real_t* restrict c,                /* [M x N] column-major, ldc */
  int M, int N, int K_pad, int N_pad, int ldc,
  int M_pad,                                /* padded M dimension = slice row stride */
  real_t alpha,
  int nslices,                              /* total number of slices */
  int cutoff,                               /* sa + sb <= cutoff */
  int first_pair,                           /* 1 if beta == 0 (overwrite C) */
  int sq)                                   /* 1: full square, 0: triangle+mirror */
{
  const int ib_idx  = (int)get_group_id(0);
  const int jb_idx  = (int)get_group_id(1);
  const int sg_lid  = (int)get_sub_group_local_id();
  const int sg_id   = (int)get_sub_group_id();
  const int tile_m  = sg_id / NTN;
  const int tile_n  = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  const int nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  const long a_stride = (long)M_pad * K_pad;
  const long b_stride = (long)K_pad * N_pad;
  SINT sa;

  /* Register-resident C accumulators: c_fp[rm*RTN*XMX_M + rn*XMX_M + m] */
  real_t c_fp[RTM * RTN * XMX_M];
  { int ci;
    if (0 != first_pair) {
      UNROLL_FORCE(RTM * RTN * XMX_M)
      for (ci = 0; ci < RTM * RTN * XMX_M; ++ci) c_fp[ci] = ZERO;
    }
    else {
      int rm, rn;
      for (ci = 0; ci < RTM * RTN * XMX_M; ++ci) c_fp[ci] = ZERO;
      UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm) {
        UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn) {
          const int col = nj_base + rn * XMX_N + sg_lid;
          int m_;
          UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) {
            const int r_ = mi_base + rm * XMX_M + m_;
            if (r_ < M && col < N) {
              c_fp[(rm * RTN + rn) * XMX_M + m_] =
                c[(long)col * ldc + r_];
            }
          }
        }
      }
    }
  }

  for (sa = 0; sa < (SINT)nslices; ++sa) {
    const int high_sa = MANT_BITS - (7 * (int)sa);
    const int low_bit_sa = MAX(0, high_sa - 6);
    CONSTANT const char* as_sa = as_base + (long)sa * a_stride;
    CONSTANT const char* bs_sa = bs_base + (long)sa * b_stride;
    const int sb_end_raw = cutoff + 1 - (int)sa;
    const SINT sb_end = (SINT)(sb_end_raw < nslices ? sb_end_raw : nslices);
    SINT sb;

    for (sb = sq ? 0 : sa; sb < sb_end; ++sb) {
      const int high_sb = MANT_BITS - (7 * (int)sb);
      const int low_bit_sb = MAX(0, high_sb - 6);
      CONSTANT const char* as_sb = as_base + (long)sb * a_stride;
      CONSTANT const char* bs_sb = bs_base + (long)sb * b_stride;

      /* (sa, sb) K-loop — unrolled by KU */
      int8 c_acc[RTM * RTN];
      { int ri;
        UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
          c_acc[ri] = (int8)(0);
        }
      }
      { int k;
        for (k = 0; k + (KU - 1) * BK < K_pad; k += KU * BK) {
          int ku;
          OZAKI_PREFETCH_TILED(as_sa, bs_sb, K_pad, N_pad,
                               M, k + KU * BK, mi_base, nj_base);
          UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku) {
            OZAKI_DPAS_TILED(as_sa, bs_sb, K_pad, N_pad,
                             mi_base, nj_base, k + ku * BK, M, c_acc);
          }
        }
        for (; k < K_pad; k += BK) {
          OZAKI_DPAS_TILED(as_sa, bs_sb, K_pad, N_pad,
                           mi_base, nj_base, k, M, c_acc);
        }
      }

      /* For off-diagonal triangle pairs, also compute (sb, sa) */
      if (0 == sq && sa != sb) {
        int8 c_mir[RTM * RTN];
        { int ri;
          UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
            c_mir[ri] = (int8)(0);
          }
        }
        { int k;
          for (k = 0; k + (KU - 1) * BK < K_pad; k += KU * BK) {
            int ku;
            OZAKI_PREFETCH_TILED(as_sb, bs_sa, K_pad, N_pad,
                                 M, k + KU * BK, mi_base, nj_base);
            UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku) {
              OZAKI_DPAS_TILED(as_sb, bs_sa, K_pad, N_pad,
                               mi_base, nj_base, k + ku * BK, M, c_mir);
            }
          }
          for (; k < K_pad; k += BK) {
            OZAKI_DPAS_TILED(as_sb, bs_sa, K_pad, N_pad,
                             mi_base, nj_base, k, M, c_mir);
          }
        }
        { int ri;
          UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
            c_acc[ri] = c_acc[ri] + c_mir[ri];
          }
        }
      }

      /* Scale and accumulate into register C */
      { int rm, rn;
        UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm) {
          UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn) {
            const int idx = rm * RTN + rn;
            const int col = nj_base + rn * XMX_N + sg_lid;
            OZAKI_GEMM_ACCUM(c_acc[idx], expa, expb,
                             c_fp + idx * XMX_M,
                             M, N, mi_base + rm * XMX_M, col,
                             alpha, low_bit_sa, low_bit_sb);
          }
        }
      }
    }
  }

  /* Final write: register C -> global C */
  { int rm, rn;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm) {
      UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn) {
        const int col = nj_base + rn * XMX_N + sg_lid;
        int m_;
        UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) {
          const int r_ = mi_base + rm * XMX_M + m_;
          if (r_ < M && col < N) {
            c[(long)col * ldc + r_] =
              c_fp[(rm * RTN + rn) * XMX_M + m_];
          }
        }
      }
    }
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
