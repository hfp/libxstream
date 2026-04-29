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

#if defined(INTEL) && (2 <= INTEL) && (RTM >= 2) && (RTN >= 2)
# define OZAKI_USE_OCL_KLOOP
#endif

/* Bounds checks are OFF by default for performance.
 * Set -DOZAKI_BOUNDS=1 to enable per-element M/N guards
 * in gemm_fused (exponent caching, C load/store, scaling).
 * OFF requires host-side C buffer padded to tile boundaries. */
#if defined(OZAKI_BOUNDS) && (OZAKI_BOUNDS)
# define OZAKI_IN_BOUNDS(R, M, COL, N) ((R) < (M) && (COL) < (N))
#else
# define OZAKI_IN_BOUNDS(R, M, COL, N) (1)
#endif


/* Composable macros (DBM-style factoring).
 * Each is a do{...}while(0) block for use in kernel functions. */

/* Extract NSLICES int8 digits from an aligned mantissa into DST buffer.
 * DST[s * SS + ROW * RS + COL] = digit(s). */
#define OZAKI_EXTRACT_SLICES(ALIGNED, SIGN, DST, SS, RS, ROW, COL) \
  do { \
    SINT s_; \
    UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) \
    { \
      (DST)[(long)(s_) * (SS) + (long)(ROW) * (RS) + (COL)] = ozaki_slice_digit((ALIGNED), (SIGN), (int)(s_)); \
    } \
  } while (0)

/* Zero NSLICES entries at the given position. */
#define OZAKI_ZERO_SLICES(DST, SS, RS, ROW, COL) \
  do { \
    SINT s_; \
    UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) \
    { \
      (DST)[(long)(s_) * (SS) + (long)(ROW) * (RS) + (COL)] = 0; \
    } \
  } while (0)

/* Scalar accumulator DPAS: individual int8 variables instead of an array.
 * Helps IGC keep accumulators in GRFs instead of lowering to stack.
 * Only for RTM=4 RTN=2 (the hot path). */
#if defined(OZAKI_SCALAR_ACC) && (OZAKI_SCALAR_ACC) && (RTM == 4) && (RTN == 2) && defined(OZAKI_USE_OCL_KLOOP)
# define OZAKI_SC_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, C00, C01, C10, C11, C20, C21, C30, C31) \
    do { \
      ushort8 a_sc_[4]; \
      uint8 b_sc_[2]; \
      intel_sub_group_2d_block_read_8b_32r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)a_sc_); \
      intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)b_sc_); \
      OZAKI_DPAS_ONE_(a_sc_[0], b_sc_[0], C00); \
      OZAKI_DPAS_ONE_(a_sc_[0], b_sc_[1], C01); \
      OZAKI_DPAS_ONE_(a_sc_[1], b_sc_[0], C10); \
      OZAKI_DPAS_ONE_(a_sc_[1], b_sc_[1], C11); \
      OZAKI_DPAS_ONE_(a_sc_[2], b_sc_[0], C20); \
      OZAKI_DPAS_ONE_(a_sc_[2], b_sc_[1], C21); \
      OZAKI_DPAS_ONE_(a_sc_[3], b_sc_[0], C30); \
      OZAKI_DPAS_ONE_(a_sc_[3], b_sc_[1], C31); \
    } while (0)

# define OZAKI_KLOOP_SC(AS, BS, K_PAD_, N_PAD_, M_, MI, NJ, C00, C01, C10, C11, C20, C21, C30, C31) \
    do { \
      int k_l_; \
      for (k_l_ = 0; k_l_ + (KU - 1) * BK < (K_PAD_); k_l_ += KU * BK) { \
        int ku_l_; \
        UNROLL_FORCE(KU) for (ku_l_ = 0; ku_l_ < KU; ++ku_l_) \
        { \
          OZAKI_SC_DPAS(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_ + ku_l_ * BK, M_, C00, C01, C10, C11, C20, C21, C30, C31); \
        } \
      } \
      for (; k_l_ < (K_PAD_); k_l_ += BK) { \
        OZAKI_SC_DPAS(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_, M_, C00, C01, C10, C11, C20, C21, C30, C31); \
      } \
    } while (0)
#endif /* OZAKI_SCALAR_ACC */

/* No-prefetch K-loop: simple DPAS_TILED loop without prefetch messages.
 * Mirrors the asm K-loop structure but in pure OpenCL C builtins. */
#if defined(OZAKI_USE_OCL_KLOOP)
# define OZAKI_KLOOP_OCL(AS, BS, K_PAD_, N_PAD_, M_, MI, NJ, ACC) \
    do { \
      int k_l_; \
      for (k_l_ = 0; k_l_ + (KU - 1) * BK < (K_PAD_); k_l_ += KU * BK) { \
        int ku_l_; \
        UNROLL_FORCE(KU) for (ku_l_ = 0; ku_l_ < KU; ++ku_l_) \
        { \
          OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_ + ku_l_ * BK, M_, ACC); \
        } \
      } \
      for (; k_l_ < (K_PAD_); k_l_ += BK) { \
        OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_, M_, ACC); \
      } \
    } while (0)
#endif

/* K-loop prefetch: opt-in via OZAKI_PREFETCH=1 (default off on PVC). */
#if defined(OZAKI_PREFETCH) && (0 < OZAKI_PREFETCH)
# define OZAKI_KLOOP_PREFETCH(AS, BS, K, N, M, KOFF, MI, NJ) OZAKI_PREFETCH_TILED(AS, BS, K, N, M, KOFF, MI, NJ)
#else
# define OZAKI_KLOOP_PREFETCH(AS, BS, K, N, M, KOFF, MI, NJ)
#endif

/* Full tiled K-loop: KU-unrolled DPAS with optional prefetch, then remainder.
 * AS, BS: slice pointers for this pair.
 * ACC: int8[RTM*RTN] accumulator array (must be pre-zeroed by caller).
 * OZAKI_PREFETCH: opt-in prefetch (default off — hurts PVC perf). */
#define OZAKI_KLOOP(AS, BS, K_PAD_, N_PAD_, M_, MI, NJ, ACC) \
  do { \
    int k_l_; \
    for (k_l_ = 0; k_l_ + (KU - 1) * BK < (K_PAD_); k_l_ += KU * BK) { \
      int ku_l_; \
      OZAKI_KLOOP_PREFETCH(AS, BS, K_PAD_, N_PAD_, M_, k_l_ + KU * BK, MI, NJ); \
      UNROLL_FORCE(KU) for (ku_l_ = 0; ku_l_ < KU; ++ku_l_) \
      { \
        OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_ + ku_l_ * BK, M_, ACC); \
      } \
    } \
    for (; k_l_ < (K_PAD_); k_l_ += BK) { \
      OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, MI, NJ, k_l_, M_, ACC); \
    } \
  } while (0)

/* Scale i32 accumulator + accumulate into register-resident fp C with
 * pre-cached FP exponent scales in registers.
 * EA is a real_t array[XMX_M], EB is a real_t scalar (FP scale factors).
 * PAIR_SCALE = alpha * EXP2I(low_sa + low_sb - 2*MANT_BITS), safe from
 * underflow because preprocessing stores 2^(exp-BIAS) not 2^exp.
 * Avoids re-reading expa/expb from global memory for every pair. */
#define OZAKI_GEMM_ACCUM_CACHED(DOT, EA, EB, C, M, N, MI, COL, PAIR_SCALE) \
  do { \
    union { \
      int8 v_; \
      int a_[8]; \
    } du_c_; \
    int m_c_; \
    du_c_.v_ = (DOT); \
    UNROLL_FORCE(XMX_M) for (m_c_ = 0; m_c_ < XMX_M; ++m_c_) \
    { \
      const int rm_c_ = (MI) + m_c_; \
      if (OZAKI_IN_BOUNDS(rm_c_, (M), (COL), (N))) { \
        const real_t sc_c_ = (PAIR_SCALE) * (EA)[m_c_] * (EB); \
        (C)[m_c_] += (real_t)du_c_.a_[m_c_] * sc_c_; \
      } \
    } \
  } while (0)

/* Tile-by-tile scale+flush: read C tile, accumulate scaled i32 result,
 * write C tile back.  ACC is an int8 array indexed by [rm*RTN+rn]. */
#define OZAKI_SCALE_FLUSH(ACC, C_PTR, LDC, EA_CACHE, EB_CACHE, MI_BASE, NJ_BASE, SG_LID, M, N, PAIR_SCALE) \
  do { \
    int rm_sf_, rn_sf_; \
    UNROLL_FORCE(RTM) for (rm_sf_ = 0; rm_sf_ < RTM; ++rm_sf_) \
    { \
      UNROLL_FORCE(RTN) for (rn_sf_ = 0; rn_sf_ < RTN; ++rn_sf_) \
      { \
        const int idx_sf_ = rm_sf_ * RTN + rn_sf_; \
        const int col_sf_ = (NJ_BASE) + rn_sf_ * XMX_N + (SG_LID); \
        real_t ct_sf_[XMX_M]; \
        int m_sf_; \
        UNROLL_FORCE(XMX_M) for (m_sf_ = 0; m_sf_ < XMX_M; ++m_sf_) \
        { \
          const int r_sf_ = (MI_BASE) + rm_sf_ * XMX_M + m_sf_; \
          ct_sf_[m_sf_] = OZAKI_IN_BOUNDS(r_sf_, (M), col_sf_, (N)) ? (C_PTR)[(long)col_sf_ * (LDC) + r_sf_] : ZERO; \
        } \
        OZAKI_GEMM_ACCUM_CACHED((ACC)[idx_sf_], (EA_CACHE) + rm_sf_ * XMX_M, (EB_CACHE)[rn_sf_], ct_sf_, (M), (N), \
          (MI_BASE) + rm_sf_ * XMX_M, col_sf_, (PAIR_SCALE)); \
        UNROLL_FORCE(XMX_M) for (m_sf_ = 0; m_sf_ < XMX_M; ++m_sf_) \
        { \
          const int r_sf_ = (MI_BASE) + rm_sf_ * XMX_M + m_sf_; \
          if (OZAKI_IN_BOUNDS(r_sf_, (M), col_sf_, (N))) { \
            (C_PTR)[(long)col_sf_ * (LDC) + r_sf_] = ct_sf_[m_sf_]; \
          } \
        } \
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
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_a_dense(CONSTANT const real_t* restrict a, int M, int K, int lda, int transa,
  global char* restrict as, /* [NSLICES * M_pad * K_pad] */
  global real_t* restrict expa, /* [M] per-row FP scale factor = 2^max_exp */
  int K_pad, /* padded K stride (>= 64) */
  int M_pad, /* padded M = nblk_m * BM_PRE */
  global int* restrict slice_occ) /* [NSLICES] per-slice nonzero flag (or NULL) */
{
  const int mi = (int)get_local_id(0);
  const int kk = (int)get_local_id(1);
  const int row = (int)get_group_id(0) * BM_PRE + mi;
  int col;

  local int row_max_exp[BM_PRE];
  local int occ_local[NSLICES];
  if (0 == kk) row_max_exp[mi] = 0;
  if (0 == kk && mi < NSLICES) occ_local[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Pass 1: find max exponent across ALL of K for this row */
  for (col = kk; col < K; col += BK_PRE) {
    if (row < M) {
      int s0;
      short e0;
      uint_repr_t m0;
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      ieee_decompose(a[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&row_max_exp[mi], (int)e0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write global max exponent as FP scale: 2^(max_exp - BIAS).
     * EXP2I arg range: [-(BPM-MANT)..BPM-MANT] i.e. [-1022..1023] for f64. */
  if (0 == kk && row < M) {
    expa[row] = EXP2I(row_max_exp[mi] - (BIAS_PLUS_MANT - MANT_BITS));
  }

  /* Pass 2: compute and store int8 slices using the true max exponent */
  if (row < M) {
    const short max_exp = (short)row_max_exp[mi];
    for (col = kk; col < K; col += BK_PRE) {
      int s1;
      short e1;
      uint_repr_t m1;
      const int idx = transa ? (row * lda + col) : (col * lda + row);
      ieee_decompose(a[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift <= MANT_BITS) ? (m1 >> shift) : 0;
        OZAKI_EXTRACT_SLICES(aligned, s1, as, M_pad * K_pad, K_pad, row, col);
        if (aligned != 0) {
          SINT s_;
          UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) {
            if (0 != ozaki_slice_digit(aligned, s1, (int)s_)) occ_local[s_] = 1;
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (0 == kk && mi < NSLICES && 0 != occ_local[mi]) {
    atomic_or(&slice_occ[mi], 1);
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
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_b_dense(CONSTANT const real_t* restrict b, int N, int K, int ldb, int transb,
  global char* restrict bs, /* [NSLICES * K_pad * N_pad] */
  global real_t* restrict expb, /* [N] per-column FP scale factor = 2^max_exp */
  int K_pad, int N_pad,
  global int* restrict slice_occ) /* [NSLICES] per-slice nonzero flag */
{
  const int nj = (int)get_local_id(0);
  const int kk = (int)get_local_id(1);
  const int col = (int)get_group_id(0) * BN_PRE + nj;
  int row;

  local int col_max_exp[BN_PRE];
  local int occ_local[NSLICES];
  if (0 == kk) col_max_exp[nj] = 0;
  if (0 == kk && nj < NSLICES) occ_local[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Pass 1: find max exponent across ALL of K for this column */
  for (row = kk; row < K; row += BK_PRE) {
    if (col < N) {
      int s0;
      short e0;
      uint_repr_t m0;
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&col_max_exp[nj], (int)e0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Write global max exponent as FP scale: 2^(max_exp - BIAS). */
  if (0 == kk && col < N) {
    expb[col] = EXP2I(col_max_exp[nj] - (BIAS_PLUS_MANT - MANT_BITS));
  }

  /* Pass 2: compute and store int8 slices using the true max exponent */
  if (col < N) {
    const short max_exp = (short)col_max_exp[nj];
    for (row = kk; row < K; row += BK_PRE) {
      int s1;
      short e1;
      uint_repr_t m1;
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift <= MANT_BITS) ? (m1 >> shift) : 0;
        OZAKI_EXTRACT_SLICES(aligned, s1, bs, K_pad * N_pad, N_pad, row, col);
        if (aligned != 0) {
          SINT s_;
          UNROLL_FORCE(NSLICES) for (s_ = 0; s_ < NSLICES; ++s_) {
            if (0 != ozaki_slice_digit(aligned, s1, (int)s_)) occ_local[s_] = 1;
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (0 == kk && nj < NSLICES && 0 != occ_local[nj]) {
    atomic_or(&slice_occ[nj], 1);
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
__attribute__((reqd_work_group_size(SG, NTM* NTN, 1)))
#if defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void gemm_fused(
  CONSTANT const char* restrict as_base, /* all slices: [nslices][M_pad][K_pad] */
  CONSTANT const char* restrict bs_base, /* all slices: [nslices][K_pad][N_pad] */
  CONSTANT const real_t* restrict expa, /* [M] per-row FP scale = 2^exp */
  CONSTANT const real_t* restrict expb, /* [N] per-col FP scale = 2^exp */
  global real_t* restrict c, /* [M x N] column-major, ldc */
  int M, int N, int K_pad, int N_pad, int ldc, int M_pad, /* padded M dimension = slice row stride */
  real_t alpha, int first_pair)
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int sg_lid = (int)LIBXS_SGLID();
  const int sg_id = (int)LIBXS_SGID();
  const int tile_m = sg_id / NTN;
  const int tile_n = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  const int nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  const long a_stride = (long)M_pad * K_pad;
  const long b_stride = (long)K_pad * N_pad;
  SINT sa;

  /* Pre-cache FP exponent scales in registers: avoid re-reading from
   * global memory per pair.  Preprocessing stores 2^(max_exp - BIAS). */
  real_t ea_cache[RTM * XMX_M];
  real_t eb_cache[RTN];
  {
    int rm;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
    {
      int m_;
      UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_)
      {
        const int r_ = mi_base + rm * XMX_M + m_;
        ea_cache[rm * XMX_M + m_] = OZAKI_IN_BOUNDS(r_, M, 0, 1) ? expa[r_] : ZERO;
      }
    }
  }
  {
    int rn;
    UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
    {
      const int col = nj_base + rn * XMX_N + sg_lid;
      eb_cache[rn] = OZAKI_IN_BOUNDS(0, 1, col, N) ? expb[col] : ZERO;
    }
  }

#if !defined(OZAKI_USE_OCL_KLOOP)
  /* Register-resident C accumulators: c_fp[rm*RTN*XMX_M + rn*XMX_M + m] */
  real_t c_fp[RTM * RTN * XMX_M];
  {
    int ci;
    if (0 != first_pair) {
      UNROLL_FORCE(RTM * RTN * XMX_M)
      for (ci = 0; ci < RTM * RTN * XMX_M; ++ci) c_fp[ci] = ZERO;
    }
    else {
      int rm, rn;
      for (ci = 0; ci < RTM * RTN * XMX_M; ++ci) c_fp[ci] = ZERO;
      UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
      {
        UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
        {
          const int col = nj_base + rn * XMX_N + sg_lid;
          int m_;
          UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_)
          {
            const int r_ = mi_base + rm * XMX_M + m_;
            if (OZAKI_IN_BOUNDS(r_, M, col, N)) {
              c_fp[(rm * RTN + rn) * XMX_M + m_] = c[(long)col * ldc + r_];
            }
          }
        }
      }
    }
  }
#endif

    for (sa = 0; sa < (SINT)NSLICES && (int)sa <= OZAKI_CUTOFF; ++sa) {
      const int high_sa = MANT_BITS - (7 * (int)sa);
      const int low_bit_sa = MAX(0, high_sa - 6);
      CONSTANT const char* as_sa = as_base + (long)sa * a_stride;
      CONSTANT const char* bs_sa = bs_base + (long)sa * b_stride;
      const int sb_end_raw = OZAKI_CUTOFF + 1 - (int)sa;
      const SINT sb_end = (SINT)(sb_end_raw < NSLICES ? sb_end_raw : NSLICES);
      SINT sb;

      for (sb = OZAKI_SQ ? 0 : sa; sb < sb_end; ++sb) {
        const int high_sb = MANT_BITS - (7 * (int)sb);
        const int low_bit_sb = MAX(0, high_sb - 6);
        const real_t pair_scale = alpha * EXP2I(low_bit_sa + low_bit_sb - 2 * MANT_BITS);
        CONSTANT const char* as_sb = as_base + (long)sb * a_stride;
        CONSTANT const char* bs_sb = bs_base + (long)sb * b_stride;

        /* (sa, sb) K-loop - unrolled by KU */
#if defined(OZAKI_SCALAR_ACC) && (OZAKI_SCALAR_ACC) && (RTM == 4) && (RTN == 2) && defined(OZAKI_USE_OCL_KLOOP)
        /* Scalar accumulator K-loop. */
        {
          int8 sc00 = (int8)(0), sc01 = (int8)(0);
          int8 sc10 = (int8)(0), sc11 = (int8)(0);
          int8 sc20 = (int8)(0), sc21 = (int8)(0);
          int8 sc30 = (int8)(0), sc31 = (int8)(0);
          int8 c_acc_sc[RTM * RTN];
          OZAKI_KLOOP_SC(as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base, sc00, sc01, sc10, sc11, sc20, sc21, sc30, sc31);
          if (0 == OZAKI_SQ && sa != sb) {
            int8 sm00 = (int8)(0), sm01 = (int8)(0);
            int8 sm10 = (int8)(0), sm11 = (int8)(0);
            int8 sm20 = (int8)(0), sm21 = (int8)(0);
            int8 sm30 = (int8)(0), sm31 = (int8)(0);
            OZAKI_KLOOP_SC(as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base, sm00, sm01, sm10, sm11, sm20, sm21, sm30, sm31);
            sc00 = sc00 + sm00;
            sc01 = sc01 + sm01;
            sc10 = sc10 + sm10;
            sc11 = sc11 + sm11;
            sc20 = sc20 + sm20;
            sc21 = sc21 + sm21;
            sc30 = sc30 + sm30;
            sc31 = sc31 + sm31;
          }
          c_acc_sc[0] = sc00;
          c_acc_sc[1] = sc01;
          c_acc_sc[2] = sc10;
          c_acc_sc[3] = sc11;
          c_acc_sc[4] = sc20;
          c_acc_sc[5] = sc21;
          c_acc_sc[6] = sc30;
          c_acc_sc[7] = sc31;
          OZAKI_SCALE_FLUSH(c_acc_sc, c, ldc, ea_cache, eb_cache, mi_base, nj_base, sg_lid, M, N, pair_scale);
        }
#elif defined(OZAKI_USE_OCL_KLOOP)
        /* OCL K-loop: per-pair compute + scale+flush. */
        {
          int8 c_acc[RTM * RTN];
          {
            int ri;
            UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
            {
              c_acc[ri] = (int8)(0);
            }
          }
          OZAKI_KLOOP_OCL(as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base, c_acc);
          if (0 == OZAKI_SQ && sa != sb) {
            int8 c_mir[RTM * RTN];
            {
              int ri;
              UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
              {
                c_mir[ri] = (int8)(0);
              }
            }
            OZAKI_KLOOP_OCL(as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base, c_mir);
            {
              int ri;
              UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
              {
                c_acc[ri] = c_acc[ri] + c_mir[ri];
              }
            }
          }
          OZAKI_SCALE_FLUSH(c_acc, c, ldc, ea_cache, eb_cache, mi_base, nj_base, sg_lid, M, N, pair_scale);
        }
#else
        {
          int8 c_acc[RTM * RTN];
          {
            int ri;
            UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
            {
              c_acc[ri] = (int8)(0);
            }
          }
          OZAKI_KLOOP(as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base, c_acc);
          if (0 == OZAKI_SQ && sa != sb) {
            int8 c_mir[RTM * RTN];
            {
              int ri;
              UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
              {
                c_mir[ri] = (int8)(0);
              }
            }
            OZAKI_KLOOP(as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base, c_mir);
            {
              int ri;
              UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri)
              {
                c_acc[ri] = c_acc[ri] + c_mir[ri];
              }
            }
          }
          /* Scale and accumulate into register C (cached exponents) */
          {
            int rm, rn;
            UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
            {
              UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
              {
                const int idx = rm * RTN + rn;
                const int col = nj_base + rn * XMX_N + sg_lid;
                OZAKI_GEMM_ACCUM_CACHED(
                  c_acc[idx], ea_cache + rm * XMX_M, eb_cache[rn], c_fp + idx * XMX_M, M, N, mi_base + rm * XMX_M, col, pair_scale);
              }
            }
          }
        }
#endif
      }
    }

#if !defined(OZAKI_USE_OCL_KLOOP)
  /* Final write: register C -> global C */
  {
    int rm, rn;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
    {
      UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
      {
        const int col = nj_base + rn * XMX_N + sg_lid;
        int m_;
        UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_)
        {
          const int r_ = mi_base + rm * XMX_M + m_;
          if (OZAKI_IN_BOUNDS(r_, M, col, N)) {
            c[(long)col * ldc + r_] = c_fp[(rm * RTN + rn) * XMX_M + m_];
          }
        }
      }
    }
  }
#endif
}
