/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_COMMON_CL
#define OZAKI_COMMON_CL

/* Shared primitives for all Ozaki kernel files.
 *
 * Provides:
 *   CONSTANT            - address-space qualifier (global or __constant)
 *   OZAKI_DPAS          - one DPAS step (2D block I/O + MAD)
 *   ieee_decompose()    - IEEE-754 -> (sign, biased exponent, mantissa)
 *   ozaki_slice_digit() - extract a 7-bit signed digit from aligned mantissa
 *
 * OZAKI_U8 (compile-time):
 *   0 or undefined: signed i8 DPAS (intel_sub_group_i8_i8_matrix_mad_k32)
 *   1:             unsigned u8 DPAS (intel_sub_group_u8_u8_matrix_mad_k32)
 *   Scheme 2 (CRT) defaults to u8 for larger moduli (<=256 vs <=128).
 *   Scheme 1 (slicing) always uses i8 (signed slice digits).
 */

#if !defined(CONSTANT)
# define CONSTANT global
#endif

/* Small integer type for loop counters (states value range) */
#if !defined(SINT)
# define SINT signed char
#endif

/* Register tiling: RTM x RTN sub-tiles per sub-group.
 * Each sub-group computes (RTM*XMX_M) x (RTN*XMX_N) output elements,
 * issuing RTM*RTN DPAS instructions per K-step.
 * RTM=1, RTN=1 reproduces the non-tiled baseline (1 DPAS per K-step).
 * Higher values (e.g. RTM=4,RTN=4) saturate the systolic pipeline and
 * require 256-GRF mode (LIBXSTREAM_BIGGRF=1). */
#if !defined(RTM)
# define RTM 1
#endif
#if !defined(RTN)
# define RTN 1
#endif

/* DPAS repeat count: 8 (default) or 4 (split for scheduling). */
#if !defined(RC)
# define RC 8
#endif

/* DPAS sub-tile dimensions (fixed for PVC XMX) */
#define XMX_M 8
#define XMX_N 16

/* Integer power of two via bit manipulation: 2^N exactly.
 * Avoids FP transcendental — one integer add, one shift, one bitcast. */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
# define EXP2I(N) as_double((long)((N) + 1023) << 52)
#else
# define EXP2I(N) as_float(((N) + 127) << 23)
#endif

/* One DPAS step: 8x32 A tile * 32x16 B tile -> 8x16 int32 accumulator.
 * Each work-item holds 8 rows; the column is get_sub_group_local_id().
 *
 * XMX path (OZAKI_U8=1 — unsigned, default for CRT):
 *   int8 intel_sub_group_u8_u8_matrix_mad_k32(ushort8 a, uint8 b, int8 acc)
 * XMX path (OZAKI_U8=0 — signed, default for slicing):
 *   int8 intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc)
 *   A tile: 8 rows x 32 cols  (read as ushort8 via 2D block read)
 *   B tile: 32 rows x 16 cols (read with VNNI transform via 2D block read)
 *   C tile: 8 x 16 int32      (int8 per WI — 8 rows, sg_lid selects column)
 *   2D block I/O requires SG=16 and surface pitch >= 64 bytes.
 *
 * Scalar path (USE_XMX not defined):
 *   Same 8x32x16 tile contract via explicit loops.
 *   Allows the GEMM kernels to run on hardware without DPAS/2D block I/O. */
#if defined(USE_XMX) && (0 < USE_XMX)

/* Prefetch next K-step's A and B tiles into cache.
 * 2D block prefetch with .ca.ca hints — writes to null, no register cost.
 * OOB prefetches are silently clamped by the hardware. */
# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI) \
    intel_sub_group_2d_block_prefetch_8b_8r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ) \
    intel_sub_group_2d_block_prefetch_8b_32r16x1c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))

# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_MAD_K32_8_(A, B, ACC) intel_sub_group_u8_u8_matrix_mad_k32(A, B, ACC)
# define OZAKI_MAD_K32_4_(A, B, ACC) intel_sub_group_u8_u8_matrix_mad_k32(A, B, ACC)
# else
# define OZAKI_MAD_K32_8_(A, B, ACC) intel_sub_group_i8_i8_matrix_mad_k32(as_short8(A), as_int8(B), ACC)
# define OZAKI_MAD_K32_4_(A, B, ACC) intel_sub_group_i8_i8_matrix_mad_k32(as_short4(A), as_int8(B), ACC)
# endif

# if (8 == RC)
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_blk_; \
        uint8 b_blk_; \
        intel_sub_group_2d_block_read_8b_8r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)&a_blk_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)&b_blk_); \
        (ACC) = OZAKI_MAD_K32_8_(a_blk_, b_blk_, (ACC)); \
      } while (0)
# elif (4 == RC)
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_blk_; \
        uint8 b_blk_; \
        int4 lo_, hi_; \
        intel_sub_group_2d_block_read_8b_8r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)&a_blk_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)&b_blk_); \
        lo_ = (ACC).lo; \
        hi_ = (ACC).hi; \
        lo_ = OZAKI_MAD_K32_4_(a_blk_.lo, b_blk_, lo_); \
        hi_ = OZAKI_MAD_K32_4_(a_blk_.hi, b_blk_, hi_); \
        (ACC) = (int8)(lo_, hi_); \
      } while (0)
# endif

/* Single-tile DPAS from pre-loaded A (ushort8) and B (uint8).
 * RC=8: one MAD(8rows). RC=4: split into two MAD(4rows). */
# if (8 == RC)
# define OZAKI_DPAS_ONE_(A, B, ACC) (ACC) = OZAKI_MAD_K32_8_(A, B, (ACC))
# elif (4 == RC)
# define OZAKI_DPAS_ONE_(A, B, ACC) \
      do { \
        int4 lo1_ = (ACC).lo, hi1_ = (ACC).hi; \
        lo1_ = OZAKI_MAD_K32_4_((A).lo, B, lo1_); \
        hi1_ = OZAKI_MAD_K32_4_((A).hi, B, hi1_); \
        (ACC) = (int8)(lo1_, hi1_); \
      } while (0)
# endif

/* Tiled DPAS: RTM x RTN sub-tiles per sub-group.
 * Loads RTM A-strips and RTN B-strips, then issues RTM*RTN DPAS.
 * ACC is an int8 array of size RTM*RTN, indexed [rm * RTN + rn].
 *
 * Coalesced-load specializations use wider 2D block reads to reduce
 * the number of load messages per K-step (matching TinyTC codegen):
 *   A: _8b_{RTM*8}r32x1c loads all RTM subtiles in one message.
 *   B: _transform_8b_32r16x{RTN}c loads all RTN subtiles in one message.
 * Fallback: per-subtile loops (generic for any RTM/RTN). */
# if (RTM == 4) && (RTN == 2)
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_rt_[4]; \
        uint8 b_rt_[2]; \
        intel_sub_group_2d_block_read_8b_32r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)a_rt_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)b_rt_); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[0], (ACC)[2]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[1], (ACC)[3]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[0], (ACC)[4]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[1], (ACC)[5]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[0], (ACC)[6]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[1], (ACC)[7]); \
      } while (0)
# elif (RTM == 4) && (RTN == 4)
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_rt_[4]; \
        uint8 b_rt_[4]; \
        intel_sub_group_2d_block_read_8b_32r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)a_rt_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x4c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)b_rt_); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[2], (ACC)[2]); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[3], (ACC)[3]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[0], (ACC)[4]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[1], (ACC)[5]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[2], (ACC)[6]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[3], (ACC)[7]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[0], (ACC)[8]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[1], (ACC)[9]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[2], (ACC)[10]); \
        OZAKI_DPAS_ONE_(a_rt_[2], b_rt_[3], (ACC)[11]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[0], (ACC)[12]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[1], (ACC)[13]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[2], (ACC)[14]); \
        OZAKI_DPAS_ONE_(a_rt_[3], b_rt_[3], (ACC)[15]); \
      } while (0)
# elif (RTM == 2) && (RTN == 2)
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_rt_[2]; \
        uint8 b_rt_[2]; \
        intel_sub_group_2d_block_read_8b_16r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)a_rt_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)b_rt_); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE_(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[0], (ACC)[2]); \
        OZAKI_DPAS_ONE_(a_rt_[1], b_rt_[1], (ACC)[3]); \
      } while (0)
# else
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_rt_[RTM]; \
        uint8 b_rt_[RTN]; \
        int rm_t_, rn_t_; \
        UNROLL_FORCE(RTM) for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) \
        { \
          intel_sub_group_2d_block_read_8b_8r32x1c( \
            (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI) + rm_t_ * XMX_M), (private ushort*)&a_rt_[rm_t_]); \
        } \
        UNROLL_FORCE(RTN) for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) \
        { \
          intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
            (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ) + rn_t_ * XMX_N, (KOFF)), (private uint*)&b_rt_[rn_t_]); \
        } \
        UNROLL_FORCE(RTM) for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) \
        { \
          UNROLL_FORCE(RTN) for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) \
          { \
            OZAKI_DPAS_ONE_(a_rt_[rm_t_], b_rt_[rn_t_], (ACC)[rm_t_ * RTN + rn_t_]); \
          } \
        } \
      } while (0)
# endif

/* Split load/compute for software pipelining.
 * OZAKI_LOAD_TILED: load A/B tiles into caller-supplied arrays.
 * OZAKI_COMPUTE_TILED: issue DPAS from pre-loaded tiles. */
# if (RTM == 4)
# define OZAKI_LOAD_A_TILED_(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
      intel_sub_group_2d_block_read_8b_32r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)(A_BUF))
# elif (RTM == 2)
# define OZAKI_LOAD_A_TILED_(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
      intel_sub_group_2d_block_read_8b_16r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)(A_BUF))
# else
# define OZAKI_LOAD_A_TILED_(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
      do { \
        int rl_m_; \
        UNROLL_FORCE(RTM) for (rl_m_ = 0; rl_m_ < RTM; ++rl_m_) \
        { \
          intel_sub_group_2d_block_read_8b_8r32x1c( \
            (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI) + rl_m_ * XMX_M), (private ushort*)&(A_BUF)[rl_m_]); \
        } \
      } while (0)
# endif
# if (RTN == 4)
# define OZAKI_LOAD_B_TILED_(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
      intel_sub_group_2d_block_read_transform_8b_32r16x4c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)(B_BUF))
# elif (RTN == 2)
# define OZAKI_LOAD_B_TILED_(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
      intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)(B_BUF))
# else
# define OZAKI_LOAD_B_TILED_(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
      do { \
        int rl_n_; \
        UNROLL_FORCE(RTN) for (rl_n_ = 0; rl_n_ < RTN; ++rl_n_) \
        { \
          intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
            (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ) + rl_n_ * XMX_N, (KOFF)), (private uint*)&(B_BUF)[rl_n_]); \
        } \
      } while (0)
# endif
# define OZAKI_LOAD_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, A_BUF, B_BUF) \
    do { \
      OZAKI_LOAD_A_TILED_(AS, K_PAD, M_HT, MI, KOFF, A_BUF); \
      OZAKI_LOAD_B_TILED_(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF); \
    } while (0)

# define OZAKI_COMPUTE_TILED(A_BUF, B_BUF, ACC) \
    do { \
      int rc_m_, rc_n_; \
      UNROLL_FORCE(RTM) for (rc_m_ = 0; rc_m_ < RTM; ++rc_m_) \
      { \
        UNROLL_FORCE(RTN) for (rc_n_ = 0; rc_n_ < RTN; ++rc_n_) \
        { \
          OZAKI_DPAS_ONE_((A_BUF)[rc_m_], (B_BUF)[rc_n_], (ACC)[rc_m_ * RTN + rc_n_]); \
        } \
      } \
    } while (0)

/* Tiled prefetch: prefetch next K-step for all RTM A and RTN B tiles.
 * Coalesced variants match the wider loads above. */
# if (RTM == 4)
# define OZAKI_PREFETCH_A_TILED_(AS, K_PAD, M_HT, KOFF, MI) \
      intel_sub_group_2d_block_prefetch_8b_32r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# elif (RTM == 2)
# define OZAKI_PREFETCH_A_TILED_(AS, K_PAD, M_HT, KOFF, MI) \
      intel_sub_group_2d_block_prefetch_8b_16r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# else
# define OZAKI_PREFETCH_A_TILED_(AS, K_PAD, M_HT, KOFF, MI) \
      do { \
        int rp_m_; \
        UNROLL_FORCE(RTM) for (rp_m_ = 0; rp_m_ < RTM; ++rp_m_) \
        { \
          OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, (MI) + rp_m_ * XMX_M); \
        } \
      } while (0)
# endif
# if (RTN == 4)
# define OZAKI_PREFETCH_B_TILED_(BS, N_PAD, K_PAD, KOFF, NJ) \
      intel_sub_group_2d_block_prefetch_8b_32r16x4c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))
# elif (RTN == 2)
# define OZAKI_PREFETCH_B_TILED_(BS, N_PAD, K_PAD, KOFF, NJ) \
      intel_sub_group_2d_block_prefetch_8b_32r16x2c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))
# else
# define OZAKI_PREFETCH_B_TILED_(BS, N_PAD, K_PAD, KOFF, NJ) \
      do { \
        int rp_n_; \
        UNROLL_FORCE(RTN) for (rp_n_ = 0; rp_n_ < RTN; ++rp_n_) \
        { \
          OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, (NJ) + rp_n_ * XMX_N); \
        } \
      } while (0)
# endif
# define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ) \
    do { \
      OZAKI_PREFETCH_A_TILED_(AS, K_PAD, M_HT, KOFF, MI); \
      OZAKI_PREFETCH_B_TILED_(BS, N_PAD, K_PAD, KOFF, NJ); \
    } while (0)
#else
# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI)
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ)
# define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ)
# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_SCALAR_BYTE_T_ uchar
# else
# define OZAKI_SCALAR_BYTE_T_ char
# endif
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int col_ = (NJ) + (int)get_sub_group_local_id(); \
      union { \
        int8 v_; \
        int a_[8]; \
      } u_; \
      int m_; \
      u_.v_ = (ACC); \
      for (m_ = 0; m_ < 8; ++m_) { \
        int k_; \
        for (k_ = 0; k_ < 32; ++k_) { \
          u_.a_[m_] += (int)((CONSTANT const OZAKI_SCALAR_BYTE_T_*)(AS))[(long)((MI) + m_) * (K_PAD) + (KOFF) + k_] * \
                       (int)((CONSTANT const OZAKI_SCALAR_BYTE_T_*)(BS))[(long)((KOFF) + k_) * (N_PAD) + col_]; \
        } \
      } \
      (ACC) = u_.v_; \
    } while (0)

/* Scalar DPAS_TILED: loop over RTM x RTN sub-tiles using scalar DPAS. */
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      int rm_t_, rn_t_; \
      for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
        for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
          OZAKI_DPAS(AS, BS, K_PAD, N_PAD, (MI) + rm_t_ * XMX_M, (NJ) + rn_t_ * XMX_N, KOFF, M_HT, (ACC)[rm_t_ * RTN + rn_t_]); \
        } \
      } \
    } while (0)
#endif


/* Decompose an IEEE-754 value into sign, biased exponent, and implicit-1 mantissa.
 * Zero, subnormal, Inf, and NaN inputs yield exp=0, mant=0.
 * real_t, uint_repr_t, EXP_MASK, and AS_UINT come from libxstream_common.h. */
inline void ieee_decompose(real_t val, int* sign, short* exp, uint_repr_t* mant)
{
  const uint_repr_t bits = AS_UINT(val);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  *sign = (int)(bits >> 63);
  *exp = (short)((bits >> 52) & EXP_MASK);
  *mant = (bits & 0x000FFFFFFFFFFFFFUL) | 0x0010000000000000UL;
#else
  *sign = (int)(bits >> 31);
  *exp = (short)((bits >> 23) & EXP_MASK);
  *mant = (bits & 0x007FFFFFU) | 0x00800000U;
#endif
  if (0 == *exp || *exp == (short)EXP_MASK) {
    *mant = 0;
    *exp = 0;
  }
}

/* Extract a 7-bit signed digit from an aligned mantissa for slice index S.
 * The mantissa ALIGNED is already right-shifted by (max_exp - elem_exp).
 * Returns a signed char: the digit with sign applied if SIGN != 0.
 * MANT_BITS must be defined by the including file. */
#if defined(MANT_BITS)
inline char ozaki_slice_digit(uint_repr_t aligned, int sign, int s)
{
  const int high = MANT_BITS - (7 * s);
  const int low = MAX(0, high - 6);
  const int width = high - low + 1;
  char digit = 0;
  if (width > 0 && high >= 0) {
    digit = (char)((aligned >> low) & ((1U << width) - 1U));
  }
  if (sign) digit = -digit;
  return digit;
}
#endif /*defined(MANT_BITS)*/


/**
 * scale_beta: Prescale C by beta before accumulation.
 *
 * Work-group: (BM_PRE, 1, 1).
 * Dispatch: global = (ceil(M, BM_PRE) * BM_PRE, N, 1).
 */
#if defined(BM_PRE)
__attribute__((reqd_work_group_size(BM_PRE, 1, 1))) kernel void scale_beta(
  global real_t* restrict c, int M, int N, int ldc, real_t beta)
{
  const int row = (int)get_global_id(0);
  const int col = (int)get_global_id(1);
  if (row < M && col < N) {
    c[col * ldc + row] *= beta;
  }
}
#endif /*defined(BM_PRE)*/

#endif /*OZAKI_COMMON_CL*/
