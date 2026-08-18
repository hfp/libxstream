/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_COMMON_CL
#define OZAKI_COMMON_CL

#include "../../../libxstream/opencl/libxstream_common.h"

/**
 * Shared primitives for all Ozaki kernel files.
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

/**
 * PTX state space matching CONSTANT, for inline asm that dereferences a
 * CONSTANT-qualified pointer. CONSTANT is supplied by the host (-DCONSTANT=)
 * and is "constant" when the operand fits constant memory (see
 * libxstream_opencl_use_cmem_size), otherwise "global". Inline asm must not
 * hard-code .global: the mismatch compiles silently and reads the wrong state
 * space. The .nc (non-coherent) qualifier is only valid on .global.
 */
#define OZAKI_CONSTANT_IS_global 1
#define OZAKI_CONSTANT_IS_constant 0
#if CAT(OZAKI_CONSTANT_IS_, CONSTANT)
# define OZAKI_PTX_LD_V4 "ld.global.nc.v4.u32"
#else
# define OZAKI_PTX_LD_V4 "ld.const.v4.u32"
#endif

/**
 * B storage layout for the preprocessed residue/slice matrices.
 *
 * Default (Intel DPAS, scalar): plain K-major [K_pad][N_pad]. The DPAS path
 * relies on this because the VNNI interleave is applied by the hardware on
 * read (2d_block_read_transform).
 *
 * OZAKI_BVNNI (NVIDIA dp4a): pre-interleaved so the 4 K-values a single dp4a
 * consumes are adjacent in memory. A column's 4-value group becomes one
 * aligned uint, turning the 4-way strided gather into a single load.
 * Layout: [K_pad/4][N_pad][4], i.e. quad-of-K major, then column, then the
 * K-phase within the quad. Producer and consumer must agree, so both go
 * through OZAKI_IDX_BS().
 *
 * OZAKI_BKMAJOR (NVIDIA warp-group MMA): fully transposed [N_pad][K_pad], i.e.
 * a column's K-values are contiguous. Required by wgmma, which sources both
 * operands from shared memory and wants them K-major, so a staging thread can
 * move 16 bytes of one column with a single load. It also serves the older
 * paths better than the interleave does: a b-fragment's two registers land 16
 * bytes apart in one segment instead of 4*N_pad apart in two, and a dp4a
 * column becomes 8 consecutive uints.
 */
#if defined(OZAKI_BKMAJOR) && (OZAKI_BKMAJOR)
# define OZAKI_IDX_BS(ROW, COL, N_PAD, K_PAD) ((long)(COL) * (K_PAD) + (ROW))
#elif defined(OZAKI_BVNNI) && (OZAKI_BVNNI)
# define OZAKI_IDX_BS(ROW, COL, N_PAD, K_PAD) \
    ((((long)(ROW) >> 2) * (N_PAD) + (COL)) * 4 + ((ROW) & 3))
#else
# define OZAKI_IDX_BS(ROW, COL, N_PAD, K_PAD) ((long)(ROW) * (N_PAD) + (COL))
#endif

/* Small integer type for loop counters (states value range) */
#if !defined(SINT)
# define SINT signed char
#endif

/**
 * Register tiling: RTM x RTN sub-tiles per sub-group.
 * Each sub-group computes (RTM*XMX_M) x (RTN*XMX_N) output elements,
 * issuing RTM*RTN DPAS instructions per K-step.
 * RTM=1, RTN=1 reproduces the non-tiled baseline (1 DPAS per K-step).
 * Higher values (e.g. RTM=4,RTN=4) saturate the systolic pipeline and
 * require 256-GRF mode (LIBXSTREAM_BIGGRF=1).
 */
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

/**
 * Hardware sub-tile dimensions.
 * Intel DPAS (PVC XMX):  8 rows x 16 cols, K=32  (SG=16)
 * NVIDIA dp4a:           8 rows x 16 cols, K=32  (SG=16)
 * NVIDIA MMA m16n8k32:  16 rows x  8 cols, K=32  (SG=32)
 * Scalar fallback:       8 rows x 16 cols, K=32  (SG=16)
 */
#if defined(NV_MMA) && (NV_MMA)
# define XMX_M 16
# define XMX_N 8
# define XMX_K 32
#else
# define XMX_M 8
# define XMX_N 16
#endif

/**
 * Accumulator fragment: the XMX_M * XMX_N / SG outputs of one sub-tile that a
 * single work-item owns, and where each of them lands.
 *
 * DPAS, dp4a and the scalar fallback give a lane one whole column: XMX_FRAG
 * equals XMX_M, element f is row f, and the column is the lane itself. MMA
 * splits the tile differently - a lane holds 2 rows x 2 cols at (lane/4,
 * (lane%4)*2), the second row +8 and the second column +1 - so element f is
 * row (f/2)*8 and column (f%2), both offset by the lane.
 *
 * Everything after the K-loop (mod-reduce, residue strides, exponent caching,
 * C store) is expressed in XMX_FRAG and this mapping, so one body serves both
 * layouts and the layout is stated exactly once.
 */
#if defined(NV_MMA) && (NV_MMA)
# define XMX_FRAG 4
# define OZAKI_ACC_T int4
# define OZAKI_ACC_ZERO ((int4)(0))
# define OZAKI_FRAG_ROW(F, LANE) (((F) / 2) * 8 + (int)(LANE) / 4)
# define OZAKI_FRAG_NCOL 2
# define OZAKI_FRAG_COLIDX(F) ((F) & 1)
# define OZAKI_FRAG_COL(CI, LANE) (((int)(LANE) % 4) * 2 + (CI))
#else
# define XMX_FRAG 8
# define OZAKI_ACC_T int8
# define OZAKI_ACC_ZERO ((int8)(0))
# define OZAKI_FRAG_ROW(F, LANE) (F)
# define OZAKI_FRAG_NCOL 1
# define OZAKI_FRAG_COLIDX(F) 0
# define OZAKI_FRAG_COL(CI, LANE) ((int)(LANE))
#endif

/* Reinterpret an accumulator fragment as XMX_FRAG scalars. */
#define OZAKI_ACC_UNION(NAME) \
  union { \
    OZAKI_ACC_T v_; \
    int a_[XMX_FRAG]; \
  } NAME


/**
 * One DPAS step: 8x32 A tile * 32x16 B tile -> 8x16 int32 accumulator.
 * Each work-item holds 8 rows; the column is get_sub_group_local_id().
 *
 * XMX path (OZAKI_U8=1 - unsigned, default for CRT):
 *   int8 intel_sub_group_u8_u8_matrix_mad_k32(ushort8 a, uint8 b, int8 acc)
 * XMX path (OZAKI_U8=0 - signed, default for slicing):
 *   int8 intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc)
 *   A tile: 8 rows x 32 cols  (read as ushort8 via 2D block read)
 *   B tile: 32 rows x 16 cols (read with VNNI transform via 2D block read)
 *   C tile: 8 x 16 int32      (int8 per WI - 8 rows, sg_lid selects column)
 *   2D block I/O requires SG=16 and surface pitch >= 64 bytes.
 *
 * Scalar path (INTEL < 2):
 *   Same 8x32x16 tile contract via explicit loops.
 *   Allows the GEMM kernels to run on hardware without DPAS/2D block I/O.
 */
#if defined(INTEL) && (2 <= INTEL)

/**
 * Prefetch next K-step's A and B tiles into cache.
 * 2D block prefetch with .ca.ca hints - writes to null, no register cost.
 * OOB prefetches are silently clamped by the hardware.
 */
# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI) \
    intel_sub_group_2d_block_prefetch_8b_8r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ) \
    intel_sub_group_2d_block_prefetch_8b_32r16x1c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))

# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_MAD_K32_8(A, B, ACC) intel_sub_group_u8_u8_matrix_mad_k32(A, B, ACC)
# define OZAKI_MAD_K32_4(A, B, ACC) intel_sub_group_u8_u8_matrix_mad_k32(A, B, ACC)
# else
# define OZAKI_MAD_K32_8(A, B, ACC) intel_sub_group_i8_i8_matrix_mad_k32(as_short8(A), as_int8(B), ACC)
# define OZAKI_MAD_K32_4(A, B, ACC) intel_sub_group_i8_i8_matrix_mad_k32(as_short4(A), as_int8(B), ACC)
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
        (ACC) = OZAKI_MAD_K32_8(a_blk_, b_blk_, (ACC)); \
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
        lo_ = OZAKI_MAD_K32_4(a_blk_.lo, b_blk_, lo_); \
        hi_ = OZAKI_MAD_K32_4(a_blk_.hi, b_blk_, hi_); \
        (ACC) = (int8)(lo_, hi_); \
      } while (0)
# endif

/**
 * Single-tile DPAS from pre-loaded A (ushort8) and B (uint8).
 * RC=8: one MAD(8rows). RC=4: split into two MAD(4rows).
 */
# if (8 == RC)
# define OZAKI_DPAS_ONE(A, B, ACC) (ACC) = OZAKI_MAD_K32_8(A, B, (ACC))
# elif (4 == RC)
# define OZAKI_DPAS_ONE(A, B, ACC) \
      do { \
        int4 lo1_ = (ACC).lo, hi1_ = (ACC).hi; \
        lo1_ = OZAKI_MAD_K32_4((A).lo, B, lo1_); \
        hi1_ = OZAKI_MAD_K32_4((A).hi, B, hi1_); \
        (ACC) = (int8)(lo1_, hi1_); \
      } while (0)
# endif

/**
 * Tiled DPAS: RTM x RTN sub-tiles per sub-group.
 * Loads RTM A-strips and RTN B-strips, then issues RTM*RTN DPAS.
 * ACC is an int8 array of size RTM*RTN, indexed [rm * RTN + rn].
 *
 * Coalesced-load specializations use wider 2D block reads to reduce
 * the number of load messages per K-step:
 *   A: _8b_{RTM*8}r32x1c loads all RTM subtiles in one message.
 *   B: _transform_8b_32r16x{RTN}c loads all RTN subtiles in one message.
 * Fallback: per-subtile loops (generic for any RTM/RTN).
 */
# if (RTM == 4) && (RTN == 2)
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
      do { \
        ushort8 a_rt_[4]; \
        uint8 b_rt_[2]; \
        intel_sub_group_2d_block_read_8b_32r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)a_rt_); \
        intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)b_rt_); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[0], (ACC)[2]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[1], (ACC)[3]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[0], (ACC)[4]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[1], (ACC)[5]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[0], (ACC)[6]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[1], (ACC)[7]); \
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
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[2], (ACC)[2]); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[3], (ACC)[3]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[0], (ACC)[4]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[1], (ACC)[5]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[2], (ACC)[6]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[3], (ACC)[7]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[0], (ACC)[8]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[1], (ACC)[9]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[2], (ACC)[10]); \
        OZAKI_DPAS_ONE(a_rt_[2], b_rt_[3], (ACC)[11]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[0], (ACC)[12]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[1], (ACC)[13]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[2], (ACC)[14]); \
        OZAKI_DPAS_ONE(a_rt_[3], b_rt_[3], (ACC)[15]); \
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
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[0], (ACC)[0]); \
        OZAKI_DPAS_ONE(a_rt_[0], b_rt_[1], (ACC)[1]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[0], (ACC)[2]); \
        OZAKI_DPAS_ONE(a_rt_[1], b_rt_[1], (ACC)[3]); \
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
            OZAKI_DPAS_ONE(a_rt_[rm_t_], b_rt_[rn_t_], (ACC)[rm_t_ * RTN + rn_t_]); \
          } \
        } \
      } while (0)
# endif

/**
 * Split load/compute for software pipelining.
 * OZAKI_LOAD_TILED: load A/B tiles into caller-supplied arrays.
 * OZAKI_COMPUTE_TILED: issue DPAS from pre-loaded tiles.
 */
# if (RTM == 4)
# define OZAKI_LOAD_A_TILED(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
      intel_sub_group_2d_block_read_8b_32r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)(A_BUF))
# elif (RTM == 2)
# define OZAKI_LOAD_A_TILED(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
      intel_sub_group_2d_block_read_8b_16r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)), (private ushort*)(A_BUF))
# else
# define OZAKI_LOAD_A_TILED(AS, K_PAD, M_HT, MI, KOFF, A_BUF) \
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
# define OZAKI_LOAD_B_TILED(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
      intel_sub_group_2d_block_read_transform_8b_32r16x4c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)(B_BUF))
# elif (RTN == 2)
# define OZAKI_LOAD_B_TILED(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
      intel_sub_group_2d_block_read_transform_8b_32r16x2c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)), (private uint*)(B_BUF))
# else
# define OZAKI_LOAD_B_TILED(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF) \
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
      OZAKI_LOAD_A_TILED(AS, K_PAD, M_HT, MI, KOFF, A_BUF); \
      OZAKI_LOAD_B_TILED(BS, N_PAD, K_PAD, NJ, KOFF, B_BUF); \
    } while (0)

# define OZAKI_COMPUTE_TILED(A_BUF, B_BUF, ACC) \
    do { \
      int rc_m_, rc_n_; \
      UNROLL_FORCE(RTM) for (rc_m_ = 0; rc_m_ < RTM; ++rc_m_) \
      { \
        UNROLL_FORCE(RTN) for (rc_n_ = 0; rc_n_ < RTN; ++rc_n_) \
        { \
          OZAKI_DPAS_ONE((A_BUF)[rc_m_], (B_BUF)[rc_n_], (ACC)[rc_m_ * RTN + rc_n_]); \
        } \
      } \
    } while (0)

/**
 * Tiled prefetch: prefetch next K-step for all RTM A and RTN B tiles.
 * Coalesced variants match the wider loads above.
 */
# if (RTM == 4)
# define OZAKI_PREFETCH_A_TILED(AS, K_PAD, M_HT, KOFF, MI) \
      intel_sub_group_2d_block_prefetch_8b_32r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# elif (RTM == 2)
# define OZAKI_PREFETCH_A_TILED(AS, K_PAD, M_HT, KOFF, MI) \
      intel_sub_group_2d_block_prefetch_8b_16r32x1c((global void*)(AS), (K_PAD), (M_HT), (K_PAD), (int2)((KOFF), (MI)))
# else
# define OZAKI_PREFETCH_A_TILED(AS, K_PAD, M_HT, KOFF, MI) \
      do { \
        int rp_m_; \
        UNROLL_FORCE(RTM) for (rp_m_ = 0; rp_m_ < RTM; ++rp_m_) \
        { \
          OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, (MI) + rp_m_ * XMX_M); \
        } \
      } while (0)
# endif
# if (RTN == 4)
# define OZAKI_PREFETCH_B_TILED(BS, N_PAD, K_PAD, KOFF, NJ) \
      intel_sub_group_2d_block_prefetch_8b_32r16x4c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))
# elif (RTN == 2)
# define OZAKI_PREFETCH_B_TILED(BS, N_PAD, K_PAD, KOFF, NJ) \
      intel_sub_group_2d_block_prefetch_8b_32r16x2c((global void*)(BS), (N_PAD), (K_PAD), (N_PAD), (int2)((NJ), (KOFF)))
# else
# define OZAKI_PREFETCH_B_TILED(BS, N_PAD, K_PAD, KOFF, NJ) \
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
      OZAKI_PREFETCH_A_TILED(AS, K_PAD, M_HT, KOFF, MI); \
      OZAKI_PREFETCH_B_TILED(BS, N_PAD, K_PAD, KOFF, NJ); \
    } while (0)
/**
 * NVIDIA MMA path (NV_MMA: warp-cooperative m16n8k32, SM>=8.0, SG=32).
 * Tile: 16 rows x 8 cols, K=32. Accumulator: 4 int32 per thread (fragment).
 * A layout in global: row-major [M_pad x K_pad] - same as dp4a/Intel path.
 * B layout in global: K-major  [K_pad x N_pad] - same as dp4a/Intel path.
 * Shared memory staging + ldmatrix for fragment generation.
 */
#elif defined(NV_MMA) && (NV_MMA)

# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI)
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ)
# define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ)

# if defined(OZAKI_U8) && (OZAKI_U8)
#   define OZAKI_BYTE_T uchar
#   define OZAKI_BYTE4_T uchar4
#   define NV_MMA_16x8x32(D0,D1,D2,D3, A0,A1,A2,A3, B0,B1) \
      asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
        : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3) \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), \
          "r"(D0), "r"(D1), "r"(D2), "r"(D3))
# else
#   define OZAKI_BYTE_T char
#   define OZAKI_BYTE4_T char4
#   define NV_MMA_16x8x32(D0,D1,D2,D3, A0,A1,A2,A3, B0,B1) \
      asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
        : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3) \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), \
          "r"(D0), "r"(D1), "r"(D2), "r"(D3))
# endif

/**
 * Load the 2-register b-fragment: b[reg] byte j = B[k = K0 + j + reg*16, COL].
 *
 * With OZAKI_BVNNI the 4 K-values of one fragment register are already adjacent
 * (layout [K_pad/4][N_pad][4]) and K0 is a multiple of 4, so each register is a
 * single aligned uint - 2 loads instead of 8 strided byte gathers. Plain
 * K-major layout keeps the gather.
 */
#if defined(OZAKI_BKMAJOR) && (OZAKI_BKMAJOR)
/* K-major: both registers are aligned uints 16 bytes apart in one segment. */
# define NV_MMA_LOAD_BFRAG(BS, N_PAD, K_PAD, NJ, KOFF, K0, COL, B0, B1) \
    do { \
      CONSTANT const OZAKI_BYTE_T* bk_ = (CONSTANT const OZAKI_BYTE_T*)(BS) \
        + OZAKI_IDX_BS((KOFF) + (K0), (NJ) + (COL), (N_PAD), (K_PAD)); \
      (B0) = *(CONSTANT const uint*)bk_; \
      (B1) = *(CONSTANT const uint*)(bk_ + 16); \
    } while (0)
#elif defined(OZAKI_BVNNI) && (OZAKI_BVNNI)
# define NV_MMA_LOAD_BFRAG(BS, N_PAD, K_PAD, NJ, KOFF, K0, COL, B0, B1) \
    do { \
      CONSTANT const uint* bq_ = (CONSTANT const uint*)((CONSTANT const OZAKI_BYTE_T*)(BS) \
        + OZAKI_IDX_BS((KOFF) + (K0), (NJ) + (COL), (N_PAD), (K_PAD))); \
      (B0) = bq_[0]; \
      (B1) = bq_[4 * (N_PAD)]; \
    } while (0)
#else
# define NV_MMA_LOAD_BFRAG(BS, N_PAD, K_PAD, NJ, KOFF, K0, COL, B0, B1) \
    do { \
      CONSTANT const OZAKI_BYTE_T* bb_ = \
        (CONSTANT const OZAKI_BYTE_T*)(BS) + (long)(KOFF) * (N_PAD) + (NJ) + (COL); \
      (B0) = as_uint((OZAKI_BYTE4_T)( \
        bb_[(long)((K0) + 0) * (N_PAD)], bb_[(long)((K0) + 1) * (N_PAD)], \
        bb_[(long)((K0) + 2) * (N_PAD)], bb_[(long)((K0) + 3) * (N_PAD)])); \
      (B1) = as_uint((OZAKI_BYTE4_T)( \
        bb_[(long)((K0) + 16) * (N_PAD)], bb_[(long)((K0) + 17) * (N_PAD)], \
        bb_[(long)((K0) + 18) * (N_PAD)], bb_[(long)((K0) + 19) * (N_PAD)])); \
    } while (0)
#endif

/**
 * One MMA step: 16x8x32 tile, accumulates into int4 fragment (D0..D3).
 * PTX ISA fragment: b[reg] byte j = B[k=threadID*4+j+reg*16, n=groupID]
 *                   a[reg] byte j = A[m=groupID+(reg%2)*8, k=threadID*4+j+(reg/2)*16]
 * The A registers alternate in m first and advance k only every second
 * register: a0=(m,k), a1=(m+8,k), a2=(m,k+16), a3=(m+8,k+16). Getting this
 * order wrong (advancing k first) still compiles and runs, but silently
 * computes a wrong product for any operand needing more than one k-half.
 * Host must set KU=1: NVIDIA OpenCL compiler evaluates all asm inputs upfront
 * when unrolling, so iteration N+1 reads stale C (not iteration N's D output).
 */
# define NV_MMA_STEP(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, LANE, D0, D1, D2, D3) \
    do { \
      const int grp_ = (LANE) / 4; \
      const int tid_ = (LANE) % 4; \
      CONSTANT const OZAKI_BYTE_T* ap0_ = \
        (CONSTANT const OZAKI_BYTE_T*)(AS) + (long)((MI) + grp_) * (K_PAD) + (KOFF) + tid_ * 4; \
      CONSTANT const OZAKI_BYTE_T* ap1_ = \
        (CONSTANT const OZAKI_BYTE_T*)(AS) + (long)((MI) + grp_ + 8) * (K_PAD) + (KOFF) + tid_ * 4; \
      uint a0_ = *(CONSTANT const uint*)ap0_; \
      uint a1_ = *(CONSTANT const uint*)ap1_; \
      uint a2_ = *(CONSTANT const uint*)(ap0_ + 16); \
      uint a3_ = *(CONSTANT const uint*)(ap1_ + 16); \
      const int k0_ = tid_ * 4; \
      uint b0_, b1_; \
      NV_MMA_LOAD_BFRAG(BS, N_PAD, K_PAD, NJ, KOFF, k0_, grp_, b0_, b1_); \
      NV_MMA_16x8x32(D0, D1, D2, D3, a0_, a1_, a2_, a3_, b0_, b1_); \
    } while (0)

/**
 * Tiled MMA: RTM x RTN sub-tiles of m16n8k32 each.
 * ACC is int4[RTM*RTN] - each int4 is one MMA fragment (4 int32 values).
 * XMX_M=16 rows, XMX_N=8 cols per MMA tile.
 */
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int lane_ = (int)SGLID(); \
      int rm_l_, rn_l_; \
      for (rm_l_ = 0; rm_l_ < RTM; ++rm_l_) { \
        for (rn_l_ = 0; rn_l_ < RTN; ++rn_l_) { \
          const int idx_ = rm_l_ * RTN + rn_l_; \
          NV_MMA_STEP(AS, BS, K_PAD, N_PAD, \
            (MI) + rm_l_ * XMX_M, (NJ) + rn_l_ * XMX_N, KOFF, lane_, \
            (ACC)[idx_].s0, (ACC)[idx_].s1, (ACC)[idx_].s2, (ACC)[idx_].s3); \
        } \
      } \
    } while (0)

/* Single-tile MMA (RTM=1, RTN=1). */
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int lane_ = (int)SGLID(); \
      NV_MMA_STEP(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, lane_, \
        (ACC).s0, (ACC).s1, (ACC).s2, (ACC).s3); \
    } while (0)

/* NVIDIA PTX dp4a path (NV>=2, SG=16, no tensor cores) */
#elif defined(NV) && (2 <= NV)

# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI)
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ)
# define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ)

/* PTX dp4a: 4-element dot product of packed bytes with int32 accumulator. */
# if defined(OZAKI_U8) && (OZAKI_U8)
#   define NV_DP4A(D, A, B, C) asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(D) : "r"(A), "r"(B), "r"(C))
#   define OZAKI_BYTE_T uchar
#   define OZAKI_BYTE4_T  uchar4
# else
#   define NV_DP4A(D, A, B, C) asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(D) : "r"(A), "r"(B), "r"(C))
#   define OZAKI_BYTE_T char
#   define OZAKI_BYTE4_T  char4
# endif

/**
 * dp4a single-tile DPAS: 8 rows x 16 cols, K=32.
 * Processes 4 K-elements per dp4a (8 dp4a calls per row).
 * A row data (8 uints) and B column data (8 uints) are pre-loaded by caller.
 */
# define NV_DP4A_8x1(AROW, BCOL, ACC) \
    do { \
      int k4d_; \
      for (k4d_ = 0; k4d_ < 8; ++k4d_) { \
        NV_DP4A((ACC), (AROW)[k4d_], (BCOL)[k4d_], (ACC)); \
      } \
    } while (0)

/* Load one B column (8 packed uints covering K=32) into BDST. */
# if defined(OZAKI_BKMAJOR) && (OZAKI_BKMAJOR)
/* K-major: the whole column-slice is 8 consecutive uints. */
# define NV_LOAD_BCOL(BS, N_PAD, K_PAD, KOFF, COL, BDST) \
    do { \
      CONSTANT const uint* bcp_ = (CONSTANT const uint*)((CONSTANT const OZAKI_BYTE_T*)(BS) \
        + OZAKI_IDX_BS((KOFF), (COL), (N_PAD), (K_PAD))); \
      int kb_; \
      UNROLL_FORCE(8) for (kb_ = 0; kb_ < 8; ++kb_) { \
        (BDST)[kb_] = bcp_[kb_]; \
      } \
    } while (0)
# elif defined(OZAKI_BVNNI) && (OZAKI_BVNNI)
/**
 * VNNI-packed: each dp4a operand is one aligned uint at [k/4][col][0..3].
 * 8 loads instead of 32 scalar byte gathers, and lanes stay coalesced
 * because consecutive COL are 4 bytes apart.
 */
# define NV_LOAD_BCOL(BS, N_PAD, K_PAD, KOFF, COL, BDST) \
    do { \
      CONSTANT const uint* bcp_ = (CONSTANT const uint*)((CONSTANT const OZAKI_BYTE_T*)(BS) \
        + OZAKI_IDX_BS((KOFF), (COL), (N_PAD), (K_PAD))); \
      int kb_; \
      UNROLL_FORCE(8) for (kb_ = 0; kb_ < 8; ++kb_) { \
        (BDST)[kb_] = bcp_[kb_ * (N_PAD)]; \
      } \
    } while (0)
# else
# define NV_LOAD_BCOL(BS, N_PAD, K_PAD, KOFF, COL, BDST) \
    do { \
      CONSTANT const OZAKI_BYTE_T* bcp_ = \
        (CONSTANT const OZAKI_BYTE_T*)(BS) + (long)(KOFF) * (N_PAD) + (COL); \
      int kb_; \
      for (kb_ = 0; kb_ < 8; ++kb_) { \
        (BDST)[kb_] = as_uint((OZAKI_BYTE4_T)(bcp_[0], bcp_[(N_PAD)], bcp_[2*(N_PAD)], bcp_[3*(N_PAD)])); \
        bcp_ += 4 * (N_PAD); \
      } \
    } while (0)
# endif

/**
 * Load one A row (8 packed uints covering K=32) into ADST.
 * A is K-contiguous and K_PAD is a multiple of BK (32), so the 32 bytes are
 * contiguous and 16B-aligned: fetch them as 2 uint4 instead of 8 scalar
 * loads. Cuts A load messages per row from 8 to 2.
 */
# if defined(NV) && (2 <= NV)
/* Force real 128-bit loads: assigning a uint4 into a scalar array lets the
 * compiler scalarize it back into 4 separate loads (verified in PTX).
 * The state space follows CONSTANT via OZAKI_PTX_LD_V4. */
# define NV_LOAD_AROW(AS, K_PAD, ROW, KOFF, ADST) \
    do { \
      CONSTANT const uint* arp_ = (CONSTANT const uint*)((CONSTANT const OZAKI_BYTE_T*)(AS) \
        + (long)(ROW) * (K_PAD) + (KOFF)); \
      asm(OZAKI_PTX_LD_V4 " {%0,%1,%2,%3}, [%4];" \
        : "=r"((ADST)[0]), "=r"((ADST)[1]), "=r"((ADST)[2]), "=r"((ADST)[3]) : "l"(arp_)); \
      asm(OZAKI_PTX_LD_V4 " {%0,%1,%2,%3}, [%4];" \
        : "=r"((ADST)[4]), "=r"((ADST)[5]), "=r"((ADST)[6]), "=r"((ADST)[7]) : "l"(arp_ + 4)); \
    } while (0)
# else
# define NV_LOAD_AROW(AS, K_PAD, ROW, KOFF, ADST) \
    do { \
      CONSTANT const uint* arp_ = (CONSTANT const uint*)((CONSTANT const OZAKI_BYTE_T*)(AS) \
        + (long)(ROW) * (K_PAD) + (KOFF)); \
      const uint4 alo_ = vload4(0, arp_), ahi_ = vload4(1, arp_); \
      (ADST)[0] = alo_.s0; (ADST)[1] = alo_.s1; (ADST)[2] = alo_.s2; (ADST)[3] = alo_.s3; \
      (ADST)[4] = ahi_.s0; (ADST)[5] = ahi_.s1; (ADST)[6] = ahi_.s2; (ADST)[7] = ahi_.s3; \
    } while (0)
# endif

/**
 * Tiled dp4a with register reuse: pre-load all RTM A-strips and RTN B-columns
 * into registers, then compute RTM*RTN dot products from registers.
 * B reuse: each B column is loaded once, used by all RTM row-tiles.
 * A reuse: each A row-set is loaded once, used by all RTN column-tiles.
 *
 * A is loaded row-identically by every lane of the sub-group (the row index
 * carries no SGLID term) and the hardware serves that from one broadcast, so
 * the buffering costs registers without saving traffic. Streaming A per row
 * instead (footprint 8 rather than 64*RTM) was measured slower at every
 * RTM/RTN/KU combination, so the pre-loaded form is kept deliberately.
 */
# define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int col0_ = (NJ) + (int)SGLID(); \
      uint b_reg_[RTN][8]; \
      uint a_reg_[RTM][8][8]; \
      int rn_l_, rm_l_; \
      /* Load all B columns (one per RTN tile, this thread's column) */ \
      for (rn_l_ = 0; rn_l_ < RTN; ++rn_l_) { \
        NV_LOAD_BCOL(BS, N_PAD, K_PAD, KOFF, col0_ + rn_l_ * XMX_N, b_reg_[rn_l_]); \
      } \
      /* Load all A rows (8 rows per RTM tile) */ \
      for (rm_l_ = 0; rm_l_ < RTM; ++rm_l_) { \
        int m_l_; \
        for (m_l_ = 0; m_l_ < 8; ++m_l_) { \
          NV_LOAD_AROW(AS, K_PAD, (MI) + rm_l_ * XMX_M + m_l_, KOFF, a_reg_[rm_l_][m_l_]); \
        } \
      } \
      /* Compute: RTM * RTN sub-tiles from registers */ \
      for (rm_l_ = 0; rm_l_ < RTM; ++rm_l_) { \
        for (rn_l_ = 0; rn_l_ < RTN; ++rn_l_) { \
          union { int8 v_; int a_[8]; } u_t_; \
          int m_c_; \
          u_t_.v_ = (ACC)[rm_l_ * RTN + rn_l_]; \
          for (m_c_ = 0; m_c_ < 8; ++m_c_) { \
            NV_DP4A_8x1(a_reg_[rm_l_][m_c_], b_reg_[rn_l_], u_t_.a_[m_c_]); \
          } \
          (ACC)[rm_l_ * RTN + rn_l_] = u_t_.v_; \
        } \
      } \
    } while (0)

/* Single-tile fallback (RTM=1, RTN=1 or direct call). */
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int col_ = (NJ) + (int)SGLID(); \
      uint b_s_[8]; \
      union { int8 v_; int a_[8]; } u_s_; \
      int m_s_; \
      NV_LOAD_BCOL(BS, N_PAD, K_PAD, KOFF, col_, b_s_); \
      u_s_.v_ = (ACC); \
      for (m_s_ = 0; m_s_ < 8; ++m_s_) { \
        uint a_s_[8]; \
        NV_LOAD_AROW(AS, K_PAD, (MI) + m_s_, KOFF, a_s_); \
        NV_DP4A_8x1(a_s_, b_s_, u_s_.a_[m_s_]); \
      } \
      (ACC) = u_s_.v_; \
    } while (0)

/* Scalar fallback (no hardware acceleration) */
#else
# define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI)
# define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ)
# define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ)
# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_BYTE_T uchar
# else
# define OZAKI_BYTE_T char
# endif
# define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
    do { \
      const int col_ = (NJ) + (int)SGLID(); \
      union { \
        int8 v_; \
        int a_[8]; \
      } u_; \
      int m_; \
      u_.v_ = (ACC); \
      for (m_ = 0; m_ < 8; ++m_) { \
        int k_; \
        for (k_ = 0; k_ < 32; ++k_) { \
          u_.a_[m_] += (int)((CONSTANT const OZAKI_BYTE_T*)(AS))[(long)((MI) + m_) * (K_PAD) + (KOFF) + k_] * \
                       (int)((CONSTANT const OZAKI_BYTE_T*)(BS))[(long)((KOFF) + k_) * (N_PAD) + col_]; \
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


/**
 * Decompose an IEEE-754 value into sign, biased exponent, and implicit-1 mantissa.
 * Zero, subnormal, Inf, and NaN inputs yield exp=0, mant=0.
 * real_t, uint_repr_t, EXP_MASK, and AS_UINT come from libxstream_common.h.
 */
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

/**
 * Extract a 7-bit signed digit from an aligned mantissa for slice index S.
 * The mantissa ALIGNED is already right-shifted by (max_exp - elem_exp).
 * Returns a signed char: the digit with sign applied if SIGN != 0.
 * MANT_BITS must be defined by the including file.
 */
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
 *
 * Every kernel here takes each buffer as a (base, index) pair and opens by
 * declaring the usable pointer as base + index: with USM the offset already
 * travels in the pointer and the index is zero, but clSetKernelArg takes a
 * cl_mem that cannot express an offset, so without USM the host passes the
 * registered base and the index carries the remainder. Resolving it on entry
 * makes both cases one code path and leaves each body addressing its operand
 * from zero, which is why panelling needs neither USM nor sub-buffers.
 */
#if defined(BM_PRE)
__attribute__((reqd_work_group_size(BM_PRE, 1, 1))) kernel void scale_beta(
  global real_t* restrict c_base, int c_index, int M, int N, int ldc, real_t beta)
{
  global real_t* restrict c = c_base + c_index;
  const int row = (int)get_global_id(0);
  const int col = (int)get_global_id(1);
  if (row < M && col < N) {
    c[col * ldc + row] *= beta;
  }
}
#endif /*defined(BM_PRE)*/

#endif /*OZAKI_COMMON_CL*/
