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
 *   OZAKI_DPAS          - one DPAS step (2D block I/O + int8 MAD)
 *   ieee_decompose()    - IEEE-754 -> (sign, biased exponent, mantissa)
 *   ozaki_slice_digit() - extract a 7-bit signed digit from aligned mantissa
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

/* Accumulator strategy: 1 = individual scalar variables, 0 = int8 array (default). */
#if !defined(OZAKI_SCALAR_ACC)
# define OZAKI_SCALAR_ACC 0
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
 * XMX path:
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
#define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI) \
  intel_sub_group_2d_block_prefetch_8b_8r32x1c( \
      (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
      (int2)((KOFF), (MI)))
#define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ) \
  intel_sub_group_2d_block_prefetch_8b_32r16x1c( \
      (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
      (int2)((NJ), (KOFF)))

#if (8 == RC)
#define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    ushort8 a_blk_; \
    uint8 b_blk_; \
    intel_sub_group_2d_block_read_8b_8r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
        (int2)((KOFF), (MI)), (private ushort*)&a_blk_); \
    intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
        (int2)((NJ), (KOFF)), (private uint*)&b_blk_); \
    (ACC) = intel_sub_group_i8_i8_matrix_mad_k32( \
                as_short8(a_blk_), as_int8(b_blk_), (ACC)); \
  } while (0)
#elif (4 == RC)
#define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    ushort8 a_blk_; \
    uint8 b_blk_; \
    int4 lo_, hi_; \
    intel_sub_group_2d_block_read_8b_8r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
        (int2)((KOFF), (MI)), (private ushort*)&a_blk_); \
    intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
        (int2)((NJ), (KOFF)), (private uint*)&b_blk_); \
    lo_ = (ACC).lo; hi_ = (ACC).hi; \
    lo_ = intel_sub_group_i8_i8_matrix_mad_k32( \
              as_short4(a_blk_.lo), as_int8(b_blk_), lo_); \
    hi_ = intel_sub_group_i8_i8_matrix_mad_k32( \
              as_short4(a_blk_.hi), as_int8(b_blk_), hi_); \
    (ACC) = (int8)(lo_, hi_); \
  } while (0)
#endif

/* Single-tile DPAS from pre-loaded A (ushort8) and B (uint8).
 * RC=8: one DPAS(short8,int8,int8). RC=4: split into two DPAS(short4,int8,int4). */
#if (8 == RC)
#define OZAKI_DPAS_ONE_(A, B, ACC) \
  (ACC) = intel_sub_group_i8_i8_matrix_mad_k32( \
              as_short8(A), as_int8(B), (ACC))
#elif (4 == RC)
#define OZAKI_DPAS_ONE_(A, B, ACC) \
  do { \
    int4 lo1_ = (ACC).lo, hi1_ = (ACC).hi; \
    lo1_ = intel_sub_group_i8_i8_matrix_mad_k32( \
               as_short4((A).lo), as_int8(B), lo1_); \
    hi1_ = intel_sub_group_i8_i8_matrix_mad_k32( \
               as_short4((A).hi), as_int8(B), hi1_); \
    (ACC) = (int8)(lo1_, hi1_); \
  } while (0)
#endif

#if OZAKI_SCALAR_ACC
/* Flat DPAS compute: preprocessor-expanded, no loops, no runtime indexing.
 * Each RTM x RTN specialization directly references accumulator scalars
 * P##0, P##1, ... via token pasting — LLVM sees plain variables from the
 * start, so SROA keeps them in SSA registers throughout the K-loop.
 * A and B are ushort8[] and uint8[] arrays (short-lived, easily SROA'd). */
#if RTM == 1 && RTN == 1
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0)
#elif RTM == 1 && RTN == 2
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0); \
  OZAKI_DPAS_ONE_((A)[0], (B)[1], P##1)
#elif RTM == 2 && RTN == 1
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0); \
  OZAKI_DPAS_ONE_((A)[1], (B)[0], P##1)
#elif RTM == 2 && RTN == 2
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0); \
  OZAKI_DPAS_ONE_((A)[0], (B)[1], P##1); \
  OZAKI_DPAS_ONE_((A)[1], (B)[0], P##2); \
  OZAKI_DPAS_ONE_((A)[1], (B)[1], P##3)
#elif RTM == 4 && RTN == 1
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0); \
  OZAKI_DPAS_ONE_((A)[1], (B)[0], P##1); \
  OZAKI_DPAS_ONE_((A)[2], (B)[0], P##2); \
  OZAKI_DPAS_ONE_((A)[3], (B)[0], P##3)
#elif RTM == 4 && RTN == 2
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0); \
  OZAKI_DPAS_ONE_((A)[0], (B)[1], P##1); \
  OZAKI_DPAS_ONE_((A)[1], (B)[0], P##2); \
  OZAKI_DPAS_ONE_((A)[1], (B)[1], P##3); \
  OZAKI_DPAS_ONE_((A)[2], (B)[0], P##4); \
  OZAKI_DPAS_ONE_((A)[2], (B)[1], P##5); \
  OZAKI_DPAS_ONE_((A)[3], (B)[0], P##6); \
  OZAKI_DPAS_ONE_((A)[3], (B)[1], P##7)
#elif RTM == 4 && RTN == 4
#define OZAKI_COMPUTE_DIRECT_(A, B, P) \
  OZAKI_DPAS_ONE_((A)[0], (B)[0], P##0);  \
  OZAKI_DPAS_ONE_((A)[0], (B)[1], P##1);  \
  OZAKI_DPAS_ONE_((A)[0], (B)[2], P##2);  \
  OZAKI_DPAS_ONE_((A)[0], (B)[3], P##3);  \
  OZAKI_DPAS_ONE_((A)[1], (B)[0], P##4);  \
  OZAKI_DPAS_ONE_((A)[1], (B)[1], P##5);  \
  OZAKI_DPAS_ONE_((A)[1], (B)[2], P##6);  \
  OZAKI_DPAS_ONE_((A)[1], (B)[3], P##7);  \
  OZAKI_DPAS_ONE_((A)[2], (B)[0], P##8);  \
  OZAKI_DPAS_ONE_((A)[2], (B)[1], P##9);  \
  OZAKI_DPAS_ONE_((A)[2], (B)[2], P##10); \
  OZAKI_DPAS_ONE_((A)[2], (B)[3], P##11); \
  OZAKI_DPAS_ONE_((A)[3], (B)[0], P##12); \
  OZAKI_DPAS_ONE_((A)[3], (B)[1], P##13); \
  OZAKI_DPAS_ONE_((A)[3], (B)[2], P##14); \
  OZAKI_DPAS_ONE_((A)[3], (B)[3], P##15)
#endif /* OZAKI_SCALAR_ACC — OZAKI_COMPUTE_DIRECT_ */
#endif /* OZAKI_SCALAR_ACC */

/* Tiled DPAS: RTM x RTN sub-tiles per sub-group.
 * Loads RTM A-strips and RTN B-strips, then issues RTM*RTN DPAS.
 * Scalar path: OZAKI_COMPUTE_DIRECT_ pastes ACC##0, ACC##1, ...
 * Array path: loop over ACC##_arr_[rm*RTN+rn]. */
#if OZAKI_SCALAR_ACC
#define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    ushort8 a_rt_[RTM]; \
    uint8 b_rt_[RTN]; \
    int rm_t_, rn_t_; \
    UNROLL_FORCE(RTM) for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
      intel_sub_group_2d_block_read_8b_8r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
          (int2)((KOFF), (MI) + rm_t_ * XMX_M), \
          (private ushort*)&a_rt_[rm_t_]); \
    } \
    UNROLL_FORCE(RTN) for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
      intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
          (int2)((NJ) + rn_t_ * XMX_N, (KOFF)), \
          (private uint*)&b_rt_[rn_t_]); \
    } \
    OZAKI_COMPUTE_DIRECT_(a_rt_, b_rt_, ACC); \
  } while (0)
#else
#define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  OZAKI_DPAS_TILED_ARR(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC)
#endif

/* Array-based DPAS tiled: ACC is an int8 array indexed [rm*RTN+rn].
 * Use for kernels where accumulators cannot be individual scalars. */
#define OZAKI_DPAS_TILED_ARR(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    ushort8 a_rt_[RTM]; \
    uint8 b_rt_[RTN]; \
    int rm_t_, rn_t_; \
    UNROLL_FORCE(RTM) for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
      intel_sub_group_2d_block_read_8b_8r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
          (int2)((KOFF), (MI) + rm_t_ * XMX_M), \
          (private ushort*)&a_rt_[rm_t_]); \
    } \
    UNROLL_FORCE(RTN) for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
      intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
          (int2)((NJ) + rn_t_ * XMX_N, (KOFF)), \
          (private uint*)&b_rt_[rn_t_]); \
    } \
    UNROLL_FORCE(RTM) for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
      UNROLL_FORCE(RTN) for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
        OZAKI_DPAS_ONE_(a_rt_[rm_t_], b_rt_[rn_t_], \
                        (ACC)[rm_t_ * RTN + rn_t_]); \
      } \
    } \
  } while (0)

/* Split load/compute for software pipelining.
 * OZAKI_LOAD_TILED: load A/B tiles into caller-supplied arrays.
 * OZAKI_COMPUTE_TILED: issue DPAS from pre-loaded tiles. */
#define OZAKI_LOAD_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, A_BUF, B_BUF) \
  do { \
    int rl_m_, rl_n_; \
    UNROLL_FORCE(RTM) for (rl_m_ = 0; rl_m_ < RTM; ++rl_m_) { \
      intel_sub_group_2d_block_read_8b_8r32x1c( \
          (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
          (int2)((KOFF), (MI) + rl_m_ * XMX_M), \
          (private ushort*)&(A_BUF)[rl_m_]); \
    } \
    UNROLL_FORCE(RTN) for (rl_n_ = 0; rl_n_ < RTN; ++rl_n_) { \
      intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
          (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
          (int2)((NJ) + rl_n_ * XMX_N, (KOFF)), \
          (private uint*)&(B_BUF)[rl_n_]); \
    } \
  } while (0)

#if OZAKI_SCALAR_ACC
#define OZAKI_COMPUTE_TILED(A_BUF, B_BUF, ACC) \
  OZAKI_COMPUTE_DIRECT_(A_BUF, B_BUF, ACC)
#else
#define OZAKI_COMPUTE_TILED(A_BUF, B_BUF, ACC) \
  do { \
    int rc_m_, rc_n_; \
    UNROLL_FORCE(RTM) for (rc_m_ = 0; rc_m_ < RTM; ++rc_m_) { \
      UNROLL_FORCE(RTN) for (rc_n_ = 0; rc_n_ < RTN; ++rc_n_) { \
        OZAKI_DPAS_ONE_((A_BUF)[rc_m_], (B_BUF)[rc_n_], \
                        (ACC)[rc_m_ * RTN + rc_n_]); \
      } \
    } \
  } while (0)
#endif

/* Tiled prefetch: prefetch next K-step for all RTM A and RTN B tiles. */
#define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ) \
  do { \
    int rp_m_, rp_n_; \
    UNROLL_FORCE(RTM) for (rp_m_ = 0; rp_m_ < RTM; ++rp_m_) { \
      OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, (MI) + rp_m_ * XMX_M); \
    } \
    UNROLL_FORCE(RTN) for (rp_n_ = 0; rp_n_ < RTN; ++rp_n_) { \
      OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, (NJ) + rp_n_ * XMX_N); \
    } \
  } while (0)
#else
#define OZAKI_PREFETCH_A(AS, K_PAD, M_HT, KOFF, MI)
#define OZAKI_PREFETCH_B(BS, N_PAD, K_PAD, KOFF, NJ)
#define OZAKI_PREFETCH_TILED(AS, BS, K_PAD, N_PAD, M_HT, KOFF, MI, NJ)
#define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    const int col_ = (NJ) + (int)get_sub_group_local_id(); \
    union { int8 v_; int a_[8]; } u_; \
    int m_; \
    u_.v_ = (ACC); \
    for (m_ = 0; m_ < 8; ++m_) { \
      int k_; \
      for (k_ = 0; k_ < 32; ++k_) { \
        u_.a_[m_] += (int)((CONSTANT const char*)(AS)) \
            [(long)((MI) + m_) * (K_PAD) + (KOFF) + k_] \
          * (int)((CONSTANT const char*)(BS)) \
            [(long)((KOFF) + k_) * (N_PAD) + col_]; \
      } \
    } \
    (ACC) = u_.v_; \
  } while (0)

/* Scalar DPAS_TILED: loop over RTM x RTN sub-tiles using scalar DPAS. */
#define OZAKI_DPAS_TILED(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    int rm_t_, rn_t_; \
    for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
      for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
        OZAKI_DPAS(AS, BS, K_PAD, N_PAD, \
                   (MI) + rm_t_ * XMX_M, (NJ) + rn_t_ * XMX_N, \
                   KOFF, M_HT, OZAKI_ACC_GET(ACC, rm_t_ * RTN + rn_t_)); \
      } \
    } \
  } while (0)

/* Scalar array-based DPAS_TILED: ACC is an int8 array. */
#define OZAKI_DPAS_TILED_ARR(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    int rm_t_, rn_t_; \
    for (rm_t_ = 0; rm_t_ < RTM; ++rm_t_) { \
      for (rn_t_ = 0; rn_t_ < RTN; ++rn_t_) { \
        OZAKI_DPAS(AS, BS, K_PAD, N_PAD, \
                   (MI) + rm_t_ * XMX_M, (NJ) + rn_t_ * XMX_N, \
                   KOFF, M_HT, (ACC)[rm_t_ * RTN + rn_t_]); \
      } \
    } \
  } while (0)
#endif


#if OZAKI_SCALAR_ACC
/* Register-resident accumulator helpers (tiling-aware).
 * Only RTM*RTN scalars are declared/touched to minimize register pressure. */
#if RTM * RTN <= 2
#define OZAKI_ACC_DECL(P) \
  int8 P##0 = (int8)(0), P##1 = (int8)(0)
#define OZAKI_ACC_ZERO(P) do { \
  (P##0)=(int8)(0); (P##1)=(int8)(0); \
} while (0)
#define OZAKI_ACC_ADD(D, S) do { \
  (D##0)+=(S##0); (D##1)+=(S##1); \
} while (0)
#define OZAKI_ACC_GET(P, I) \
  (*((I)==0?&(P##0):&(P##1)))
#elif RTM * RTN <= 4
#define OZAKI_ACC_DECL(P) \
  int8 P##0 = (int8)(0), P##1 = (int8)(0), \
       P##2 = (int8)(0), P##3 = (int8)(0)
#define OZAKI_ACC_ZERO(P) do { \
  (P##0)=(int8)(0); (P##1)=(int8)(0); \
  (P##2)=(int8)(0); (P##3)=(int8)(0); \
} while (0)
#define OZAKI_ACC_ADD(D, S) do { \
  (D##0)+=(S##0); (D##1)+=(S##1); \
  (D##2)+=(S##2); (D##3)+=(S##3); \
} while (0)
#define OZAKI_ACC_GET(P, I) \
  (*((I)==0?&(P##0):(I)==1?&(P##1):(I)==2?&(P##2):&(P##3)))
#else
#define OZAKI_ACC_DECL(P) \
  int8 P##0 = (int8)(0), P##1 = (int8)(0), \
       P##2 = (int8)(0), P##3 = (int8)(0), \
       P##4 = (int8)(0), P##5 = (int8)(0), \
       P##6 = (int8)(0), P##7 = (int8)(0)
#define OZAKI_ACC_ZERO(P) do { \
  (P##0)=(int8)(0); (P##1)=(int8)(0); \
  (P##2)=(int8)(0); (P##3)=(int8)(0); \
  (P##4)=(int8)(0); (P##5)=(int8)(0); \
  (P##6)=(int8)(0); (P##7)=(int8)(0); \
} while (0)
#define OZAKI_ACC_ADD(D, S) do { \
  (D##0)+=(S##0); (D##1)+=(S##1); \
  (D##2)+=(S##2); (D##3)+=(S##3); \
  (D##4)+=(S##4); (D##5)+=(S##5); \
  (D##6)+=(S##6); (D##7)+=(S##7); \
} while (0)
#define OZAKI_ACC_GET(P, I) \
  (*((I)==0?&(P##0):(I)==1?&(P##1):(I)==2?&(P##2):(I)==3?&(P##3): \
     (I)==4?&(P##4):(I)==5?&(P##5):(I)==6?&(P##6):&(P##7)))
#endif
/* Pack scalar accumulators into an int8 array (for indexed access
 * outside the DPAS hot loop). Avoids the ternary chain of ACC_GET. */
#if RTM * RTN <= 2
#define OZAKI_ACC_PACK(P, ARR) do { \
  (ARR)[0] = P##0; (ARR)[1] = P##1; \
} while (0)
#elif RTM * RTN <= 4
#define OZAKI_ACC_PACK(P, ARR) do { \
  (ARR)[0] = P##0; (ARR)[1] = P##1; \
  (ARR)[2] = P##2; (ARR)[3] = P##3; \
} while (0)
#else
#define OZAKI_ACC_PACK(P, ARR) do { \
  (ARR)[0] = P##0; (ARR)[1] = P##1; \
  (ARR)[2] = P##2; (ARR)[3] = P##3; \
  (ARR)[4] = P##4; (ARR)[5] = P##5; \
  (ARR)[6] = P##6; (ARR)[7] = P##7; \
} while (0)
#endif
#else /* !OZAKI_SCALAR_ACC — array-based accumulators */
#define OZAKI_ACC_DECL(P) \
  int8 P##_arr_[RTM * RTN]; \
  do { int zi_; for (zi_ = 0; zi_ < RTM * RTN; ++zi_) P##_arr_[zi_] = (int8)(0); } while (0)
#define OZAKI_ACC_ZERO(P) do { \
  int zi_; for (zi_ = 0; zi_ < RTM * RTN; ++zi_) (P##_arr_)[zi_] = (int8)(0); \
} while (0)
#define OZAKI_ACC_ADD(D, S) do { \
  int zi_; for (zi_ = 0; zi_ < RTM * RTN; ++zi_) (D##_arr_)[zi_] += (S##_arr_)[zi_]; \
} while (0)
#define OZAKI_ACC_GET(P, I) ((P##_arr_)[I])
#endif


/* Decompose an IEEE-754 value into sign, biased exponent, and implicit-1 mantissa.
 * Zero and subnormal inputs yield exp=0, mant=0.
 * real_t, uint_repr_t, EXP_MASK, and AS_UINT come from libxstream_common.h. */
inline void ieee_decompose(real_t val, int* sign, short* exp, uint_repr_t* mant)
{
  const uint_repr_t bits = AS_UINT(val);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  *sign = (int)(bits >> 63);
  *exp  = (short)((bits >> 52) & EXP_MASK);
  *mant = (bits & 0x000FFFFFFFFFFFFFUL) | 0x0010000000000000UL;
#else
  *sign = (int)(bits >> 31);
  *exp  = (short)((bits >> 23) & EXP_MASK);
  *mant = (bits & 0x007FFFFFU) | 0x00800000U;
#endif
  if (0 == *exp) {
    *mant = 0;
    *exp  = 0;
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
  const int low  = MAX(0, high - 6);
  const int width = high - low + 1;
  char digit = 0;
  if (width > 0 && high >= 0) {
    digit = (char)((aligned >> low) & ((1U << width) - 1U));
  }
  if (sign) digit = -digit;
  return digit;
}
#endif /*defined(MANT_BITS)*/

#endif /*OZAKI_COMMON_CL*/
