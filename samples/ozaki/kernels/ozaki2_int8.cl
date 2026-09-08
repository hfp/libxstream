/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../libxstream/opencl/libxstream_common.h"
#include "ozaki_common.cl"

/**
 * Ozaki Scheme 2: CRT residue GEMM.
 *
 * A and B are decomposed once over the full K into dense per-prime residue
 * planes (preprocess_*_crt_dense), then one tiled GEMM accumulates every prime
 * with full-K int8 MMA passes. Reconstruction is hierarchical: leaf groups of
 * HIER_GS primes combine by explicit CRT (or Garner), the group values by
 * level-2 Garner, then Horner, scaling and the C update. It runs either fused
 * into the GEMM's epilogue or, the default on GPUs (OZAKI_UNFUSE), as a second
 * kernel over the residue bytes the GEMM stores.
 *
 * OZAKI_U8 (default): unsigned residues with moduli up to 256 (fp64 16 primes,
 * fp32 9), sign folded as the modular additive inverse; OZAKI_U8=0 keeps signed
 * i8 with moduli up to 128. KGROUPS > 0 inserts a Barrett reduction every
 * KGROUPS * BK steps for a K the int32 accumulator cannot cover.
 *
 * Compile-time parameters (-D): BM, BN (output tile), BK (K per MMA step),
 * RTM, RTN (register tiling), KU, SG, NPRIMES, MANT_BITS, BIAS_PLUS_MANT,
 * USE_DOUBLE, BM_PRE, BN_PRE, BK_PRE, and the hierarchical tables the host
 * emits (HIER_*).
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
#if !defined(NPRIMES)
# define NPRIMES 20
#endif
#if !defined(MANT_BITS)
# define MANT_BITS 52
#endif
#if !defined(BIAS_PLUS_MANT)
# define BIAS_PLUS_MANT 1075
#endif
#if !defined(MANT_TRUNC)
# define MANT_TRUNC 0
#endif
#if !defined(KGROUPS)
# define KGROUPS 0
#endif
#if !defined(KU)
# define KU 1
#endif
#if !defined(SG)
# define SG 16
#endif
#if !defined(OZ2_HORNER_GROUP)
/* Primes per Horner group whose product fits a ulong: the 8 (u8) or 9 (i8) largest. */
# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZ2_HORNER_GROUP 8
# else
# define OZ2_HORNER_GROUP 9
# endif
#endif
#if !defined(PB)
# define PB 1
#endif

/**
 * Hierarchical CRT: leaf groups of HIER_GS primes reconstruct to a group value
 * (explicit CRT, or Garner), and level 2 combines the HIER_NGROUPS group values
 * by Garner, so the peak live state is max(HIER_GS, HIER_NGROUPS) rather than
 * NPRIMES. Fractional-CRT mode 1 replaces the whole reconstruction and needs the
 * flat path; mode 2 reconstructs per group and keeps the hierarchy.
 */
#if defined(OZAKI_FRACCRT) && (1 == OZAKI_FRACCRT)
# undef OZAKI_HIER
# define OZAKI_HIER 0
#endif
#if defined(OZAKI_FRACCRT) && (2 == OZAKI_FRACCRT) && !defined(OZAKI_HIER)
# define OZAKI_HIER 1
#endif
#if !defined(OZAKI_HIER)
# define OZAKI_HIER 0
#endif
#define POW2_PIDX 3
#if OZAKI_HIER
/**
 * Leaf group size, at most 4: the level-2 datapath is 32-bit, so group products
 * and group values must fit uint32. The host lowers it to the largest divisor of
 * NPRIMES that still fits, since a group holding a single prime is pathological
 * (NPRIMES=9: the 4,4,1 split costs 1.21 ms against 0.58 for three full groups).
 * The level-2 tables arrive from the host as brace lists sized to the group
 * count, which the build checks (see oz2g_hier_tables_check).
 */
# if !defined(HIER_GS)
#   define HIER_GS 4
# endif
# define HIER_NGROUPS ((NPRIMES + HIER_GS - 1) / HIER_GS)
# define HIER_L2_HORNER_GROUP 2
# if !defined(OZAKI_HIER_L2)
#   define OZAKI_HIER_L2 0
# endif
#endif

/* DPAS tile dimensions and the accumulator fragment layout are in ozaki_common.cl */

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

#if defined(OZAKI_BOUNDS) && (OZAKI_BOUNDS)
# define OZAKI_IN_BOUNDS(R, M, COL, N) ((R) < (M) && (COL) < (N))
#else
/* consumes its arguments so a caller's index is not "unused" here */
# define OZAKI_IN_BOUNDS(R, M, COL, N) ((void)(R), (void)(M), (void)(COL), (void)(N), 1)
#endif

/**
 * OZAKI_FIRST: compile-time specialization for first-tile (C = 0 + result)
 * vs accumulate (C = C_old + result).  When defined, the kernel ignores
 * the `first` runtime argument and uses this value instead.
 */
#if defined(OZAKI_FIRST)
# define OZAKI_IS_FIRST(ARG) (OZAKI_FIRST)
#else
# define OZAKI_IS_FIRST(ARG) (ARG)
#endif

/**
 * OZAKI_ALPHA_ONE: compile-time specialization for alpha==1.0.
 * Eliminates the multiply when alpha is known to be unity.
 */
#if defined(OZAKI_ALPHA_ONE) && (OZAKI_ALPHA_ONE)
# define OZAKI_ALPHA_MUL(A, X) (X)
#else
# define OZAKI_ALPHA_MUL(A, X) ((A) * (X))
#endif

/**
 * Transpose specialization: when OZAKI_TRANSA / OZAKI_TRANSB are defined
 * at compile time, the ternary index computation becomes straight-line.
 */
#if defined(OZAKI_TRANSA)
# define OZAKI_IDX_A(ROW, COL, LD) ((OZAKI_TRANSA) ? ((ROW) * (LD) + (COL)) : ((COL) * (LD) + (ROW)))
#else
# define OZAKI_IDX_A(ROW, COL, LD) (transa ? ((ROW) * (LD) + (COL)) : ((COL) * (LD) + (ROW)))
#endif
#if defined(OZAKI_TRANSB)
# define OZAKI_IDX_B(ROW, COL, LD) ((OZAKI_TRANSB) ? ((ROW) * (LD) + (COL)) : ((COL) * (LD) + (ROW)))
#else
# define OZAKI_IDX_B(ROW, COL, LD) (transb ? ((ROW) * (LD) + (COL)) : ((COL) * (LD) + (ROW)))
#endif

/**
 * Extract NPRIMES CRT residues from aligned mantissa into DST buffer.
 * DST[p * SS + ROW * RS + COL] = (aligned mod m_p), sign-folded.
 * u8: sign via modular additive inverse (p - r), stored as uchar [0, p-1].
 * i8: sign via negation (-r), stored as char [-(p-1), p-1].
 */
#define OZAKI_EXTRACT_CRT(ALIGNED, SIGN, DST, SS, RS, ROW, COL) \
  OZAKI_EXTRACT_CRT_AT(ALIGNED, SIGN, DST, SS, (long)(ROW) * (RS) + (COL))
/* Store B via the (possibly VNNI-packed) index so producer and consumer agree. */
#define OZAKI_EXTRACT_CRT_B(ALIGNED, SIGN, DST, SS, N_PAD, K_PAD, ROW, COL) \
  OZAKI_EXTRACT_CRT_AT(ALIGNED, SIGN, DST, SS, OZAKI_IDX_BS(ROW, COL, N_PAD, K_PAD))
/**
 * Hierarchical extraction, the dual of the hierarchical reconstruction: reduce the
 * aligned mantissa once per leaf group modulo the group product (one mul_hi, the
 * Barrett level 2 already uses), then take the group's residues from that 32-bit
 * value with the cheap Barrett, since m_i divides M_g. Exact, and it replaces
 * NPRIMES two-step 64-bit reductions by HIER_NGROUPS one-step ones plus NPRIMES
 * 32-bit ones. fp32 mantissas already fit 32 bits, so there it would only add the
 * group step. OZAKI_EXTRACT_FLAT=1 keeps the direct form for comparison.
 */
#if OZAKI_HIER && defined(USE_DOUBLE) && (1 == USE_DOUBLE) && defined(HIER_GPROD) \
  && (!defined(OZAKI_EXTRACT_FLAT) || (0 == OZAKI_EXTRACT_FLAT))
# define OZAKI_EXTRACT_HIER 1
#else
# define OZAKI_EXTRACT_HIER 0
#endif
#if OZAKI_EXTRACT_HIER
#define OZAKI_EXTRACT_CRT_AT(ALIGNED, SIGN, DST, SS, OFF) \
  do { \
    const long off_ = (OFF); \
    SINT g_; \
    UNROLL_FORCE(HIER_NGROUPS) for (g_ = 0; g_ < HIER_NGROUPS; ++g_) \
    { \
      const uint gr_ = oz2g_mod_l2((ulong)(ALIGNED), (int)g_); \
      SINT j_; \
      UNROLL_FORCE(HIER_GS) for (j_ = 0; j_ < HIER_GS; ++j_) \
      { \
        const SINT p_ = g_ * HIER_GS + j_; \
        if (p_ < NPRIMES) { \
          uint r_ = oz2g_mod(gr_, p_); \
          if ((SIGN) && 0 != r_) OZAKI_SIGN_FOLD(r_, p_); \
          (DST)[(long)(p_) * (SS) + off_] = (char)r_; \
        } \
      } \
    } \
  } while (0)
#else
#define OZAKI_EXTRACT_CRT_AT(ALIGNED, SIGN, DST, SS, OFF) \
  do { \
    const long off_ = (OFF); \
    SINT p_; \
    UNROLL_FORCE(NPRIMES) for (p_ = 0; p_ < NPRIMES; ++p_) \
    { \
      uint r_ = oz2g_mod64((ulong)(ALIGNED), p_); \
      if ((SIGN) && 0 != r_) OZAKI_SIGN_FOLD(r_, p_); \
      (DST)[(long)(p_) * (SS) + off_] = (char)r_; \
    } \
  } while (0)
#endif
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_SIGN_FOLD(R, P) (R) = oz2g_moduli[(P)] - (R)
#else
# define OZAKI_SIGN_FOLD(R, P) (R) = -(R)
#endif

/**
 * Mod-reduce one accumulator fragment into the uint residues of prime PIDX,
 * stored at slot LIDX (the prime's index within the array, which is the global
 * index for the flat path and the index within the group for the hierarchical
 * one). u8 accumulators are non-negative, so the reduction is branchless; i8
 * accumulators need the sign-aware form.
 */
#define OZAKI_CRT_MOD_REDUCE(ACC, PIDX, LIDX, RESIDUES) \
  do { \
    OZAKI_ACC_UNION(du_); \
    int mr_; \
    du_.v_ = (ACC); \
    UNROLL_FORCE(XMX_FRAG) for (mr_ = 0; mr_ < XMX_FRAG; ++mr_) \
    { \
      uint r_; \
      OZAKI_MOD_REDUCE_ELEM(du_.a_[mr_], (PIDX), r_); \
      { \
        const uint prev_ = (RESIDUES)[(int)(LIDX) * XMX_FRAG + mr_]; \
        const uint sum_ = prev_ + r_; \
        (RESIDUES)[(int)(LIDX) * XMX_FRAG + mr_] = (sum_ >= oz2g_moduli[(PIDX)]) ? (sum_ - oz2g_moduli[(PIDX)]) : sum_; \
      } \
    } \
  } while (0)
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_MOD_REDUCE_ELEM(VAL, PIDX, R) (R) = oz2g_mod((uint)(VAL), (PIDX))
#else
# define OZAKI_MOD_REDUCE_ELEM(VAL, PIDX, R) \
    if ((VAL) >= 0) { \
      (R) = oz2g_mod((uint)(VAL), (PIDX)); \
    } \
    else { \
      const uint nr_ = oz2g_mod((uint)(-(VAL)), (PIDX)); \
      (R) = (0 != nr_) ? (oz2g_moduli[(PIDX)] - nr_) : 0; \
    }
#endif

/**
 * Mod-reduce the PB batched accumulators into RESIDUES, which holds NRES primes
 * per sub-tile starting at prime LO (0/NPRIMES for the flat path, the group's
 * first prime/HIER_GS for the hierarchical one). ZERO_ACC clears them afterwards.
 */
#define OZAKI_CRT_REDUCE_BATCH(ACC, PIDX_BASE, LO, NRES, RESIDUES, ZERO_ACC) \
  do { \
    SINT bi_r_; \
    UNROLL_FORCE(PB) for (bi_r_ = 0; bi_r_ < PB; ++bi_r_) \
    { \
      if ((PIDX_BASE) + bi_r_ < NPRIMES) { \
        int rm_r_, rn_r_; \
        UNROLL_FORCE(RTM) for (rm_r_ = 0; rm_r_ < RTM; ++rm_r_) \
        { \
          UNROLL_FORCE(RTN) for (rn_r_ = 0; rn_r_ < RTN; ++rn_r_) \
          { \
            OZAKI_CRT_MOD_REDUCE((ACC)[bi_r_ * RTM * RTN + rm_r_ * RTN + rn_r_], (PIDX_BASE) + bi_r_, \
              (PIDX_BASE) + bi_r_ - (LO), (RESIDUES) + (rm_r_ * RTN + rn_r_) * (NRES) * XMX_FRAG); \
            if (ZERO_ACC) { \
              (ACC)[bi_r_ * RTM * RTN + rm_r_ * RTN + rn_r_] = OZAKI_ACC_ZERO; \
            } \
          } \
        } \
      } \
    } \
  } while (0)

/**
 * Cache the exponent scales a work-item needs for one sub-tile: XMX_FRAG row
 * exponents (one per fragment element) and OZAKI_FRAG_NCOL column exponents
 * (one per distinct column the fragment touches - 1 for DPAS/dp4a, 2 for MMA).
 */
#define OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, EA_C, EB_C) \
  do { \
    int fe_; \
    UNROLL_FORCE(XMX_FRAG) for (fe_ = 0; fe_ < XMX_FRAG; ++fe_) \
    { \
      (EA_C)[fe_] = (EXPA)[(MI) + OZAKI_FRAG_ROW(fe_, (LANE))]; \
    } \
    UNROLL_FORCE(OZAKI_FRAG_NCOL) for (fe_ = 0; fe_ < OZAKI_FRAG_NCOL; ++fe_) \
    { \
      const int cc_ = (NJ) + OZAKI_FRAG_COL(fe_, (LANE)); \
      (EB_C)[fe_] = (cc_ < (N)) ? (EXPB)[cc_] : 0; \
    } \
  } while (0)

/**
 * The final reconstruction, one function per variant with one signature: from
 * OZAKI_CRT_NSRC values (per-prime residues on the flat path, group values on the
 * hierarchical one) to the scaled update of a C element. Flat: Garner over all
 * primes, or the fractional CRT of mode 1. Hierarchical: level-2 Garner, or the
 * tree merge (OZAKI_HIER_L2=1). All are exact except mode 1.
 */
#if !OZAKI_HIER
# define OZAKI_CRT_NSRC NPRIMES
# if defined(OZAKI_FRACCRT) && (1 == OZAKI_FRACCRT)
#   define OZAKI_CRT_EPILOGUE oz2g_frac_accumulate
# else
#   define OZAKI_CRT_EPILOGUE oz2g_garner_accumulate
# endif
#else
# define OZAKI_CRT_NSRC HIER_NGROUPS
# if defined(OZAKI_HIER_L2) && (1 == OZAKI_HIER_L2)
#   define OZAKI_CRT_EPILOGUE oz2g_hier_tree_accumulate
# else
#   define OZAKI_CRT_EPILOGUE oz2g_hier_garner_accumulate
# endif
/**
 * Level-1 reconstruction of a group value: explicit CRT (the default, one dot
 * product and one reduction), exact per-group fractional CRT (OZAKI_FRACCRT=2), or
 * the sequential group Garner (OZAKI_L1_GARNER=1). All three are exact.
 */
# if defined(OZAKI_FRACCRT) && (2 == OZAKI_FRACCRT)
#   define OZAKI_L1_RECONSTRUCT(R, GIDX) oz2g_frac_l1((R), (GIDX))
# elif defined(HIER_L1W) && (!defined(OZAKI_L1_GARNER) || (0 == OZAKI_L1_GARNER))
#   define OZAKI_L1_RECONSTRUCT(R, GIDX) oz2g_hier_l1_crt((R), (GIDX))
# else
#   define OZAKI_L1_RECONSTRUCT(R, GIDX) oz2g_hier_l1_garner((R), (GIDX))
# endif
/* Level 1 for one sub-tile: group-local residues [HIER_GS * XMX_FRAG] to GVAL_ALL[GIDX]. */
# define OZAKI_CRT_L1_STORE(GROUP_RES, GVAL_ALL, GIDX) \
  do { \
    int ms_l1_; \
    UNROLL_FORCE(XMX_FRAG) for (ms_l1_ = 0; ms_l1_ < XMX_FRAG; ++ms_l1_) \
    { \
      uint r_[HIER_GS]; \
      SINT pg_l1_; \
      UNROLL_FORCE(HIER_GS) for (pg_l1_ = 0; pg_l1_ < HIER_GS; ++pg_l1_) \
      { \
        r_[pg_l1_] = (GROUP_RES)[(int)pg_l1_ * XMX_FRAG + ms_l1_]; \
      } \
      (GVAL_ALL)[(GIDX) * XMX_FRAG + ms_l1_] = OZAKI_L1_RECONSTRUCT(r_, (GIDX)); \
    } \
  } while (0)
#endif

/* Reconstruct, scale and store one sub-tile of C from SRC [OZAKI_CRT_NSRC * XMX_FRAG]. */
#define OZAKI_CRT_STORE(SRC, EXPA, EXPB, C_PTR, M, N, MI, NJ, LANE, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_FRAG], eb_c_[OZAKI_FRAG_NCOL]; \
    int ms_; \
    OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, ea_c_, eb_c_); \
    UNROLL_FORCE(XMX_FRAG) for (ms_ = 0; ms_ < XMX_FRAG; ++ms_) \
    { \
      const int rm_ = (MI) + OZAKI_FRAG_ROW(ms_, (LANE)); \
      const int col_ = (NJ) + OZAKI_FRAG_COL(OZAKI_FRAG_COLIDX(ms_), (LANE)); \
      if (OZAKI_IN_BOUNDS(rm_, (M), col_, (N))) { \
        const int sh_ = (int)ea_c_[ms_] + (int)eb_c_[OZAKI_FRAG_COLIDX(ms_)] - (2 * BIAS_PLUS_MANT); \
        real_t cv_ = OZAKI_IS_FIRST(FIRST) ? ZERO : (C_PTR)[(long)col_ * (LDC) + rm_]; \
        uint r_[OZAKI_CRT_NSRC]; \
        SINT pg_; \
        UNROLL_FORCE(OZAKI_CRT_NSRC) for (pg_ = 0; pg_ < OZAKI_CRT_NSRC; ++pg_) \
        { \
          r_[pg_] = (SRC)[(int)pg_ * XMX_FRAG + ms_]; \
        } \
        OZAKI_CRT_EPILOGUE(r_, (ALPHA), sh_, &cv_); \
        (C_PTR)[(long)col_ * (LDC) + rm_] = cv_; \
      } \
    } \
  } while (0)

/**
 * Work-group rasterization. The launch order runs group_id(0) fastest, so the
 * work-groups resident at any moment form a column strip of the tile grid: they
 * share one B panel, which L2 then serves, and each reads its own A panel from
 * DRAM. That is why the A-term of the residue traffic is the only one that
 * measures - halving it (BN 64 -> 128) is worth 44% while halving the B-term
 * (BM 128 -> 256) is worth nothing.
 *
 * OZAKI_SWIZZLE walks the grid in strips OZAKI_SWIZZLE tiles wide instead, so a
 * wave covers a block rather than a column and re-reads both panels a factor
 * fewer times. Both kernels derive it from the same ids, which is what keeps the
 * blocked residue layout consistent between them.
 */
#if defined(OZAKI_SWIZZLE) && (0 < OZAKI_SWIZZLE)
# define OZAKI_TILES_M(M_) (((M_) + BM - 1) / BM)
# define OZAKI_TILES_N(N_) (((N_) + BN - 1) / BN)
# define OZAKI_SWIZZLE_IDX(M_, N_, IB, JB) \
    do { \
      const int tm_sw_ = OZAKI_TILES_M(M_); \
      const int lin_sw_ = (int)get_group_id(0) + tm_sw_ * (int)get_group_id(1); \
      const int gid_sw_ = lin_sw_ / (OZAKI_SWIZZLE * OZAKI_TILES_N(N_)); \
      const int rem_sw_ = lin_sw_ % (OZAKI_SWIZZLE * OZAKI_TILES_N(N_)); \
      const int lo_sw_ = gid_sw_ * (OZAKI_SWIZZLE); \
      const int wid_sw_ = ((tm_sw_ - lo_sw_) < (OZAKI_SWIZZLE)) ? (tm_sw_ - lo_sw_) : (OZAKI_SWIZZLE); \
      (IB) = lo_sw_ + rem_sw_ % wid_sw_; \
      (JB) = rem_sw_ / wid_sw_; \
    } while (0)
#else
# define OZAKI_SWIZZLE_IDX(M_, N_, IB, JB) \
    do { \
      (IB) = (int)get_group_id(0); \
      (JB) = (int)get_group_id(1); \
    } while (0)
#endif

/**
 * Unfused reconstruction (OZAKI_UNFUSE): the GEMM writes one residue byte per
 * prime and output, a second kernel reconstructs. The point is not the extra
 * kernel but what it removes - with the prime loop outermost the fused kernel
 * has to keep every output's group values live across it, which is 2 KB per
 * work-item of dynamically indexed arrays, 512 KB per work-group against a 256 KB
 * L1. A separate pass can put the output loop outermost instead and keeps only
 * HIER_NGROUPS group values live, i.e. registers. Measured cost of the epilogue
 * inside the fused kernel: 4.44 ms of 13.06 at n=4096.
 *
 * Residues are bytes because a reduced residue is below its modulus (<=256), so
 * the round trip is nprimes*M*N bytes each way - 536 MB at n=4096, ~0.27 ms.
 *
 * The plane layout is tile-blocked and lane-contiguous rather than row/column
 * major: consecutive lanes hold columns two apart within a row of the MMA
 * fragment, so a C-layout store would scatter a warp over 32 sectors. Blocking by
 * (tile, sub-group, fragment, lane) makes both the store here and the read in
 * gemm_crt_reduce fully coalesced, and the two kernels agree on it by construction
 * - same launch geometry, same compile-time tile.
 */
#define OZAKI_RES_UPDIV(X, Y) (((X) + (Y) - 1) / (Y))
#define OZAKI_RES_TILE (BM * BN)
#define OZAKI_RES_PLANE(M_, N_) \
  ((long)OZAKI_RES_UPDIV(M_, BM) * OZAKI_RES_UPDIV(N_, BN) * OZAKI_RES_TILE)
#define OZAKI_RES_BASE(IB, JB, N_, SGI, LANE) \
  ((long)((IB) * OZAKI_RES_UPDIV(N_, BN) + (JB)) * OZAKI_RES_TILE \
    + (long)(SGI) * (RTM * RTN) * XMX_FRAG * SG + (LANE))
#define OZAKI_RES_OFF(RM, RN, MS) ((long)(((RM) * RTN + (RN)) * XMX_FRAG + (MS)) * SG)

/* Mod-reduce the whole register tile for one prime and store it as bytes. */
#define OZAKI_CRT_STORE_RESIDUES(ACC, PIDX, RES) \
  do { \
    int rm_sr_, rn_sr_; \
    UNROLL_FORCE(RTM) for (rm_sr_ = 0; rm_sr_ < RTM; ++rm_sr_) \
    { \
      UNROLL_FORCE(RTN) for (rn_sr_ = 0; rn_sr_ < RTN; ++rn_sr_) \
      { \
        OZAKI_ACC_UNION(dsr_); \
        int ms_sr_; \
        dsr_.v_ = (ACC)[rm_sr_ * RTN + rn_sr_]; \
        UNROLL_FORCE(XMX_FRAG) for (ms_sr_ = 0; ms_sr_ < XMX_FRAG; ++ms_sr_) \
        { \
          uint rsr_; \
          OZAKI_MOD_REDUCE_ELEM(dsr_.a_[ms_sr_], (PIDX), rsr_); \
          (RES)[OZAKI_RES_OFF(rm_sr_, rn_sr_, ms_sr_)] = (uchar)rsr_; \
        } \
      } \
    } \
  } while (0)

/**
 * K-loop inner body: prefetch + DPAS for PB batched primes.
 * AS_BASE, BS_BASE: base pointers for all prime planes.
 * A_PLANE, B_PLANE: per-prime plane offsets.
 * PIDX_BASE: first prime in current batch.
 * ACC: OZAKI_ACC_T array of PB*RTM*RTN accumulators.
 */
#define OZAKI_CRT_KSTEP(AS_BASE, BS_BASE, A_PLANE, B_PLANE, K_PAD_, N_PAD_, M_, MI, NJ, KOFF, PIDX_BASE, ACC) \
  do { \
    SINT bi_k_; \
    UNROLL_FORCE(PB) for (bi_k_ = 0; bi_k_ < PB; ++bi_k_) \
    { \
      if ((PIDX_BASE) + bi_k_ < NPRIMES) { \
        CONSTANT const char* as_k_ = (AS_BASE) + (long)((PIDX_BASE) + bi_k_) * (A_PLANE); \
        CONSTANT const char* bs_k_ = (BS_BASE) + (long)((PIDX_BASE) + bi_k_) * (B_PLANE); \
        OZAKI_PREFETCH_TILED(as_k_, bs_k_, K_PAD_, N_PAD_, M_, (KOFF) + BK, MI, NJ); \
        OZAKI_DPAS_TILED(as_k_, bs_k_, K_PAD_, N_PAD_, MI, NJ, KOFF, M_, (ACC) + bi_k_ * RTM * RTN); \
      } \
    } \
  } while (0)

#if defined(OZAKI_WGMMA) && (OZAKI_WGMMA)

/**
 * Warp-group MMA path (Hopper). A warp group is four warps computing 64 rows:
 * warp w owns rows w*16..w*16+15 across all BN columns, which is exactly what
 * NTM=BM/16, NTN=1, RTM=1, RTN=BN/8 make the shared mi_base/nj_base indexing
 * produce. The accumulator fragments therefore land where the existing epilogue
 * expects them and everything after the K-loop - mod-reduce, hierarchical
 * Garner, Horner, store - is reused unchanged.
 *
 * BM selects how many warp groups a work-group runs (WG_NGROUPS = BM/64). Two of
 * them (BM=128, 256 work-items) halve the residue-plane traffic per output
 * because both read the same staged B tile, at no cost in accumulators per
 * thread: the rows are added by adding warps, not registers. Warp-group rank in
 * the CTA is warp rank / 4, and SGID() is the warp rank here (get_local_id(1)
 * with a work-group of (32, NTM*NTN, 1)), so sub-groups 0-3 form the first warp
 * group and 4-7 the second, each issuing its own wgmma over its own A tile half.
 *
 * wgmma cannot be written in OpenCL C: the front-end emits .target sm_90 while
 * the instruction needs sm_90a. OZAKI_WGMMA_ISSUE is therefore a comment-only asm
 * carrying the real operands, so the compiler allocates and names the registers,
 * and the host splices the instruction into the PTX by those names (see
 * ozaki_wgmma_splice in ozaki_gemm.c). Operands: the 32 accumulators as "+r",
 * then the two shared-memory tile pointers. The descriptor's layout fields are
 * compile-time constants and are baked into the spliced text by the host.
 */
# if (1 != RTM) || ((8 != RTN) && (16 != RTN) && (32 != RTN))
#   error OZAKI_WGMMA implies RTM=1 and RTN=8, 16 or 32 (n64 / n128 / n128 twice).
# endif
/**
 * RTN=32 is a 256-column tile issued as two n128 instructions over the same A
 * fragments, not as m64n256k32: the second only shifts its B descriptor, so the
 * splice keeps the width it already handles and 64 operands still cover one
 * instruction's accumulators. It doubles the MMA work behind an A load, a barrier
 * and a drain, which is why columns pay where warp groups (BM=256) do not. A must
 * be in registers: the SS form would stage 2*(BM+BN)*WBK and not fit.
 */
# if (32 == RTN) && (!defined(OZAKI_WGMMA_RS) || (0 == OZAKI_WGMMA_RS))
#   error RTN=32 implies OZAKI_WGMMA_RS (staging both operands does not fit).
# endif
# if (64 != BM) && (128 != BM) && (256 != BM)
#   error OZAKI_WGMMA implies BM=64, 128 or 256 (one, two or four warp groups).
# endif
# if (1 != PB) || (KGROUPS > 0) || (0 == OZAKI_HIER)
#   error OZAKI_WGMMA implies PB=1, no K-grouping and hierarchical CRT.
# endif
# if (32 != SG)
#   error OZAKI_WGMMA implies SG=32 (one warp per sub-group).
# endif

/**
 * Bytes of K staged per round, work-items per work-group, warp groups per CTA.
 *
 * The staging depth is tile-specialized rather than global (OZAKI_WGMMA_KU, set per
 * specialization like BM and BN): depth costs shared memory, and shared memory is
 * what decides how many work-groups stay resident, so the right depth depends on
 * whether the tile grid fills the device. KU is the fallback for a build that does
 * not specialize it.
 */
# if defined(OZAKI_WGMMA_KU) && (0 < OZAKI_WGMMA_KU)
#   define WKU OZAKI_WGMMA_KU
# else
#   define WKU KU
# endif
# define WBK (WKU * BK)
# define WGS (SG * (BM / (XMX_M * RTM)) * (BN / (XMX_N * RTN)))
# define WG_NGROUPS (BM / 64)
# define WG_NSUB (64 / (XMX_M * RTM))

/**
 * One issue per K-chunk; the marker names the shape so the host need not assume it.
 * Two forms, distinguished by the marker itself so the splice needs no flag: SS
 * takes both operands from shared memory through descriptors, RS takes A from
 * registers (OZAKI_WGMMA_RS) and only B keeps a descriptor. The operand lists are
 * built here once; the marker text they produce is what the splice parses.
 */
# define OZAKI_WGMMA_ACC4(A, I) "+r"((A)[(I)]), "+r"((A)[(I) + 1]), "+r"((A)[(I) + 2]), "+r"((A)[(I) + 3])
# define OZAKI_WGMMA_ACC16(A, I) \
    OZAKI_WGMMA_ACC4(A, I), OZAKI_WGMMA_ACC4(A, (I) + 4), OZAKI_WGMMA_ACC4(A, (I) + 8), OZAKI_WGMMA_ACC4(A, (I) + 12)
# define OZAKI_WGMMA_ACC32(A) OZAKI_WGMMA_ACC16(A, 0), OZAKI_WGMMA_ACC16(A, 16)
# define OZAKI_WGMMA_ACC64(A) OZAKI_WGMMA_ACC32(A), OZAKI_WGMMA_ACC16(A, 32), OZAKI_WGMMA_ACC16(A, 48)
# define OZAKI_WGMMA_D32 "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22," \
    "%23,%24,%25,%26,%27,%28,%29,%30,%31"
# define OZAKI_WGMMA_D64 OZAKI_WGMMA_D32 ",%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43," \
    "%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
# if (16 == RTN)
# define OZAKI_WGMMA_ISSUE(ACCS, PA, PB_) \
    asm volatile("// WGMMA_SLOT n128 d={" OZAKI_WGMMA_D64 "} pa=%64 pb=%65" \
      : OZAKI_WGMMA_ACC64(ACCS) : "l"(PA), "l"(PB_))
# else
# define OZAKI_WGMMA_ISSUE(ACCS, PA, PB_) \
    asm volatile("// WGMMA_SLOT n64 d={" OZAKI_WGMMA_D32 "} pa=%32 pb=%33" \
      : OZAKI_WGMMA_ACC32(ACCS) : "l"(PA), "l"(PB_))
# endif

# if defined(OZAKI_WGMMA_RS) && (OZAKI_WGMMA_RS)
# if (16 == RTN) || (32 == RTN)
# define OZAKI_WGMMA_ISSUE_RS_N128(ACCS, A0, A1, A2, A3, PB_) \
    asm volatile("// WGMMA_SLOT n128 d={" OZAKI_WGMMA_D64 "} a={%64,%65,%66,%67} pb=%68" \
      : OZAKI_WGMMA_ACC64(ACCS) : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "l"(PB_))
# else
# define OZAKI_WGMMA_ISSUE_RS(ACCS, A0, A1, A2, A3, PB_) \
    asm volatile("// WGMMA_SLOT n64 d={" OZAKI_WGMMA_D32 "} a={%32,%33,%34,%35} pb=%36" \
      : OZAKI_WGMMA_ACC32(ACCS) : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "l"(PB_))
# endif
/**
 * Columns 128..255 begin halfway through the staged tile, which groups whole k-blocks
 * by column-block (OZAKI_WGMMA_BSTAGE); sharing A registers is a read-read.
 */
# if (32 == RTN)
# define OZAKI_WGMMA_BHALF (((BN) * WBK) / 32)
# define OZAKI_WGMMA_ISSUE_RS(ACCS, A0, A1, A2, A3, PB_) \
    do { \
      OZAKI_WGMMA_ISSUE_RS_N128(ACCS, A0, A1, A2, A3, PB_); \
      OZAKI_WGMMA_ISSUE_RS_N128((ACCS) + 64, A0, A1, A2, A3, (PB_) + OZAKI_WGMMA_BHALF); \
    } while (0)
# elif (16 == RTN)
# define OZAKI_WGMMA_ISSUE_RS(ACCS, A0, A1, A2, A3, PB_) \
    OZAKI_WGMMA_ISSUE_RS_N128(ACCS, A0, A1, A2, A3, PB_)
# endif
# endif

/**
 * Asynchronous staging. cp.async copies global to shared directly, without the
 * register round-trip that an ordinary load/store pair pays, and its completion is
 * tracked per group so the copies for the next K-round overlap the current round's
 * MMAs. It assembles on the plain target, unlike wgmma, so it needs no splice.
 */
# define OZAKI_WGMMA_COPY16(DST, SRC) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"l"(DST), "l"(SRC) : "memory")
# define OZAKI_WGMMA_COPY4(DST, SRC) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"l"(DST), "l"(SRC) : "memory")
# define OZAKI_WGMMA_COMMIT() asm volatile("cp.async.commit_group;" ::: "memory")
/**
 * The MMA group wait, hoisted out of the chunk loop: the issues of one round are
 * committed back to back and awaited once, so the MMA pipeline stays fed instead of
 * draining per instruction. Spliced like the issue marker (see ozaki_wgmma_splice).
 */
# define OZAKI_WGMMA_MMAWAIT() asm volatile("// WGMMA_WAIT" ::: "memory")
# define OZAKI_WGMMA_WAIT() asm volatile("cp.async.wait_group 0;" ::: "memory")
# if !defined(OZAKI_WGMMA_STAGES)
#   define OZAKI_WGMMA_STAGES 2
# endif

# if defined(OZAKI_WGMMA_RS) && (OZAKI_WGMMA_RS)
/**
 * A straight from global memory into registers, which is what the RS form exists
 * for: no staging, no shared memory and no barrier on the A side, so the tile's
 * shared footprint and its copy count both halve.
 *
 * The layout the instruction expects is mma.sync's m16n8k32 A fragment repeated
 * per warp - a0=(r,k), a1=(r+8,k), a2=(r,k+16), a3=(r+8,k+16) with r = MI + lane/4
 * and k = 4*(lane%4) - derived empirically, not assumed (wgmma-opencl-rs.c). The
 * check that matters is not that it computes a correct product but that it agrees
 * with OZAKI_FRAG_ROW/COL: exchanging a0 with a1 also multiplies correctly and
 * merely relabels the rows, which would silently transpose halves of the tile past
 * an epilogue that reads the accumulators by convention.
 *
 * Four 4-byte loads rather than one 16-byte load, yet still one sector per row and
 * lane: lanes 0-3 cover bytes 0-15 of a row through a0 and bytes 16-31 through a2,
 * so a warp's two instructions together consume 16 whole 32-byte sectors. The
 * copy-count slope that governs cp.async does not apply here - that one is paid per
 * asynchronous transaction, this one coalesces.
 */
# define OZAKI_WGMMA_ALOAD(AS_K, K_PAD_, MI, KOFF, LANE, A0, A1, A2, A3) \
    do { \
      CONSTANT const char* ap_ = (AS_K) + (long)((MI) + ((LANE) >> 2)) * (K_PAD_) + (KOFF) + ((LANE) & 3) * 4; \
      (A0) = *(CONSTANT const uint*)ap_; \
      (A1) = *(CONSTANT const uint*)(ap_ + (long)8 * (K_PAD_)); \
      (A2) = *(CONSTANT const uint*)(ap_ + 16); \
      (A3) = *(CONSTANT const uint*)(ap_ + (long)8 * (K_PAD_) + 16); \
    } while (0)
# endif

/**
 * Stage one K-round of A into shared memory in wgmma's core-matrix layout: 8x16
 * byte core matrices stored contiguously, blocks ordered (m_block, k_block)
 * row-major. Global A is [M_pad][K_pad], so rows of the tile are contiguous in K:
 * one work-item moves 16 bytes and consecutive work-items cover consecutive
 * chunks of a row. With two warp groups the m-blocks of the second are simply the
 * upper half of the same array, which is why staging needs no notion of them.
 */
# define OZAKI_WGMMA_ASTAGE(AS_K, K_PAD_, MB, KOFF, SA, WT) \
    do { \
      int ia_; \
      for (ia_ = (WT); ia_ < (BM * WBK) / 16; ia_ += WGS) { \
        const int m_ = ia_ / (WBK / 16); \
        const int j_ = ia_ % (WBK / 16); \
        OZAKI_WGMMA_COPY16((SA) + (((m_ >> 3) * (WBK / 16) + j_) * 8) + (m_ & 7), \
          (AS_K) + (long)((MB) + m_) * (K_PAD_) + (KOFF) + j_ * 16); \
      } \
    } while (0)

# if defined(OZAKI_BBLOCK) && (OZAKI_BBLOCK)
/**
 * B blocked: 16 consecutive K-values of a column are contiguous, so one work-item
 * moves 16 bytes like it does for A, and consecutive work-items read consecutive
 * columns 16 bytes apart - a warp covers 512 contiguous bytes. This is the variant
 * that removes three quarters of the copies without giving up coalescing.
 *
 * The lane must walk columns, not k-blocks: it is the column index that the global
 * layout makes contiguous here, the opposite of OZAKI_BKMAJOR. Mapping lanes to
 * k-blocks instead reads 64 KB apart and measured 9.5 against 8.1 ms.
 */
# define OZAKI_WGMMA_BSTAGE(BS_K, N_PAD_, K_PAD_, NB, KOFF, SB, WT) \
    do { \
      int ib_; \
      for (ib_ = (WT); ib_ < (BN * WBK) / 16; ib_ += WGS) { \
        const int c_ = ib_ % BN; \
        const int j_ = ib_ / BN; \
        OZAKI_WGMMA_COPY16((SB) + (((c_ >> 3) * (WBK / 16) + j_) * 8) + (c_ & 7), \
          (BS_K) + ((long)(((KOFF) >> 4) + j_) * (N_PAD_) + (NB) + c_) * 16); \
      } \
    } while (0)
# elif defined(OZAKI_BKMAJOR) && (OZAKI_BKMAJOR)
/* B transposed: a column's K is contiguous, so B stages exactly like A. */
# define OZAKI_WGMMA_BSTAGE(BS_K, N_PAD_, K_PAD_, NB, KOFF, SB, WT) \
    do { \
      int ib_; \
      for (ib_ = (WT); ib_ < (BN * WBK) / 16; ib_ += WGS) { \
        const int c_ = ib_ / (WBK / 16); \
        const int j_ = ib_ % (WBK / 16); \
        OZAKI_WGMMA_COPY16((SB) + (((c_ >> 3) * (WBK / 16) + j_) * 8) + (c_ & 7), \
          (BS_K) + (long)((NB) + c_) * (K_PAD_) + (KOFF) + j_ * 16); \
      } \
    } while (0)
# else
/**
 * B interleaved (OZAKI_BVNNI): a K-quad of one column is one aligned uint, so the
 * global side is coalesced across columns but the copies are 4 bytes wide - four
 * times the instructions of the transposed layout, which is the trade OZAKI_BKMAJOR
 * exists to make.
 */
# define OZAKI_WGMMA_BSTAGE(BS_K, N_PAD_, K_PAD_, NB, KOFF, SB, WT) \
    do { \
      int ib_; \
      for (ib_ = (WT); ib_ < (BN * WBK) / 4; ib_ += WGS) { \
        const int c_ = ib_ % BN; \
        const int q_ = ib_ / BN; \
        OZAKI_WGMMA_COPY4(((local uint*)(SB)) \
            + ((((c_ >> 3) * (WBK / 16)) + (q_ >> 2)) * 8 + (c_ & 7)) * 4 + (q_ & 3), \
          ((CONSTANT const uint*)(BS_K)) + (long)(((KOFF) >> 2) + q_) * (N_PAD_) + (NB) + c_); \
      } \
    } while (0)
# endif

/**
 * The whole K-loop for one prime, double-buffered: wait for the round staged last
 * time, publish it with one barrier (which also proves every warp has finished
 * reading the other buffer), start the next round's copies, then issue this
 * round's MMAs so the copies overlap them. One barrier per round instead of two,
 * and the global-to-shared latency is hidden rather than waited on.
 *
 * WG is the warp-group rank. Staging is over the whole work-group, so the barrier
 * publishes both A halves and the single B tile at once; only the issue is
 * per-warp-group, reading its own 64 rows of A (WG * nawg_) and the shared B.
 */
# define OZAKI_CRT_KLOOP_W(AS_BASE, BS_BASE, A_PLANE, B_PLANE, K_PAD_, N_PAD_, MB, NB, PIDX, ACCS, SA, SB, WT, WG) \
    do { \
      CONSTANT const char* asw_ = (AS_BASE) + (long)(PIDX) * (A_PLANE); \
      CONSTANT const char* bsw_ = (BS_BASE) + (long)(PIDX) * (B_PLANE); \
      const int nasz_ = (BM * WBK) / 16; \
      const int nbsz_ = (BN * WBK) / 16; \
      const int nawg_ = nasz_ / WG_NGROUPS; \
      int kw_, buf_ = 0; \
      OZAKI_WGMMA_ASTAGE(asw_, K_PAD_, MB, 0, SA, WT); \
      OZAKI_WGMMA_BSTAGE(bsw_, N_PAD_, K_PAD_, NB, 0, SB, WT); \
      OZAKI_WGMMA_COMMIT(); \
      for (kw_ = 0; kw_ < (K_PAD_); kw_ += WBK) { \
        const int next_ = kw_ + WBK; \
        int cw_; \
        OZAKI_WGMMA_WAIT(); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (next_ < (K_PAD_)) { \
          OZAKI_WGMMA_ASTAGE(asw_, K_PAD_, MB, next_, (SA) + (1 - buf_) * nasz_, WT); \
          OZAKI_WGMMA_BSTAGE(bsw_, N_PAD_, K_PAD_, NB, next_, (SB) + (1 - buf_) * nbsz_, WT); \
          OZAKI_WGMMA_COMMIT(); \
        } \
        UNROLL_FORCE(WBK / 32) for (cw_ = 0; cw_ < WBK / 32; ++cw_) { \
          OZAKI_WGMMA_ISSUE(ACCS, (SA) + buf_ * nasz_ + (WG) * nawg_ + cw_ * 16, \
            (SB) + buf_ * nbsz_ + cw_ * 16); \
        } \
        OZAKI_WGMMA_MMAWAIT(); \
        buf_ = 1 - buf_; \
      } \
    } while (0)

# if defined(OZAKI_WGMMA_RS) && (OZAKI_WGMMA_RS)
/**
 * A in registers: only B is staged, so a round costs one cp.async group and one
 * barrier for half the shared memory. OZAKI_WGMMA_DEFER moves the MMA wait from
 * after the issues to before the next round's barrier, which is where the B buffer
 * and the A registers it protects are first overwritten; the A sets alternate by
 * round and are named, since a runtime index would put them in local memory. The
 * host selects it for the full tile only: -7% GEMM at n=4096 there, but every
 * narrowed tile measured a loss (4-20%, registers cost occupancy or spill).
 * OZAKI_WGMMA_STAGES=3 (third B buffer and A set, wait_group leaving one round
 * in flight) is refuted: 3.09 ms at half depth, 2.93 at KU=6, 9.67 at equal depth
 * (the SLM/L1 cliff, since RS fetches A through L1) against 2.86 for two stages.
 */
# if 3 == OZAKI_WGMMA_STAGES
#   if !defined(OZAKI_WGMMA_DEFER) || (0 == OZAKI_WGMMA_DEFER) || !defined(OZAKI_WGMMA_ROUND_GROUPS)
#     error OZAKI_WGMMA_STAGES=3 needs the deferred wait and OZAKI_WGMMA_ROUND_GROUPS (the host emits both).
#   endif
#   define OZAKI_WGMMA_STR_(X) #X
#   define OZAKI_WGMMA_STR(X) OZAKI_WGMMA_STR_(X)
#   define OZAKI_WGMMA_MMAWAIT_ROUND() \
      asm volatile("// WGMMA_WAIT " OZAKI_WGMMA_STR(OZAKI_WGMMA_ROUND_GROUPS) ::: "memory")
#   define OZAKI_WGMMA_AF3 , af2_[(WBK / 32) * 4]
#   define OZAKI_CRT_ROUND3_WRS(ASW, BSW, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, KW, NBSZ) \
      do { \
        if ((KW) + 2 * WBK < (K_PAD_)) { \
          OZAKI_CRT_ROUND_WRS(ASW, BSW, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, (KW) + 2 * WBK, af2_, 2, NBSZ); \
        } \
      } while (0)
# elif 2 == OZAKI_WGMMA_STAGES
#   define OZAKI_WGMMA_MMAWAIT_ROUND() OZAKI_WGMMA_MMAWAIT()
#   define OZAKI_WGMMA_AF3
#   define OZAKI_CRT_ROUND3_WRS(ASW, BSW, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, KW, NBSZ) ((void)0)
# else
#   error OZAKI_WGMMA_STAGES must be 2 or 3.
# endif
# if defined(OZAKI_WGMMA_DEFER) && (OZAKI_WGMMA_DEFER)
#   define OZAKI_WGMMA_WAIT_PRE() OZAKI_WGMMA_MMAWAIT_ROUND()
#   define OZAKI_WGMMA_WAIT_POST()
#   define OZAKI_WGMMA_DRAIN() OZAKI_WGMMA_MMAWAIT()
# else
#   define OZAKI_WGMMA_WAIT_PRE()
#   define OZAKI_WGMMA_WAIT_POST() OZAKI_WGMMA_MMAWAIT()
#   define OZAKI_WGMMA_DRAIN()
# endif
# define OZAKI_CRT_ROUND_WRS(ASW, BSW, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, KW, AF, BUF, NBSZ) \
    do { \
      const int next_ = (KW) + WBK; \
      int cw_; \
      UNROLL_FORCE(WBK / 32) for (cw_ = 0; cw_ < WBK / 32; ++cw_) { \
        OZAKI_WGMMA_ALOAD(ASW, K_PAD_, MI, (KW) + cw_ * 32, LANE, \
          AF[cw_ * 4], AF[cw_ * 4 + 1], AF[cw_ * 4 + 2], AF[cw_ * 4 + 3]); \
      } \
      OZAKI_WGMMA_WAIT(); \
      OZAKI_WGMMA_WAIT_PRE(); \
      barrier(CLK_LOCAL_MEM_FENCE); \
      if (next_ < (K_PAD_)) { \
        OZAKI_WGMMA_BSTAGE(BSW, N_PAD_, K_PAD_, NB, next_, (SB) + (((BUF) + 1) % OZAKI_WGMMA_STAGES) * (NBSZ), WT); \
        OZAKI_WGMMA_COMMIT(); \
      } \
      UNROLL_FORCE(WBK / 32) for (cw_ = 0; cw_ < WBK / 32; ++cw_) { \
        OZAKI_WGMMA_ISSUE_RS(ACCS, AF[cw_ * 4], AF[cw_ * 4 + 1], AF[cw_ * 4 + 2], AF[cw_ * 4 + 3], \
          (SB) + (BUF) * (NBSZ) + cw_ * 16); \
      } \
      OZAKI_WGMMA_WAIT_POST(); \
    } while (0)
# define OZAKI_CRT_KLOOP_WRS(AS_BASE, BS_BASE, A_PLANE, B_PLANE, K_PAD_, N_PAD_, MI, NB, PIDX, ACCS, SB, WT, LANE) \
    do { \
      CONSTANT const char* asw_ = (AS_BASE) + (long)(PIDX) * (A_PLANE); \
      CONSTANT const char* bsw_ = (BS_BASE) + (long)(PIDX) * (B_PLANE); \
      const int nbsz_ = (BN * WBK) / 16; \
      uint af0_[(WBK / 32) * 4], af1_[(WBK / 32) * 4] OZAKI_WGMMA_AF3; \
      int kw_; \
      OZAKI_WGMMA_BSTAGE(bsw_, N_PAD_, K_PAD_, NB, 0, SB, WT); \
      OZAKI_WGMMA_COMMIT(); \
      for (kw_ = 0; kw_ < (K_PAD_); kw_ += OZAKI_WGMMA_STAGES * WBK) { \
        OZAKI_CRT_ROUND_WRS(asw_, bsw_, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, kw_, af0_, 0, nbsz_); \
        if (kw_ + WBK < (K_PAD_)) { \
          OZAKI_CRT_ROUND_WRS(asw_, bsw_, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, kw_ + WBK, af1_, 1, nbsz_); \
        } \
        OZAKI_CRT_ROUND3_WRS(asw_, bsw_, K_PAD_, N_PAD_, MI, NB, ACCS, SB, WT, LANE, kw_, nbsz_); \
      } \
      OZAKI_WGMMA_DRAIN(); \
    } while (0)
# endif

#endif /* OZAKI_WGMMA */

/**
 * The full K-loop for one prime batch, whichever matrix engine is in use. Named
 * once so the fused and unfused prime loops below cannot drift apart; it reads the
 * kernel's own operands (as, bs, the padded extents, the tile bases) by name.
 */
#if defined(OZAKI_WGMMA_RS) && (OZAKI_WGMMA_RS)
# define OZAKI_CRT_KLOOP_RUN(ACC, PIDX) \
    OZAKI_CRT_KLOOP_WRS(as, bs, a_plane, b_plane, K_pad, N_pad, mi_base, nb_base, PIDX, \
      (ACC).s_, wg_sb, wt, sg_lid)
#elif defined(OZAKI_WGMMA) && (OZAKI_WGMMA)
# define OZAKI_CRT_KLOOP_RUN(ACC, PIDX) \
    OZAKI_CRT_KLOOP_W(as, bs, a_plane, b_plane, K_pad, N_pad, mb_base, nb_base, PIDX, \
      (ACC).s_, wg_sa, wg_sb, wt, wg_id)
#else
# define OZAKI_CRT_KLOOP_RUN(ACC, PIDX) \
    do { \
      int kr_; \
      for (kr_ = 0; kr_ < K_pad; kr_ += KU * BK) { \
        int kur_; \
        UNROLL_FORCE(KU) for (kur_ = 0; kur_ < KU; ++kur_) \
        { \
          OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, kr_ + kur_ * BK, PIDX, ACC); \
        } \
      } \
    } while (0)
#endif

/**
 * Accumulator storage. The wgmma path needs the same registers addressable both
 * as a flat int array (the instruction takes one register vector) and as XMX_FRAG
 * fragments (what the epilogue consumes); a union provides that at no cost.
 */
#if defined(OZAKI_WGMMA) && (OZAKI_WGMMA)
# define OZAKI_ACC_DECL(NAME) \
  union { \
    OZAKI_ACC_T v_[PB * RTM * RTN]; \
    int s_[PB * RTM * RTN * XMX_FRAG]; \
  } NAME
# define OZAKI_ACC_FRAGS(NAME) ((NAME).v_)
# define OZAKI_ACC_ZERO_ALL(NAME) \
  do { \
    int az_; \
    UNROLL_FORCE(PB * RTM * RTN * XMX_FRAG) \
    for (az_ = 0; az_ < PB * RTM * RTN * XMX_FRAG; ++az_) { \
      (NAME).s_[az_] = 0; \
    } \
  } while (0)
#else
# define OZAKI_ACC_DECL(NAME) OZAKI_ACC_T NAME[PB * RTM * RTN]
# define OZAKI_ACC_FRAGS(NAME) (NAME)
# define OZAKI_ACC_ZERO_ALL(NAME) \
  do { \
    int az_; \
    UNROLL_FORCE(PB * RTM * RTN) \
    for (az_ = 0; az_ < PB * RTM * RTN; ++az_) { \
      (NAME)[az_] = OZAKI_ACC_ZERO; \
    } \
  } while (0)
#endif

/**
 * One prime batch of the fused kernel: the K-loop, then the mod-reduce into
 * RESIDUES (see OZAKI_CRT_REDUCE_BATCH for LO/NRES). With KGROUPS the reduction
 * also fires every KGROUPS steps, which is what keeps a long K inside int32.
 */
#if KGROUPS > 0
# define OZAKI_CRT_KLOOP_REDUCE(ACC, PIDX, LO, NRES, RESIDUES) \
    do { \
      int k_, steps_ = 0; \
      for (k_ = 0; k_ < K_pad; k_ += KU * BK) { \
        int ku_; \
        UNROLL_FORCE(KU) for (ku_ = 0; ku_ < KU; ++ku_) \
        { \
          OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k_ + ku_ * BK, PIDX, \
            OZAKI_ACC_FRAGS(ACC)); \
        } \
        steps_ += KU; \
        if (steps_ >= KGROUPS) { \
          OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(ACC), PIDX, LO, NRES, RESIDUES, 1); \
          steps_ = 0; \
        } \
      } \
      if (0 != steps_) { \
        OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(ACC), PIDX, LO, NRES, RESIDUES, 0); \
      } \
    } while (0)
#else
# define OZAKI_CRT_KLOOP_REDUCE(ACC, PIDX, LO, NRES, RESIDUES) \
    do { \
      OZAKI_CRT_KLOOP_RUN(ACC, PIDX); \
      OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(ACC), PIDX, LO, NRES, RESIDUES, 0); \
    } while (0)
#endif


/**
 * CRT moduli, Barrett constants, pow32_mod, and Garner inverse table.
 *
 * Snake-draft interleaving balances HIER group products.
 * Power-of-2 modulus at POW2_PIDX (last in group 0) for bitmask fast path.
 *
 * u8 (OZAKI_U8=1, default): 20 pairwise coprime integers <= 256.
 *   Prime powers: 256=2^8, 243=3^5, 169=13^2.  Rest are primes.
 *   Safe K without KGROUPS: ~33K (255^2 * 32 per DPAS step).
 *
 * i8 (OZAKI_U8=0): 20 pairwise coprime integers <= 128.
 *   Prime powers: 128=2^7, 125=5^3, 121=11^2, 81=3^4.  119=7*17.
 *   Safe K without KGROUPS: ~133K (127^2 * 32 per DPAS step).
 */

#if defined(OZAKI_U8) && (OZAKI_U8)

constant ushort oz2g_moduli[] = {211, 199, 163, 256, 251, 223, 197, 167, 243, 227, 193, 169, 241, 229, 191, 173, 239, 233, 181, 179};

constant uint oz2g_barrett_inv[] = {20355295, 21582750, 26349492, 16777216, 17111423, 19259943, 21801864, 25718367, 17674762, 18920560,
  22253716, 25414007, 17821441, 18755315, 22486739, 24826400, 17970574, 18433336, 23729101, 23994230};

constant ushort oz2g_pow32_mod[] = {51, 46, 100, 0, 123, 7, 88, 7, 130, 176, 108, 113, 15, 161, 147, 96, 110, 8, 15, 126};

constant uint oz2g_garner_inv[][20] = {
  /* m 0=211 */ {0, 83, 17, 91, 69, 130, 183, 19, 205, 156, 118, 165, 8, 89, 86, 41, 128, 180, 175, 28},
  /* m 1=199 */ {0, 0, 77, 247, 111, 65, 99, 47, 127, 154, 161, 62, 109, 145, 24, 20, 233, 185, 171, 9},
  /* m 2=163 */ {0, 0, 0, 11, 77, 26, 168, 125, 82, 39, 45, 28, 207, 170, 75, 121, 22, 223, 10, 123},
  /* m 3=256 */ {0, 0, 0, 0, 201, 196, 187, 152, 187, 47, 144, 68, 225, 17, 144, 148, 225, 152, 70, 93},
  /* m 4=251 */ {0, 0, 0, 0, 0, 8, 135, 2, 152, 123, 10, 101, 217, 177, 156, 122, 20, 13, 75, 92},
  /* m 5=223 */ {0, 0, 0, 0, 0, 0, 144, 3, 85, 170, 148, 72, 174, 38, 6, 45, 224, 163, 125, 118},
  /* m 6=197 */ {0, 0, 0, 0, 0, 0, 0, 39, 206, 174, 145, 163, 115, 93, 32, 137, 165, 110, 34, 10},
  /* m 7=167 */ {0, 0, 0, 0, 0, 0, 0, 0, 227, 87, 141, 84, 127, 48, 183, 144, 156, 60, 168, 164},
  /* m 8=243 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 166, 16, 121, 180, 180, 131, 60, 70, 73, 14},
  /* m 9=227 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 102, 86, 114, 69, 157, 219, 194, 122, 138},
  /* m10=193 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 5, 159, 96, 26, 213, 99, 166, 64},
  /* m11=169 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 164, 187, 26, 43, 99, 91, 15, 161},
  /* m12=241 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 210, 149, 28, 120, 204, 178, 26},
  /* m13=229 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 186, 34, 215, 58, 132, 111},
  /* m14=191 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125, 234, 61, 163, 15},
  /* m15=173 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 66, 113, 149},
  /* m16=239 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 103, 3},
  /* m17=233 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 63},
  /* m18=181 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90},
  /* m19=179 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

#else /* i8 fallback */

constant ushort oz2g_moduli[] = {101, 97, 59, 128, 127, 103, 89, 61, 125, 107, 83, 67, 121, 109, 81, 71, 119, 113, 79, 73};

constant uint oz2g_barrett_inv[] = {42524428, 44278013, 72796055, 33554432, 33818640, 41698711, 48258059, 70409299, 34359738, 40139881,
  51746593, 64103989, 35495597, 39403369, 53024287, 60492497, 36092162, 38008560, 54366674, 58835168};

constant ushort oz2g_pow32_mod[] = {68, 35, 51, 0, 16, 63, 45, 57, 46, 29, 77, 33, 59, 75, 49, 9, 18, 16, 50, 32};

constant uint oz2g_garner_inv[][20] = {
  /* m 0=101 */ {0, 73, 52, 109, 83, 51, 52, 29, 26, 89, 60, 2, 6, 68, 77, 45, 33, 47, 18, 60},
  /* m 1= 97 */ {0, 0, 14, 33, 55, 17, 78, 39, 58, 32, 6, 38, 5, 9, 76, 41, 27, 7, 22, 70},
  /* m 2= 59 */ {0, 0, 0, 115, 28, 7, 86, 30, 89, 78, 38, 25, 80, 85, 11, 65, 117, 23, 75, 26},
  /* m 3=128 */ {0, 0, 0, 0, 1, 33, 16, 51, 42, 51, 24, 11, 52, 23, 50, 5, 53, 98, 50, 4},
  /* m 4=127 */ {0, 0, 0, 0, 0, 73, 82, 49, 63, 91, 17, 19, 101, 103, 37, 52, 15, 105, 28, 23},
  /* m 5=103 */ {0, 0, 0, 0, 0, 0, 70, 16, 17, 80, 54, 54, 47, 18, 70, 20, 52, 79, 56, 56},
  /* m 6= 89 */ {0, 0, 0, 0, 0, 0, 0, 24, 59, 101, 14, 64, 34, 49, 71, 4, 115, 80, 8, 32},
  /* m 7= 61 */ {0, 0, 0, 0, 0, 0, 0, 0, 41, 100, 49, 11, 2, 84, 4, 7, 80, 63, 57, 6},
  /* m 8=125 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 52, 91, 75, 35, 25, 20, 66, 67, 66},
  /* m 9=107 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 62, 95, 54, 53, 2, 109, 94, 48, 58},
  /* m10= 83 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 35, 88, 41, 6, 76, 64, 20, 22},
  /* m11= 67 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 96, 52, 53, 16, 27, 46, 12},
  /* m12=121 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 79, 27, 60, 99, 32, 35},
  /* m13=109 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 43, 107, 28, 29, 71},
  /* m14= 81 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 72, 60, 40, 64},
  /* m15= 71 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 78, 69, 36},
  /* m16=119 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 2, 27},
  /* m17=113 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 42},
  /* m18= 79 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61},
  /* m19= 73 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

#endif /* OZAKI_U8 */

#define OZ2G_BARRETT_SHIFT 32

/**
 * Barrett modular reduction: x mod oz2g_moduli[pidx].
 * POW2_PIDX is the power-of-2 modulus (bitmask fast path).
 * u8: 256 = 2^8 -> mask 0xFF.  i8: 128 = 2^7 -> mask 0x7F.
 */
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZ2G_POW2_MASK 0xFFu
# define OZ2G_POW2_MASK64 0xFFul
#else
# define OZ2G_POW2_MASK 0x7Fu
# define OZ2G_POW2_MASK64 0x7Ful
#endif
inline uint oz2g_mod(uint x, SINT pidx)
{
  if (POW2_PIDX == pidx) return x & OZ2G_POW2_MASK;
  {
    const uint q = (uint)(((ulong)x * oz2g_barrett_inv[pidx]) >> OZ2G_BARRETT_SHIFT);
    uint r = x - q * oz2g_moduli[pidx];
    return (r >= oz2g_moduli[pidx]) ? (r - oz2g_moduli[pidx]) : r;
  }
}

/**
 * Modular reduction for aligned mantissa (up to 53 bits for FP64, 24 for FP32).
 * Decomposes x = hi*2^32 + lo, reduces each part via 32-bit Barrett,
 * then combines.  Avoids expensive 64-bit integer division.
 */
inline uint oz2g_mod64(ulong x, SINT pidx)
{
  if (POW2_PIDX == pidx) return (uint)(x & OZ2G_POW2_MASK64);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  {
    const uint hi = (uint)(x >> 32);
    const uint lo = (uint)x;
    const uint partial = hi * oz2g_pow32_mod[pidx] + oz2g_mod(lo, pidx);
    return oz2g_mod(partial, pidx);
  }
#else
  /* FP32: aligned mantissa <= 24 bits, direct 32-bit Barrett. */
  return oz2g_mod((uint)x, pidx);
#endif
}


#if defined(OZAKI_FRACCRT) && (OZAKI_FRACCRT)
/**
 * Fractional CRT: x/M = frac(sum_i alpha_i / m_i) with alpha_i = (r_i * k_i) mod
 * m_i and 1/m_i expanded into OZ2G_FRAC_L base-256 limbs, so the sum is O(P*L)
 * independent MACs instead of Garner's O(P^2) chain, combined in double-double.
 * Mode 1 applies it over all P moduli with a centered lift for the sign (opt-in:
 * a magnitude-domain bound); mode 2 per hierarchical group, where M_g < 2^53
 * keeps it exact. Both need the error-free transformations intact, hence no
 * fast-relaxed-math on these builds. Tables come from ozaki_emit_fraccrt.
 */
constant uint oz2g_frac_k[NPRIMES] = OZ2G_FRAC_K;
constant uchar oz2g_frac_climb[NPRIMES][OZ2G_FRAC_L] = OZ2G_FRAC_CLIMB;

inline double oz2g_two_sum(double a, double b, double* err)
{
  const double s = a + b;
  const double bb = s - a;
  *err = (a - (s - bb)) + (b - bb);
  return s;
}


inline double oz2g_two_prod(double a, double b, double* err)
{
  const double p = a * b;
  *err = fma(a, b, -p);
  return p;
}

/* Fractional part of the sum over COUNT primes from LO with residues R, as the double-double (FRH, FRL). */
#define OZAKI_FRAC_SUM(R, LO, COUNT, FRH, FRL) \
  do { \
    double sl_[OZ2G_FRAC_L], fh_ = 0.0, fl_ = 0.0, e_, s_; \
    SINT li_, l_; \
    UNROLL_FORCE(OZ2G_FRAC_L) for (l_ = 0; l_ < OZ2G_FRAC_L; ++l_) sl_[l_] = 0.0; \
    UNROLL_FORCE(COUNT) for (li_ = 0; li_ < (COUNT); ++li_) { \
      const int p_ = (LO) + (int)li_; \
      if (p_ < NPRIMES) { \
        const uint a_ = oz2g_mod((R)[li_] * oz2g_frac_k[p_], p_); \
        UNROLL_FORCE(OZ2G_FRAC_L) for (l_ = 0; l_ < OZ2G_FRAC_L; ++l_) { \
          sl_[l_] += (double)(a_ * (uint)oz2g_frac_climb[p_][l_]); \
        } \
      } \
    } \
    UNROLL_FORCE(OZ2G_FRAC_L) for (l_ = 0; l_ < OZ2G_FRAC_L; ++l_) { \
      const double term_ = sl_[l_] * EXP2I(-8 * (l_ + 1)); \
      s_ = oz2g_two_sum(fh_, term_, &e_); \
      fh_ = oz2g_two_sum(s_, e_ + fl_, &e_); \
      fl_ = e_; \
    } \
    s_ = floor(fh_); \
    (FRH) = oz2g_two_sum(fh_, -s_, &e_); \
    (FRL) = e_ + fl_; \
    s_ = oz2g_two_sum((FRH), (FRL), &e_); \
    (FRH) = s_; \
    (FRL) = e_; \
  } while (0)

# if 1 == OZAKI_FRACCRT
inline void oz2g_frac_accumulate(const uint* restrict r, real_t alpha, int base_sh, real_t* cval)
{
  double frh, frl, e, eh, vh, vl;
  OZAKI_FRAC_SUM(r, 0, NPRIMES, frh, frl);
  /* frac lies in [0,1) and the negative half above 1/2, so floor(frac + 0.5) folds the sign in. */
  vh = floor(frh + 0.5);
  frh = oz2g_two_sum(frh, -vh, &e);
  frl += e;
  vh = oz2g_two_sum(frh, frl, &e);
  frh = vh;
  frl = e;
  vh = oz2g_two_prod(frh, OZ2G_FRAC_MH, &eh);
  vl = frh * OZ2G_FRAC_ML + frl * OZ2G_FRAC_MH + eh;
  { const double val = vh + vl;
    if (0.0 != val && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
      *cval += (real_t)(val * (double)scale);
    }
  }
}
# else
constant double oz2g_frac_gmh[HIER_NGROUPS] = OZ2G_FRAC_GMH;

/* Group value x mod M_g from the group's residues; M_g < 2^53, so exact for every value. */
inline uint oz2g_frac_l1(const uint* restrict group_residues, int g)
{
  double frh, frl, eh, vh, vl;
  OZAKI_FRAC_SUM(group_residues, g * HIER_GS, HIER_GS, frh, frl);
  vh = oz2g_two_prod(frh, oz2g_frac_gmh[g], &eh);
  vl = frl * oz2g_frac_gmh[g] + eh;
  return (uint)(vh + vl + 0.5);
}
# endif
#endif /* OZAKI_FRACCRT */


/**
 * Garner chain: mixed-radix digits V from N residues R at index offset OFF. The
 * flat, leaf and level-2 chains differ only in the modulus table, the reduction
 * of a digit to the current modulus and the product reduction, which the hooks
 * supply (OZ2G_FLAT_*, OZ2G_L2_*).
 */
#define OZAKI_GARNER_CHAIN(R, V, N, OFF, MODULUS, REDUCE_DIGIT, REDUCE_PROD, INV) \
  do { \
    SINT i_, j_; \
    for (i_ = 0; i_ < (N); ++i_) { \
      uint u_ = (R)[i_]; \
      const uint m_ = MODULUS((OFF) + i_); \
      for (j_ = 0; j_ < i_; ++j_) { \
        uint vj_ = (V)[j_]; \
        REDUCE_DIGIT(vj_, m_, (OFF) + i_); \
        { \
          const uint diff_ = (u_ >= vj_) ? (u_ - vj_) : (m_ + u_ - vj_); \
          u_ = REDUCE_PROD(diff_, INV((OFF) + j_, (OFF) + i_), (OFF) + i_); \
        } \
      } \
      (V)[i_] = u_; \
    } \
  } while (0)
/**
 * Sign from the top digit, then two's complement with the +1 carry propagated in
 * integer space: added after the float conversion it would be inexact beyond
 * 2^MANT_BITS, so Horner already yields |x|.
 */
#define OZAKI_GARNER_SIGN(V, N, MODULUS, IS_NEG) \
  do { \
    SINT i_; \
    (IS_NEG) = ((V)[(N) - 1] >= ((uint)MODULUS((N) - 1) + 1u) / 2u) ? 1 : 0; \
    if (0 != (IS_NEG)) { \
      UNROLL_FORCE(N) for (i_ = 0; i_ < (N); ++i_) \
      { \
        (V)[i_] = (uint)MODULUS(i_) - 1u - (V)[i_]; \
      } \
      for (i_ = 0; i_ < (N); ++i_) { \
        if ((V)[i_] + 1u < (uint)MODULUS(i_)) { \
          (V)[i_] += 1u; \
          break; \
        } \
        (V)[i_] = 0; \
      } \
    } \
  } while (0)
/* Horner over N mixed-radix digits D, GROUP moduli at a time in ulong, into the real_t RESULT. */
#define OZAKI_HORNER(D, N, GROUP, MODULUS, RESULT) \
  do { \
    const int ngroups_ = ((N) + (GROUP) - 1) / (GROUP); \
    int g_, i_; \
    { \
      const int lo_ = (ngroups_ - 1) * (GROUP); \
      ulong r_ = (ulong)(D)[(N) - 1]; \
      for (i_ = (N) - 2; i_ >= lo_; --i_) r_ = r_ * (ulong)MODULUS(i_) + (ulong)(D)[i_]; \
      (RESULT) = (real_t)r_; \
    } \
    for (g_ = ngroups_ - 2; g_ >= 0; --g_) { \
      const int lo_ = g_ * (GROUP), hi_ = lo_ + (GROUP) - 1; \
      ulong gval_ = (ulong)(D)[hi_], gprod_ = 1; \
      for (i_ = lo_; i_ <= hi_; ++i_) gprod_ *= (ulong)MODULUS(i_); \
      for (i_ = hi_ - 1; i_ >= lo_; --i_) gval_ = gval_ * (ulong)MODULUS(i_) + (ulong)(D)[i_]; \
      (RESULT) = (RESULT) * (real_t)gprod_ + (real_t)gval_; \
    } \
  } while (0)
#define OZ2G_FLAT_MOD(I) oz2g_moduli[(I)]
#define OZ2G_FLAT_DIGIT(V, M, I) \
  do { \
    if ((V) >= (M)) (V) -= (M); \
    if ((V) >= (M)) (V) -= (M); \
  } while (0)
#define OZ2G_FLAT_PROD(D, INV, I) oz2g_mod((D) * (INV), (I))
#define OZ2G_FLAT_INV(J, I) oz2g_garner_inv[(J)][(I)]

/* Scale the reconstructed |x| by alpha * 2^base_sh, sign it and add it to C. */
inline void oz2g_accumulate(real_t result, int is_negative, real_t alpha, int base_sh, real_t* cval)
{
  result = (0 != is_negative) ? -result : result;
  if (ZERO != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
    const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
    *cval += result * scale;
  }
}


/* Flat reconstruction: Garner over all NPRIMES residues, Horner in groups that fit a ulong. */
inline void oz2g_garner_accumulate(const uint* restrict r, real_t alpha, int base_sh, real_t* cval)
{
  uint v[NPRIMES];
  real_t result;
  int is_negative;
  OZAKI_GARNER_CHAIN(r, v, NPRIMES, 0, OZ2G_FLAT_MOD, OZ2G_FLAT_DIGIT, OZ2G_FLAT_PROD, OZ2G_FLAT_INV);
  OZAKI_GARNER_SIGN(v, NPRIMES, OZ2G_FLAT_MOD, is_negative);
  OZAKI_HORNER(v, NPRIMES, OZ2_HORNER_GROUP, OZ2G_FLAT_MOD, result);
  oz2g_accumulate(result, is_negative, alpha, base_sh, cval);
}


#if OZAKI_HIER

/**
 * Level-2 tables from the host: group products, their Barrett constants
 * floor(2^64 / gprod), and gprod_j^-1 mod gprod_i at [j][i] above the diagonal.
 * A list shorter than the group count would be read past silently (17 primes at
 * HIER_GS=3 once did), so the sizes are checked at build time.
 */
# if !defined(HIER_GPROD)
/* Standalone build (no host): the tables of the 20-prime, four-per-group layout above. */
#   if defined(OZAKI_U8) && (OZAKI_U8)
#     define HIER_GPROD {1752116992u, 1841455727u, 1799186337u, 1823610127u, 1804203113u}
#     define HIER_L2B {10528260474ul, 10017478999ul, 10252825788ul, 10115508682ul, 10224316730ul}
#     define HIER_L2INV {0u, 828768696u, 1255745875u, 96929798u, 430518282u, 0u, 0u, 1062200843u, 1479311133u, 742073819u, \
        0u, 0u, 0u, 1583419479u, 1296690879u, 0u, 0u, 0u, 0u, 1036097590u, 0u, 0u, 0u, 0u, 0u}
#   else
#     define HIER_GPROD {73986944u, 71016749u, 74378375u, 75849939u, 77548849u}
#     define HIER_L2B {249324314215ul, 259752020945ul, 248012195395ul, 243200512972ul, 237872570793ul}
#     define HIER_L2INV {0u, 16740944u, 25622404u, 62222726u, 40198002u, 0u, 0u, 20777749u, 7009982u, 11759761u, \
        0u, 0u, 0u, 1845215u, 15543578u, 0u, 0u, 0u, 0u, 54885903u, 0u, 0u, 0u, 0u, 0u}
#   endif
# endif
# if !defined(HIER_L2B) || !defined(HIER_L2INV)
#   error hierarchical CRT needs all three level-2 tables (HIER_GPROD, HIER_L2B, HIER_L2INV).
# endif
constant uint oz2g_hier_gprod[] = HIER_GPROD;
constant ulong oz2g_hier_l2b[] = HIER_L2B;
constant uint oz2g_hier_l2inv[] = HIER_L2INV;
typedef char oz2g_hier_tables_check[(sizeof(oz2g_hier_gprod) == HIER_NGROUPS * sizeof(uint)
  && sizeof(oz2g_hier_l2b) == HIER_NGROUPS * sizeof(ulong)
  && sizeof(oz2g_hier_l2inv) == HIER_NGROUPS * HIER_NGROUPS * sizeof(uint)) ? 1 : -1];

/* Level-2 Barrett reduction: x mod gprod[gidx]. */
inline uint oz2g_mod_l2(ulong x, int gidx)
{
  const uint m = oz2g_hier_gprod[gidx];
  const ulong q = mul_hi(x, oz2g_hier_l2b[gidx]);
  uint r = (uint)(x - q * (ulong)m);
  return (r >= m) ? (r - m) : r;
}


#if defined(HIER_L1W)
constant uint oz2g_hier_l1w[] = HIER_L1W;
/**
 * Level-1 explicit CRT: the group value is (sum_i r_i * w_i) mod gprod, which is one
 * dot product and the same Barrett level 2 already uses. Residues are below 256 and
 * the weights below gprod, so HIER_GS terms stay under 2^10 * gprod and the sum fits
 * a ulong for any group product that fits uint32 - which is exactly the leaf's
 * property, and the reason Garner's chain of dependent reductions is only needed
 * where the modulus is the full product.
 */
inline uint oz2g_hier_l1_crt(const uint* restrict group_residues, int g)
{
  ulong s = 0;
  SINT li;
  UNROLL_FORCE(HIER_GS) for (li = 0; li < HIER_GS; ++li)
  {
    s += (ulong)group_residues[li] * (ulong)oz2g_hier_l1w[g * HIER_GS + (int)li];
  }
  return oz2g_mod_l2(s, g);
}
#endif


/* Level-1 Garner for group g (OZAKI_L1_GARNER=1): the flat chain at the group's offset. */
inline uint oz2g_hier_l1_garner(const uint* restrict group_residues, int g)
{
  const int lo = g * HIER_GS;
  const int gsz = ((lo + HIER_GS <= NPRIMES) ? (lo + HIER_GS) : NPRIMES) - lo;
  uint v[HIER_GS];
  ulong hval;
  SINT li;
  OZAKI_GARNER_CHAIN(group_residues, v, gsz, lo, OZ2G_FLAT_MOD, OZ2G_FLAT_DIGIT, OZ2G_FLAT_PROD, OZ2G_FLAT_INV);
  hval = (ulong)v[gsz - 1];
  for (li = gsz - 2; li >= 0; --li) hval = hval * (ulong)oz2g_moduli[lo + li] + (ulong)v[li];
  return (uint)hval;
}

#define OZ2G_L2_MOD(I) oz2g_hier_gprod[(I)]
#define OZ2G_L2_DIGIT(V, M, I) \
  do { \
    if ((V) >= (M)) (V) = oz2g_mod_l2((ulong)(V), (I)); \
  } while (0)
#define OZ2G_L2_PROD(D, INV, I) oz2g_mod_l2((ulong)(D) * (ulong)(INV), (I))
#define OZ2G_L2_INV(J, I) oz2g_hier_l2inv[(J) * HIER_NGROUPS + (I)]

#if !defined(OZAKI_HIER_L2) || (0 == OZAKI_HIER_L2)
/* Level 2: Garner over the group values, Horner over its digits two groups at a time. */
inline void oz2g_hier_garner_accumulate(const uint* restrict gval, real_t alpha, int base_sh, real_t* cval)
{
  uint d[HIER_NGROUPS];
  real_t result;
  int is_negative;
  OZAKI_GARNER_CHAIN(gval, d, HIER_NGROUPS, 0, OZ2G_L2_MOD, OZ2G_L2_DIGIT, OZ2G_L2_PROD, OZ2G_L2_INV);
  OZAKI_GARNER_SIGN(d, HIER_NGROUPS, OZ2G_L2_MOD, is_negative);
  OZAKI_HORNER(d, HIER_NGROUPS, HIER_L2_HORNER_GROUP, OZ2G_L2_MOD, result);
  oz2g_accumulate(result, is_negative, alpha, base_sh, cval);
}
#endif /* OZAKI_HIER_L2 == 0 */


#if defined(OZAKI_HIER_L2) && (1 == OZAKI_HIER_L2)
/**
 * Tree merge for one or two groups: a single pairwise merge instead of the Garner
 * chain, yielding |x| directly rather than complemented digits. More groups would
 * compile with no branch taken and leave the value unassigned.
 */
# if 2 < HIER_NGROUPS
#   error OZAKI_HIER_L2 (tree merge) supports at most 2 groups; use OZAKI_HIER_L2=0.
# endif
inline void oz2g_hier_tree_accumulate(const uint* restrict gval, real_t alpha, int base_sh, real_t* cval)
{
  ulong combined = (ulong)gval[0], modulus = (ulong)oz2g_hier_gprod[0];
# if 2 == HIER_NGROUPS
  { const uint gp1 = oz2g_hier_gprod[1];
    const uint v0 = oz2g_mod_l2(combined, 1);
    const uint diff = (gval[1] >= v0) ? (gval[1] - v0) : (gp1 + gval[1] - v0);
    const uint t = oz2g_mod_l2((ulong)diff * (ulong)OZ2G_L2_INV(0, 1), 1);
    combined += modulus * (ulong)t;
    modulus *= (ulong)gp1;
  }
# endif
  { const int is_negative = (combined > modulus / 2) ? 1 : 0;
    oz2g_accumulate((real_t)((0 != is_negative) ? (modulus - combined) : combined), is_negative, alpha, base_sh, cval);
  }
}
#endif /* OZAKI_HIER_L2 == 1 */


#endif /* OZAKI_HIER */


/**
 * preprocess_a_crt_dense: decompose A into dense per-prime CRT residue matrices.
 *
 * Output layout: As[pidx][M_pad][K_pad] - one dense M_pad x K_pad int8 matrix
 * per prime, with residues in [0, m_pidx-1] and sign folded in.
 *
 * Work-group: (BK_PRE, BM_PRE, 1) - K on dim 0, so the lanes of a sub-group walk col
 * and the NPRIMES stores per element land on consecutive bytes of As[p][row][col].
 * The read wants the opposite mapping, because A is contiguous along row for
 * transa=0, so it uses a rank remapped to walk rows and hands the block over through
 * shared memory. A work-item cannot have both directions dense, and neither can be
 * given up: with lanes on col the read spends a 32-byte sector per element, twice
 * over because the exponent pass re-reads the input, so the kernel moves 1.34 GB of
 * sectors for 0.40 GB of data; with lanes on row the NPRIMES stores scatter instead.
 *
 * APRE_R columns per lane per block is what makes the exchange pay. At one column the
 * two barriers per block land on every 32 columns of K and cost exactly what the
 * dense read saves (measured 0.355 either way); at four they amortize over 128.
 *
 * Dispatch: global[0] = BK_PRE (single WG in K) - loops internally.
 */
#if !defined(APRE_R)
# define APRE_R 4
#endif
#define APRE_TC (BK_PRE * APRE_R)
/* The sign rides the top bit of the aligned mantissa, which is below 2^MANT_BITS, so
 * the exchange needs one tile rather than two. */
#define OZAKI_APRE_SGN (1ul << 63)
__attribute__((reqd_work_group_size(BK_PRE, BM_PRE, 1)))
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_a_crt_dense(CONSTANT const real_t* restrict a_base, int a_index, int M, int K, int lda, int transa,
  global char* restrict as_base, /* [NPRIMES * M_pad * K_pad] */ long as_index,
  global int* restrict expa_base, /* [M] per-row max exponent (int for atomic_max) */ int expa_index,
  int K_pad, int M_pad)
{
  CONSTANT const real_t* restrict a = a_base + a_index;
  global char* restrict as = as_base + as_index;
  global int* restrict expa = expa_base + expa_index;
  const int kk = (int)get_local_id(0);
  const int mi = (int)get_local_id(1);
  const int row_base = (int)get_group_id(1) * BM_PRE;
  const int row = row_base + mi;
  /* Rank remapped so the read walks rows, the axis A is contiguous along. */
  const int rt = kk + BK_PRE * mi;
  const int rrow = rt % BM_PRE;
  const int rcol = rt / BM_PRE;
  const int rrow_ok = (row_base + rrow < M);
  int cb, e, emax = 0;

  local int row_max_exp[BM_PRE];
  local ulong tile[APRE_TC][BM_PRE + 1]; /* [col][row], padded against bank conflicts */
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /**
   * Pass 1: max exponent across ALL of K for this row, read with the same remapped
   * rank. The lanes sharing a row would serialize on one SLM address, so the maximum
   * accumulates privately and contributes once.
   */
  for (cb = rcol; cb < K; cb += BK_PRE) {
    if (rrow_ok) {
      int s0;
      short e0;
      uint_repr_t m0;
      ieee_decompose(a[OZAKI_IDX_A(row_base + rrow, cb, lda)], &s0, &e0, &m0);
      if (e0 > emax) emax = (int)e0;
    }
  }
  if (rrow_ok && 0 < emax) atomic_max(&row_max_exp[rrow], emax);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && row < M) expa[row] = row_max_exp[mi];

  /**
   * Pass 2: align a block of APRE_TC columns, exchange it, then store its residues.
   * Padding columns hold zero and are stored as such, which is what the GEMM needs.
   */
  for (cb = 0; cb < K_pad; cb += APRE_TC) {
    barrier(CLK_LOCAL_MEM_FENCE);
    UNROLL_FORCE(APRE_R) for (e = 0; e < APRE_R; ++e)
    {
      const int lcol = rcol + BK_PRE * e;
      const int col = cb + lcol;
      ulong v = 0;
      if (rrow_ok && col < K) {
        int s1;
        short e1;
        uint_repr_t m1;
        ieee_decompose(a[OZAKI_IDX_A(row_base + rrow, col, lda)], &s1, &e1, &m1);
        if (m1 != 0) {
          const int shift = (int)(row_max_exp[rrow] - e1);
          v = (shift + MANT_TRUNC <= MANT_BITS) ? (ulong)(m1 >> (shift + MANT_TRUNC)) : 0;
          if (0 != s1 && 0 != v) v |= OZAKI_APRE_SGN;
        }
      }
      tile[lcol][rrow] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < M) {
      UNROLL_FORCE(APRE_R) for (e = 0; e < APRE_R; ++e)
      {
        const int lcol = kk + BK_PRE * e;
        if (cb + lcol < K_pad) {
          const ulong t = tile[lcol][mi];
          OZAKI_EXTRACT_CRT_AT(t & ~OZAKI_APRE_SGN, 0 != (t & OZAKI_APRE_SGN), as, M_pad * K_pad,
            (long)row * K_pad + cb + lcol);
        }
      }
    }
  }
}


/**
 * preprocess_b_crt_dense: decompose B into dense per-prime CRT residue matrices.
 *
 * Output layout: Bs[pidx][K_pad][N_pad] - K-major, N_pad >= 64 for 2D block I/O.
 *
 * Work-group: (BN_PRE, BK_PRE, 1).
 * Dispatch: global[1] = BK_PRE (single WG in K) - loops internally.
 */
__attribute__((reqd_work_group_size(BN_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_b_crt_dense(CONSTANT const real_t* restrict b_base, int b_index, int N, int K, int ldb, int transb,
  global char* restrict bs_base, /* [NPRIMES * K_pad * N_pad] */ long bs_index,
  global int* restrict expb_base, /* [N] per-column max exponent (int for atomic_max) */ int expb_index,
  int K_pad, int N_pad)
{
  CONSTANT const real_t* restrict b = b_base + b_index;
  global char* restrict bs = bs_base + bs_index;
  global int* restrict expb = expb_base + expb_index;
  const int nj = (int)get_local_id(0);
  const int kk = (int)get_local_id(1);
  const int col = (int)get_group_id(0) * BN_PRE + nj;
  int row, emax = 0;

  local int col_max_exp[BN_PRE];
  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /**
   * Pass 1: max exponent over all of K, in the same 16-K blocks pass 2 stores, so a
   * lane reads 128 contiguous bytes instead of one element per 32-byte sector.
   */
  for (row = kk << 4; row < K; row += BK_PRE << 4) {
    int i;
    UNROLL_FORCE(16) for (i = 0; i < 16; ++i)
    {
      const int krow = row + i;
      if (krow < K && col < N) {
        int s0;
        short e0;
        uint_repr_t m0;
        const int idx = OZAKI_IDX_B(krow, col, ldb);
        ieee_decompose(b[idx], &s0, &e0, &m0);
        if (e0 > emax) emax = (int)e0;
      }
    }
  }
  if (col < N && 0 < emax) atomic_max(&col_max_exp[nj], emax);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && col < N) expb[col] = col_max_exp[nj];

  /* Pass 2: compute and store CRT residues using the true max exponent */
#if defined(OZAKI_BBLOCK) && (OZAKI_BBLOCK)
  /**
   * Blocked layout: a work-item owns whole 16-K blocks of its column, so the
   * NPRIMES stores per block are one 16-byte store each instead of 16 scattered
   * bytes, and consecutive work-items still write consecutive columns. Emitting the
   * padding as zeros is what the tail of a partial block needs anyway.
   */
  if (col < N) {
    const short max_exp = (short)col_max_exp[nj];
    int kb;
    for (kb = kk; kb < (K_pad >> 4); kb += BK_PRE) {
      ulong aligned[16];
      int sign[16];
      int i;
      SINT p;
      UNROLL_FORCE(16) for (i = 0; i < 16; ++i)
      {
        const int krow = (kb << 4) + i;
        aligned[i] = 0;
        sign[i] = 0;
        if (krow < K) {
          int s1;
          short e1;
          uint_repr_t m1;
          const int idx = OZAKI_IDX_B(krow, col, ldb);
          ieee_decompose(b[idx], &s1, &e1, &m1);
          if (m1 != 0) {
            const int shift = (int)(max_exp - e1);
            aligned[i] = (shift + MANT_TRUNC <= MANT_BITS) ? (ulong)(m1 >> (shift + MANT_TRUNC)) : 0;
            sign[i] = s1;
          }
        }
      }
#if OZAKI_EXTRACT_HIER
      /* Group-outer, so the 64-bit reduction runs once per group and block. */
      { SINT g;
        UNROLL_FORCE(HIER_NGROUPS) for (g = 0; g < HIER_NGROUPS; ++g)
        {
          uint gr[16];
          SINT j;
          UNROLL_FORCE(16) for (i = 0; i < 16; ++i)
          {
            gr[i] = oz2g_mod_l2(aligned[i], (int)g);
          }
          UNROLL_FORCE(HIER_GS) for (j = 0; j < HIER_GS; ++j)
          {
            p = g * HIER_GS + j;
            if (p < NPRIMES) {
              union {
                uchar b[16];
                uint4 v;
              } blk;
              UNROLL_FORCE(16) for (i = 0; i < 16; ++i)
              {
                uint r = oz2g_mod(gr[i], p);
                if (sign[i] && 0 != r) OZAKI_SIGN_FOLD(r, p);
                blk.b[i] = (uchar)r;
              }
              *(global uint4*)(bs + (long)p * K_pad * N_pad + ((long)kb * N_pad + col) * 16) = blk.v;
            }
          }
        }
      }
#else
      UNROLL_FORCE(NPRIMES) for (p = 0; p < NPRIMES; ++p)
      {
        union {
          uchar b[16];
          uint4 v;
        } blk;
        UNROLL_FORCE(16) for (i = 0; i < 16; ++i)
        {
          uint r = oz2g_mod64(aligned[i], p);
          if (sign[i] && 0 != r) OZAKI_SIGN_FOLD(r, p);
          blk.b[i] = (uchar)r;
        }
        *(global uint4*)(bs + (long)p * K_pad * N_pad + ((long)kb * N_pad + col) * 16) = blk.v;
      }
#endif
    }
#else
  if (col < N) {
    const short max_exp = (short)col_max_exp[nj];
    for (row = kk; row < K; row += BK_PRE) {
      int s1;
      short e1;
      uint_repr_t m1;
      const int idx = OZAKI_IDX_B(row, col, ldb);
      ieee_decompose(b[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift + MANT_TRUNC <= MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
        OZAKI_EXTRACT_CRT_B(aligned, s1, bs, K_pad * N_pad, N_pad, K_pad, row, col);
      }
    }
#endif
  }
}


/**
 * gemm_crt_fused: one launch over all primes, full-K accumulation per prime, then
 * either the residue store (OZAKI_UNFUSE, reconstructed by gemm_crt_reduce) or the
 * fused reconstruction. Work-group (SG, NTM * NTN, 1), one per output tile.
 */
__attribute__((reqd_work_group_size(SG, NTM* NTN, 1)))
#if defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void gemm_crt_fused(
  CONSTANT const char* restrict as_base, /* As: [NPRIMES * M_pad * K_pad] */ long as_index,
  CONSTANT const char* restrict bs_base, /* Bs: [NPRIMES * K_pad * N_pad] */ long bs_index,
  CONSTANT const int* restrict expa_base, /* [M] per-row max exponent */ int expa_index,
  CONSTANT const int* restrict expb_base, /* [N] per-col max exponent */ int expb_index,
  global real_t* restrict c_base, int c_index, int M, int N, int K_pad, int N_pad, int ldc, int M_pad, real_t alpha,
  int first
#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE)
  /* Residue planes, last so the preceding arguments keep their indices. */
  , global uchar* restrict res_base, long res_index
#endif
)
{
  CONSTANT const char* restrict as = as_base + as_index;
  CONSTANT const char* restrict bs = bs_base + bs_index;
  CONSTANT const int* restrict expa = expa_base + expa_index;
  CONSTANT const int* restrict expb = expb_base + expb_index;
  global real_t* restrict c = c_base + c_index;
  const int sg_lid = (int)SGLID();
  const int sg_id = (int)SGID();
  const int tile_m = sg_id / NTN;
  const int tile_n = sg_id % NTN;
  const long a_plane = (long)M_pad * K_pad;
  int ib_idx, jb_idx, mi_base, nj_base;
  OZAKI_SWIZZLE_IDX(M, N, ib_idx, jb_idx);
  mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  const long b_plane = (long)K_pad * N_pad;
#if defined(OZAKI_WGMMA) && (OZAKI_WGMMA)
  /* Work-group tile base (staging is cooperative, unlike the per-sub-group MI/NJ). */
  const int nb_base = jb_idx * BN;
  const int wt = sg_id * SG + sg_lid;
# if defined(OZAKI_WGMMA_RS) && (OZAKI_WGMMA_RS)
  local uint4 wg_sb[OZAKI_WGMMA_STAGES * ((BN * WBK) / 16)]; /* staged B only; A needs none */
# else
  const int mb_base = ib_idx * BM;
  const int wg_id = sg_id / WG_NSUB;
  local uint4 wg_sa[2 * ((BM * WBK) / 16)]; /* double-buffered */
  local uint4 wg_sb[2 * ((BN * WBK) / 16)];
# endif
#endif
#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE)
  /**
   * Unfused: accumulate one prime, reduce it to a byte per output, store, move on.
   * No group values are kept, so the 2 KB per-work-item frame the fused epilogue
   * needs never exists - which is the whole reason for the second kernel.
   */
  { SINT pidx_base;
    global uchar* const res = res_base + res_index + OZAKI_RES_BASE(ib_idx, jb_idx, N, sg_id, sg_lid);
    const long rplane = OZAKI_RES_PLANE(M, N);
    UNROLL_OUTER(1) for (pidx_base = 0; pidx_base < NPRIMES; ++pidx_base) {
      OZAKI_ACC_DECL(acc);
      OZAKI_ACC_ZERO_ALL(acc);
      OZAKI_CRT_KLOOP_RUN(acc, pidx_base);
      OZAKI_CRT_STORE_RESIDUES(OZAKI_ACC_FRAGS(acc), pidx_base, res + (long)pidx_base * rplane);
    }
  }
#elif OZAKI_HIER
  /* One group of HIER_GS primes at a time into group_res; level 1 folds each into gval_all. */
#define GRP_RES_STRIDE (RTM * RTN * HIER_GS * XMX_FRAG)
#define GVAL_ALL_STRIDE (RTM * RTN * HIER_NGROUPS * XMX_FRAG)
  uint group_res[GRP_RES_STRIDE];
  uint gval_all[GVAL_ALL_STRIDE];
  { SINT gidx;
    for (gidx = 0; gidx < HIER_NGROUPS; ++gidx) {
      const int group_lo = gidx * HIER_GS;
      SINT pidx_base;
      int ri;
      for (ri = 0; ri < GRP_RES_STRIDE; ++ri) group_res[ri] = 0;
      UNROLL_OUTER(1) for (pidx_base = group_lo; pidx_base < group_lo + HIER_GS && pidx_base < NPRIMES; pidx_base += PB)
      {
        OZAKI_ACC_DECL(acc);
        OZAKI_ACC_ZERO_ALL(acc);
        OZAKI_CRT_KLOOP_REDUCE(acc, pidx_base, group_lo, HIER_GS, group_res);
      }
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
      { int rm, rn;
        UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
        {
          UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
          {
            OZAKI_CRT_L1_STORE(group_res + (rm * RTN + rn) * HIER_GS * XMX_FRAG,
              gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_FRAG, gidx);
          }
        }
      }
#endif
    }
  }
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
  { int rm, rn;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
    {
      UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
      {
        OZAKI_CRT_STORE(gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_FRAG, expa, expb, c, M, N,
          mi_base + rm * XMX_M, nj_base + rn * XMX_N, sg_lid, ldc, alpha, first);
      }
    }
  }
#endif
#else /* !OZAKI_HIER */
  /**
   * Per-prime residues stay private: lanes accumulate different columns, so SLM
   * would need a lane dimension and NTM*NTN*SG*RES_STRIDE exceeds it. Private lets
   * the compiler spill by liveness (cold in the K-loop, hot in the epilogue).
   */
#define RES_STRIDE (RTM * RTN * NPRIMES * XMX_FRAG)
  uint residues[RES_STRIDE];
  { SINT pidx_base;
    int ri;
    for (ri = 0; ri < RES_STRIDE; ++ri) residues[ri] = 0;
    UNROLL_OUTER(1) for (pidx_base = 0; pidx_base < NPRIMES; pidx_base += PB)
    {
      OZAKI_ACC_DECL(acc);
      OZAKI_ACC_ZERO_ALL(acc);
      OZAKI_CRT_KLOOP_REDUCE(acc, pidx_base, 0, NPRIMES, residues);
    }
  }
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
  { int rm, rn;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
    {
      UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
      {
        OZAKI_CRT_STORE(residues + (rm * RTN + rn) * NPRIMES * XMX_FRAG, expa, expb, c, M, N,
          mi_base + rm * XMX_M, nj_base + rn * XMX_N, sg_lid, ldc, alpha, first);
      }
    }
  }
#endif
#endif /* OZAKI_HIER */
}


#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE) && OZAKI_HIER
/**
 * Reconstruct C from the residue planes gemm_crt_fused wrote. Launched with the
 * identical geometry, so a work-item reconstructs exactly the outputs it
 * accumulated and the blocked residue layout needs no index translation.
 *
 * The loop order is the whole point: outputs outside, primes inside, so only
 * HIER_NGROUPS group values are ever live and the reconstruction stays in
 * registers. What remains is memory-bound by construction - NPRIMES bytes read
 * and one element written per output.
 *
 * Primes past NPRIMES in a partial group contribute zero, exactly as the fused
 * path's cleared group_res does, which is what keeps the two bit-identical (and
 * what keeps this from reading past the last plane at NPRIMES=9 in fp32).
 */
__attribute__((reqd_work_group_size(SG, NTM* NTN, 1)))
#if defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void gemm_crt_reduce(CONSTANT const uchar* restrict res_base, /* [NPRIMES * tiles * BM * BN] */ long res_index,
  CONSTANT const int* restrict expa_base, int expa_index, CONSTANT const int* restrict expb_base, int expb_index,
  global real_t* restrict c_base, int c_index, int M, int N, int ldc, real_t alpha, int first)
{
  CONSTANT const uchar* restrict res = res_base + res_index;
  CONSTANT const int* restrict expa = expa_base + expa_index;
  CONSTANT const int* restrict expb = expb_base + expb_index;
  global real_t* restrict c = c_base + c_index;
  const int sg_lid = (int)SGLID();
  const int sg_id = (int)SGID();
  const int tile_m = sg_id / NTN;
  const int tile_n = sg_id % NTN;
  int ib_idx, jb_idx, mi_base, nj_base;
  long rbase;
  const long rplane = OZAKI_RES_PLANE(M, N);
  OZAKI_SWIZZLE_IDX(M, N, ib_idx, jb_idx);
  mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  rbase = OZAKI_RES_BASE(ib_idx, jb_idx, N, sg_id, sg_lid);
  { int rm, rn;
    for (rm = 0; rm < RTM; ++rm) {
      for (rn = 0; rn < RTN; ++rn) {
        uint gval_all[HIER_NGROUPS * XMX_FRAG];
        int gidx;
        UNROLL_FORCE(HIER_NGROUPS) for (gidx = 0; gidx < HIER_NGROUPS; ++gidx)
        {
          int ms;
          UNROLL_FORCE(XMX_FRAG) for (ms = 0; ms < XMX_FRAG; ++ms)
          {
            const long off = rbase + OZAKI_RES_OFF(rm, rn, ms);
            uint r[HIER_GS];
            int pg;
            UNROLL_FORCE(HIER_GS) for (pg = 0; pg < HIER_GS; ++pg)
            {
              const int pidx = gidx * HIER_GS + pg;
              r[pg] = (pidx < NPRIMES) ? (uint)res[off + (long)pidx * rplane] : 0u;
            }
            gval_all[gidx * XMX_FRAG + ms] = OZAKI_L1_RECONSTRUCT(r, gidx);
          }
        }
        OZAKI_CRT_STORE(gval_all, expa, expb, c, M, N, mi_base + rm * XMX_M, nj_base + rn * XMX_N, sg_lid, ldc, alpha, first);
      }
    }
  }
}
#endif /* OZAKI_UNFUSE */
