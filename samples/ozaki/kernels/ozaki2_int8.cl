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
 * Ozaki Scheme 2 - GEMM-based XMX path (CRT).
 *
 * Unlike the panel-batched dotprod path, this approach:
 *   1. Preprocesses the FULL K dimension of A and B into dense per-prime
 *      CRT residue matrices (one M_pad x K_pad or K_pad x N_pad per prime)
 *   2. Runs a single tiled GEMM kernel that loops over all NPRIMES internally,
 *      performing full-K DPAS accumulation per prime, then fuses Garner
 *      CRT reconstruction + Horner evaluation + scaling into the store
 *
 * OZAKI_U8 (default 1 for Scheme 2):
 *   Uses unsigned u8 DPAS with moduli up to 256 (vs 128 for signed i8).
 *   Larger moduli reduce the number of primes: fp64 16 (vs 19), fp32 9 (vs 10).
 *   Sign is encoded via modular additive inverse: (p - r) == -r (mod p).
 *   Trade-off: safe K without KGROUPS drops from ~133K to ~33K.
 *
 * The KGROUPS tunable controls intermediate int32 mod reductions within
 * the K-loop.  When 0 (default), no intermediate reductions - the int32
 * accumulator covers the full K.  When > 0, a Barrett mod reduction fires
 * every KGROUPS * BK steps, preventing int32 overflow for large K.
 * Garner reconstruction always runs once per C element regardless.
 *
 * Compile-time parameters (-D):
 *   BM, BN          - output tile per work-group (256x256 default)
 *   BK              - DPAS K-unroll (32 for int8)
 *   NPRIMES         - number of CRT moduli (up to 20)
 *   MANT_BITS       - mantissa bits (52=fp64, 23=fp32)
 *   BIAS_PLUS_MANT  - exponent bias + mantissa bits
 *   KGROUPS         - intermediate mod reduction period (0 = full K)
 *   OZAKI_U8        - 1: unsigned u8 DPAS (default), 0: signed i8 DPAS
 *   USE_DOUBLE      - 1: fp64, 0: fp32
 *   SG              - sub-group size (16)
 *   BM_PRE, BN_PRE, BK_PRE - preprocessing work-group sizes
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
/**
 * Max primes per Horner group that fit ulong accumulation:
 * u8 (moduli<=256): product of 8 largest < 2^64 (group=8)
 * i8 (moduli<=128): product of 9 largest < 2^64 (group=9)
 */
# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZ2_HORNER_GROUP 8
# else
# define OZ2_HORNER_GROUP 9
# endif
#endif
#define OZ2_HORNER_NGROUPS ((NPRIMES + OZ2_HORNER_GROUP - 1) / OZ2_HORNER_GROUP)
#if !defined(PB)
# define PB 1
#endif

/**
 * Hierarchical CRT: two-level Garner reconstruction.
 * Level 1: HIER_GS primes per group (small Garner, 32-bit).
 * Level 2: Garner over HIER_NGROUPS group-moduli (32-bit, ulong intermediate).
 * Reduces peak live registers from ~NPRIMES to ~max(HIER_GS, HIER_NGROUPS).
 */
/**
 * Fractional-CRT mode 1 replaces the whole reconstruction and needs the raw
 * per-prime residues, so it forces the flat (non-hierarchical) path. Mode 2
 * reconstructs per group and keeps the hierarchical path (leaf fractional CRT
 * feeding the exact level-2 combine).
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
 * and group values must fit uint32 (Barrett tables, gval_all). The host lowers it
 * to the largest divisor of NPRIMES that still fits, because a group holding a
 * single prime is pathological - at NPRIMES=9 the 4,4,1 split costs the
 * reconstruction 1.21 ms against 0.58 for a 12-prime run of three full groups.
 * Fractional-CRT leaves could reconstruct up to 6 primes per group (group
 * product below 2^53 rather than 2^32), which would shorten level 2, but that
 * requires widening the whole level-2 datapath to 64-bit.
 */
# if !defined(HIER_GS)
#   define HIER_GS 4
# endif
# define HIER_NGROUPS ((NPRIMES + HIER_GS - 1) / HIER_GS)
# define HIER_L2_HORNER_GROUP 2
# define HIER_L2_HORNER_NGROUPS ((HIER_NGROUPS + HIER_L2_HORNER_GROUP - 1) / HIER_L2_HORNER_GROUP)
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
# define OZAKI_IN_BOUNDS(R, M, COL, N) (1)
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

/* Alias the shared DPAS primitive from ozaki_common.cl */
#define OZAKI_CRT_DPAS OZAKI_DPAS

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
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_SIGN_FOLD(R, P) (R) = oz2g_moduli[(P)] - (R)
#else
# define OZAKI_SIGN_FOLD(R, P) (R) = -(R)
#endif

/* Zero NPRIMES entries at the given position. */
#define OZAKI_ZERO_CRT(DST, SS, RS, ROW, COL) \
  do { \
    SINT p_; \
    UNROLL_FORCE(NPRIMES) for (p_ = 0; p_ < NPRIMES; ++p_) \
    { \
      (DST)[(long)(p_) * (SS) + (long)(ROW) * (RS) + (COL)] = 0; \
    } \
  } while (0)

/**
 * Mod-reduce DPAS accumulator into uint residue array.
 * RESIDUES[pidx * XMX_FRAG + f] accumulates the unsigned residue of the
 * fragment element f this work-item owns (XMX_FRAG per sub-tile).
 * u8: accumulator is always non-negative (unsigned products) - branchless.
 * i8: accumulator can be negative - requires sign-aware reduction.
 */
#define OZAKI_CRT_MOD_REDUCE(ACC, PIDX, RESIDUES) \
  do { \
    OZAKI_ACC_UNION(du_); \
    int mr_; \
    du_.v_ = (ACC); \
    UNROLL_FORCE(XMX_FRAG) for (mr_ = 0; mr_ < XMX_FRAG; ++mr_) \
    { \
      uint r_; \
      OZAKI_MOD_REDUCE_ELEM(du_.a_[mr_], (PIDX), r_); \
      { \
        const uint prev_ = (RESIDUES)[(int)(PIDX) * XMX_FRAG + mr_]; \
        const uint sum_ = prev_ + r_; \
        (RESIDUES)[(int)(PIDX) * XMX_FRAG + mr_] = (sum_ >= oz2g_moduli[(PIDX)]) ? (sum_ - oz2g_moduli[(PIDX)]) : sum_; \
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

#if !OZAKI_HIER
#if defined(OZAKI_FRACCRT) && (OZAKI_FRACCRT)
/* Fractional-CRT store (experiment): reconstruct signed value, scale, write C */
#define OZAKI_CRT_STORE(RESIDUES, EXPA, EXPB, C_PTR, M, N, MI, NJ, LANE, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_FRAG], eb_c_[OZAKI_FRAG_NCOL]; \
    int ms_; \
    OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, ea_c_, eb_c_); \
    UNROLL_FORCE(XMX_FRAG) for (ms_ = 0; ms_ < XMX_FRAG; ++ms_) \
    { \
      const int rm_ = (MI) + OZAKI_FRAG_ROW(ms_, (LANE)); \
      const int col_ = (NJ) + OZAKI_FRAG_COL(OZAKI_FRAG_COLIDX(ms_), (LANE)); \
      if (OZAKI_IN_BOUNDS(rm_, (M), col_, (N))) { \
        SINT pg_; \
        double val_; \
        UNROLL_FORCE(NPRIMES) for (pg_ = 0; pg_ < NPRIMES; ++pg_) \
        { \
          dot_r_[pg_] = (RESIDUES)[(int)pg_ * XMX_FRAG + ms_]; \
        } \
        val_ = oz2g_frac_reconstruct(dot_r_); \
        { \
          const int sh_ = (int)ea_c_[ms_] + (int)eb_c_[OZAKI_FRAG_COLIDX(ms_)] - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = OZAKI_IS_FIRST(FIRST) ? ZERO : (C_PTR)[(long)col_ * (LDC) + rm_]; \
          if (0.0 != val_ && ZERO != (ALPHA) && sh_ >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) { \
            const real_t scale_ = OZAKI_ALPHA_MUL((ALPHA), EXP2I(sh_)); \
            cv_ += (real_t)(val_ * (double)scale_); \
          } \
          (C_PTR)[(long)col_ * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#else
/* Garner + Horner store: reconstruct from per-prime residues, scale, write C */
#define OZAKI_CRT_STORE(RESIDUES, EXPA, EXPB, C_PTR, M, N, MI, NJ, LANE, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_FRAG], eb_c_[OZAKI_FRAG_NCOL]; \
    int ms_; \
    OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, ea_c_, eb_c_); \
    UNROLL_FORCE(XMX_FRAG) for (ms_ = 0; ms_ < XMX_FRAG; ++ms_) \
    { \
      const int rm_ = (MI) + OZAKI_FRAG_ROW(ms_, (LANE)); \
      const int col_ = (NJ) + OZAKI_FRAG_COL(OZAKI_FRAG_COLIDX(ms_), (LANE)); \
      if (OZAKI_IN_BOUNDS(rm_, (M), col_, (N))) { \
        int is_neg_; \
        SINT pg_; \
        UNROLL_FORCE(NPRIMES) for (pg_ = 0; pg_ < NPRIMES; ++pg_) \
        { \
          dot_r_[pg_] = (RESIDUES)[(int)pg_ * XMX_FRAG + ms_]; \
        } \
        is_neg_ = oz2g_garner_reconstruct(dot_r_, vg_); \
        { \
          const int sh_ = (int)ea_c_[ms_] + (int)eb_c_[OZAKI_FRAG_COLIDX(ms_)] - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = OZAKI_IS_FIRST(FIRST) ? ZERO : (C_PTR)[(long)col_ * (LDC) + rm_]; \
          oz2g_horner_accumulate(vg_, is_neg_, (ALPHA), sh_, &cv_); \
          (C_PTR)[(long)col_ * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#endif /* OZAKI_FRACCRT */
#else /* OZAKI_HIER */

/**
 * Level-1 reconstruction of a group value: exact per-group fractional CRT
 * (OZAKI_FRACCRT=2) or the sequential group Garner (default).
 */
#if defined(OZAKI_FRACCRT) && (2 == OZAKI_FRACCRT)
# define OZAKI_L1_RECONSTRUCT(DOT_R, GIDX) oz2g_frac_l1((DOT_R), (GIDX))
#else
# define OZAKI_L1_RECONSTRUCT(DOT_R, GIDX) oz2g_hier_l1_garner((DOT_R), (GIDX))
#endif

/**
 * Level-1 reconstruction from group-local residues -> gval_all.
 * GROUP_RES: base of group-local residues for this tile [HIER_GS * XMX_FRAG].
 * GVAL_ALL: base of gval_all for this tile [HIER_NGROUPS * XMX_FRAG].
 * GIDX: group index.
 */
#define OZAKI_CRT_L1_STORE(GROUP_RES, GVAL_ALL, GIDX) \
  do { \
    int ms_l1_; \
    UNROLL_FORCE(XMX_FRAG) for (ms_l1_ = 0; ms_l1_ < XMX_FRAG; ++ms_l1_) \
    { \
      SINT pg_l1_; \
      UNROLL_FORCE(HIER_GS) for (pg_l1_ = 0; pg_l1_ < HIER_GS; ++pg_l1_) \
      { \
        dot_r_[pg_l1_] = (GROUP_RES)[(int)pg_l1_ * XMX_FRAG + ms_l1_]; \
      } \
      (GVAL_ALL)[(GIDX) * XMX_FRAG + ms_l1_] = OZAKI_L1_RECONSTRUCT(dot_r_, (GIDX)); \
    } \
  } while (0)

/* Level-2 Garner + Horner + store C from gval_all. */
#if !defined(OZAKI_HIER_L2) || (0 == OZAKI_HIER_L2)
#define OZAKI_CRT_L2_STORE(GVAL_ALL, EXPA, EXPB, C_PTR, M, N, MI, NJ, LANE, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_FRAG], eb_c_[OZAKI_FRAG_NCOL]; \
    int ms_l2_; \
    OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, ea_c_, eb_c_); \
    UNROLL_FORCE(XMX_FRAG) for (ms_l2_ = 0; ms_l2_ < XMX_FRAG; ++ms_l2_) \
    { \
      const int rm_ = (MI) + OZAKI_FRAG_ROW(ms_l2_, (LANE)); \
      const int col_ = (NJ) + OZAKI_FRAG_COL(OZAKI_FRAG_COLIDX(ms_l2_), (LANE)); \
      if (OZAKI_IN_BOUNDS(rm_, (M), col_, (N))) { \
        int is_neg_; \
        SINT pg_l2_; \
        UNROLL_FORCE(HIER_NGROUPS) for (pg_l2_ = 0; pg_l2_ < HIER_NGROUPS; ++pg_l2_) \
        { \
          gval_[pg_l2_] = (GVAL_ALL)[(int)pg_l2_ * XMX_FRAG + ms_l2_]; \
        } \
        is_neg_ = oz2g_hier_l2_garner(gval_, vg_); \
        { \
          const int sh_ = (int)ea_c_[ms_l2_] + (int)eb_c_[OZAKI_FRAG_COLIDX(ms_l2_)] - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = OZAKI_IS_FIRST(FIRST) ? ZERO : (C_PTR)[(long)col_ * (LDC) + rm_]; \
          oz2g_hier_horner_accumulate(vg_, is_neg_, (ALPHA), sh_, &cv_); \
          (C_PTR)[(long)col_ * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#else /* OZAKI_HIER_L2 == 1: tree-merge */
#define OZAKI_CRT_L2_STORE(GVAL_ALL, EXPA, EXPB, C_PTR, M, N, MI, NJ, LANE, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_FRAG], eb_c_[OZAKI_FRAG_NCOL]; \
    int ms_l2_; \
    OZAKI_CRT_EXP_CACHE(EXPA, EXPB, N, MI, NJ, LANE, ea_c_, eb_c_); \
    UNROLL_FORCE(XMX_FRAG) for (ms_l2_ = 0; ms_l2_ < XMX_FRAG; ++ms_l2_) \
    { \
      const int rm_ = (MI) + OZAKI_FRAG_ROW(ms_l2_, (LANE)); \
      const int col_ = (NJ) + OZAKI_FRAG_COL(OZAKI_FRAG_COLIDX(ms_l2_), (LANE)); \
      if (OZAKI_IN_BOUNDS(rm_, (M), col_, (N))) { \
        ulong tree_val_; \
        int is_neg_; \
        SINT pg_l2_; \
        UNROLL_FORCE(HIER_NGROUPS) for (pg_l2_ = 0; pg_l2_ < HIER_NGROUPS; ++pg_l2_) \
        { \
          gval_[pg_l2_] = (GVAL_ALL)[(int)pg_l2_ * XMX_FRAG + ms_l2_]; \
        } \
        is_neg_ = oz2g_hier_l2_tree(gval_, &tree_val_); \
        { \
          const int sh_ = (int)ea_c_[ms_l2_] + (int)eb_c_[OZAKI_FRAG_COLIDX(ms_l2_)] - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = OZAKI_IS_FIRST(FIRST) ? ZERO : (C_PTR)[(long)col_ * (LDC) + rm_]; \
          oz2g_hier_tree_accumulate(tree_val_, is_neg_, (ALPHA), sh_, &cv_); \
          (C_PTR)[(long)col_ * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#endif /* OZAKI_HIER_L2 */
#endif /* OZAKI_HIER */

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
# if (1 != RTM) || ((8 != RTN) && (16 != RTN))
#   error OZAKI_WGMMA implies RTM=1 and RTN=8 or 16 (m64n64k32 / m64n128k32).
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

/* Bytes of K staged per round, work-items per work-group, warp groups per CTA. */
# define WBK (KU * BK)
# define WGS (SG * (BM / (XMX_M * RTM)) * (BN / (XMX_N * RTN)))
# define WG_NGROUPS (BM / 64)
# define WG_NSUB (64 / (XMX_M * RTM))

/* One issue per K-chunk; the marker names the shape so the host need not assume it. */
# if (16 == RTN)
# define OZAKI_WGMMA_ISSUE(ACCS, PA, PB_) \
    asm volatile("// WGMMA_SLOT n128 d={%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"\
      "%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,"\
      "%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63} pa=%64 pb=%65" \
      : "+r"((ACCS)[0]), "+r"((ACCS)[1]), "+r"((ACCS)[2]), "+r"((ACCS)[3]), \
      "+r"((ACCS)[4]), "+r"((ACCS)[5]), "+r"((ACCS)[6]), "+r"((ACCS)[7]), \
      "+r"((ACCS)[8]), "+r"((ACCS)[9]), "+r"((ACCS)[10]), "+r"((ACCS)[11]), \
      "+r"((ACCS)[12]), "+r"((ACCS)[13]), "+r"((ACCS)[14]), "+r"((ACCS)[15]), \
      "+r"((ACCS)[16]), "+r"((ACCS)[17]), "+r"((ACCS)[18]), "+r"((ACCS)[19]), \
      "+r"((ACCS)[20]), "+r"((ACCS)[21]), "+r"((ACCS)[22]), "+r"((ACCS)[23]), \
      "+r"((ACCS)[24]), "+r"((ACCS)[25]), "+r"((ACCS)[26]), "+r"((ACCS)[27]), \
      "+r"((ACCS)[28]), "+r"((ACCS)[29]), "+r"((ACCS)[30]), "+r"((ACCS)[31]), \
      "+r"((ACCS)[32]), "+r"((ACCS)[33]), "+r"((ACCS)[34]), "+r"((ACCS)[35]), \
      "+r"((ACCS)[36]), "+r"((ACCS)[37]), "+r"((ACCS)[38]), "+r"((ACCS)[39]), \
      "+r"((ACCS)[40]), "+r"((ACCS)[41]), "+r"((ACCS)[42]), "+r"((ACCS)[43]), \
      "+r"((ACCS)[44]), "+r"((ACCS)[45]), "+r"((ACCS)[46]), "+r"((ACCS)[47]), \
      "+r"((ACCS)[48]), "+r"((ACCS)[49]), "+r"((ACCS)[50]), "+r"((ACCS)[51]), \
      "+r"((ACCS)[52]), "+r"((ACCS)[53]), "+r"((ACCS)[54]), "+r"((ACCS)[55]), \
      "+r"((ACCS)[56]), "+r"((ACCS)[57]), "+r"((ACCS)[58]), "+r"((ACCS)[59]), \
      "+r"((ACCS)[60]), "+r"((ACCS)[61]), "+r"((ACCS)[62]), "+r"((ACCS)[63]) \
      : "l"(PA), "l"(PB_))
# else
# define OZAKI_WGMMA_ISSUE(ACCS, PA, PB_) \
    asm volatile("// WGMMA_SLOT n64 d={%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"\
      "%23,%24,%25,%26,%27,%28,%29,%30,%31} pa=%32 pb=%33" \
      : "+r"((ACCS)[0]), "+r"((ACCS)[1]), "+r"((ACCS)[2]), "+r"((ACCS)[3]), \
      "+r"((ACCS)[4]), "+r"((ACCS)[5]), "+r"((ACCS)[6]), "+r"((ACCS)[7]), \
      "+r"((ACCS)[8]), "+r"((ACCS)[9]), "+r"((ACCS)[10]), "+r"((ACCS)[11]), \
      "+r"((ACCS)[12]), "+r"((ACCS)[13]), "+r"((ACCS)[14]), "+r"((ACCS)[15]), \
      "+r"((ACCS)[16]), "+r"((ACCS)[17]), "+r"((ACCS)[18]), "+r"((ACCS)[19]), \
      "+r"((ACCS)[20]), "+r"((ACCS)[21]), "+r"((ACCS)[22]), "+r"((ACCS)[23]), \
      "+r"((ACCS)[24]), "+r"((ACCS)[25]), "+r"((ACCS)[26]), "+r"((ACCS)[27]), \
      "+r"((ACCS)[28]), "+r"((ACCS)[29]), "+r"((ACCS)[30]), "+r"((ACCS)[31]) \
      : "l"(PA), "l"(PB_))
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

#endif /* OZAKI_WGMMA */

/**
 * The full K-loop for one prime batch, whichever matrix engine is in use. Named
 * once so the fused and unfused prime loops below cannot drift apart; it reads the
 * kernel's own operands (as, bs, the padded extents, the tile bases) the way the
 * reconstruction macros read dot_r_ and gval_.
 */
#if defined(OZAKI_WGMMA) && (OZAKI_WGMMA)
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

#if OZAKI_HIER
/**
 * Mod-reduce with separate global prime index (for moduli lookup)
 * and local storage index (for residue array offset).
 */
#define OZAKI_CRT_MOD_REDUCE_LOCAL(ACC, PIDX, LOCAL_IDX, RESIDUES) \
  do { \
    OZAKI_ACC_UNION(dul_); \
    int mrl_; \
    dul_.v_ = (ACC); \
    UNROLL_FORCE(XMX_FRAG) for (mrl_ = 0; mrl_ < XMX_FRAG; ++mrl_) \
    { \
      uint rl_; \
      OZAKI_MOD_REDUCE_ELEM(dul_.a_[mrl_], (PIDX), rl_); \
      { \
        const uint prevl_ = (RESIDUES)[(int)(LOCAL_IDX) * XMX_FRAG + mrl_]; \
        const uint suml_ = prevl_ + rl_; \
        (RESIDUES)[(int)(LOCAL_IDX) * XMX_FRAG + mrl_] = (suml_ >= oz2g_moduli[(PIDX)]) ? (suml_ - oz2g_moduli[(PIDX)]) : suml_; \
      } \
    } \
  } while (0)

/**
 * HIER variant: mod-reduce into group-local residues (stride HIER_GS * XMX_FRAG).
 * PIDX_BASE: global prime index.  GROUP_LO: first prime in group.
 */
#define OZAKI_CRT_REDUCE_BATCH_GROUP(ACC, PIDX_BASE, GROUP_LO, GROUP_RES, ZERO_ACC) \
  do { \
    SINT bi_rg_; \
    UNROLL_FORCE(PB) for (bi_rg_ = 0; bi_rg_ < PB; ++bi_rg_) \
    { \
      if ((PIDX_BASE) + bi_rg_ < NPRIMES) { \
        int rm_rg_, rn_rg_; \
        const SINT lpidx_ = (PIDX_BASE) + bi_rg_ - (GROUP_LO); \
        UNROLL_FORCE(RTM) for (rm_rg_ = 0; rm_rg_ < RTM; ++rm_rg_) \
        { \
          UNROLL_FORCE(RTN) for (rn_rg_ = 0; rn_rg_ < RTN; ++rn_rg_) \
          { \
            OZAKI_CRT_MOD_REDUCE_LOCAL((ACC)[bi_rg_ * RTM * RTN + rm_rg_ * RTN + rn_rg_], (PIDX_BASE) + bi_rg_, lpidx_, \
              (GROUP_RES) + (rm_rg_ * RTN + rn_rg_) * HIER_GS * XMX_FRAG); \
            if (ZERO_ACC) { \
              (ACC)[bi_rg_ * RTM * RTN + rm_rg_ * RTN + rn_rg_] = OZAKI_ACC_ZERO; \
            } \
          } \
        } \
      } \
    } \
  } while (0)
#endif

/**
 * Mod-reduce all PB batched primes' accumulators into residues.
 * If ZERO_ACC is non-zero, also zero the accumulators after reduction.
 */
#define OZAKI_CRT_REDUCE_BATCH(ACC, PIDX_BASE, RESIDUES, ZERO_ACC) \
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
              (RESIDUES) + (rm_r_ * RTN + rn_r_) * NPRIMES * XMX_FRAG); \
            if (ZERO_ACC) { \
              (ACC)[bi_r_ * RTM * RTN + rm_r_ * RTN + rn_r_] = OZAKI_ACC_ZERO; \
            } \
          } \
        } \
      } \
    } \
  } while (0)


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

#if OZAKI_HIER
constant uint oz2g_hier_gprod[] = {1752116992u, 1841455727u, 1799186337u, 1823610127u, 1804203113u};
constant ulong oz2g_hier_l2_barrett[] = {10528260474ul, 10017478999ul, 10252825788ul, 10115508682ul, 10224316730ul};
constant uint oz2g_hier_l2_garner_inv[][5] = {
  {0u, 828768696u, 1255745875u, 96929798u, 430518282u},
  {0u, 0u, 1062200843u, 1479311133u, 742073819u},
  {0u, 0u, 0u, 1583419479u, 1296690879u},
  {0u, 0u, 0u, 0u, 1036097590u},
  {0u, 0u, 0u, 0u, 0u}};
#endif

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

#if OZAKI_HIER
constant uint oz2g_hier_gprod[] = {73986944u, 71016749u, 74378375u, 75849939u, 77548849u};
constant ulong oz2g_hier_l2_barrett[] = {249324314215ul, 259752020945ul, 248012195395ul, 243200512972ul, 237872570793ul};
constant uint oz2g_hier_l2_garner_inv[][5] = {
  {0u, 16740944u, 25622404u, 62222726u, 40198002u},
  {0u, 0u, 20777749u, 7009982u, 11759761u},
  {0u, 0u, 0u, 1845215u, 15543578u},
  {0u, 0u, 0u, 0u, 54885903u},
  {0u, 0u, 0u, 0u, 0u}};
#endif

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
 * Fractional / rank-based CRT reconstruction.
 * x/M = frac(sum_i alpha_i / m_i), alpha_i = (residue_i * k_i) mod m_i.
 * 1/m_i is expanded into OZ2G_FRAC_L base-256 limbs so the fractional sum
 * becomes sum_l (sum_i alpha_i * climb[i][l]) * 2^-8(l+1). The inner limb
 * sums are O(P*L) parallel int MACs (no sequential dependency), replacing the
 * O(P^2) sequential Garner chain. Combine in double-double.
 * Mode 1 (oz2g_frac_reconstruct) applies this over all P moduli, dividing by
 * the full product M, with a centered lift for sign; it is opt-in due to a
 * magnitude domain bound. Mode 2 (oz2g_frac_l1) applies it per hierarchical
 * group, dividing by the group product M_g < 2^53, so it is exact for every
 * group value and feeds the exact level-2 combine, which makes it exact over
 * the whole range. Tables come as -D initializers from
 * ozaki_emit_fraccrt()/ozaki_emit_fraccrt2().
 */
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
#endif /* OZAKI_FRACCRT */


#if defined(OZAKI_FRACCRT) && (1 == OZAKI_FRACCRT)
constant uint oz2g_frac_k[NPRIMES] = OZ2G_FRAC_K;

inline double oz2g_frac_reconstruct(const uint* restrict dot_residues)
{
  double sl[OZ2G_FRAC_L];
  double fh = 0.0, fl = 0.0;
  double fi, e, eh, frh, frl, s2, corr, vh, vl;
  SINT i, l;
  UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) sl[l] = 0.0;
  UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i) {
    const uint a = oz2g_mod(dot_residues[i] * oz2g_frac_k[i], i);
    UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) {
      sl[l] += (double)(a * (uint)oz2g_frac_climb[i][l]);
    }
  }
  UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) {
    const double term = sl[l] * EXP2I(-8 * (l + 1));
    const double s = oz2g_two_sum(fh, term, &e);
    const double lo = e + fl;
    fh = oz2g_two_sum(s, lo, &e);
    fl = e;
  }
  fi = floor(fh);
  frh = oz2g_two_sum(fh, -fi, &e);
  frl = e + fl;
  s2 = oz2g_two_sum(frh, frl, &e);
  frh = s2;
  frl = e;
  /**
   * Branchless centered lift: frac in [0,1), so corr = floor(frac + 0.5) is 0
   * (|x| in the lower half, value >= 0) or 1 (upper half, value < 0). Folding
   * corr into the fractional part before scaling by M avoids the M-subtract and
   * the sign branch.
   */
  corr = floor(frh + 0.5);
  frh = oz2g_two_sum(frh, -corr, &e);
  frl += e;
  s2 = oz2g_two_sum(frh, frl, &e);
  frh = s2;
  frl = e;
  vh = oz2g_two_prod(frh, OZ2G_FRAC_MH, &eh);
  vl = frh * OZ2G_FRAC_ML + frl * OZ2G_FRAC_MH + eh;
  return vh + vl;
}
#endif /* OZAKI_FRACCRT == 1 */


#if defined(OZAKI_FRACCRT) && (2 == OZAKI_FRACCRT)
constant uint oz2g_frac_kg[NPRIMES] = OZ2G_FRAC_KG;
constant double oz2g_frac_gmh[HIER_NGROUPS] = OZ2G_FRAC_GMH;

/**
 * Leaf fractional CRT for group g: reconstruct the group value V_g = x mod M_g
 * from its HIER_GS residues (group_residues[0..gsz-1], global prime index
 * lo+li). M_g < 2^53, so V_g is a non-negative integer exactly representable in
 * a double; the result is exact for all V_g. Returns V_g as a uint.
 */
inline uint oz2g_frac_l1(const uint* restrict group_residues, int g)
{
  const int lo = g * HIER_GS;
  const int hi = (lo + HIER_GS <= NPRIMES) ? (lo + HIER_GS) : NPRIMES;
  double sl[OZ2G_FRAC_L];
  double fh = 0.0, fl = 0.0;
  double fi, e, eh, frh, frl, s2, vh, vl;
  SINT li, l;
  UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) sl[l] = 0.0;
  UNROLL_FORCE(HIER_GS) for (li = 0; li < HIER_GS; ++li) {
    const int pidx = lo + li;
    if (pidx < hi) {
      const uint a = oz2g_mod(group_residues[li] * oz2g_frac_kg[pidx], pidx);
      UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) {
        sl[l] += (double)(a * (uint)oz2g_frac_climb[pidx][l]);
      }
    }
  }
  UNROLL_FORCE(OZ2G_FRAC_L) for (l = 0; l < OZ2G_FRAC_L; ++l) {
    const double term = sl[l] * EXP2I(-8 * (l + 1));
    const double s = oz2g_two_sum(fh, term, &e);
    const double lo_ = e + fl;
    fh = oz2g_two_sum(s, lo_, &e);
    fl = e;
  }
  fi = floor(fh);
  frh = oz2g_two_sum(fh, -fi, &e);
  frl = e + fl;
  s2 = oz2g_two_sum(frh, frl, &e);
  frh = s2;
  frl = e;
  vh = oz2g_two_prod(frh, oz2g_frac_gmh[g], &eh);
  vl = frl * oz2g_frac_gmh[g] + eh;
  return (uint)(vh + vl + 0.5);
}
#endif /* OZAKI_FRACCRT >= 2 */


/* Garner CRT reconstruction: residues -> mixed-radix digits + sign */
inline int oz2g_garner_reconstruct(const uint* restrict dot_residues, uint* restrict v)
{
  SINT i, j;
  int is_negative;

  UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i)
  {
    uint u = dot_residues[i];
    const uint pi = oz2g_moduli[i];
    for (j = 0; j < i; ++j) {
      uint vj = v[j];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      {
        const uint diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz2g_mod(diff * oz2g_garner_inv[j][i], i);
      }
    }
    v[i] = u;
  }

  is_negative = (v[NPRIMES - 1] >= (uint)(oz2g_moduli[NPRIMES - 1] + 1) / 2) ? 1 : 0;

  /**
   * Two's-complement the mixed-radix digits and complete the negation with a
   * +1 carry propagated in integer space. Adding the 1 to the reconstructed
   * value in floating point instead would be inexact once the magnitude
   * exceeds 2^MANT_BITS, so the Horner evaluation below already yields |x|.
   */
  if (0 != is_negative) {
    UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i)
    {
      v[i] = oz2g_moduli[i] - 1 - v[i];
    }
    for (i = 0; i < NPRIMES; ++i) {
      if (v[i] + 1 < oz2g_moduli[i]) {
        v[i] += 1;
        break;
      }
      v[i] = 0;
    }
  }
  return is_negative;
}


/* Horner evaluation + exponent scaling + C accumulation */
inline void oz2g_horner_accumulate(const uint* restrict v, int is_negative, real_t alpha, int base_sh, real_t* cval)
{
  SINT i;
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  {
    const int ngroups = OZ2_HORNER_NGROUPS;
    double result;
    int g;

    {
      const int lo = (ngroups - 1) * OZ2_HORNER_GROUP;
      ulong r = (ulong)v[NPRIMES - 1];
      for (i = NPRIMES - 2; i >= lo; --i) {
        r = r * (ulong)oz2g_moduli[i] + (ulong)v[i];
      }
      result = (double)r;
    }

    for (g = ngroups - 2; g >= 0; --g) {
      const int lo = g * OZ2_HORNER_GROUP;
      const int hi = lo + OZ2_HORNER_GROUP - 1;
      ulong gval, gprod = 1;
      for (i = lo; i <= hi; ++i) gprod *= (ulong)oz2g_moduli[i];
      gval = (ulong)v[hi];
      for (i = hi - 1; i >= lo; --i) {
        gval = gval * (ulong)oz2g_moduli[i] + (ulong)v[i];
      }
      result = result * (double)gprod + (double)gval;
    }

    result = (0 != is_negative) ? -result : result;
    if (0.0 != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  {
    const int ngroups = OZ2_HORNER_NGROUPS;
    float result;
    int g;

    {
      const int lo = (ngroups - 1) * OZ2_HORNER_GROUP;
      ulong r = (ulong)v[NPRIMES - 1];
      for (i = NPRIMES - 2; i >= lo; --i) {
        r = r * (ulong)oz2g_moduli[i] + (ulong)v[i];
      }
      result = (float)r;
    }

    for (g = ngroups - 2; g >= 0; --g) {
      const int lo = g * OZ2_HORNER_GROUP;
      const int hi = lo + OZ2_HORNER_GROUP - 1;
      ulong gval, gprod = 1;
      for (i = lo; i <= hi; ++i) gprod *= (ulong)oz2g_moduli[i];
      gval = (ulong)v[hi];
      for (i = hi - 1; i >= lo; --i) {
        gval = gval * (ulong)oz2g_moduli[i] + (ulong)v[i];
      }
      result = result * (float)gprod + (float)gval;
    }

    {
      const float fresult = (0 != is_negative) ? -result : result;
      if (0.0f != fresult && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
        const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
        *cval += fresult * scale;
      }
    }
  }
#endif
}


#if OZAKI_HIER

/**
 * Host-precomputed HIER L2 tables (passed as -D defines).
 * HIER_GPROD_g: actual group product for group g.
 * HIER_L2B_g: Barrett constant floor(2^64/gprod_g).
 * HIER_L2INV_j_i: gprod_j^{-1} mod gprod_i.
 */
#if defined(HIER_GPROD_0)
constant uint oz2g_hier_gprod_actual[] = {
  HIER_GPROD_0,
# if defined(HIER_GPROD_1)
  HIER_GPROD_1,
# endif
# if defined(HIER_GPROD_2)
  HIER_GPROD_2,
# endif
# if defined(HIER_GPROD_3)
  HIER_GPROD_3,
# endif
# if defined(HIER_GPROD_4)
  HIER_GPROD_4,
# endif
};
constant ulong oz2g_hier_l2b_actual[] = {
  HIER_L2B_0,
# if defined(HIER_L2B_1)
  HIER_L2B_1,
# endif
# if defined(HIER_L2B_2)
  HIER_L2B_2,
# endif
# if defined(HIER_L2B_3)
  HIER_L2B_3,
# endif
# if defined(HIER_L2B_4)
  HIER_L2B_4,
# endif
};
#else
# define oz2g_hier_gprod_actual oz2g_hier_gprod
# define oz2g_hier_l2b_actual oz2g_hier_l2_barrett
#endif

/* Level-2 Barrett reduction using host-precomputed tables. */
inline uint oz2g_mod_l2(ulong x, int gidx)
{
  const uint m = oz2g_hier_gprod_actual[gidx];
  const ulong q = mul_hi(x, oz2g_hier_l2b_actual[gidx]);
  uint r = (uint)(x - q * (ulong)m);
  return (r >= m) ? (r - m) : r;
}


/**
 * Level-1 Garner: reconstruct HIER_GS residues for group g -> uint group value.
 * group_residues[0..gsz-1] are the per-prime residues within this group.
 * g: group index (for moduli/garner_inv offset = g * HIER_GS).
 */
inline uint oz2g_hier_l1_garner(const uint* restrict group_residues, int g)
{
  const int lo = g * HIER_GS;
  const int hi = (lo + HIER_GS <= NPRIMES) ? (lo + HIER_GS) : NPRIMES;
  const int gsz = hi - lo;
  uint v[HIER_GS];
  SINT li, lj;
  ulong hval;

  for (li = 0; li < gsz; ++li) {
    uint u = group_residues[li];
    const uint pi = oz2g_moduli[lo + li];
    for (lj = 0; lj < li; ++lj) {
      uint vj = v[lj];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      {
        const uint diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz2g_mod(diff * oz2g_garner_inv[lo + lj][lo + li], lo + li);
      }
    }
    v[li] = u;
  }

  hval = (ulong)v[gsz - 1];
  for (li = gsz - 2; li >= 0; --li) {
    hval = hval * (ulong)oz2g_moduli[lo + li] + (ulong)v[li];
  }
  return (uint)hval;
}

#if !defined(OZAKI_HIER_L2) || (0 == OZAKI_HIER_L2)
/**
 * Level-2 Garner: reconstruct HIER_NGROUPS group values -> mixed-radix digits + sign.
 * Uses host-precomputed tables (oz2g_hier_gprod_actual, oz2g_hier_l2inv) to handle
 * partial last group without runtime modular-inverse computation.
 */
/**
 * L2 Garner inverse lookup: use host-precomputed defines when available,
 * fall back to hardcoded table otherwise.
 */
#if defined(HIER_L2INV_0_1)
inline uint oz2g_hier_l2inv(int j, int i)
{
  /* max 5 groups -> 10 unique (j,i) pairs with j<i */
# if defined(HIER_L2INV_0_4)
  if (0 == j && 4 == i) return HIER_L2INV_0_4;
# endif
# if defined(HIER_L2INV_1_4)
  if (1 == j && 4 == i) return HIER_L2INV_1_4;
# endif
# if defined(HIER_L2INV_2_4)
  if (2 == j && 4 == i) return HIER_L2INV_2_4;
# endif
# if defined(HIER_L2INV_3_4)
  if (3 == j && 4 == i) return HIER_L2INV_3_4;
# endif
# if defined(HIER_L2INV_0_3)
  if (0 == j && 3 == i) return HIER_L2INV_0_3;
# endif
# if defined(HIER_L2INV_1_3)
  if (1 == j && 3 == i) return HIER_L2INV_1_3;
# endif
# if defined(HIER_L2INV_2_3)
  if (2 == j && 3 == i) return HIER_L2INV_2_3;
# endif
# if defined(HIER_L2INV_0_2)
  if (0 == j && 2 == i) return HIER_L2INV_0_2;
# endif
# if defined(HIER_L2INV_1_2)
  if (1 == j && 2 == i) return HIER_L2INV_1_2;
# endif
  return HIER_L2INV_0_1;
}
#else
inline uint oz2g_hier_l2inv(int j, int i) { return oz2g_hier_l2_garner_inv[j][i]; }
#endif

inline int oz2g_hier_l2_garner(const uint* restrict gval, uint* restrict d)
{
  SINT i, j;
  int is_negative;

  for (i = 0; i < HIER_NGROUPS; ++i) {
    uint u = gval[i];
    const uint mi = oz2g_hier_gprod_actual[i];
    for (j = 0; j < i; ++j) {
      uint dj = d[j];
      if (dj >= mi) dj = oz2g_mod_l2((ulong)dj, i);
      {
        const uint diff = (u >= dj) ? (u - dj) : (mi + u - dj);
        u = oz2g_mod_l2((ulong)diff * (ulong)oz2g_hier_l2inv(j, i), i);
      }
    }
    d[i] = u;
  }

  is_negative = (d[HIER_NGROUPS - 1] >= (oz2g_hier_gprod_actual[HIER_NGROUPS - 1] + 1) / 2) ? 1 : 0;

  /**
   * Two's-complement the level-2 digits and complete the negation with a +1
   * carry propagated in integer space, so the Horner evaluation already yields
   * |x|. Adding the 1 to the reconstructed value in floating point instead
   * would be inexact once the magnitude exceeds 2^MANT_BITS.
   */
  if (0 != is_negative) {
    for (i = 0; i < HIER_NGROUPS; ++i) {
      d[i] = oz2g_hier_gprod_actual[i] - 1 - d[i];
    }
    for (i = 0; i < HIER_NGROUPS; ++i) {
      if (d[i] + 1 < oz2g_hier_gprod_actual[i]) {
        d[i] += 1;
        break;
      }
      d[i] = 0;
    }
  }
  return is_negative;
}
#endif /* OZAKI_HIER_L2 == 0 */


#if defined(OZAKI_HIER_L2) && (1 == OZAKI_HIER_L2)
/**
 * Tree-merge CRT: binary tree of pairwise merges over group values.
 * Depth = ceil(log2(HIER_NGROUPS)) instead of HIER_NGROUPS-1 (sequential Garner).
 * Each merge: val = gval[a] + gp[a] * ((gval[b] - gval[a]) * inv_a_b % gp[b])
 * Host precomputes: HIER_TREE_INV_a_b = gp[a]^{-1} mod gp[b] for each merge pair.
 * HIER_TREE_PROD_ab = gp[a] * gp[b] (uint64) for merged modulus.
 */

/**
 * Tree-merge L2 reconstruction.
 * Returns combined integer value and sign. Result written to *out_val.
 * Uses host-precomputed merge inverses (HIER_TREE_INV_a_b defines).
 * Only 1 and 2 groups are implemented; without this guard a larger group count
 * would compile with no branch taken, leaving the result unassigned.
 */
#if 2 < HIER_NGROUPS
# error "OZAKI_HIER_L2 (tree-merge) supports at most 2 groups; use OZAKI_HIER_L2=0"
#endif
inline int oz2g_hier_l2_tree(const uint* restrict gval, ulong* out_val)
{
  int is_negative;
#if HIER_NGROUPS == 1
  { const ulong combined = (ulong)gval[0];
    const ulong half_m = (ulong)oz2g_hier_gprod_actual[0] / 2;
    is_negative = (combined > half_m) ? 1 : 0;
    *out_val = (0 != is_negative) ? ((ulong)oz2g_hier_gprod_actual[0] - combined) : combined;
  }
#elif HIER_NGROUPS == 2
  { const uint gp1 = oz2g_hier_gprod_actual[1];
    const uint v0_mod1 = oz2g_mod_l2((ulong)gval[0], 1);
    const uint diff = (gval[1] >= v0_mod1)
      ? (gval[1] - v0_mod1) : (gp1 + gval[1] - v0_mod1);
    const uint t = oz2g_mod_l2((ulong)diff * (ulong)HIER_TREE_INV_0_1, 1);
    const ulong combined = (ulong)gval[0] + (ulong)oz2g_hier_gprod_actual[0] * (ulong)t;
    const ulong half_m = HIER_TREE_PROD_01 / 2;
    is_negative = (combined > half_m) ? 1 : 0;
    *out_val = (0 != is_negative) ? (HIER_TREE_PROD_01 - combined) : combined;
  }
#endif
  return is_negative;
}

/**
 * Tree-merge accumulate: direct FP conversion from combined ulong value.
 * Unlike Horner over complemented digits, the tree merge returns the
 * absolute value directly (M - combined), so negation is -(val).
 */
inline void oz2g_hier_tree_accumulate(ulong val, int is_negative,
                                      real_t alpha, int base_sh, real_t* cval)
{
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  { double result = (double)val;
    if (0 != is_negative) result = -result;
    if (0.0 != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  { float result = (float)val;
    if (0 != is_negative) result = -result;
    if (0.0f != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
      *cval += result * scale;
    }
  }
#endif
}
#endif /* OZAKI_HIER_L2 == 1 */


#if !defined(OZAKI_HIER_L2) || (0 == OZAKI_HIER_L2)
/* Horner evaluation over level-2 mixed-radix digits. */
inline void oz2g_hier_horner_accumulate(const uint* restrict d, int is_negative,
                                        real_t alpha, int base_sh, real_t* cval)
{
  SINT i;
#define gp oz2g_hier_gprod_actual
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  {
    const int nsuper = HIER_L2_HORNER_NGROUPS;
    double result;
    int sg;

    {
      const int lo = (nsuper - 1) * HIER_L2_HORNER_GROUP;
      ulong r = (ulong)d[HIER_NGROUPS - 1];
      for (i = HIER_NGROUPS - 2; i >= lo; --i) {
        r = r * (ulong)gp[i] + (ulong)d[i];
      }
      result = (double)r;
    }

    for (sg = nsuper - 2; sg >= 0; --sg) {
      const int lo = sg * HIER_L2_HORNER_GROUP;
      const int hi = lo + HIER_L2_HORNER_GROUP - 1;
      ulong sgval, sgprod = 1;
      for (i = lo; i <= hi; ++i) sgprod *= (ulong)gp[i];
      sgval = (ulong)d[hi];
      for (i = hi - 1; i >= lo; --i) {
        sgval = sgval * (ulong)gp[i] + (ulong)d[i];
      }
      result = result * (double)sgprod + (double)sgval;
    }

    result = (0 != is_negative) ? -result : result;
    if (0.0 != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  {
    const int nsuper_s = HIER_L2_HORNER_NGROUPS;
    float result_s;
    int sg_s;

    {
      const int lo = (nsuper_s - 1) * HIER_L2_HORNER_GROUP;
      ulong r = (ulong)d[HIER_NGROUPS - 1];
      for (i = HIER_NGROUPS - 2; i >= lo; --i) {
        r = r * (ulong)gp[i] + (ulong)d[i];
      }
      result_s = (float)r;
    }

    for (sg_s = nsuper_s - 2; sg_s >= 0; --sg_s) {
      const int lo = sg_s * HIER_L2_HORNER_GROUP;
      const int hi = lo + HIER_L2_HORNER_GROUP - 1;
      ulong sgval, sgprod = 1;
      for (i = lo; i <= hi; ++i) sgprod *= (ulong)gp[i];
      sgval = (ulong)d[hi];
      for (i = hi - 1; i >= lo; --i) {
        sgval = sgval * (ulong)gp[i] + (ulong)d[i];
      }
      result_s = result_s * (float)sgprod + (float)sgval;
    }

    {
      const float fresult = (0 != is_negative) ? -result_s : result_s;
      if (0.0f != fresult && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
        const real_t scale = OZAKI_ALPHA_MUL(alpha, EXP2I(base_sh));
        *cval += fresult * scale;
      }
    }
  }
#endif
#undef gp
}
#endif /* OZAKI_HIER_L2 == 0 */
#endif /* OZAKI_HIER */


/**
 * preprocess_a_crt_dense: decompose A into dense per-prime CRT residue matrices.
 *
 * Output layout: As[pidx][M_pad][K_pad] - one dense M_pad x K_pad int8 matrix
 * per prime, with residues in [0, m_pidx-1] and sign folded in.
 *
 * Work-group: (BK_PRE, BM_PRE, 1) - K on dim 0, so the lanes of a sub-group
 * walk col and the NPRIMES stores per element land on consecutive bytes of
 * As[p][row][col]. Every element is loaded once but stored NPRIMES times, so
 * coalescing the store outweighs coalescing the load: mapping lanes to row
 * instead (as the operand layout would suggest) measured 7.5 ms against 1.6 ms
 * for preprocess_b on the same element count at m=n=k=4096, purely from the
 * 16-way store scatter. The load becomes stride-lda for transa=0, which is what
 * preprocess_b already pays.
 *
 * Dispatch: global[0] = BK_PRE (single WG in K) - loops internally.
 */
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
  const int row = (int)get_group_id(1) * BM_PRE + mi;
  int col, emax = 0;

  local int row_max_exp[BM_PRE];
  if (0 == kk) row_max_exp[mi] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /**
   * Pass 1: max exponent across ALL of K for this row. A sub-group now shares
   * one row, so a per-element atomic_max would serialize all its lanes on the
   * same SLM address; accumulate privately and contribute once instead.
   */
  for (col = kk; col < K; col += BK_PRE) {
    if (row < M) {
      int s0;
      short e0;
      uint_repr_t m0;
      const int idx = OZAKI_IDX_A(row, col, lda);
      ieee_decompose(a[idx], &s0, &e0, &m0);
      if (e0 > emax) emax = (int)e0;
    }
  }
  if (row < M && 0 < emax) atomic_max(&row_max_exp[mi], emax);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && row < M) expa[row] = row_max_exp[mi];

  /* Pass 2: compute and store CRT residues using the true max exponent */
  if (row < M) {
    const short max_exp = (short)row_max_exp[mi];
    for (col = kk; col < K; col += BK_PRE) {
      int s1;
      short e1;
      uint_repr_t m1;
      const int idx = OZAKI_IDX_A(row, col, lda);
      ieee_decompose(a[idx], &s1, &e1, &m1);
      if (m1 != 0) {
        const int shift = (int)(max_exp - e1);
        const uint_repr_t aligned = (shift + MANT_TRUNC <= MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
        OZAKI_EXTRACT_CRT(aligned, s1, as, M_pad * K_pad, K_pad, row, col);
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
  int row;

  local int col_max_exp[BN_PRE];
  if (0 == kk) col_max_exp[nj] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Pass 1: find max exponent across ALL of K for this column */
  for (row = kk; row < K; row += BK_PRE) {
    if (col < N) {
      int s0;
      short e0;
      uint_repr_t m0;
      const int idx = OZAKI_IDX_B(row, col, ldb);
      ieee_decompose(b[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&col_max_exp[nj], (int)e0);
    }
  }
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
 * gemm_crt_fused: all-primes tiled GEMM with fused Garner + Horner store.
 *
 * Loops over all NPRIMES internally.  For each prime:
 *   1. Full K-loop DPAS accumulation in int32
 *   2. Mod-reduce into per-prime uint residue (with optional KGROUPS
 *      intermediate reductions for large-K overflow safety)
 * After all primes: Garner CRT + Horner evaluation + scaled C store.
 *
 * This eliminates the host-side per-prime kernel dispatch entirely.
 * No symmetrize variant needed (CRT has no cross-prime products).
 *
 * Work-group: (SG, NTM * NTN, 1).
 * Dispatch: global = (nblk_m * SG, nblk_n * NTM * NTN, 1).
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
  const int mb_base = ib_idx * BM;
  const int nb_base = jb_idx * BN;
  const int wt = sg_id * SG + sg_lid;
  const int wg_id = sg_id / WG_NSUB;
  local uint4 wg_sa[2 * ((BM * WBK) / 16)]; /* double-buffered */
  local uint4 wg_sb[2 * ((BN * WBK) / 16)];
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
  uint dot_r_[HIER_GS];
  uint vg_[HIER_NGROUPS];
  uint gval_[HIER_NGROUPS];

#define GRP_RES_STRIDE (RTM * RTN * HIER_GS * XMX_FRAG)
#define GVAL_ALL_STRIDE (RTM * RTN * HIER_NGROUPS * XMX_FRAG)
  uint group_res[GRP_RES_STRIDE];
  uint gval_all[GVAL_ALL_STRIDE];

  /**
   * Group-at-a-time: for each group, accumulate HIER_GS primes into
   * group_res (reused), then level-1 Garner into gval_all (persistent).
   */
  {
    SINT gidx;
    for (gidx = 0; gidx < HIER_NGROUPS; ++gidx) {
      const int group_lo = gidx * HIER_GS;
      {
        int ri;
        for (ri = 0; ri < GRP_RES_STRIDE; ++ri) {
          group_res[ri] = 0;
        }
      }

      {
        SINT pidx_base;
        UNROLL_OUTER(1) for (pidx_base = group_lo; pidx_base < group_lo + HIER_GS && pidx_base < NPRIMES; pidx_base += PB)
        {
          OZAKI_ACC_DECL(acc);
          OZAKI_ACC_ZERO_ALL(acc);

#if KGROUPS > 0
          {
            int k, steps = 0;
            for (k = 0; k < K_pad; k += KU * BK) {
              int ku;
              UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
              {
                OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
              }
              steps += KU;
              if (steps >= KGROUPS) {
                OZAKI_CRT_REDUCE_BATCH_GROUP(OZAKI_ACC_FRAGS(acc), pidx_base, group_lo, group_res, 1);
                steps = 0;
              }
            }
            if (0 != steps) {
              OZAKI_CRT_REDUCE_BATCH_GROUP(OZAKI_ACC_FRAGS(acc), pidx_base, group_lo, group_res, 0);
            }
          }
#else
          {
#if defined(OZAKI_WGMMA) && (OZAKI_WGMMA)
            OZAKI_CRT_KLOOP_W(as, bs, a_plane, b_plane, K_pad, N_pad, mb_base, nb_base, pidx_base,
              acc.s_, wg_sa, wg_sb, wt, wg_id);
#else
            int k;
            for (k = 0; k < K_pad; k += KU * BK) {
              int ku;
              UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
              {
                OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
              }
            }
#endif
            OZAKI_CRT_REDUCE_BATCH_GROUP(OZAKI_ACC_FRAGS(acc), pidx_base, group_lo, group_res, 0);
          }
#endif
        }
      }

      /* Level-1 Garner: group_res -> gval_all[gidx] per tile element */
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
      {
        int rm, rn;
        UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
        {
          UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
          {
            OZAKI_CRT_L1_STORE(
              group_res + (rm * RTN + rn) * HIER_GS * XMX_FRAG,
              gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_FRAG, gidx);
          }
        }
      }
#endif
    }
  }

  /* Level-2 Garner + Horner evaluation + store C */
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
  {
    int rm, rn;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm)
    {
      UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn)
      {
        OZAKI_CRT_L2_STORE(gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_FRAG, expa, expb, c, M, N,
          mi_base + rm * XMX_M, nj_base + rn * XMX_N, sg_lid, ldc, alpha, first);
      }
    }
  }
#endif

#else /* !OZAKI_HIER */
  uint dot_r_[NPRIMES];
  uint vg_[NPRIMES];

  /**
   * Per-prime residue accumulators (private, per work-item).
   * Each SIMD lane accumulates a different output column, so this
   * cannot be shared across lanes in SLM without a lane dimension --
   * but NTM*NTN*SG*RES_STRIDE exceeds SLM capacity.  Private lets
   * the compiler spill to scratch with liveness-aware scheduling
   * (residues are cold during the DPAS K-loop, hot during reduce/store).
   */
#define RES_STRIDE (RTM * RTN * NPRIMES * XMX_FRAG)
  uint residues[RES_STRIDE];

  {
    int ri;
    for (ri = 0; ri < RES_STRIDE; ++ri) {
      residues[ri] = 0;
    }
  }

  /**
   * Loop over primes in batches of PB for improved ILP.
   * PB=1 reproduces the original per-prime loop.
   * PB=2 interleaves two primes in the K-loop, hiding memory
   * latency behind independent DPAS chains.
   */
  {
    SINT pidx_base;
    UNROLL_OUTER(1) for (pidx_base = 0; pidx_base < NPRIMES; pidx_base += PB)
    {
      OZAKI_ACC_DECL(acc);
      OZAKI_ACC_ZERO_ALL(acc);

#if KGROUPS > 0
      {
        int k, steps = 0;
        for (k = 0; k < K_pad; k += KU * BK) {
          int ku;
          UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
          {
            OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
          }
          steps += KU;
          if (steps >= KGROUPS) {
            OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(acc), pidx_base, residues, 1);
            steps = 0;
          }
        }
        if (0 != steps) {
          OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(acc), pidx_base, residues, 0);
        }
      }
#else
      {
        int k;
        for (k = 0; k < K_pad; k += KU * BK) {
          int ku;
          UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
          {
            OZAKI_CRT_KSTEP(as, bs, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
          }
        }
        OZAKI_CRT_REDUCE_BATCH(OZAKI_ACC_FRAGS(acc), pidx_base, residues, 0);
      }
#endif
    }
  }

  /* Garner CRT reconstruction + Horner evaluation + store */
#if !defined(SKIP_GARNER) || (0 == SKIP_GARNER)
  {
    int rm, rn;
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
  uint dot_r_[HIER_GS];
  uint vg_[HIER_NGROUPS];
  uint gval_[HIER_NGROUPS];
  int rm, rn;
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
          int pg;
          UNROLL_FORCE(HIER_GS) for (pg = 0; pg < HIER_GS; ++pg)
          {
            const int pidx = gidx * HIER_GS + pg;
            dot_r_[pg] = (pidx < NPRIMES) ? (uint)res[off + (long)pidx * rplane] : 0u;
          }
          gval_all[gidx * XMX_FRAG + ms] = OZAKI_L1_RECONSTRUCT(dot_r_, gidx);
        }
      }
      OZAKI_CRT_L2_STORE(
        gval_all, expa, expb, c, M, N, mi_base + rm * XMX_M, nj_base + rn * XMX_N, sg_lid, ldc, alpha, first);
    }
  }
}
#endif /* OZAKI_UNFUSE */
