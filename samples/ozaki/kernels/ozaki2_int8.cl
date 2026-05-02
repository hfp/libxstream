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

/* Ozaki Scheme 2 — GEMM-based XMX path (CRT).
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
 *   Sign is encoded via modular additive inverse: (p - r) ≡ -r (mod p).
 *   Trade-off: safe K without KGROUPS drops from ~133K to ~33K.
 *
 * The KGROUPS tunable controls intermediate int32 mod reductions within
 * the K-loop.  When 0 (default), no intermediate reductions — the int32
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
/* Max primes per Horner group that fit ulong accumulation:
   * u8 (moduli<=256): product of 8 largest < 2^64 (group=8)
   * i8 (moduli<=128): product of 9 largest < 2^64 (group=9) */
# if defined(OZAKI_U8) && (OZAKI_U8)
# define OZ2_HORNER_GROUP 8
# else
# define OZ2_HORNER_GROUP 9
# endif
#endif
#if !defined(PB)
# define PB 1
#endif

/* Hierarchical CRT: two-level Garner reconstruction.
 * Level 1: HIER_GS primes per group (small Garner, 32-bit).
 * Level 2: Garner over HIER_NGROUPS group-moduli (32-bit, ulong intermediate).
 * Reduces peak live registers from ~NPRIMES to ~max(HIER_GS, HIER_NGROUPS). */
#if !defined(OZAKI_HIER)
# define OZAKI_HIER 0
#endif
#define POW2_PIDX 3
#if OZAKI_HIER
# define HIER_GS 4
# define HIER_NGROUPS ((NPRIMES + HIER_GS - 1) / HIER_GS)
# define HIER_L2_HORNER_GROUP 2
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

/* Alias the shared DPAS primitive from ozaki_common.cl */
#define OZAKI_CRT_DPAS OZAKI_DPAS

/* Extract NPRIMES CRT residues from aligned mantissa into DST buffer.
 * DST[p * SS + ROW * RS + COL] = (aligned mod m_p), sign-folded.
 * u8: sign via modular additive inverse (p - r), stored as uchar [0, p-1].
 * i8: sign via negation (-r), stored as char [-(p-1), p-1]. */
#define OZAKI_EXTRACT_CRT(ALIGNED, SIGN, DST, SS, RS, ROW, COL) \
  do { \
    SINT p_; \
    UNROLL_FORCE(NPRIMES) for (p_ = 0; p_ < NPRIMES; ++p_) \
    { \
      uint r_ = oz2g_mod64((ulong)(ALIGNED), p_); \
      if ((SIGN) && 0 != r_) OZAKI_SIGN_FOLD_(r_, p_); \
      (DST)[(long)(p_) * (SS) + (long)(ROW) * (RS) + (COL)] = (char)r_; \
    } \
  } while (0)
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_SIGN_FOLD_(R, P) (R) = oz2g_moduli[(P)] - (R)
#else
# define OZAKI_SIGN_FOLD_(R, P) (R) = -(R)
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

/* Mod-reduce DPAS accumulator into uint residue array.
 * RESIDUES[pidx * XMX_M + m] accumulates the unsigned residue.
 * u8: accumulator is always non-negative (unsigned products) — branchless.
 * i8: accumulator can be negative — requires sign-aware reduction. */
#define OZAKI_CRT_MOD_REDUCE(ACC, PIDX, RESIDUES) \
  do { \
    union { \
      int8 v_; \
      int a_[8]; \
    } du_; \
    int mr_; \
    du_.v_ = (ACC); \
    UNROLL_FORCE(XMX_M) for (mr_ = 0; mr_ < XMX_M; ++mr_) \
    { \
      uint r_; \
      OZAKI_MOD_REDUCE_ELEM_(du_.a_[mr_], (PIDX), r_); \
      { \
        const uint prev_ = (RESIDUES)[(int)(PIDX) * XMX_M + mr_]; \
        const uint sum_ = prev_ + r_; \
        (RESIDUES)[(int)(PIDX) * XMX_M + mr_] = (sum_ >= oz2g_moduli[(PIDX)]) ? (sum_ - oz2g_moduli[(PIDX)]) : sum_; \
      } \
    } \
  } while (0)
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZAKI_MOD_REDUCE_ELEM_(VAL, PIDX, R) (R) = oz2g_mod((uint)(VAL), (PIDX))
#else
# define OZAKI_MOD_REDUCE_ELEM_(VAL, PIDX, R) \
    if ((VAL) >= 0) { \
      (R) = oz2g_mod((uint)(VAL), (PIDX)); \
    } \
    else { \
      const uint nr_ = oz2g_mod((uint)(-(VAL)), (PIDX)); \
      (R) = (0 != nr_) ? (oz2g_moduli[(PIDX)] - nr_) : 0; \
    }
#endif

#if !OZAKI_HIER
/* Garner + Horner store: reconstruct from per-prime residues, scale, write C */
#define OZAKI_CRT_STORE(RESIDUES, EXPA, EXPB, C_PTR, M, N, MI, COL, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_M]; \
    const short eb_c_ = ((COL) < (N)) ? (EXPB)[(COL)] : 0; \
    int ms_; \
    UNROLL_FORCE(XMX_M) for (ms_ = 0; ms_ < XMX_M; ++ms_) \
    { \
      ea_c_[ms_] = (EXPA)[(MI) + ms_]; \
    } \
    UNROLL_FORCE(XMX_M) for (ms_ = 0; ms_ < XMX_M; ++ms_) \
    { \
      const int rm_ = (MI) + ms_; \
      if (rm_ < (M) && (COL) < (N)) { \
        int is_neg_; \
        SINT pg_; \
        UNROLL_FORCE(NPRIMES) for (pg_ = 0; pg_ < NPRIMES; ++pg_) \
        { \
          dot_r_[pg_] = (RESIDUES)[(int)pg_ * XMX_M + ms_]; \
        } \
        is_neg_ = oz2g_garner_reconstruct(dot_r_, vg_); \
        { \
          const int sh_ = (int)ea_c_[ms_] + (int)eb_c_ - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = (FIRST) ? ZERO : (C_PTR)[(COL) * (LDC) + rm_]; \
          oz2g_horner_accumulate(vg_, is_neg_, (ALPHA), sh_, &cv_); \
          (C_PTR)[(COL) * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#else /* OZAKI_HIER */

/* Level-1 Garner from group-local residues -> gval_all.
 * GROUP_RES: base of group-local residues for this tile [HIER_GS * XMX_M].
 * GVAL_ALL: base of gval_all for this tile [HIER_NGROUPS * XMX_M].
 * GIDX: group index. */
#define OZAKI_CRT_L1_STORE(GROUP_RES, GVAL_ALL, GIDX) \
  do { \
    int ms_l1_; \
    UNROLL_FORCE(XMX_M) for (ms_l1_ = 0; ms_l1_ < XMX_M; ++ms_l1_) \
    { \
      SINT pg_l1_; \
      UNROLL_FORCE(HIER_GS) for (pg_l1_ = 0; pg_l1_ < HIER_GS; ++pg_l1_) \
      { \
        dot_r_[pg_l1_] = (GROUP_RES)[(int)pg_l1_ * XMX_M + ms_l1_]; \
      } \
      (GVAL_ALL)[(GIDX) * XMX_M + ms_l1_] = oz2g_hier_l1_garner(dot_r_, (GIDX)); \
    } \
  } while (0)

/* Level-2 Garner + Horner + store C from gval_all. */
#define OZAKI_CRT_L2_STORE(GVAL_ALL, EXPA, EXPB, C_PTR, M, N, MI, COL, LDC, ALPHA, FIRST) \
  do { \
    short ea_c_[XMX_M]; \
    const short eb_c_ = ((COL) < (N)) ? (EXPB)[(COL)] : 0; \
    int ms_l2_; \
    UNROLL_FORCE(XMX_M) for (ms_l2_ = 0; ms_l2_ < XMX_M; ++ms_l2_) \
    { \
      ea_c_[ms_l2_] = (EXPA)[(MI) + ms_l2_]; \
    } \
    UNROLL_FORCE(XMX_M) for (ms_l2_ = 0; ms_l2_ < XMX_M; ++ms_l2_) \
    { \
      const int rm_ = (MI) + ms_l2_; \
      if (rm_ < (M) && (COL) < (N)) { \
        int is_neg_; \
        SINT pg_l2_; \
        UNROLL_FORCE(HIER_NGROUPS) for (pg_l2_ = 0; pg_l2_ < HIER_NGROUPS; ++pg_l2_) \
        { \
          gval_[pg_l2_] = (GVAL_ALL)[(int)pg_l2_ * XMX_M + ms_l2_]; \
        } \
        is_neg_ = oz2g_hier_l2_garner(gval_, vg_); \
        { \
          const int sh_ = (int)ea_c_[ms_l2_] + (int)eb_c_ - (2 * BIAS_PLUS_MANT); \
          real_t cv_ = (FIRST) ? ZERO : (C_PTR)[(COL) * (LDC) + rm_]; \
          oz2g_hier_horner_accumulate(vg_, is_neg_, (ALPHA), sh_, &cv_); \
          (C_PTR)[(COL) * (LDC) + rm_] = cv_; \
        } \
      } \
    } \
  } while (0)
#endif /* OZAKI_HIER */

/* K-loop inner body: prefetch + DPAS for PB batched primes.
 * AS_BASE, BS_BASE: base pointers for all prime planes.
 * A_PLANE, B_PLANE: per-prime plane offsets.
 * PIDX_BASE: first prime in current batch.
 * ACC: int8 array of PB*RTM*RTN accumulators. */
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

#if OZAKI_HIER
/* Mod-reduce with separate global prime index (for moduli lookup)
 * and local storage index (for residue array offset). */
#define OZAKI_CRT_MOD_REDUCE_LOCAL(ACC, PIDX, LOCAL_IDX, RESIDUES) \
  do { \
    union { \
      int8 v_; \
      int a_[8]; \
    } dul_; \
    int mrl_; \
    dul_.v_ = (ACC); \
    UNROLL_FORCE(XMX_M) for (mrl_ = 0; mrl_ < XMX_M; ++mrl_) \
    { \
      uint rl_; \
      OZAKI_MOD_REDUCE_ELEM_(dul_.a_[mrl_], (PIDX), rl_); \
      { \
        const uint prevl_ = (RESIDUES)[(int)(LOCAL_IDX) * XMX_M + mrl_]; \
        const uint suml_ = prevl_ + rl_; \
        (RESIDUES)[(int)(LOCAL_IDX) * XMX_M + mrl_] = (suml_ >= oz2g_moduli[(PIDX)]) ? (suml_ - oz2g_moduli[(PIDX)]) : suml_; \
      } \
    } \
  } while (0)

/* HIER variant: mod-reduce into group-local residues (stride HIER_GS * XMX_M).
 * PIDX_BASE: global prime index.  GROUP_LO: first prime in group. */
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
              (GROUP_RES) + (rm_rg_ * RTN + rn_rg_) * HIER_GS * XMX_M); \
            if (ZERO_ACC) { \
              (ACC)[bi_rg_ * RTM * RTN + rm_rg_ * RTN + rn_rg_] = (int8)(0); \
            } \
          } \
        } \
      } \
    } \
  } while (0)
#endif

/* Mod-reduce all PB batched primes' accumulators into residues.
 * If ZERO_ACC is non-zero, also zero the accumulators after reduction. */
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
              (RESIDUES) + (rm_r_ * RTN + rn_r_) * NPRIMES * XMX_M); \
            if (ZERO_ACC) { \
              (ACC)[bi_r_ * RTM * RTN + rm_r_ * RTN + rn_r_] = (int8)(0); \
            } \
          } \
        } \
      } \
    } \
  } while (0)


/* CRT moduli, Barrett constants, pow32_mod, and Garner inverse table.
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
 *   Safe K without KGROUPS: ~133K (127^2 * 32 per DPAS step). */

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

/* Barrett modular reduction: x mod oz2g_moduli[pidx].
 * POW2_PIDX is the power-of-2 modulus (bitmask fast path).
 * u8: 256 = 2^8 -> mask 0xFF.  i8: 128 = 2^7 -> mask 0x7F. */
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

/* Modular reduction for aligned mantissa (up to 53 bits for FP64, 24 for FP32).
 * Decomposes x = hi*2^32 + lo, reduces each part via 32-bit Barrett,
 * then combines.  Avoids expensive 64-bit integer division. */
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

  if (0 != is_negative) {
    UNROLL_FORCE(NPRIMES) for (i = 0; i < NPRIMES; ++i)
    {
      v[i] = oz2g_moduli[i] - 1 - v[i];
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
    const int ngroups = (NPRIMES + OZ2_HORNER_GROUP - 1) / OZ2_HORNER_GROUP;
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

    result = (0 != is_negative) ? -(result + 1.0) : result;
    if (0.0 != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = alpha * EXP2I(base_sh);
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  {
    const int ngroups = (NPRIMES + OZ2_HORNER_GROUP - 1) / OZ2_HORNER_GROUP;
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
      const float fresult = (0 != is_negative) ? -(result + 1.0f) : result;
      if (0.0f != fresult && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
        const real_t scale = alpha * EXP2I(base_sh);
        *cval += fresult * scale;
      }
    }
  }
#endif
}


#if OZAKI_HIER
/* Actual group product for group g (handles partial last group). */
inline uint oz2g_hier_actual_gprod(int g)
{
  const int lo = g * HIER_GS;
  const int hi = (lo + HIER_GS <= NPRIMES) ? (lo + HIER_GS) : NPRIMES;
  uint p = 1;
  int i;
  for (i = lo; i < hi; ++i) p *= (uint)oz2g_moduli[i];
  return p;
}

/* Modular inverse: a^{-1} mod m via extended Euclidean algorithm. */
inline uint oz2g_mod_inverse(uint a, uint m)
{
  long r0 = (long)m, r1 = (long)(a % m);
  long s0 = 0, s1 = 1;
  while (0 != r1) {
    const long q = r0 / r1;
    const long tmp_r = r0 - q * r1; r0 = r1; r1 = tmp_r;
    { const long tmp_s = s0 - q * s1; s0 = s1; s1 = tmp_s; }
  }
  return (uint)((s0 % (long)m + (long)m) % (long)m);
}

/* Level-2 Barrett reduction: (ulong)x mod group product for group gidx. */
inline uint oz2g_mod_l2(ulong x, int gidx)
{
#if (0 != (NPRIMES % HIER_GS))
  const uint m = oz2g_hier_actual_gprod(gidx);
  const ulong q = mul_hi(x, (ulong)(-1) / (ulong)m);
#else
  const uint m = oz2g_hier_gprod[gidx];
  const ulong q = mul_hi(x, oz2g_hier_l2_barrett[gidx]);
#endif
  uint r = (uint)(x - q * (ulong)m);
  return (r >= m) ? (r - m) : r;
}

/* Level-1 Garner: reconstruct HIER_GS residues for group g -> uint group value.
 * group_residues[0..gsz-1] are the per-prime residues within this group.
 * g: group index (for moduli/garner_inv offset = g * HIER_GS). */
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

/* Level-2 Garner: reconstruct HIER_NGROUPS group values -> mixed-radix digits + sign.
 * Uses oz2g_hier_actual_gprod() for group moduli to handle partial last group.
 * L2 Garner inverses are computed inline via oz2g_mod_l2 when the precomputed
 * table may not match (partial group changes the modulus). */
inline int oz2g_hier_l2_garner(const uint* restrict gval, uint* restrict d)
{
  SINT i, j;
  int is_negative;
#if (0 != (NPRIMES % HIER_GS))
  uint gprod_actual[HIER_NGROUPS];
  for (i = 0; i < HIER_NGROUPS; ++i) gprod_actual[i] = oz2g_hier_actual_gprod(i);
#else
# define gprod_actual oz2g_hier_gprod
#endif

  for (i = 0; i < HIER_NGROUPS; ++i) {
    uint u = gval[i];
    const uint mi = gprod_actual[i];
    for (j = 0; j < i; ++j) {
      uint inv_ji;
      uint dj = d[j];
      if (dj >= mi) dj = oz2g_mod_l2((ulong)dj, i);
#if (0 != (NPRIMES % HIER_GS))
      inv_ji = (i == HIER_NGROUPS - 1 || j == HIER_NGROUPS - 1)
        ? oz2g_mod_inverse(gprod_actual[j], mi)
        : oz2g_hier_l2_garner_inv[j][i];
#else
      inv_ji = oz2g_hier_l2_garner_inv[j][i];
#endif
      {
        const uint diff = (u >= dj) ? (u - dj) : (mi + u - dj);
        u = oz2g_mod_l2((ulong)diff * (ulong)inv_ji, i);
      }
    }
    d[i] = u;
  }

  is_negative = (d[HIER_NGROUPS - 1] >= (gprod_actual[HIER_NGROUPS - 1] + 1) / 2) ? 1 : 0;

  if (0 != is_negative) {
    for (i = 0; i < HIER_NGROUPS; ++i) {
      d[i] = gprod_actual[i] - 1 - d[i];
    }
  }
#if (0 == (NPRIMES % HIER_GS))
# undef gprod_actual
#endif
  return is_negative;
}

/* Horner evaluation over level-2 mixed-radix digits. */
inline void oz2g_hier_horner_accumulate(const uint* restrict d, int is_negative,
                                        real_t alpha, int base_sh, real_t* cval)
{
  SINT i;
#if (0 != (NPRIMES % HIER_GS))
  uint gp[HIER_NGROUPS];
  for (i = 0; i < HIER_NGROUPS; ++i) gp[i] = oz2g_hier_actual_gprod(i);
#else
# define gp oz2g_hier_gprod
#endif
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  {
    const int nsuper = (HIER_NGROUPS + HIER_L2_HORNER_GROUP - 1) / HIER_L2_HORNER_GROUP;
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

    result = (0 != is_negative) ? -(result + 1.0) : result;
    if (0.0 != result && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
      const real_t scale = alpha * EXP2I(base_sh);
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  {
    const int nsuper_s = (HIER_NGROUPS + HIER_L2_HORNER_GROUP - 1) / HIER_L2_HORNER_GROUP;
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
      const float fresult = (0 != is_negative) ? -(result_s + 1.0f) : result_s;
      if (0.0f != fresult && ZERO != alpha && base_sh >= -(BIAS_PLUS_MANT - MANT_BITS - 1)) {
        const real_t scale = alpha * EXP2I(base_sh);
        *cval += fresult * scale;
      }
    }
  }
#endif
#if (0 == (NPRIMES % HIER_GS))
# undef gp
#endif
}
#endif /* OZAKI_HIER */


/**
 * preprocess_a_crt_dense: decompose A into dense per-prime CRT residue matrices.
 *
 * Output layout: As[pidx][M_pad][K_pad] — one dense M_pad x K_pad int8 matrix
 * per prime, with residues in [0, m_pidx-1] and sign folded in.
 *
 * Work-group: (BM_PRE, BK_PRE, 1).
 * Dispatch: global[1] = BK_PRE (single WG in K) — loops internally.
 */
__attribute__((reqd_work_group_size(BM_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_a_crt_dense(CONSTANT const real_t* restrict a, int M, int K, int lda, int transa,
  global char* restrict as, /* [NPRIMES * M_pad * K_pad] */
  global int* restrict expa, /* [M] per-row max exponent (int for atomic_max) */
  int K_pad, int M_pad)
{
  const int mi = (int)get_local_id(0);
  const int kk = (int)get_local_id(1);
  const int row = (int)get_group_id(0) * BM_PRE + mi;
  int col;

  local int row_max_exp[BM_PRE];
  if (0 == kk) row_max_exp[mi] = 0;
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

  if (0 == kk && row < M) expa[row] = row_max_exp[mi];

  /* Pass 2: compute and store CRT residues using the true max exponent */
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
        const uint_repr_t aligned = (shift + MANT_TRUNC <= MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
        OZAKI_EXTRACT_CRT(aligned, s1, as, M_pad * K_pad, K_pad, row, col);
      }
    }
  }
}


/**
 * preprocess_b_crt_dense: decompose B into dense per-prime CRT residue matrices.
 *
 * Output layout: Bs[pidx][K_pad][N_pad] — K-major, N_pad >= 64 for 2D block I/O.
 *
 * Work-group: (BN_PRE, BK_PRE, 1).
 * Dispatch: global[1] = BK_PRE (single WG in K) — loops internally.
 */
__attribute__((reqd_work_group_size(BN_PRE, BK_PRE, 1)))
#if defined(SG) && (0 < SG) && defined(INTEL) && (0 != INTEL)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void
preprocess_b_crt_dense(CONSTANT const real_t* restrict b, int N, int K, int ldb, int transb,
  global char* restrict bs, /* [NPRIMES * K_pad * N_pad] */
  global int* restrict expb, /* [N] per-column max exponent (int for atomic_max) */
  int K_pad, int N_pad)
{
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
      const int idx = transb ? (row * ldb + col) : (col * ldb + row);
      ieee_decompose(b[idx], &s0, &e0, &m0);
      if (e0 > 0) atomic_max(&col_max_exp[nj], (int)e0);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (0 == kk && col < N) expb[col] = col_max_exp[nj];

  /* Pass 2: compute and store CRT residues using the true max exponent */
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
        const uint_repr_t aligned = (shift + MANT_TRUNC <= MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
        OZAKI_EXTRACT_CRT(aligned, s1, bs, K_pad * N_pad, N_pad, row, col);
      }
    }
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
  CONSTANT const char* restrict as_base, /* As: [NPRIMES * M_pad * K_pad] */
  CONSTANT const char* restrict bs_base, /* Bs: [NPRIMES * K_pad * N_pad] */
  CONSTANT const int* restrict expa, /* [M] per-row max exponent */
  CONSTANT const int* restrict expb, /* [N] per-col max exponent */
  global real_t* restrict c, int M, int N, int K_pad, int N_pad, int ldc, int M_pad, real_t alpha,
  int first)
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int sg_lid = (int)LIBXS_SGLID();
  const int sg_id = (int)LIBXS_SGID();
  const int tile_m = sg_id / NTN;
  const int tile_n = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  const int nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  const long a_plane = (long)M_pad * K_pad;
  const long b_plane = (long)K_pad * N_pad;
#if OZAKI_HIER
  uint dot_r_[HIER_GS];
  uint vg_[HIER_NGROUPS];
  uint gval_[HIER_NGROUPS];

#define GRP_RES_STRIDE (RTM * RTN * HIER_GS * XMX_M)
#define GVAL_ALL_STRIDE (RTM * RTN * HIER_NGROUPS * XMX_M)
  uint group_res[GRP_RES_STRIDE];
  uint gval_all[GVAL_ALL_STRIDE];

  /* Group-at-a-time: for each group, accumulate HIER_GS primes into
   * group_res (reused), then level-1 Garner into gval_all (persistent). */
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
          int8 acc[PB * RTM * RTN];
          {
            int ai;
            UNROLL_FORCE(PB * RTM * RTN)
            for (ai = 0; ai < PB * RTM * RTN; ++ai) {
              acc[ai] = (int8)(0);
            }
          }

#if KGROUPS > 0
          {
            int k, steps = 0;
            for (k = 0; k < K_pad; k += KU * BK) {
              int ku;
              UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
              {
                OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
              }
              steps += KU;
              if (steps >= KGROUPS) {
                OZAKI_CRT_REDUCE_BATCH_GROUP(acc, pidx_base, group_lo, group_res, 1);
                steps = 0;
              }
            }
            if (0 != steps) {
              OZAKI_CRT_REDUCE_BATCH_GROUP(acc, pidx_base, group_lo, group_res, 0);
            }
          }
#else
          {
            int k;
            for (k = 0; k < K_pad; k += KU * BK) {
              int ku;
              UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
              {
                OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
              }
            }
            OZAKI_CRT_REDUCE_BATCH_GROUP(acc, pidx_base, group_lo, group_res, 0);
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
              group_res + (rm * RTN + rn) * HIER_GS * XMX_M,
              gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_M, gidx);
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
        const int col = nj_base + rn * XMX_N + sg_lid;
        OZAKI_CRT_L2_STORE(
          gval_all + (rm * RTN + rn) * HIER_NGROUPS * XMX_M, expa, expb, c, M, N, mi_base + rm * XMX_M, col, ldc, alpha, first);
      }
    }
  }
#endif

#else /* !OZAKI_HIER */
  uint dot_r_[NPRIMES];
  uint vg_[NPRIMES];

  /* Per-prime residue accumulators (private, per work-item).
   * Each SIMD lane accumulates a different output column, so this
   * cannot be shared across lanes in SLM without a lane dimension —
   * but NTM*NTN*SG*RES_STRIDE exceeds SLM capacity.  Private lets
   * the compiler spill to scratch with liveness-aware scheduling
   * (residues are cold during the DPAS K-loop, hot during reduce/store). */
#define RES_STRIDE (RTM * RTN * NPRIMES * XMX_M)
  uint residues[RES_STRIDE];

  {
    int ri;
    for (ri = 0; ri < RES_STRIDE; ++ri) {
      residues[ri] = 0;
    }
  }

  /* Loop over primes in batches of PB for improved ILP.
   * PB=1 reproduces the original per-prime loop.
   * PB=2 interleaves two primes in the K-loop, hiding memory
   * latency behind independent DPAS chains. */
  {
    SINT pidx_base;
    UNROLL_OUTER(1) for (pidx_base = 0; pidx_base < NPRIMES; pidx_base += PB)
    {
      int8 acc[PB * RTM * RTN];
      {
        int ai;
        UNROLL_FORCE(PB * RTM * RTN)
        for (ai = 0; ai < PB * RTM * RTN; ++ai) {
          acc[ai] = (int8)(0);
        }
      }

#if KGROUPS > 0
      {
        int k, steps = 0;
        for (k = 0; k < K_pad; k += KU * BK) {
          int ku;
          UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
          {
            if (k + ku * BK < K_pad) {
              OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
            }
          }
          steps += KU;
          if (steps >= KGROUPS) {
            OZAKI_CRT_REDUCE_BATCH(acc, pidx_base, residues, 1);
            steps = 0;
          }
        }
        if (0 != steps) {
          OZAKI_CRT_REDUCE_BATCH(acc, pidx_base, residues, 0);
        }
      }
#else
      {
        int k;
        for (k = 0; k < K_pad; k += KU * BK) {
          int ku;
          UNROLL_FORCE(KU) for (ku = 0; ku < KU; ++ku)
          {
            if (k + ku * BK < K_pad) {
              OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
            }
          }
        }
        OZAKI_CRT_REDUCE_BATCH(acc, pidx_base, residues, 0);
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
        const int col = nj_base + rn * XMX_N + sg_lid;
        OZAKI_CRT_STORE(
          residues + (rm * RTN + rn) * NPRIMES * XMX_M, expa, expb, c, M, N, mi_base + rm * XMX_M, col, ldc, alpha, first);
      }
    }
  }
#endif
#endif /* OZAKI_HIER */
}
