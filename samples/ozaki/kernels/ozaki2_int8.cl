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
 * u8 (OZAKI_U8=1, default): 20 pairwise coprime integers <= 256.
 *   Prime powers: 256=2^8, 243=3^5, 169=13^2.  Rest are primes.
 *   Larger moduli -> fewer primes for the same cumulative product.
 *   Safe K without KGROUPS: ~33K (255^2 * 32 per DPAS step).
 *
 * i8 (OZAKI_U8=0): 20 pairwise coprime integers <= 128.
 *   Prime powers: 128=2^7, 125=5^3, 121=11^2, 81=3^4.
 *   119=7*17 (composite).
 *   Safe K without KGROUPS: ~133K (127^2 * 32 per DPAS step). */

#if defined(OZAKI_U8) && (OZAKI_U8)

constant ushort oz2g_moduli[] = {256, 251, 243, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181, 179, 173, 169, 167, 163};

constant uint oz2g_barrett_inv[] = {16777216, 17111423, 17674762, 17821441, 17970574, 18433336, 18755315, 18920560, 19259943,
  20355295, 21582750, 21801864, 22253716, 22486739, 23729101, 23994230, 24826400, 25414007, 25718367, 26349492};

constant ushort oz2g_pow32_mod[] = {0, 123, 130, 15, 110, 8, 161, 176, 7, 51, 46, 88, 108, 147, 15, 126, 96, 113, 7, 100};

constant uint oz2g_garner_inv[][20] = {
  /* m_0=256 */ {0, 201, 187, 225, 225, 152, 17, 47, 196, 136, 7, 187, 144, 144, 70, 93, 148, 68, 152, 156},
  /* m_1=251 */ {0, 0, 152, 217, 20, 13, 177, 123, 8, 153, 111, 135, 10, 156, 75, 92, 122, 101, 2, 113},
  /* m_2=243 */ {0, 0, 0, 121, 60, 70, 180, 71, 145, 33, 95, 30, 166, 180, 73, 14, 131, 16, 11, 108},
  /* m_3=241 */ {0, 0, 0, 0, 120, 204, 210, 146, 62, 204, 109, 103, 189, 149, 178, 26, 28, 54, 79, 23},
  /* m_4=239 */ {0, 0, 0, 0, 0, 39, 23, 19, 14, 98, 5, 61, 21, 4, 103, 3, 97, 99, 58, 148},
  /* m_5=233 */ {0, 0, 0, 0, 0, 0, 172, 38, 67, 48, 41, 104, 111, 141, 94, 63, 124, 103, 124, 7},
  /* m_6=229 */ {0, 0, 0, 0, 0, 0, 0, 114, 186, 129, 73, 117, 59, 186, 132, 111, 34, 31, 132, 42},
  /* m_7=227 */ {0, 0, 0, 0, 0, 0, 0, 0, 56, 66, 64, 46, 176, 69, 122, 138, 157, 102, 103, 135},
  /* m_8=223 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 141, 144, 148, 6, 125, 118, 45, 72, 3, 144},
  /* m_9=211 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 183, 118, 86, 175, 28, 41, 165, 19, 17},
  /* m10=199 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 161, 24, 171, 9, 20, 62, 47, 77},
  /* m11=197 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 145, 32, 34, 10, 137, 163, 39, 24},
  /* m12=193 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 166, 64, 26, 162, 45, 125},
  /* m13=191 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 15, 125, 146, 7, 99},
  /* m14=181 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 65, 155, 12, 154},
  /* m15=179 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 17, 14, 51},
  /* m16=173 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 28, 49},
  /* m17=169 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 136},
  /* m18=167 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41},
  /* m19=163 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

#else /* i8 fallback */

constant ushort oz2g_moduli[] = {128, 127, 125, 121, 119, 113, 109, 107, 103, 101, 97, 89, 83, 81, 79, 73, 71, 67, 61, 59};

constant uint oz2g_barrett_inv[] = {33554432, 33818640, 34359738, 35495597, 36092162, 38008560, 39403369, 40139881, 41698711,
  42524428, 44278013, 48258059, 51746593, 53024287, 54366674, 58835168, 60492497, 64103989, 70409299, 72796055};

constant ushort oz2g_pow32_mod[] = {0, 16, 46, 59, 18, 16, 75, 29, 63, 68, 35, 45, 77, 49, 50, 32, 9, 33, 57, 51};

constant uint oz2g_garner_inv[][20] = {
  /* m_0=128 */ {0, 1, 42, 52, 53, 98, 23, 51, 33, 15, 72, 16, 24, 50, 50, 4, 5, 11, 51, 6},
  /* m_1=127 */ {0, 0, 63, 101, 15, 105, 103, 91, 73, 35, 55, 82, 17, 37, 28, 23, 52, 19, 49, 46},
  /* m_2=125 */ {0, 0, 0, 91, 20, 66, 75, 6, 89, 80, 52, 47, 2, 35, 67, 66, 25, 52, 41, 17},
  /* m_3=121 */ {0, 0, 0, 0, 60, 99, 100, 23, 63, 96, 93, 64, 59, 79, 32, 35, 27, 36, 60, 20},
  /* m_4=119 */ {0, 0, 0, 0, 0, 19, 11, 9, 58, 73, 75, 3, 30, 32, 2, 27, 37, 58, 20, 1},
  /* m_5=113 */ {0, 0, 0, 0, 0, 0, 82, 18, 31, 59, 91, 26, 36, 38, 7, 42, 22, 51, 27, 47},
  /* m_6=109 */ {0, 0, 0, 0, 0, 0, 0, 54, 86, 38, 89, 49, 16, 55, 29, 71, 43, 8, 14, 13},
  /* m_7=107 */ {0, 0, 0, 0, 0, 0, 0, 0, 26, 17, 68, 5, 45, 53, 48, 58, 2, 62, 4, 16},
  /* m_8=103 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 81, 70, 54, 70, 56, 56, 20, 54, 16, 55},
  /* m_9=101 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 52, 60, 77, 18, 60, 45, 2, 29, 52},
  /* m10= 97 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 6, 76, 22, 70, 41, 38, 39, 14},
  /* m11= 89 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 71, 8, 32, 4, 64, 24, 2},
  /* m12= 83 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 20, 22, 6, 21, 25, 32},
  /* m13= 81 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 64, 64, 24, 58, 51},
  /* m14= 79 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 9, 28, 17, 3},
  /* m15= 73 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 56, 56, 38},
  /* m16= 71 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 55, 5},
  /* m17= 67 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 37},
  /* m18= 61 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30},
  /* m19= 59 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

#endif /* OZAKI_U8 */

#define OZ2G_BARRETT_SHIFT 32

/* Barrett modular reduction: x mod oz2g_moduli[pidx].
 * For pidx==0 (power-of-2 modulus), a simple bitmask suffices:
 * u8: 256 = 2^8 -> mask 0xFF.  i8: 128 = 2^7 -> mask 0x7F. */
#if defined(OZAKI_U8) && (OZAKI_U8)
# define OZ2G_MOD0_MASK 0xFFu
# define OZ2G_MOD0_MASK64 0xFFul
#else
# define OZ2G_MOD0_MASK 0x7Fu
# define OZ2G_MOD0_MASK64 0x7Ful
#endif
inline uint oz2g_mod(uint x, SINT pidx)
{
  if (0 == pidx) return x & OZ2G_MOD0_MASK;
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
  if (0 == pidx) return (uint)(x & OZ2G_MOD0_MASK64);
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
    if (0.0 != result && ZERO != alpha) {
      const real_t scale = alpha * EXP2I(base_sh);
      *cval += (real_t)(result * (double)scale);
    }
  }
#else
  {
    long r = (long)v[NPRIMES - 1];
    for (i = NPRIMES - 2; i >= 0; --i) {
      r = r * (long)oz2g_moduli[i] + (long)v[i];
    }
    {
      const long result = (0 != is_negative) ? -(r + 1) : r;
      if (0 != result && ZERO != alpha) {
        const real_t scale = alpha * EXP2I(base_sh);
        *cval += (real_t)result * scale;
      }
    }
  }
#endif
}


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
#if defined(SG) && (0 < SG)
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
        const uint_repr_t aligned = (shift + MANT_TRUNC < MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
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
#if defined(SG) && (0 < SG)
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
        const uint_repr_t aligned = (shift + MANT_TRUNC < MANT_BITS) ? (m1 >> (shift + MANT_TRUNC)) : 0;
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
__attribute__((reqd_work_group_size(SG, NTM* NTN, 1))) __attribute__((intel_reqd_sub_group_size(SG))) kernel void gemm_crt_fused(
  CONSTANT const char* restrict as_base, /* As: [NPRIMES * M_pad * K_pad] */
  CONSTANT const char* restrict bs_base, /* Bs: [NPRIMES * K_pad * N_pad] */
  CONSTANT const int* restrict expa, /* [M] per-row max exponent */
  CONSTANT const int* restrict expb, /* [N] per-col max exponent */
  global real_t* restrict c, int M, int N, int K_pad, int N_pad, int ldc, int M_pad, real_t alpha,
  int first) /* 1: overwrite C, 0: accumulate */
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int sg_lid = (int)get_sub_group_local_id();
  const int sg_id = (int)get_sub_group_id();
  const int tile_m = sg_id / NTN;
  const int tile_n = sg_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * XMX_M * RTM;
  const int nj_base = jb_idx * BN + tile_n * XMX_N * RTN;
  const long a_plane = (long)M_pad * K_pad;
  const long b_plane = (long)K_pad * N_pad;
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
            OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
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
            OZAKI_CRT_KSTEP(as_base, bs_base, a_plane, b_plane, K_pad, N_pad, M, mi_base, nj_base, k + ku * BK, pidx_base, acc);
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
}
