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

/* Auto-enable inline asm K-loop for tested RTM=4 RTN=2 KU=4 config.
 * Uses vISA .decl for zero-copy dpas register allocation, SLM extraction.
 * Disable with -DNO_ASM_KLOOP. */
#if !defined(NO_ASM_KLOOP) && defined(USE_XMX) && (0 < USE_XMX) \
    && (RTM == 4) && (RTN == 2) && (KU == 4) && (RC == 8) && (BK == 32)
# define OZAKI_USE_ASM_KLOOP
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

/* Full tiled K-loop: prefetch + KU-unrolled DPAS, then remainder.
 * AS, BS: slice pointers for this pair.
 * ACC: int8[RTM*RTN] accumulator array (must be pre-zeroed by caller). */
#define OZAKI_KLOOP(AS, BS, K_PAD_, N_PAD_, M_, MI, NJ, ACC) \
  do { \
    int k_l_; \
    for (k_l_ = 0; k_l_ + (KU - 1) * BK < (K_PAD_); k_l_ += KU * BK) { \
      int ku_l_; \
      OZAKI_PREFETCH_TILED(AS, BS, K_PAD_, N_PAD_, \
                           M_, k_l_ + KU * BK, MI, NJ); \
      UNROLL_FORCE(KU) for (ku_l_ = 0; ku_l_ < KU; ++ku_l_) { \
        OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, \
                         MI, NJ, k_l_ + ku_l_ * BK, M_, ACC); \
      } \
    } \
    for (; k_l_ < (K_PAD_); k_l_ += BK) { \
      OZAKI_DPAS_TILED(AS, BS, K_PAD_, N_PAD_, \
                       MI, NJ, k_l_, M_, ACC); \
    } \
  } while (0)

/* vISA inline asm K-loop for RTM=4 RTN=2: full K-loop inside one asm
 * block with KU=4-step unroll and goto-based loop. Only one SLM
 * extraction at the end of the entire K dimension.
 * dpas convention: src1 = B (8 GRFs, .offset), src2 = A (4 GRFs, (row,col)).
 * TAG is a string literal suffix for unique .decl names per call site. */
#if defined(OZAKI_USE_ASM_KLOOP)

/* K-step with zero-init: load A/B blocks, 8 dpas with %%null src0, advance */
#define OZAKI_ASM_KSTEP_INIT(T) \
  "lsc_load_block2d.ugm (M1, 1) ab" T ":d8.32x32nn flat[pa" T " + (0,0)]\n" \
  "lsc_load_block2d.ugm (M1, 1) bb" T ":d8.2x16x32nt flat[pb" T " + (0,0)]\n" \
  "dpas.s8.s8.8.8 (M1,16) c0" T ".0 %%null.0 bb" T ".0   ab" T "(0,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c1" T ".0 %%null.0 bb" T ".512 ab" T "(0,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c2" T ".0 %%null.0 bb" T ".0   ab" T "(4,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c3" T ".0 %%null.0 bb" T ".512 ab" T "(4,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c4" T ".0 %%null.0 bb" T ".0   ab" T "(8,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c5" T ".0 %%null.0 bb" T ".512 ab" T "(8,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c6" T ".0 %%null.0 bb" T ".0   ab" T "(12,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c7" T ".0 %%null.0 bb" T ".512 ab" T "(12,0)\n" \
  "add (M1_NM, 1) pa" T "(0,5)<1> pa" T "(0,5)<0;1,0> 32:d\n" \
  "add (M1_NM, 1) pb" T "(0,6)<1> pb" T "(0,6)<0;1,0> 32:d\n"

/* K-step with accumulation: load A/B blocks, 8 dpas accumulating, advance */
#define OZAKI_ASM_KSTEP(T) \
  "lsc_load_block2d.ugm (M1, 1) ab" T ":d8.32x32nn flat[pa" T " + (0,0)]\n" \
  "lsc_load_block2d.ugm (M1, 1) bb" T ":d8.2x16x32nt flat[pb" T " + (0,0)]\n" \
  "dpas.s8.s8.8.8 (M1,16) c0" T ".0 c0" T ".0 bb" T ".0   ab" T "(0,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c1" T ".0 c1" T ".0 bb" T ".512 ab" T "(0,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c2" T ".0 c2" T ".0 bb" T ".0   ab" T "(4,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c3" T ".0 c3" T ".0 bb" T ".512 ab" T "(4,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c4" T ".0 c4" T ".0 bb" T ".0   ab" T "(8,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c5" T ".0 c5" T ".0 bb" T ".512 ab" T "(8,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c6" T ".0 c6" T ".0 bb" T ".0   ab" T "(12,0)\n" \
  "dpas.s8.s8.8.8 (M1,16) c7" T ".0 c7" T ".0 bb" T ".512 ab" T "(12,0)\n" \
  "add (M1_NM, 1) pa" T "(0,5)<1> pa" T "(0,5)<0;1,0> 32:d\n" \
  "add (M1_NM, 1) pb" T "(0,6)<1> pb" T "(0,6)<0;1,0> 32:d\n"

/* SLM store for one tile: store halves a/b, advance sp by 512 */
#define OZAKI_ASM_SLM_TILE(T, I) \
  "lsc_store.slm (M1_NM, 1) flat[sp" T "]:a32        c" #I "a" T ".0:d32x64t\n" \
  "lsc_store.slm (M1_NM, 1) flat[sp" T "+0x100]:a32  c" #I "b" T ".0:d32x64t\n" \
  "add (M1_NM, 1) sp" T "(0,0)<1> sp" T "(0,0)<0;1,0> 512:d\n"

#define OZAKI_KLOOP_ASM(TAG, AS, BS, K_PAD_, N_PAD_, M_, MI, NJ, ACC, SLM_PTR) \
  do { \
    const long as_q_ = (long)(AS); \
    const long bs_q_ = (long)(BS); \
    const long slm_kp_ = \
      ((long)(unsigned int)((K_PAD_) - 1) << 32) | \
       (unsigned int)(int)((local char*)(SLM_PTR) - (local char*)0); \
    const long np_mh_ = \
      ((long)(unsigned int)((M_) - 1) << 32) | \
       (unsigned int)((N_PAD_) - 1); \
    const long ko_mi_ = \
      ((long)(unsigned int)(MI) << 32) | (unsigned int)0; \
    const long nj_q_ = ((long)0x11F0FU << 32) | (unsigned int)(NJ); \
    const int kpad_ = (int)(K_PAD_); \
    __asm__ volatile( \
      ".decl ab" TAG " v_type=G type=d num_elts=256 align=wordx32\n" \
      ".decl bb" TAG " v_type=G type=d num_elts=256 align=wordx32\n" \
      ".decl c0" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c0a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c0" TAG ",0>\n" \
      ".decl c0b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c0" TAG ",256>\n" \
      ".decl c1" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c1a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c1" TAG ",0>\n" \
      ".decl c1b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c1" TAG ",256>\n" \
      ".decl c2" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c2a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c2" TAG ",0>\n" \
      ".decl c2b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c2" TAG ",256>\n" \
      ".decl c3" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c3a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c3" TAG ",0>\n" \
      ".decl c3b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c3" TAG ",256>\n" \
      ".decl c4" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c4a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c4" TAG ",0>\n" \
      ".decl c4b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c4" TAG ",256>\n" \
      ".decl c5" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c5a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c5" TAG ",0>\n" \
      ".decl c5b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c5" TAG ",256>\n" \
      ".decl c6" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c6a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c6" TAG ",0>\n" \
      ".decl c6b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c6" TAG ",256>\n" \
      ".decl c7" TAG " v_type=G type=d num_elts=128 align=wordx32\n" \
      ".decl c7a" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c7" TAG ",0>\n" \
      ".decl c7b" TAG " v_type=G type=d num_elts=64 align=wordx32 alias=<c7" TAG ",256>\n" \
      /* Unpack constraint words */ \
      ".decl skq" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      ".decl skd" TAG " v_type=G type=ud num_elts=2 align=qword alias=<skq" TAG ",0>\n" \
      "mov (M1_NM, 1) skq" TAG "(0,0)<1> %2(0,0)<0;1,0>\n" \
      ".decl nmq" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      ".decl nmd" TAG " v_type=G type=ud num_elts=2 align=qword alias=<nmq" TAG ",0>\n" \
      "mov (M1_NM, 1) nmq" TAG "(0,0)<1> %3(0,0)<0;1,0>\n" \
      ".decl kmq" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      ".decl kmd" TAG " v_type=G type=ud num_elts=2 align=qword alias=<kmq" TAG ",0>\n" \
      "mov (M1_NM, 1) kmq" TAG "(0,0)<1> %4(0,0)<0;1,0>\n" \
      ".decl njq" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      ".decl njd" TAG " v_type=G type=ud num_elts=2 align=qword alias=<njq" TAG ",0>\n" \
      "mov (M1_NM, 1) njq" TAG "(0,0)<1> %5(0,0)<0;1,0>\n" \
      ".decl sa" TAG " v_type=G type=ud num_elts=1 align=dword\n" \
      "mov (M1_NM, 1) sa" TAG "(0,0)<1> skd" TAG "(0,0)<0;1,0>\n" \
      /* K-loop end value */ \
      ".decl ke" TAG " v_type=G type=ud num_elts=1 align=dword\n" \
      "mov (M1_NM, 1) ke" TAG "(0,0)<1> %6(0,0)<0;1,0>\n" \
      /* Address payloads: A (pa) and B (pb) */ \
      ".decl ap" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      "mov (M1_NM, 1) ap" TAG "(0,0)<1> %0(0,0)<0;1,0>\n" \
      ".decl bp" TAG " v_type=G type=uq num_elts=1 align=qword\n" \
      "mov (M1_NM, 1) bp" TAG "(0,0)<1> %1(0,0)<0;1,0>\n" \
      ".decl pa" TAG " v_type=G type=ud num_elts=8 align=wordx32\n" \
      ".decl paq" TAG " v_type=G type=uq num_elts=4 align=wordx32 alias=<pa" TAG ",0>\n" \
      "mov (M1_NM, 1) paq" TAG "(0,0)<1> ap" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,2)<1> skd" TAG "(0,1)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,3)<1> nmd" TAG "(0,1)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,4)<1> skd" TAG "(0,1)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,5)<1> kmd" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,6)<1> kmd" TAG "(0,1)<0;1,0>\n" \
      "mov (M1_NM, 1) pa" TAG "(0,7)<1> 0x1F1F:d\n" \
      ".decl pb" TAG " v_type=G type=ud num_elts=8 align=wordx32\n" \
      ".decl pbq" TAG " v_type=G type=uq num_elts=4 align=wordx32 alias=<pb" TAG ",0>\n" \
      "mov (M1_NM, 1) pbq" TAG "(0,0)<1> bp" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,2)<1> nmd" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,3)<1> skd" TAG "(0,1)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,4)<1> nmd" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,5)<1> njd" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,6)<1> kmd" TAG "(0,0)<0;1,0>\n" \
      "mov (M1_NM, 1) pb" TAG "(0,7)<1> njd" TAG "(0,1)<0;1,0>\n" \
      /* K-loop counter */ \
      ".decl kc" TAG " v_type=G type=ud num_elts=1 align=dword\n" \
      "mov (M1_NM, 1) kc" TAG "(0,0)<1> 0:d\n" \
      /* First KU=4 block: zero-init + 3 accumulating steps */ \
      OZAKI_ASM_KSTEP_INIT(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      "add (M1_NM, 1) kc" TAG "(0,0)<1> kc" TAG "(0,0)<0;1,0> 128:d\n" \
      /* Loop: remaining KU=4 blocks with accumulating dpas */ \
      ".decl P0" TAG " v_type=P num_elts=1\n" \
      "kloop" TAG ":\n" \
      "cmp.ge (M1_NM, 1) P0" TAG " kc" TAG "(0,0)<0;1,0> ke" TAG "(0,0)<0;1,0>\n" \
      "(P0" TAG ") goto (M1_NM, 1) kexit" TAG "\n" \
      OZAKI_ASM_KSTEP(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      OZAKI_ASM_KSTEP(TAG) \
      "add (M1_NM, 1) kc" TAG "(0,0)<1> kc" TAG "(0,0)<0;1,0> 128:d\n" \
      "goto (M1_NM, 1) kloop" TAG "\n" \
      "kexit" TAG ":\n" \
      /* SLM extraction: 2 stores per tile x 8 tiles */ \
      ".decl sp" TAG " v_type=G type=ud num_elts=1 align=wordx32\n" \
      "mov (M1_NM, 1) sp" TAG "(0,0)<1> sa" TAG "(0,0)<0;1,0>\n" \
      OZAKI_ASM_SLM_TILE(TAG, 0) \
      OZAKI_ASM_SLM_TILE(TAG, 1) \
      OZAKI_ASM_SLM_TILE(TAG, 2) \
      OZAKI_ASM_SLM_TILE(TAG, 3) \
      OZAKI_ASM_SLM_TILE(TAG, 4) \
      OZAKI_ASM_SLM_TILE(TAG, 5) \
      OZAKI_ASM_SLM_TILE(TAG, 6) \
      OZAKI_ASM_SLM_TILE(TAG, 7) \
      : /* no outputs */ \
      : "rw"(as_q_), "rw"(bs_q_), "rw"(slm_kp_), \
        "rw"(np_mh_), "rw"(ko_mi_), "rw"(nj_q_), "rw"(kpad_) \
      : "memory" \
    ); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    { int t_a_; \
      UNROLL_FORCE(RTM * RTN) \
      for (t_a_ = 0; t_a_ < RTM * RTN; ++t_a_) { \
        (ACC)[t_a_] += as_int8(intel_sub_group_block_read8( \
          (const local uint*)((SLM_PTR) + t_a_ * 128))); \
      } \
    } \
  } while (0)
#endif

/* Scale i32 accumulator + accumulate into register-resident fp C with
 * pre-cached exponents in registers.
 * EA_CACHE is a short array[XMX_M], EB_CACHE is a short scalar.
 * Avoids re-reading expa/expb from global memory for every pair. */
#define OZAKI_GEMM_ACCUM_CACHED(DOT, EA_CACHE, EB_CACHE, C_REG, M, N, MI, COL, \
                                ALPHA, LOW_SA, LOW_SB) \
  do { \
    union { int8 v_; int a_[8]; } du_c_; \
    int m_c_; \
    du_c_.v_ = (DOT); \
    UNROLL_FORCE(XMX_M) for (m_c_ = 0; m_c_ < XMX_M; ++m_c_) { \
      const int rm_c_ = (MI) + m_c_; \
      if (rm_c_ < (M) && (COL) < (N)) { \
        const int sh_c_ = (int)(EA_CACHE)[m_c_] + (int)(EB_CACHE) \
                         - (2 * BIAS_PLUS_MANT) + (LOW_SA) + (LOW_SB); \
        const real_t sc_c_ = (ALPHA) * EXP2I(sh_c_); \
        (C_REG)[m_c_] += (real_t)du_c_.a_[m_c_] * sc_c_; \
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
#if defined(OZAKI_USE_ASM_KLOOP)
  local int acc_slm_[NTM * NTN * RTM * RTN * 128];
  local int* my_slm_ = acc_slm_ + sg_id * (RTM * RTN * 128);
#endif

  /* Pre-cache exponents in registers: avoid re-reading from global per pair.
   * ea_cache[rm][m] = expa[mi_base + rm*XMX_M + m]
   * eb_cache[rn]    = expb[nj_base + rn*XMX_N + sg_lid] */
  short ea_cache[RTM * XMX_M];
  short eb_cache[RTN];
  { int rm;
    UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm) {
      int m_;
      UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) {
        const int r_ = mi_base + rm * XMX_M + m_;
        ea_cache[rm * XMX_M + m_] = (r_ < M) ? expa[r_] : 0;
      }
    }
  }
  { int rn;
    UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn) {
      const int col = nj_base + rn * XMX_N + sg_lid;
      eb_cache[rn] = (col < N) ? expb[col] : 0;
    }
  }

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
#if OZAKI_SCALAR_ACC
      /* Scalar accumulators for DPAS, packed to array for scaling. */
      OZAKI_ACC_DECL(c_acc_);
      OZAKI_KLOOP(as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base, c_acc_);

      if (0 == sq && sa != sb) {
        OZAKI_ACC_DECL(c_mir_);
        OZAKI_KLOOP(as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base, c_mir_);
        OZAKI_ACC_ADD(c_acc_, c_mir_);
      }

      { int8 c_acc[RTM * RTN];
        OZAKI_ACC_PACK(c_acc_, c_acc);
#else
      int8 c_acc[RTM * RTN];
      { int ri;
        UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
          c_acc[ri] = (int8)(0);
        }
      }
#if defined(OZAKI_USE_ASM_KLOOP)
      OZAKI_KLOOP_ASM("P", as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base,
                      c_acc, my_slm_);
#else
      OZAKI_KLOOP(as_sa, bs_sb, K_pad, N_pad, M, mi_base, nj_base, c_acc);
#endif

      /* For off-diagonal triangle pairs, also compute (sb, sa) */
      if (0 == sq && sa != sb) {
        int8 c_mir[RTM * RTN];
        { int ri;
          UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
            c_mir[ri] = (int8)(0);
          }
        }
#if defined(OZAKI_USE_ASM_KLOOP)
        OZAKI_KLOOP_ASM("M", as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base,
                        c_mir, my_slm_);
#else
        OZAKI_KLOOP(as_sb, bs_sa, K_pad, N_pad, M, mi_base, nj_base, c_mir);
#endif
        { int ri;
          UNROLL_FORCE(RTM * RTN) for (ri = 0; ri < RTM * RTN; ++ri) {
            c_acc[ri] = c_acc[ri] + c_mir[ri];
          }
        }
      }

      {
#endif

      /* Scale and accumulate into register C (cached exponents) */
      { int rm, rn;
        UNROLL_FORCE(RTM) for (rm = 0; rm < RTM; ++rm) {
          UNROLL_FORCE(RTN) for (rn = 0; rn < RTN; ++rn) {
            const int idx = rm * RTN + rn;
            const int col = nj_base + rn * XMX_N + sg_lid;
            OZAKI_GEMM_ACCUM_CACHED(c_acc[idx],
                             ea_cache + rm * XMX_M, eb_cache[rn],
                             c_fp + idx * XMX_M,
                             M, N, mi_base + rm * XMX_M, col,
                             alpha, low_bit_sa, low_bit_sb);
          }
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
