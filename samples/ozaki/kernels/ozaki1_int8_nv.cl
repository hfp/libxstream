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

/* Ozaki Scheme 1 -- NVIDIA Tensor Core MMA path (NV>=3, SM>=8.0).
 *
 * Uses PTX mma.sync.aligned.m16n8k32 for the int8 GEMM.
 * SG=32 (one warp = one sub-group).  Native MMA tile: 16 rows x 8 cols.
 *
 * Fragment layout for m16n8k32 D/C (32 threads, 4 regs each = 128 values = 16x8):
 *   thread t, group g=t/4, lane l=t%4:
 *     d[0] = D[g*2  ][l*2]      d[1] = D[g*2  ][l*2+1]
 *     d[2] = D[g*2+1][l*2]      d[3] = D[g*2+1][l*2+1]
 *   i.e. each thread holds a 2x2 sub-block at rows (g*2, g*2+1), cols (l*2, l*2+1).
 *
 * After the K-loop, fragments are redistributed via shfl.sync so that each
 * thread holds a full column of 16 rows, ready for the FP scaling phase.
 *
 * Compile-time parameters:
 *   BM, BN    - output tile per work-group
 *   BK=32     - K per MMA step
 *   KU        - K-loop unroll depth
 *   NSLICES, MANT_BITS, BIAS_PLUS_MANT, USE_DOUBLE - Ozaki parameters
 *   SG=32     - sub-group size (warp)
 *   NV>=3     - NVIDIA SM>=8.0
 */

#if !defined(BM)
# define BM 128
#endif
#if !defined(BN)
# define BN 64
#endif
#if !defined(BK)
# define BK 32
#endif
#if !defined(KU)
# define KU 2
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
# define SG 32
#endif

/* MMA tile: 16 rows x 8 cols per warp.  With register tiling (RTM x RTN):
 * each warp computes (RTM * 16) rows x (RTN * 8) cols per K-step. */
#define MMA_M 16
#define MMA_N 8

/* Sub-groups (warps) per work-group dimension */
#define NTM (BM / (MMA_M * RTM))
#define NTN (BN / (MMA_N * RTN))

#if defined(OZAKI_BOUNDS) && (OZAKI_BOUNDS)
# define OZAKI_IN_BOUNDS(R, M, COL, N) ((R) < (M) && (COL) < (N))
#else
# define OZAKI_IN_BOUNDS(R, M, COL, N) (1)
#endif

#if !defined(OZAKI_CUTOFF)
# define OZAKI_CUTOFF 14
#endif
#if !defined(OZAKI_SQ)
# define OZAKI_SQ 0
#endif

/* PTX mma.sync.aligned.m16n8k32 */
#if defined(OZAKI_U8) && (OZAKI_U8)
# define NV_MMA(D0,D1,D2,D3, A0,A1,A2,A3, B0,B1, C0,C1,C2,C3) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 " \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
      : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3) \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), \
        "r"(C0), "r"(C1), "r"(C2), "r"(C3))
# define MMA_BYTE_T uchar
# define MMA_BYTE4_T uchar4
#else
# define NV_MMA(D0,D1,D2,D3, A0,A1,A2,A3, B0,B1, C0,C1,C2,C3) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" \
      : "=r"(D0), "=r"(D1), "=r"(D2), "=r"(D3) \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), \
        "r"(C0), "r"(C1), "r"(C2), "r"(C3))
# define MMA_BYTE_T char
# define MMA_BYTE4_T char4
#endif

/* Pack A fragment for thread t from row-major A[M_pad][K_pad].
 * A operand: 16 rows x 32 cols.  Thread t, group g=t/4, lane l=t%4:
 *   a0 = A[g*2  ][ l*4 .. l*4+3 ]  (first 16 K-cols, 4 bytes)
 *   a1 = A[g*2+1][ l*4 .. l*4+3 ]
 *   a2 = A[g*2  ][ 16+l*4 .. 16+l*4+3 ]  (second 16 K-cols)
 *   a3 = A[g*2+1][ 16+l*4 .. 16+l*4+3 ] */
#define MMA_LOAD_A(AS, K_PAD, MI, KOFF, LANE, A0, A1, A2, A3) \
  do { \
    const int g_ = (LANE) >> 2, l_ = (LANE) & 3; \
    CONSTANT const MMA_BYTE_T* ra0_ = (CONSTANT const MMA_BYTE_T*)(AS) + (long)((MI) + g_ * 2) * (K_PAD) + (KOFF) + l_ * 4; \
    CONSTANT const MMA_BYTE_T* ra1_ = ra0_ + (K_PAD); \
    (A0) = as_uint(vload4(0, ra0_)); \
    (A1) = as_uint(vload4(0, ra1_)); \
    (A2) = as_uint(vload4(0, ra0_ + 16)); \
    (A3) = as_uint(vload4(0, ra1_ + 16)); \
  } while (0)

/* Pack B fragment for thread t from row-major B[K_pad][N_pad].
 * B operand (col-major 32x8).  Thread t, group g=t/4, lane l=t%4:
 *   b0 = B[l*4..l*4+3][g]       (4 K-rows from K-half 0, col g)
 *   b1 = B[16+l*4..16+l*4+3][g] (4 K-rows from K-half 1, col g)
 * With row-major storage B[k*N_pad + c]: */
#define MMA_LOAD_B(BS, N_PAD, NJ, KOFF, LANE, B0, B1) \
  do { \
    const int g_ = (LANE) >> 2, l_ = (LANE) & 3; \
    CONSTANT const MMA_BYTE_T* rb0_ = (CONSTANT const MMA_BYTE_T*)(BS) + (long)((KOFF) + l_ * 4) * (N_PAD) + (NJ) + g_; \
    CONSTANT const MMA_BYTE_T* rb1_ = (CONSTANT const MMA_BYTE_T*)(BS) + (long)((KOFF) + 16 + l_ * 4) * (N_PAD) + (NJ) + g_; \
    (B0) = as_uint((MMA_BYTE4_T)(rb0_[0], rb0_[(N_PAD)], rb0_[2*(N_PAD)], rb0_[3*(N_PAD)])); \
    (B1) = as_uint((MMA_BYTE4_T)(rb1_[0], rb1_[(N_PAD)], rb1_[2*(N_PAD)], rb1_[3*(N_PAD)])); \
  } while (0)

/* Preprocessing kernels are shared with ozaki1_int8.cl -- the preprocess_a_dense
 * and preprocess_b_dense kernels produce the same int8 slice layout regardless
 * of the GEMM path (dp4a vs MMA).  This file only provides the GEMM kernel. */

__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
kernel void gemm_fused_nv(
  CONSTANT const char* restrict as_base,
  CONSTANT const char* restrict bs_base,
  CONSTANT const real_t* restrict expa,
  CONSTANT const real_t* restrict expb,
  global real_t* restrict c,
  int M, int N, int K_pad, int N_pad, int ldc, int M_pad,
  real_t alpha, int first_pair)
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int lane = (int)get_local_id(0);  /* 0..31 within warp */
  const int warp_id = (int)get_local_id(1);
  const int tile_m = warp_id / NTN;
  const int tile_n = warp_id % NTN;
  const int mi_base = ib_idx * BM + tile_m * MMA_M * RTM;
  const int nj_base = jb_idx * BN + tile_n * MMA_N * RTN;
  const long a_stride = (long)M_pad * K_pad;
  const long b_stride = (long)K_pad * N_pad;

  /* Column this thread is responsible for after redistribution.
   * Within an MMA_N=8 tile: lane%4 gives column-pair, lane/8 selects even/odd.
   * my_col = (lane%4)*2 + (lane>=16 ? 1 : 0) ... but wait, we have RTN tiles.
   * For RTN=1: my_col = (lane&3)*2 + (lane>>3) ... but lane>>3 gives 0..3
   *   for lane 0..31. That's not right for selecting even/odd within 8 cols.
   *
   * Correct mapping: thread t owns column (t%4)*2 + (t/16)%2 in the first
   * MMA_N-wide tile. But with 32 threads and 8 columns, that's 4 threads per
   * column -- each holding 4 of the 16 rows (a 2x2 block). After redistribution
   * only 8 threads get actual work (one per column), the other 24 are idle.
   *
   * BETTER: don't redistribute per-thread. Instead, in the scaling phase,
   * each thread scales its own 2x2 fragment directly.
   * Thread t holds D[g*2][l*2], D[g*2][l*2+1], D[g*2+1][l*2], D[g*2+1][l*2+1].
   * It can directly multiply by the correct ea[row] and eb[col] values.
   * No shuffle needed! */

  const int frag_g = lane >> 2;       /* 0..7: row group */
  const int frag_l = lane & 3;        /* 0..3: col pair index */
  const int frag_row0 = frag_g * 2;   /* rows: g*2, g*2+1 */
  const int frag_col0 = frag_l * 2;   /* cols: l*2, l*2+1 */

  /* Pre-cache exponent scales for this thread's 2 rows x 2 cols x RTM x RTN. */
  real_t ea_cache[RTM * 2];
  real_t eb_cache[RTN * 2];
  {
    int rm;
    for (rm = 0; rm < RTM; ++rm) {
      int fr;
      for (fr = 0; fr < 2; ++fr) {
        const int r_ = mi_base + rm * MMA_M + frag_row0 + fr;
        ea_cache[rm * 2 + fr] = OZAKI_IN_BOUNDS(r_, M, 0, 1) ? expa[r_] : ZERO;
      }
    }
  }
  {
    int rn;
    for (rn = 0; rn < RTN; ++rn) {
      int fc;
      for (fc = 0; fc < 2; ++fc) {
        const int c_ = nj_base + rn * MMA_N + frag_col0 + fc;
        eb_cache[rn * 2 + fc] = OZAKI_IN_BOUNDS(0, 1, c_, N) ? expb[c_] : ZERO;
      }
    }
  }

  /* Register-resident FP C: 2 rows x 2 cols x RTM x RTN per thread.
   * Layout: [base + fr*2 + fc] where fr=row offset (0,1), fc=col offset (0,1).
   * Actual cols: fc=0 -> frag_l, fc=1 -> frag_l+4. */
  real_t c_fp[RTM * RTN * 4];
  {
    int ci;
    if (0 != first_pair) {
      for (ci = 0; ci < RTM * RTN * 4; ++ci) c_fp[ci] = ZERO;
    }
    else {
      for (ci = 0; ci < RTM * RTN * 4; ++ci) c_fp[ci] = ZERO;
      { int rm, rn;
        for (rm = 0; rm < RTM; ++rm) {
          for (rn = 0; rn < RTN; ++rn) {
            int fr, fc;
            for (fr = 0; fr < 2; ++fr) {
              for (fc = 0; fc < 2; ++fc) {
                const int r_ = mi_base + rm * MMA_M + frag_row0 + fr;
                const int c_ = nj_base + rn * MMA_N + frag_col0 + fc;
                if (OZAKI_IN_BOUNDS(r_, M, c_, N)) {
                  c_fp[(rm * RTN + rn) * 4 + fr * 2 + fc] = c[(long)c_ * ldc + r_];
                }
              }
            }
          }
        }
      }
    }
  }

  /* Slice-pair iteration */
  { SINT sa;
    for (sa = 0; sa < (SINT)NSLICES; ++sa) {
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

        /* MMA accumulators: 4 int32 per sub-tile */
        int acc[RTM * RTN * 4];
        { int ai;
          for (ai = 0; ai < RTM * RTN * 4; ++ai) acc[ai] = 0;
        }

        /* K-loop: MMA m16n8k32, one step = 32 K-elements */
        { int k;
          for (k = 0; k < K_pad; k += BK) {
            int rm, rn;
            for (rm = 0; rm < RTM; ++rm) {
              uint a0, a1, a2, a3;
              MMA_LOAD_A(as_sa, K_pad, mi_base + rm * MMA_M, k, lane, a0, a1, a2, a3);
              for (rn = 0; rn < RTN; ++rn) {
                uint bb0, bb1;
                const int base = (rm * RTN + rn) * 4;
                MMA_LOAD_B(bs_sb, N_pad, nj_base + rn * MMA_N, k, lane, bb0, bb1);
                { int d0, d1, d2, d3;
                  NV_MMA(d0, d1, d2, d3, a0, a1, a2, a3, bb0, bb1,
                         acc[base], acc[base+1], acc[base+2], acc[base+3]);
                  acc[base] = d0; acc[base+1] = d1;
                  acc[base+2] = d2; acc[base+3] = d3;
                }
              }
            }
          }
        }

        /* Mirror pair (sa,sb) -> (sb,sa) for triangular iteration */
        if (0 == OZAKI_SQ && sa != sb) {
          int mir[RTM * RTN * 4];
          { int ai;
            for (ai = 0; ai < RTM * RTN * 4; ++ai) mir[ai] = 0;
          }
          { int k;
            for (k = 0; k < K_pad; k += BK) {
              int rm, rn;
              for (rm = 0; rm < RTM; ++rm) {
                uint a0, a1, a2, a3;
                MMA_LOAD_A(as_sb, K_pad, mi_base + rm * MMA_M, k, lane, a0, a1, a2, a3);
                for (rn = 0; rn < RTN; ++rn) {
                  uint bb0, bb1;
                  const int base = (rm * RTN + rn) * 4;
                  MMA_LOAD_B(bs_sa, N_pad, nj_base + rn * MMA_N, k, lane, bb0, bb1);
                  { int d0, d1, d2, d3;
                    NV_MMA(d0, d1, d2, d3, a0, a1, a2, a3, bb0, bb1,
                           mir[base], mir[base+1], mir[base+2], mir[base+3]);
                    mir[base] = d0; mir[base+1] = d1;
                    mir[base+2] = d2; mir[base+3] = d3;
                  }
                }
              }
            }
          }
          { int ai;
            for (ai = 0; ai < RTM * RTN * 4; ++ai) acc[ai] += mir[ai];
          }
        }

        /* Scale and accumulate: each thread handles its 2x2 fragment directly.
         * acc[base+0]=D[row0][col0], acc[base+1]=D[row0][col0+1],
         * acc[base+2]=D[row1][col0], acc[base+3]=D[row1][col0+1]. */
        { int rm, rn;
          for (rm = 0; rm < RTM; ++rm) {
            for (rn = 0; rn < RTN; ++rn) {
              const int base = (rm * RTN + rn) * 4;
              int fr, fc;
              for (fr = 0; fr < 2; ++fr) {
                const int r_ = mi_base + rm * MMA_M + frag_row0 + fr;
                const real_t ea = ea_cache[rm * 2 + fr];
                for (fc = 0; fc < 2; ++fc) {
                  const int c_ = nj_base + rn * MMA_N + frag_col0 + fc;
                  if (OZAKI_IN_BOUNDS(r_, M, c_, N)) {
                    const real_t sc = pair_scale * ea * eb_cache[rn * 2 + fc];
                    c_fp[base + fr * 2 + fc] += (real_t)acc[base + fr * 2 + fc] * sc;
                  }
                }
              }
            }
          }
        }
      } /* sb */
    } /* sa */
  }

  /* Final write: register C -> global C */
  { int rm, rn;
    for (rm = 0; rm < RTM; ++rm) {
      for (rn = 0; rn < RTN; ++rn) {
        const int base = (rm * RTN + rn) * 4;
        int fr, fc;
        for (fr = 0; fr < 2; ++fr) {
          for (fc = 0; fc < 2; ++fc) {
            const int r_ = mi_base + rm * MMA_M + frag_row0 + fr;
            const int c_ = nj_base + rn * MMA_N + frag_col0 + fc;
            if (OZAKI_IN_BOUNDS(r_, M, c_, N)) {
              c[(long)c_ * ldc + r_] = c_fp[base + fr * 2 + fc];
            }
          }
        }
      }
    }
  }
}
