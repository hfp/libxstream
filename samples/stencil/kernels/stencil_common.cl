/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_COMMON_CL
#define STENCIL_COMMON_CL

#include "../../../libxstream/opencl/libxstream_ozaki.h"

/* Block dimension (cube side length). */
#if !defined(BLK)
# define BLK 32
#endif

/* FD half-order: full stencil width is 2*RADIUS+1. */
#if !defined(RADIUS)
# define RADIUS 4
#endif

#define STENCIL_WIDTH (2 * RADIUS + 1)

/* Cascade parameters: K sub-steps each with R_PER_STEP radius. */
#if !defined(K_STEPS)
# define K_STEPS 1
#endif
#if !defined(R_PER_STEP)
# define R_PER_STEP RADIUS
#endif
#if !defined(METHOD)
# define METHOD 0
#endif
#if !defined(TRIM)
# define TRIM 0
#endif
#if !defined(NTERMS)
# define NTERMS 3
#endif

/* Number of Dekker BF16 digits for operator (A) and wavefield (X). */
#if !defined(NDIGITS_A)
# define NDIGITS_A 2
#endif
#if !defined(NDIGITS_X)
# define NDIGITS_X 3
#endif

/* Sub-group size. */
#if !defined(SG)
# define SG 16
#endif

/* N-strip width: DPAS produces 8x16 tiles, we tile N=BLK*BLK
 * in strips of XMX_N=16 columns. */
#define XMX_M 8
#define XMX_N 16
#define N_TOTAL (BLK * BLK)
#define N_STRIPS (N_TOTAL / XMX_N)
#define M_TILES (BLK / XMX_M)

/* Padded K dimension for the operator surface.
 * D is BLK rows over a haloed K dimension. K_PAD is rounded to a
 * multiple of 16 because each DPAS step consumes 16 K entries. */
#define STENCIL_ALIGN16(VALUE) (((VALUE) + 15) & ~15)
#define K_BASE (BLK + 2 * RADIUS)
#define K_PAD STENCIL_ALIGN16(K_BASE)
#define WG_M_TILES M_TILES

/* Padded N dimension for X surface (must be >= 32 bf16 = 64 bytes). */
#define N_PAD ((N_TOTAL < 32) ? 32 : N_TOTAL)

/* Super-block dimension including halo for one time step. */
#define SUPER_BLK (BLK + 2 * RADIUS)

/* N-strips batched per work-group (1 = no batching). */
#if !defined(STRIPS_PER_WG)
# define STRIPS_PER_WG 2
#endif
#define N_STRIP_GROUPS (N_STRIPS / STRIPS_PER_WG)
#if (32 == N_STRIP_GROUPS)
# define STENCIL_NSTRIP_SHIFT 5
#elif (16 == N_STRIP_GROUPS)
# define STENCIL_NSTRIP_SHIFT 4
#elif (64 == N_STRIP_GROUPS)
# define STENCIL_NSTRIP_SHIFT 6
#endif

/* BF16 split mode (STENCIL_BF16=2): float D*X without Dekker or DPAS. */
#if defined(STENCIL_BF16) && (2 <= STENCIL_BF16)
# define STENCIL_D_ELEM float
# define STENCIL_X_ELEM float
# define STENCIL_X_SLM_COUNT (K_PAD * XMX_N)
# define STENCIL_D_BAND STENCIL_WIDTH
#else
# define STENCIL_D_ELEM ushort
# define STENCIL_X_ELEM ushort
# define STENCIL_X_SLM_COUNT (NDIGITS_X * K_PAD * XMX_N)
#endif

#if defined(STENCIL_BF16S) && (0 < STENCIL_BF16S)
# define STENCIL_P_ELEM ushort
# if !defined(STENCIL_P_N)
#   define STENCIL_P_N (STENCIL_NX * STENCIL_NY * STENCIL_NZ)
# endif
# define STENCIL_BF16S_NDIGITS 2
# define STENCIL_LOAD_P(PTR, IDX) \
    (BF16_TO_F32(((global const ushort*)(PTR))[(IDX)]) \
   + BF16_TO_F32(((global const ushort*)(PTR))[(IDX) + STENCIL_P_N]))
# define STENCIL_LOAD_P_BITS(PTR, IDX) as_uint(STENCIL_LOAD_P(PTR, IDX))
# define STENCIL_STORE_P(PTR, IDX, VALUE) \
    do { \
      global ushort* ptr_sp_ = (global ushort*)(PTR); \
      const float val_sp_ = (VALUE); \
      const ushort hi_sp_ = ROUND_TO_BF16(val_sp_); \
      ptr_sp_[(IDX)] = hi_sp_; \
      ptr_sp_[(IDX) + STENCIL_P_N] = ROUND_TO_BF16(val_sp_ - BF16_TO_F32(hi_sp_)); \
    } while (0)
#else
# define STENCIL_P_ELEM float
# define STENCIL_LOAD_P(PTR, IDX) ((PTR)[(IDX)])
# define STENCIL_LOAD_P_BITS(PTR, IDX) as_uint((PTR)[(IDX)])
# define STENCIL_STORE_P(PTR, IDX, VALUE) ((PTR)[(IDX)] = (VALUE))
#endif

#if defined(STENCIL_BF16S) && (0 < STENCIL_BF16S)
# define STENCIL_TTI_X_NDIGITS STENCIL_BF16S_NDIGITS
#else
# define STENCIL_TTI_X_NDIGITS NDIGITS_X
#endif

/* Gather coordinate: maps (dim, block-origin, k-index, local i/j) to grid (gx, gy, gz). */
#define STENCIL_GATHER_COORD(DIM, OX, OY, OZ, K, CI, CJ, GX, GY, GZ) \
  do { \
    if (0 == (DIM))      { (GX) = (OX) + (K) - RADIUS; (GY) = (OY) + (CI); (GZ) = (OZ) + (CJ); } \
    else if (1 == (DIM)) { (GX) = (OX) + (CI); (GY) = (OY) + (K) - RADIUS; (GZ) = (OZ) + (CJ); } \
    else                 { (GX) = (OX) + (CI); (GY) = (OY) + (CJ); (GZ) = (OZ) + (K) - RADIUS; } \
  } while (0)

#if defined(STENCIL_PADDED) && (0 < STENCIL_PADDED)
# define STENCIL_CLAMP_COORD(GX, GY, GZ, NX, NY, NZ) ((void)0)
#else
# define STENCIL_CLAMP_COORD(GX, GY, GZ, NX, NY, NZ) \
  do { \
    if ((GX) < 0) (GX) = 0; else if ((GX) >= STENCIL_NX) (GX) = STENCIL_NX - 1; \
    if ((GY) < 0) (GY) = 0; else if ((GY) >= STENCIL_NY) (GY) = STENCIL_NY - 1; \
    if ((GZ) < 0) (GZ) = 0; else if ((GZ) >= STENCIL_NZ) (GZ) = STENCIL_NZ - 1; \
  } while (0)
#endif

/* Layout identifiers (passed as -DSTENCIL_LAYOUT=N). */
#if !defined(STENCIL_LAYOUT)
# define STENCIL_LAYOUT 0
#endif
#define STENCIL_LAYOUT_XYZ 0
#define STENCIL_LAYOUT_BLK 1
#define STENCIL_LAYOUT_ZYX 2

/* Compile-time grid dimensions (passed via -DSTENCIL_NX/NY/NZ). */
#if !defined(STENCIL_NX)
# define STENCIL_NX 256
#endif
#if !defined(STENCIL_NY)
# define STENCIL_NY 256
#endif
#if !defined(STENCIL_NZ)
# define STENCIL_NZ 256
#endif

/* X-innermost: [gz][gy][gx], stride-x=1. */
#define STENCIL_GRID_IDX(GZ, GY, GX, NY, NX) \
  ((long)(GZ) * STENCIL_NY * STENCIL_NX + (long)(GY) * STENCIL_NX + (GX))

/* Z-innermost with per-array halo: [gx+lx][gy+ly][gz+lz], stride-z=1.
 * Per-array strides (P=wavefield, V=velocity, E=eta) passed via -D flags. */
#if !defined(STENCIL_P_SX)
# define STENCIL_P_SX 1
#endif
#if !defined(STENCIL_P_SY)
# define STENCIL_P_SY 1
#endif
#if !defined(STENCIL_P_LX)
# define STENCIL_P_LX 0
#endif
#if !defined(STENCIL_P_LY)
# define STENCIL_P_LY 0
#endif
#if !defined(STENCIL_P_LZ)
# define STENCIL_P_LZ 0
#endif
#if !defined(STENCIL_V_SX)
# define STENCIL_V_SX STENCIL_P_SX
#endif
#if !defined(STENCIL_V_SY)
# define STENCIL_V_SY STENCIL_P_SY
#endif
#if !defined(STENCIL_V_LX)
# define STENCIL_V_LX STENCIL_P_LX
#endif
#if !defined(STENCIL_V_LY)
# define STENCIL_V_LY STENCIL_P_LY
#endif
#if !defined(STENCIL_V_LZ)
# define STENCIL_V_LZ STENCIL_P_LZ
#endif
#if !defined(STENCIL_E_SX)
# define STENCIL_E_SX STENCIL_P_SX
#endif
#if !defined(STENCIL_E_SY)
# define STENCIL_E_SY STENCIL_P_SY
#endif
#if !defined(STENCIL_E_LX)
# define STENCIL_E_LX STENCIL_P_LX
#endif
#if !defined(STENCIL_E_LY)
# define STENCIL_E_LY STENCIL_P_LY
#endif
#if !defined(STENCIL_E_LZ)
# define STENCIL_E_LZ STENCIL_P_LZ
#endif

#define STENCIL_ZYX_P_IDX(GZ, GY, GX) \
  ((long)((GX) + STENCIL_P_LX) * STENCIL_P_SX \
   + (long)((GY) + STENCIL_P_LY) * STENCIL_P_SY \
   + ((GZ) + STENCIL_P_LZ))
#define STENCIL_ZYX_V_IDX(GZ, GY, GX) \
  ((long)((GX) + STENCIL_V_LX) * STENCIL_V_SX \
   + (long)((GY) + STENCIL_V_LY) * STENCIL_V_SY \
   + ((GZ) + STENCIL_V_LZ))
#define STENCIL_ZYX_E_IDX(GZ, GY, GX) \
  ((long)((GX) + STENCIL_E_LX) * STENCIL_E_SX \
   + (long)((GY) + STENCIL_E_LY) * STENCIL_E_SY \
   + ((GZ) + STENCIL_E_LZ))

/* Blocked (tiled) grid index: data stored as [bz][by][bx][lz][ly][lx].
 * BLK must be a power of 2. All dims require nbx, nby as runtime params. */
#define K_PAD_I8 64

#define STENCIL_BLK_SHIFT 5
#define STENCIL_BLK_MASK 31
#define STENCIL_BLOCKED_IDX(GX, GY, GZ, NBX, NBY) \
  (((long)((GZ) >> STENCIL_BLK_SHIFT) * (NBY) * (NBX) \
    + (long)((GY) >> STENCIL_BLK_SHIFT) * (NBX) \
    + ((GX) >> STENCIL_BLK_SHIFT)) * (BLK * BLK * BLK) \
   + (long)((GZ) & STENCIL_BLK_MASK) * (BLK * BLK) \
   + (long)((GY) & STENCIL_BLK_MASK) * BLK \
   + ((GX) & STENCIL_BLK_MASK))

#if (STENCIL_LAYOUT_BLK == STENCIL_LAYOUT)
# define STENCIL_P_IDX(GZ, GY, GX, NY, NX, NBX, NBY) \
    STENCIL_BLOCKED_IDX(GX, GY, GZ, NBX, NBY)
# define STENCIL_V_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_BLOCKED_IDX(GX, GY, GZ, NBX, NBY)
# define STENCIL_E_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_BLOCKED_IDX(GX, GY, GZ, NBX, NBY)
#elif (STENCIL_LAYOUT_ZYX == STENCIL_LAYOUT)
# define STENCIL_P_IDX(GZ, GY, GX, NY, NX, NBX, NBY) \
    STENCIL_ZYX_P_IDX(GZ, GY, GX)
# define STENCIL_V_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_ZYX_V_IDX(GZ, GY, GX)
# define STENCIL_E_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_ZYX_E_IDX(GZ, GY, GX)
#else
# define STENCIL_P_IDX(GZ, GY, GX, NY, NX, NBX, NBY) \
    STENCIL_GRID_IDX(GZ, GY, GX, NY, NX)
# define STENCIL_V_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_GRID_IDX(GZ, GY, GX, NY, NX)
# define STENCIL_E_IDX(GZ, GY, GX, NY, NX) \
    STENCIL_GRID_IDX(GZ, GY, GX, NY, NX)
#endif

/* Dim iteration order: gather the memory-sequential axis first.
 * XYZ/BLK: dim 0 (X) is fastest → order {0,1,2}.
 * ZYX: dim 2 (Z) is fastest → order {2,1,0}. */
#if (STENCIL_LAYOUT_ZYX == STENCIL_LAYOUT)
# define STENCIL_DIM(ITER) (NTERMS - 1 - (ITER))
#else
# define STENCIL_DIM(ITER) (ITER)
#endif

/* Kstep range for a given MI based on operator non-zero band [MI, MI + 7 + 2*RADIUS]. */
#define KSTEP_LO(MI) ((MI) & ~15)
#define KSTEP_HI(MI) (((MI) + XMX_M - 1 + 2 * RADIUS) & ~15)
#define KSTEP_MAX_COUNT (((15 + XMX_M - 1 + 2 * RADIUS) >> 4) + 1)
#define STENCIL_BF16_PACK_B(X_DIGIT, KS, BK, LID) \
  ((uint)(X_DIGIT)[((KS) + 2 * (BK)) * XMX_N + (LID)] \
   | ((uint)(X_DIGIT)[((KS) + 2 * (BK) + 1) * XMX_N + (LID)] << 16))

/* INT8 K-step range: k32-aligned (8 int-quads per DPAS step).
 * Non-zero band of operator row MI: columns [MI .. MI+XMX_M-1+2*RADIUS].
 * DPAS k32 step starts at a multiple of 8 int-quads (32 bytes). */
#define KSTEP_I8_LO(MI) (((MI) >> 2) & ~7)
#define KSTEP_I8_HI(MI) ((((MI) + XMX_M - 1 + 2 * RADIUS) >> 2) & ~7)
#define KSTEP_I8_MAX_COUNT ((((XMX_M - 1 + 2 * RADIUS + 31) >> 5) + 1))

/* DPAS accumulation from SLM strip: iterates (sa, sb, kstep).
 * B tiles are preloaded from SLM into registers before DPAS. */
#define STENCIL_DPAS_ACC_ROWS(DK, NDIGITS_EFF, X_SLM, D_WB, A_ROWS, MI, ACC) \
  do { \
    const int ks_lo_ = KSTEP_LO(MI); \
    const int ks_hi_ = KSTEP_HI(MI); \
    const int b_col_ = (int)SGLID(); \
    int sa_, sb_, ks_; \
    UNROLL_FORCE(NDIGITS_A) for (sa_ = 0; sa_ < NDIGITS_A; ++sa_) { \
      global const ushort* d_digit_ = (DK) + (long)sa_ * (A_ROWS) * K_PAD; \
      UNROLL_AUTO for (sb_ = 0; sb_ < (NDIGITS_EFF); ++sb_) { \
        local const ushort* x_digit_; \
        STENCIL_TRIM_CHECK(sa_, sb_, NDIGITS_EFF); \
        x_digit_ = (X_SLM) + sb_ * K_PAD * XMX_N; \
        UNROLL_FORCE(KSTEP_MAX_COUNT) \
        for (ks_ = ks_lo_; ks_ <= ks_hi_; ks_ += 16) { \
          ushort8 a_bf_; \
          uint8 b_bf_; \
          int bk_; \
          UNROLL_FORCE(8) \
          for (bk_ = 0; bk_ < 8; ++bk_) { \
            ((private uint*)&b_bf_)[bk_] = STENCIL_BF16_PACK_B(x_digit_, ks_, bk_, b_col_); \
          } \
          BF16_LOAD_A(d_digit_, (D_WB), (A_ROWS), (MI), ks_, &a_bf_); \
          BF16_DPAS_ONE(a_bf_, b_bf_, (ACC)); \
        } \
      } \
    } \
  } while (0)

#define STENCIL_DPAS_ACC(DK, NDIGITS_EFF, X_SLM, D_WB, MI, ACC) \
  STENCIL_DPAS_ACC_ROWS(DK, NDIGITS_EFF, X_SLM, D_WB, BLK, MI, ACC)

/* FP32 accumulation: D[row,col] * X[col,sg_lid] via banded FMA.
 * D is stored as float[BLK][STENCIL_D_BAND], X in SLM as float[K_PAD][XMX_N].
 * Each WI accumulates 8 rows (MI..MI+7), reading one X column (sg_lid). */
#if defined(STENCIL_BF16) && (2 <= STENCIL_BF16)
#define STENCIL_FP32_ACC(DK, X_SLM, MI, ACC) \
  do { \
    const int sg_lid_fp32_ = (int)SGLID(); \
    if (sg_lid_fp32_ < XMX_N) { \
      union { float8 v; float a[8]; } u_fp32_; \
      int m_fp32_, r_fp32_; \
      u_fp32_.v = (ACC); \
      UNROLL_FORCE(XMX_M) for (m_fp32_ = 0; m_fp32_ < XMX_M; ++m_fp32_) { \
        const int row_ = (MI) + m_fp32_; \
        global const float* d_row_ = (DK) + (long)row_ * STENCIL_D_BAND; \
        UNROLL_FORCE(STENCIL_D_BAND) \
        for (r_fp32_ = 0; r_fp32_ < STENCIL_D_BAND; ++r_fp32_) { \
          const int col_ = row_ + r_fp32_; \
          u_fp32_.a[m_fp32_] += d_row_[r_fp32_] \
            * (X_SLM)[col_ * XMX_N + sg_lid_fp32_]; \
        } \
      } \
      (ACC) = u_fp32_.v; \
    } \
  } while (0)
#endif

/* Gather-store macros: write fetched value into SLM (type-switched). */
#if defined(STENCIL_BF16) && (2 <= STENCIL_BF16)
# define STENCIL_GATHER_STORE(SLM, K, COL, VAL) \
    (SLM)[(K) * XMX_N + (COL)] = (VAL)
# define STENCIL_GATHER_STORE_ZERO(SLM, K, COL) \
    (SLM)[(K) * XMX_N + (COL)] = 0.0f
#elif defined(STENCIL_BF16S) && (0 < STENCIL_BF16S)
# define STENCIL_GATHER_STORE_BF16S(SLM, K, COL, PTR, IDX) \
  do { \
    (SLM)[(K) * XMX_N + (COL)] = ((global const ushort*)(PTR))[(IDX)]; \
    (SLM)[K_PAD * XMX_N + (K) * XMX_N + (COL)] = \
    ((global const ushort*)(PTR))[(IDX) + STENCIL_P_N]; \
  } while (0)
# define STENCIL_GATHER_STORE(SLM, K, COL, VAL) \
  (SLM)[(K) * XMX_N + (COL)] = ROUND_TO_BF16(VAL)
# define STENCIL_GATHER_STORE_ZERO(SLM, K, COL) \
  do { \
    (SLM)[(K) * XMX_N + (COL)] = (ushort)0; \
    (SLM)[K_PAD * XMX_N + (K) * XMX_N + (COL)] = (ushort)0; \
  } while (0)
#else
# define STENCIL_GATHER_STORE(SLM, K, COL, VAL) \
    do { \
      float residual_gs_ = (VAL); \
      int s_gs_; \
      UNROLL_FORCE(NDIGITS_X) for (s_gs_ = 0; s_gs_ < NDIGITS_X; ++s_gs_) { \
        const ushort bf_gs_ = ROUND_TO_BF16(residual_gs_); \
        (SLM)[s_gs_ * K_PAD * XMX_N + (K) * XMX_N + (COL)] = bf_gs_; \
        residual_gs_ -= BF16_TO_F32(bf_gs_); \
      } \
    } while (0)
# define STENCIL_GATHER_STORE_ZERO(SLM, K, COL) \
    do { \
      int s_gs_; \
      UNROLL_FORCE(NDIGITS_X) for (s_gs_ = 0; s_gs_ < NDIGITS_X; ++s_gs_) { \
        (SLM)[s_gs_ * K_PAD * XMX_N + (K) * XMX_N + (COL)] = (ushort)0; \
      } \
    } while (0)
#endif

/* Re-split an FP32 intermediate into BF16 digits for a following DPAS stage. */
#define STENCIL_SPLIT_F32_TO_SLM(SLM, NDIGITS, ROW, COL, VALUE) \
  do { \
    float residual_ = (VALUE); \
    int digit_; \
    UNROLL_FORCE(NDIGITS) for (digit_ = 0; digit_ < (NDIGITS); ++digit_) { \
      ushort bf_ = ROUND_TO_BF16(residual_); \
      (SLM)[digit_ * K_PAD * XMX_N + (ROW) * XMX_N + (COL)] = bf_; \
      if (digit_ + 1 < (NDIGITS)) residual_ -= BF16_TO_F32(bf_); \
    } \
  } while (0)

#if defined(NDIGITS_A) && (1 == NDIGITS_A)
# define STENCIL_SPLIT_F32_TO_SLM_A(SLM, ROW, COL, VALUE) \
    ((SLM)[(ROW) * XMX_N + (COL)] = ROUND_TO_BF16(VALUE))
#else
# define STENCIL_SPLIT_F32_TO_SLM_A(SLM, ROW, COL, VALUE) \
    STENCIL_SPLIT_F32_TO_SLM(SLM, NDIGITS_A, ROW, COL, VALUE)
#endif

#if defined(TRIM) && (0 < TRIM)
# define STENCIL_TRIM_CHECK(SA, SB, ND) \
    if ((SA) + (SB) >= NDIGITS_A + (ND) - 1 - (TRIM - 1)) continue
#else
# define STENCIL_TRIM_CHECK(SA, SB, ND) ((void)0)
#endif

#endif /*STENCIL_COMMON_CL*/
