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
 * D is BLK x BLK (banded), stored as BLK rows x K_PAD bf16.
 * K_PAD >= BLK and must be >= 32 bytes (16 bf16) for 2D block I/O.
 * For BLK=32: K_PAD=32, surface width = 64 bytes -- meets minimum. */
#define K_PAD ((BLK < 16) ? 16 : BLK)

/* Padded N dimension for X surface (must be >= 32 bf16 = 64 bytes). */
#define N_PAD ((N_TOTAL < 32) ? 32 : N_TOTAL)

/* Super-block dimension including halo for one time step. */
#define SUPER_BLK (BLK + 2 * RADIUS)

/* Gather coordinate: maps (dim, block-origin, k-index, local i/j) to grid (gx, gy, gz). */
#define STENCIL_GATHER_COORD(DIM, OX, OY, OZ, K, CI, CJ, GX, GY, GZ) \
  do { \
    if (0 == (DIM))      { (GX) = (OX) + (K) - RADIUS; (GY) = (OY) + (CI); (GZ) = (OZ) + (CJ); } \
    else if (1 == (DIM)) { (GX) = (OX) + (CI); (GY) = (OY) + (K) - RADIUS; (GZ) = (OZ) + (CJ); } \
    else                 { (GX) = (OX) + (CI); (GY) = (OY) + (CJ); (GZ) = (OZ) + (K) - RADIUS; } \
  } while (0)

#define STENCIL_CLAMP_COORD(GX, GY, GZ, NX, NY, NZ) \
  do { \
    if ((GX) < 0) (GX) = 0; else if ((GX) >= (NX)) (GX) = (NX) - 1; \
    if ((GY) < 0) (GY) = 0; else if ((GY) >= (NY)) (GY) = (NY) - 1; \
    if ((GZ) < 0) (GZ) = 0; else if ((GZ) >= (NZ)) (GZ) = (NZ) - 1; \
  } while (0)

#define STENCIL_GRID_IDX(GZ, GY, GX, NY, NX) \
  ((long)(GZ) * (NY) * (NX) + (long)(GY) * (NX) + (GX))

/* DPAS accumulation from SLM strip: iterates (sa, sb, kstep). */
#define STENCIL_DPAS_ACC(DK, NDIGITS_EFF, X_SLM, D_WB, MI, ACC) \
  do { \
    int sa_, sb_, ks_; \
    for (sa_ = 0; sa_ < NDIGITS_A; ++sa_) { \
      global const ushort* d_digit_ = (DK) + (long)sa_ * BLK * K_PAD; \
      for (sb_ = 0; sb_ < (NDIGITS_EFF); ++sb_) { \
        local const ushort* x_digit_; \
        STENCIL_TRIM_CHECK(sa_, sb_, NDIGITS_EFF); \
        x_digit_ = (X_SLM) + sb_ * K_PAD * XMX_N; \
        for (ks_ = 0; ks_ < K_PAD; ks_ += 16) { \
          ushort8 a_bf_; \
          uint8 b_bf_; \
          BF16_LOAD_A(d_digit_, (D_WB), BLK, (MI), ks_, &a_bf_); \
          b_bf_ = *(local const uint8*)(x_digit_ + ks_ * XMX_N); \
          BF16_DPAS_ONE(a_bf_, b_bf_, (ACC)); \
        } \
      } \
    } \
  } while (0)

#if defined(TRIM) && (0 < TRIM)
# define STENCIL_TRIM_CHECK(SA, SB, ND) \
    if ((SA) + (SB) >= NDIGITS_A + (ND) - 1 - (TRIM - 1)) continue
#else
# define STENCIL_TRIM_CHECK(SA, SB, ND) ((void)0)
#endif

#endif /*STENCIL_COMMON_CL*/
