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

#endif /*STENCIL_COMMON_CL*/
