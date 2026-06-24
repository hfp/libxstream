/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_common.cl"

#if !defined(WG_X)
# define WG_X 32
#endif
#if !defined(WG_Y)
# define WG_Y 8
#endif
#define WG_SIZE (WG_X * WG_Y)
#define SLM_X (WG_X + 2 * RADIUS)

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
kernel void stencil_apply_direct(
  global const float* restrict p_grid,
  global const float* restrict p_old,
  global float* restrict p_new,
  global const float* restrict vel,
  global const float* restrict coeff,
  int nterms, float dt2,
  int nx, int ny, int nz)
{
  const int ix = (int)get_global_id(0);
  const int iy = (int)get_global_id(1);
  const int iz = (int)get_group_id(2);
  const int lx = (int)get_local_id(0);
  const int ly = (int)get_local_id(1);
  const int lid = ly * WG_X + lx;

  local float x_slm[WG_Y * SLM_X];

  const long slice_base = (long)iz * ny * nx;
  const long row_base = slice_base + (long)iy * nx;
  float lap = 0.0f;
  int r;

  { const int total = WG_Y * SLM_X;
    int idx;
    for (idx = lid; idx < total; idx += WG_SIZE) {
      const int sy = idx / SLM_X;
      const int sx = idx % SLM_X;
      const int gy = (int)get_group_id(1) * WG_Y + sy;
      int gx = (int)get_group_id(0) * WG_X + sx - RADIUS;
      if (gx < 0) gx = 0; else if (gx >= nx) gx = nx - 1;
      if (gy < ny) {
        x_slm[idx] = p_grid[slice_base + (long)gy * nx + gx];
      }
      else {
        x_slm[idx] = p_grid[slice_base + (long)(ny - 1) * nx + gx];
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (ix >= nx || iy >= ny || iz >= nz) return;

  UNROLL_FORCE(STENCIL_WIDTH) for (r = -RADIUS; r <= RADIUS; ++r) {
    lap += coeff[r + RADIUS] * x_slm[ly * SLM_X + lx + RADIUS + r];
  }

  if (1 < nterms) {
    UNROLL_FORCE(STENCIL_WIDTH) for (r = -RADIUS; r <= RADIUS; ++r) {
      int cy = iy + r;
      if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
      lap += coeff[r + RADIUS] * p_grid[slice_base + (long)cy * nx + ix];
    }
  }

  if (2 < nterms) {
    UNROLL_FORCE(STENCIL_WIDTH) for (r = -RADIUS; r <= RADIUS; ++r) {
      int cz = iz + r;
      if (cz < 0) cz = 0; else if (cz >= nz) cz = nz - 1;
      lap += coeff[r + RADIUS] * p_grid[(long)cz * ny * nx + (long)iy * nx + ix];
    }
  }

  { const long i = row_base + ix;
    p_new[i] = 2.0f * p_grid[i] - p_old[i] + dt2 * vel[i] * lap;
  }
}
