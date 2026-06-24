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
#define SLM_Y (WG_Y + 2 * RADIUS)
#define SLM_TOTAL (SLM_Y * SLM_X)
#define Z_WINDOW (2 * RADIUS + 1)

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
  const int iz_base = (int)get_group_id(2) * BLK;
  const int lx = (int)get_local_id(0);
  const int ly = (int)get_local_id(1);
  const int lid = ly * WG_X + lx;
  const int valid_xy = (ix < nx && iy < ny);

  local float xy_slm[SLM_TOTAL];

  const int gx0 = (int)get_group_id(0) * WG_X - RADIUS;
  const int gy0 = (int)get_group_id(1) * WG_Y - RADIUS;
  const long xy_off = (long)iy * nx + ix;
  float z_win[Z_WINDOW];
  int iz, r, idx, w;

  if (valid_xy) {
    UNROLL_FORCE(Z_WINDOW) for (w = 0; w < Z_WINDOW - 1; ++w) {
      int cz = iz_base - RADIUS + w;
      if (cz < 0) cz = 0; else if (cz >= nz) cz = nz - 1;
      z_win[w] = p_grid[(long)cz * ny * nx + xy_off];
    }
  }

  UNROLL_OUTER(1) for (iz = iz_base; iz < iz_base + BLK && iz < nz; ++iz) {
    const long slice_base = (long)iz * ny * nx;
    float lap, p_center;

    for (idx = lid; idx < SLM_TOTAL; idx += WG_SIZE) {
      const int sy = idx / SLM_X;
      const int sx = idx % SLM_X;
      int gx = gx0 + sx;
      int gy = gy0 + sy;
      if (gx < 0) gx = 0; else if (gx >= nx) gx = nx - 1;
      if (gy < 0) gy = 0; else if (gy >= ny) gy = ny - 1;
      xy_slm[idx] = p_grid[slice_base + (long)gy * nx + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid_xy) {
      { int cz = iz + RADIUS;
        if (cz >= nz) cz = nz - 1;
        z_win[Z_WINDOW - 1] = p_grid[(long)cz * ny * nx + xy_off];
      }

      { const int c = (ly + RADIUS) * SLM_X + lx + RADIUS;
        p_center = xy_slm[c];
        lap = coeff[RADIUS] * p_center;
        UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
          lap += coeff[RADIUS + r] * (xy_slm[c + r] + xy_slm[c - r]);
        }
        if (1 < nterms) {
          lap += coeff[RADIUS] * p_center;
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += coeff[RADIUS + r] * (xy_slm[c + r * SLM_X] + xy_slm[c - r * SLM_X]);
          }
        }
      }

      if (2 < nterms) {
        lap += coeff[RADIUS] * z_win[RADIUS];
        UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
          lap += coeff[RADIUS + r] * (z_win[RADIUS + r] + z_win[RADIUS - r]);
        }
      }

      { const long i = slice_base + xy_off;
        p_new[i] = 2.0f * p_center - p_old[i] + dt2 * vel[i] * lap;
      }

      UNROLL_FORCE(Z_WINDOW - 1) for (w = 0; w < Z_WINDOW - 1; ++w) {
        z_win[w] = z_win[w + 1];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
