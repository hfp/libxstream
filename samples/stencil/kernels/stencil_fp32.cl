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
  CONSTANT float* restrict coeff,
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  global const float* restrict eta,
  global float* restrict phi,
  float hdx_2, float hdy_2, float hdz_2,
#endif
  float dt2,
  int nx, int ny, int nz)
{
  local float xy_slm[SLM_TOTAL];
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  local float eta_slm[SLM_TOTAL];
  float eta_z[3];
#endif

  const int ix = (int)get_global_id(0);
  const int iy = (int)get_global_id(1);
  const int iz_base = (int)get_group_id(2) * BLK;
  const int lx = (int)get_local_id(0);
  const int ly = (int)get_local_id(1);
  const int lid = ly * WG_X + lx;
  const int valid_xy = (ix < nx && iy < ny);

  const int gx0 = (int)get_group_id(0) * WG_X - RADIUS;
  const int gy0 = (int)get_group_id(1) * WG_Y - RADIUS;
  const long xy_off = (long)iy * nx + ix;
  float z_win[Z_WINDOW];
  int iz, r, idx, w;

#if !defined(NTERMS) || (2 < NTERMS)
  if (valid_xy) {
    UNROLL_FORCE(Z_WINDOW) for (w = 0; w < Z_WINDOW - 1; ++w) {
      int cz = iz_base - RADIUS + w;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
      if (cz < 0) cz = 0; else if (cz >= nz) cz = nz - 1;
#endif
      z_win[w] = p_grid[(long)cz * ny * nx + xy_off];
    }
  }
#endif

#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  { const int pml_w = 20;
    const int blk_x0 = (int)get_group_id(0) * WG_X;
    const int blk_y0 = (int)get_group_id(1) * WG_Y;
    const int blk_interior =
      (blk_x0 >= pml_w && blk_x0 + WG_X <= nx - pml_w &&
       blk_y0 >= pml_w && blk_y0 + WG_Y <= ny - pml_w &&
       iz_base >= pml_w && iz_base + BLK <= nz - pml_w) ? 1 : 0;

    if (valid_xy && 0 == blk_interior) {
      int cz0 = (iz_base > 0) ? iz_base - 1 : 0;
      int cz1 = iz_base;
      eta_z[0] = eta[(long)cz0 * ny * nx + xy_off];
      eta_z[1] = eta[(long)cz1 * ny * nx + xy_off];
    }
#endif

    UNROLL_OUTER(1) for (iz = iz_base; iz < iz_base + BLK && iz < nz; ++iz) {
      const long slice_base = (long)iz * ny * nx;
      float lap, p_center;

      for (idx = lid; idx < SLM_TOTAL; idx += WG_SIZE) {
        const int sy = idx / SLM_X;
        const int sx = idx % SLM_X;
        int gx = gx0 + sx;
        int gy = gy0 + sy;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
        if (gx < 0) gx = 0; else if (gx >= nx) gx = nx - 1;
        if (gy < 0) gy = 0; else if (gy >= ny) gy = ny - 1;
#endif
        { const long gi = slice_base + (long)gy * nx + gx;
          xy_slm[idx] = p_grid[gi];
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
          if (0 == blk_interior) eta_slm[idx] = eta[gi];
#endif
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if (valid_xy) {
#if !defined(NTERMS) || (2 < NTERMS)
        { int cz = iz + RADIUS;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
          if (cz >= nz) cz = nz - 1;
#endif
          z_win[Z_WINDOW - 1] = p_grid[(long)cz * ny * nx + xy_off];
        }
#endif
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
        if (0 == blk_interior) {
          int cz = iz + 1;
          if (cz >= nz) cz = nz - 1;
          eta_z[2] = eta[(long)cz * ny * nx + xy_off];
        }
#endif
        { const int c = (ly + RADIUS) * SLM_X + lx + RADIUS;
          CONSTANT const float* cx = coeff;
          CONSTANT const float* cy = coeff + STENCIL_WIDTH;
          CONSTANT const float* cz = coeff + 2 * STENCIL_WIDTH;
          p_center = xy_slm[c];
          lap = cx[RADIUS] * p_center;
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cx[RADIUS + r] * (xy_slm[c + r] + xy_slm[c - r]);
          }
#if !defined(NTERMS) || (1 < NTERMS)
          lap += cy[RADIUS] * p_center;
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cy[RADIUS + r] * (xy_slm[c + r * SLM_X] + xy_slm[c - r * SLM_X]);
          }
#endif
        }
#if !defined(NTERMS) || (2 < NTERMS)
        { CONSTANT const float* cz = coeff + 2 * STENCIL_WIDTH;
          lap += cz[RADIUS] * z_win[RADIUS];
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cz[RADIUS + r] * (z_win[RADIUS + r] + z_win[RADIUS - r]);
          }
        }
#endif
        { const long i = slice_base + xy_off;
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
          if (0 != blk_interior) {
            p_new[i] = 2.0f * p_center - p_old[i] + vel[i] * lap;
          }
          else {
            const int c = (ly + RADIUS) * SLM_X + lx + RADIUS;
            const float eta1 = eta_slm[c];
            const float phi_val = phi[i];
            const float numerator =
              (2.0f - eta1 * eta1 + 2.0f * eta1) * p_center - p_old[i]
              + vel[i] * (lap + phi_val);
            p_new[i] = numerator / (1.0f + 2.0f * eta1);
            { const float ux_p = xy_slm[c + 1];
              const float ux_m = xy_slm[c - 1];
              const float uy_p = xy_slm[c + SLM_X];
              const float uy_m = xy_slm[c - SLM_X];
              const float uz_p = z_win[RADIUS + 1];
              const float uz_m = z_win[RADIUS - 1];
              const float eta_xp = eta_slm[c + 1];
              const float eta_xm = eta_slm[c - 1];
              const float eta_yp = eta_slm[c + SLM_X];
              const float eta_ym = eta_slm[c - SLM_X];
              const float tmp =
                (eta_xp - eta_xm) * (ux_p - ux_m) * hdx_2
                + (eta_yp - eta_ym) * (uy_p - uy_m) * hdy_2
                + (eta_z[2] - eta_z[0]) * (uz_p - uz_m) * hdz_2;
              phi[i] = (phi_val - tmp) / (1.0f + eta1);
            }
          }
#else
          p_new[i] = 2.0f * p_center - p_old[i] + dt2 * vel[i] * lap;
#endif
        }

#if !defined(NTERMS) || (2 < NTERMS)
        UNROLL_FORCE(Z_WINDOW - 1) for (w = 0; w < Z_WINDOW - 1; ++w) {
          z_win[w] = z_win[w + 1];
        }
#endif

#if defined(STENCIL_PML) && (0 < STENCIL_PML)
        if (0 == blk_interior) {
          eta_z[0] = eta_z[1];
          eta_z[1] = eta_z[2];
        }
#endif
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  }
#endif
}
