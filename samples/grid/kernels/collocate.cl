/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "../grid_opencl.h"
#include "grid_prepare_pab.h"
#include "grid_cab_cxyz.h"


static inline void cab_add(cab_store* cab, const orbital a, const orbital b, const double value)
{
  cab->data[idx(b) * cab->n1 + idx(a)] += value;
}


/* C_xyz of a task about the center of the product, triangular (lz, ly, lx ascending) */
__kernel void grid_collocate_coef(const int ntasks, __global const double* restrict pab_blocks,
  const int pab_offset, __global const double* restrict sphi, __global const int* restrict tint,
  __global const double* restrict tdbl, __global const int* restrict cinfo,
  __global double* restrict coef)
{
  const int itask = (int)get_global_id(0);
  if (itask < ntasks) {
    __global const int* const ti = tint + GRID_OPENCL_NI * itask;
    __global const double* const td = tdbl + GRID_OPENCL_ND * itask;
    __global const double* const block = pab_blocks + pab_offset + ti[4];
    const int la_max = ti[0], la_min = ti[1], lb_max = ti[2], lb_min = ti[3];
    const int ps = ti[5], pt = ti[6], nsgf_seta = ti[7], nsgf_setb = ti[8];
    const int sphia = ti[9], maxcoa = ti[10], sphib = ti[11], maxcob = ti[12];
    const prepare_ldiffs ld = prepare_get_ldiffs((enum grid_func)GRID_FUNC);
    const int la_max_cab = la_max + ld.la_max_diff, lb_max_cab = lb_max + ld.lb_max_diff;
    const int la_min_cab = imax(la_min + ld.la_min_diff, 0);
    const int lb_min_cab = imax(lb_min + ld.lb_min_diff, 0);
    const int ncoa = ncoset(la_max), ncob = ncoset(lb_max);
    const int n1 = ncoset(la_max_cab), ncab = n1 * ncoset(lb_max_cab);
    const int lp = la_max_cab + lb_max_cab, lp1 = lp + 1;
    const double zeta = td[9], zetb = td[10];
    double pab[NPAB * NPAB], w[NPAB], cabd[NCAB * NCAB];
    double alpha[3 * LMAX1 * LMAX1 * LP1], cxyz[LP1 * LP1 * LP1], tx[LP1], txy[LP1 * LP1];
    double ra[3], rb[3], rp[3];
    __global double* const c = coef + cinfo[2 * itask];
    cab_store cab;
    int i, s, t, lxa, lxb, lya, lyb, lza, lzb, lx, ly, lz, n = 0;
    for (i = 0; i < ncob * ncoa; ++i) pab[i] = 0.0;
    for (t = 0; t < nsgf_setb; ++t) {
      int ico, jco;
      for (ico = 0; ico < ncoa; ++ico) {
        double sum = 0.0;
        for (s = 0; s < nsgf_seta; ++s) {
          sum += block[s * ps + t * pt] * sphi[sphia + s * maxcoa + ico];
        }
        w[ico] = sum;
      }
      for (jco = 0; jco < ncob; ++jco) {
        const double b = sphi[sphib + t * maxcob + jco];
        for (ico = 0; ico < ncoa; ++ico) pab[jco * ncoa + ico] += b * w[ico];
      }
    }
    for (i = 0; i < ncab; ++i) cabd[i] = 0.0;
    cab.n1 = n1;
    cab.data = cabd;
    for (lxa = 0; lxa <= la_max; ++lxa) {
      for (lxb = 0; lxb <= lb_max; ++lxb) {
        for (lya = 0; lya <= la_max - lxa; ++lya) {
          for (lyb = 0; lyb <= lb_max - lxb; ++lyb) {
            for (lza = imax(la_min - lxa - lya, 0); lza <= la_max - lxa - lya; ++lza) {
              for (lzb = imax(lb_min - lxb - lyb, 0); lzb <= lb_max - lxb - lyb; ++lzb) {
                const orbital a = {{lxa, lya, lza}}, b = {{lxb, lyb, lzb}};
                const double pab_val = pab[idx(b) * ncoa + idx(a)];
                prepare_pab((enum grid_func)GRID_FUNC, a, b, zeta, zetb, pab_val, &cab);
              }
            }
          }
        }
      }
    }
    for (i = 0; i < 3; ++i) {
      ra[i] = td[i];
      rb[i] = td[3 + i];
      rp[i] = td[6 + i];
    }
    for (i = 0; i < lp1 * lp1 * lp1; ++i) cxyz[i] = 0.0;
    grid_cab_alpha(la_max_cab, lb_max_cab, ra, rb, rp, alpha);
    grid_cab_to_cxyz(la_max_cab, la_min_cab, lb_max_cab, lb_min_cab, td[11], alpha, cabd, cxyz, tx, txy);
    for (lz = 0; lz <= lp; ++lz) {
      for (ly = 0; ly <= lp - lz; ++ly) {
        for (lx = 0; lx <= lp - lz - ly; ++lx) c[n++] = cxyz[(lz * lp1 + ly) * lp1 + lx];
      }
    }
  }
}


static inline int grid_image(const int g, const int lo, const int n)
{
  int r = (g - lo) % n;
  if (0 > r) r += n;
  return lo + r;
}


/* Each point is owned by one work-item: no atomics, and the order of the tile's list fixes the sum */
__kernel __attribute__((reqd_work_group_size(TX, TY, TZ))) void grid_collocate_tile(
  const int n0, const int n1, const int n2, const double16 dh, const int tile_base,
  __global const int* restrict tile_start, __global const int* restrict tile_task,
  __global const double* restrict geo, __global const int* restrict box,
  __global const int* restrict cinfo, __global const double* restrict coef,
  __global double* restrict grid, const int grid_offset)
{
  const int g0 = (int)get_global_id(0), g1 = (int)get_global_id(1), g2 = (int)get_global_id(2);
  if (g0 < n0 && g1 < n1 && g2 < n2) {
    const int tile = tile_base +
                     ((int)get_group_id(2) * (int)get_num_groups(1) + (int)get_group_id(1)) *
                       (int)get_num_groups(0) + (int)get_group_id(0);
    const int end = tile_start[tile + 1];
    double acc = 0.0;
    int i;
    for (i = tile_start[tile]; i < end; ++i) {
      const int t = tile_task[i];
      __global const double* const gt = geo + GRID_OPENCL_NG * t;
      __global const int* const bt = box + GRID_OPENCL_NB * t;
      __global const double* const c = coef + cinfo[2 * t];
      const int lp = cinfo[2 * t + 1];
      const double s0 = gt[0], s1 = gt[1], s2 = gt[2], zetp = gt[3], r2max = gt[4];
      int u0, u1, u2;
      for (u2 = grid_image(g2, bt[2], n2); u2 <= bt[5]; u2 += n2) {
        const double f2 = u2 - s2;
        for (u1 = grid_image(g1, bt[1], n1); u1 <= bt[4]; u1 += n1) {
          const double f1 = u1 - s1;
          for (u0 = grid_image(g0, bt[0], n0); u0 <= bt[3]; u0 += n0) {
            const double f0 = u0 - s0;
#if GRID_ORTHO
            const double dx = f0 * dh.s0, dy = f1 * dh.s4, dz = f2 * dh.s8;
#else
            const double dx = f0 * dh.s0 + f1 * dh.s3 + f2 * dh.s6;
            const double dy = f0 * dh.s1 + f1 * dh.s4 + f2 * dh.s7;
            const double dz = f0 * dh.s2 + f1 * dh.s5 + f2 * dh.s8;
#endif
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < r2max) {
              double sum = 0.0, pz = 1.0;
              int lx, ly, lz, k = 0;
              for (lz = 0; lz <= lp; ++lz) {
                double sy = 0.0, py = 1.0;
                for (ly = 0; ly <= lp - lz; ++ly) {
                  double sx = 0.0, px = 1.0;
                  for (lx = 0; lx <= lp - lz - ly; ++lx) {
                    sx += c[k++] * px;
                    px *= dx;
                  }
                  sy += sx * py;
                  py *= dy;
                }
                sum += sy * pz;
                pz *= dz;
              }
              acc += exp(-zetp * r2) * sum;
            }
          }
        }
      }
    }
    grid[grid_offset + ((size_t)g2 * n1 + g1) * n0 + g0] = acc;
  }
}
