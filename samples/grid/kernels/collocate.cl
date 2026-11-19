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
#include "grid_blocks.h"


static inline void cab_add(cab_store* cab, const orbital a, const orbital b, const double value)
{
  cab->data[idx(b) * cab->n1 + idx(a)] += value;
}


/* C_xyz of a task about the center of the product, triangular (lx, ly, lz ascending) */
__kernel void grid_collocate_coef(const int ntasks, __global const double* restrict pab_blocks,
  const int pab_offset, __global const double* restrict sphi, __global const int* restrict tint,
  __global const double* restrict tdbl, __global const int* restrict cinfo,
  __global double* restrict coef)
{
  const int itask = (int)get_global_id(0);
  if (itask < ntasks) {
    __global const int* const ti = tint + GRID_OPENCL_NI * itask;
    __global const double* const td = tdbl + GRID_OPENCL_ND * itask;
    const int la_max = ti[0], la_min = ti[1], lb_max = ti[2], lb_min = ti[3];
    const prepare_ldiffs ld = prepare_get_ldiffs((enum grid_func)GRID_FUNC);
    const int la_max_cab = la_max + ld.la_max_diff, lb_max_cab = lb_max + ld.lb_max_diff;
    const int la_min_cab = imax(la_min + ld.la_min_diff, 0);
    const int lb_min_cab = imax(lb_min + ld.lb_min_diff, 0);
    const int ncoa = ncoset(la_max);
    const int n1 = ncoset(la_max_cab), ncab = n1 * ncoset(lb_max_cab);
    const int lp = la_max_cab + lb_max_cab, lp1 = lp + 1;
    const double zeta = td[9], zetb = td[10];
    double pab[NPAB * NPAB], w[NPAB], cabd[NCAB * NCAB];
    double alpha[3 * LMAX1 * LMAX1 * LP1], cxyz[LP1 * LP1 * LP1], tx[LP1], txy[LP1 * LP1];
    double ra[3], rb[3], rp[3];
    __global double* const c = coef + cinfo[2 * itask];
    cab_store cab;
    int i, lxa, lxb, lya, lyb, lza, lzb, lx, ly, lz, n = 0;
    grid_load_pab(pab_blocks + pab_offset, sphi, ti, pab, w);
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
    for (lx = 0; lx <= lp; ++lx) {
      for (ly = 0; ly <= lp - lx; ++ly) {
        for (lz = 0; lz <= lp - lx - ly; ++lz) c[n++] = cxyz[(lz * lp1 + ly) * lp1 + lx];
      }
    }
  }
}


/* Polynomial in dz of a run of coefficients (Horner) */
static inline double grid_run(__global const double* restrict c, const int m, const double dz)
{
  double s = c[m];
  int lz;
  for (lz = m - 1; 0 <= lz; --lz) s = s * dz + c[lz];
  return s;
}


static inline double grid_point(__global const double* restrict c, const int lp, const double zetp,
  const double r2, const double dx, const double dy, const double dz)
{
  double sum = 0.0, px = 1.0;
  int lx, ly, k = 0;
  for (lx = 0; lx <= lp; ++lx) {
    double sy = 0.0, py = 1.0;
    for (ly = 0; ly <= lp - lx; ++ly) {
      sy += grid_run(c + k, lp - lx - ly, dz) * py;
      k += lp - lx - ly + 1;
      py *= dy;
    }
    sum += sy * px;
    px *= dx;
  }
  return exp(-zetp * r2) * sum;
}


/* A work-item owns a row of TX points (no atomics), and the order of the tile's list fixes the sum */
__kernel __attribute__((reqd_work_group_size(1, TY, TZ))) void grid_collocate_tile(
  const int n0, const int n1, const int n2, const double16 dh, const int tile_base,
  __global const int* restrict tile_start, __global const int* restrict tile_task,
  __global const double* restrict geo, __global const int* restrict box,
  __global const int* restrict cinfo, __global const double* restrict coef,
  __global double* restrict grid, const int grid_offset)
{
  const int x0 = (int)get_group_id(0) * TX, g1 = (int)get_global_id(1), g2 = (int)get_global_id(2);
  const int tile = tile_base + ((int)get_group_id(2) * (int)get_num_groups(1) + (int)get_group_id(1)) *
                                 (int)get_num_groups(0) + (int)get_group_id(0);
  double acc[TX];
  int i, j;
  for (j = 0; j < TX; ++j) acc[j] = 0.0;
  if (g1 < n1 && g2 < n2) {
    const int end = tile_start[tile + 1];
    for (i = tile_start[tile]; i < end; ++i) {
      const int t = tile_task[i];
      __global const double* const gt = geo + GRID_OPENCL_NG * t;
      __global const int* const bt = box + GRID_OPENCL_NB * t;
      __global const double* const c = coef + cinfo[2 * t];
      const int lp = cinfo[2 * t + 1];
      const double s0 = gt[0], s1 = gt[1], s2 = gt[2], zetp = gt[3], r2max = gt[4];
      int u0, u1, u2;
      if (0 != GRID_ORTHO && bt[3] - bt[0] < n0) { /* along x at most one image: recurrence of exp */
        const double h0 = dh.s0, q = exp(-2.0 * zetp * h0 * h0);
        for (u2 = grid_image(g2, bt[2], n2); u2 <= bt[5]; u2 += n2) {
          const double dz = (u2 - s2) * dh.s8;
          for (u1 = grid_image(g1, bt[1], n1); u1 <= bt[4]; u1 += n1) {
            const double dy = (u1 - s1) * dh.s4, ryz = dy * dy + dz * dz;
            if (ryz < r2max) {
              const double eyz = exp(-zetp * ryz);
              double cx[LP1], ex = 0.0, rx = 0.0;
              int lx, ly, k = 0, have = 0;
              for (lx = 0; lx <= lp; ++lx) {
                double sy = 0.0, py = eyz;
                for (ly = 0; ly <= lp - lx; ++ly) {
                  sy += grid_run(c + k, lp - lx - ly, dz) * py;
                  k += lp - lx - ly + 1;
                  py *= dy;
                }
                cx[lx] = sy;
              }
              u0 = grid_image(x0, bt[0], n0);
#pragma unroll
              for (j = 0; j < TX; ++j) {
                if (0 != j && ++u0 == bt[0] + n0) {
                  u0 = bt[0];
                  have = 0;
                }
                if (x0 + j < n0 && u0 <= bt[3]) {
                  const double dx = (u0 - s0) * h0;
                  if (0 == have) {
                    ex = exp(-zetp * dx * dx);
                    rx = exp(-zetp * (2.0 * dx * h0 + h0 * h0));
                    have = 1;
                  }
                  else {
                    ex *= rx;
                    rx *= q;
                  }
                  if (dx * dx + ryz < r2max) {
                    double p = cx[lp];
                    for (lx = lp - 1; 0 <= lx; --lx) p = p * dx + cx[lx];
                    acc[j] += ex * p;
                  }
                }
                else have = 0;
              }
            }
          }
        }
      }
      else {
#pragma unroll
        for (j = 0; j < TX; ++j) {
          const int g0 = x0 + j;
          if (g0 < n0) {
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
                  if (r2 < r2max) acc[j] += grid_point(c, lp, zetp, r2, dx, dy, dz);
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (j = 0; j < TX; ++j) {
      if (x0 + j < n0) grid[grid_offset + ((size_t)g2 * n1 + g1) * n0 + x0 + j] = acc[j];
    }
  }
}
