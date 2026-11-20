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

#include <opencl/libxstream_common.h>
#include "../grid_opencl.h"
#include "grid_prepare_pab.h"
#include "grid_cab_cxyz.h"
#include "grid_blocks.h"

/* address space of the read-only tables in the hot loops (global or constant) */
#if !defined(CONSTANT)
#  define CONSTANT global
#endif


static inline void cab_add(cab_store* cab, const orbital a, const orbital b, const double value)
{
  cab->data[idx(b) * cab->n1 + idx(a)] += value;
}


/* C_xyz of a task about the center of the product, triangular (lx, ly, lz ascending) */
__kernel void grid_collocate_coef(const int ntasks, __global const double* restrict pab_blocks,
  const int pab_offset, __global const double* restrict sphi, __global const int* restrict tint,
  __global const double* restrict tdbl, __global int* restrict cinfo,
  __global double* restrict coef, __global const double16* restrict ldh)
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
    int i, lxa, lxb, lya, lyb, lza, lzb, lx, ly, lz, n = 0, nonzero = 0;
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
#if GRID_ORTHO
    for (lx = 0; lx <= lp; ++lx) {
      for (ly = 0; ly <= lp - lx; ++ly) {
        for (lz = 0; lz <= lp - lx - ly; ++lz) {
          const double v = cxyz[(lz * lp1 + ly) * lp1 + lx];
          nonzero |= (0.0 != v);
          c[n++] = v;
        }
      }
    }
#else
    { /* general cells: in lattice coordinates, the rows take the orthorhombic form */
      double latt[LP1 * LP1 * LP1], px[LP1 * LP1 * LP1], pxy[LP1 * LP1 * LP1], pxyz[LP1 * LP1 * LP1];
      double tmp[LP1 * LP1 * LP1];
      grid_lattice(lp, lp1, ldh[ti[14]], cxyz, latt, 1, px, pxy, pxyz, tmp);
      for (lx = 0; lx <= lp; ++lx) {
        for (ly = 0; ly <= lp - lx; ++ly) {
          for (lz = 0; lz <= lp - lx - ly; ++lz) {
            const double v = latt[(lx * LP1 + ly) * LP1 + lz];
            nonzero |= (0.0 != v);
            c[n++] = v;
          }
        }
      }
    }
#endif
    /* zero coefficients in this call (e.g., the block-diagonal atomic guess): the tiles skip the task */
    if (0 == nonzero) cinfo[2 * itask + 1] = -1;
  }
}


/* Polynomial in dz of a run of coefficients (Horner) */
static inline double grid_run(CONSTANT const double* restrict c, const int m, const double dz)
{
  double s = c[m];
  int lz;
  UNROLL_AUTO for (lz = m - 1; 0 <= lz; --lz) s = s * dz + c[lz];
  return s;
}


/* As grid_run, in single precision (a row of the tail) */
static inline float grid_run_f32(CONSTANT const double* restrict c, const int m, const float dz)
{
  float s = (float)c[m];
  int lz;
  UNROLL_AUTO for (lz = m - 1; 0 <= lz; --lz) s = s * dz + (float)c[lz];
  return s;
}


static inline double grid_point(CONSTANT const double* restrict c, const int lp, const double zetp,
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


/* A row: C_xyz contracted into C_x (y and z), then the points by Horner in x times the Gaussian along x
 * (table). Inlined with a constant lp (switch), the loops unroll: the runtime loops cost more than the math. */
static inline void grid_row64(const int lp, CONSTANT const double* restrict c, const double eyz, const int2 rg,
  const double dy, const double dz, const double h0, CONSTANT const double* restrict gx, const double s0,
  const int x0, const int n0, CONSTANT const int* restrict bt, const double rmax2, const double winv, double* acc)
{
  double cx[LP1], dx;
  int lx, ly, j, u0, k = 0;
#if WINDOW
  const float wc = (float)((rmax2 - dy * dy - dz * dz) * winv), wv = (float)winv; /* t = wc - dx^2 wv */
#endif
  UNROLL_AUTO for (lx = 0; lx <= lp; ++lx) {
    double sy = 0.0, py = eyz;
    UNROLL_AUTO for (ly = 0; ly <= lp - lx; ++ly) {
      sy += grid_run(c + k, lp - lx - ly, dz) * py;
      k += lp - lx - ly + 1;
      py *= dy;
    }
    cx[lx] = sy;
  }
  u0 = grid_image(x0, bt[0], n0);
  dx = (u0 - s0) * h0;
  UNROLL_AUTO for (j = 0; j < TX; ++j) {
    if (0 != j) { /* incremental: one add on the FP64 pipe instead of three operations */
      if (++u0 == bt[0] + n0) {
        u0 = bt[0];
        dx = (u0 - s0) * h0;
      }
      else dx += h0;
    }
    if (x0 + j < n0 && rg.x <= u0 && u0 <= rg.y) {
      double p = cx[lp];
      UNROLL_AUTO for (lx = lp - 1; 0 <= lx; --lx) p = p * dx + cx[lx];
#if WINDOW
      { const float dxf = (float)dx;
        p *= (double)grid_window(wc - dxf * dxf * wv);
      }
#endif
      acc[j] += gx[u0 - bt[0]] * p;
    }
  }
}


/* As grid_row64 in single precision (tail): the error of single precision is below the truncation */
static inline void grid_row32(const int lp, CONSTANT const double* restrict c, const double eyz, const int2 rg,
  const double dy, const double dz, const double h0, CONSTANT const double* restrict gx, const double s0,
  const int x0, const int n0, CONSTANT const int* restrict bt, const double rmax2, const double winv, double* acc)
{
  const float dyf = (float)dy, dzf = (float)dz;
  float cx[LP1];
  double dx;
  int lx, ly, j, u0, k = 0;
#if WINDOW
  const float wc = (float)((rmax2 - dy * dy - dz * dz) * winv), wv = (float)winv; /* t = wc - dx^2 wv */
#endif
  UNROLL_AUTO for (lx = 0; lx <= lp; ++lx) {
    float sy = 0.0f, py = (float)eyz;
    UNROLL_AUTO for (ly = 0; ly <= lp - lx; ++ly) {
      sy += grid_run_f32(c + k, lp - lx - ly, dzf) * py;
      k += lp - lx - ly + 1;
      py *= dyf;
    }
    cx[lx] = sy;
  }
  u0 = grid_image(x0, bt[0], n0);
  dx = (u0 - s0) * h0;
  UNROLL_AUTO for (j = 0; j < TX; ++j) {
    if (0 != j) {
      if (++u0 == bt[0] + n0) {
        u0 = bt[0];
        dx = (u0 - s0) * h0;
      }
      else dx += h0;
    }
    if (x0 + j < n0 && rg.x <= u0 && u0 <= rg.y) {
      const float dxf = (float)dx;
      float p = cx[lp];
      UNROLL_AUTO for (lx = lp - 1; 0 <= lx; --lx) p = p * dxf + cx[lx];
#if WINDOW
      p *= grid_window(wc - dxf * dxf * wv);
#endif
      acc[j] += (double)((float)gx[u0 - bt[0]] * p);
    }
  }
}


/* A row of a general cell: coefficients in lattice coordinates (f = u - s), contracted with f1, f2, then Horner in
 * f0 times the Gaussian along the row by recurrence (|d|^2 is quadratic in u0), restarted where the row wraps. */
static inline void grid_row_gen(const int lp, CONSTANT const double* restrict c, const int2 rg, const double f1,
  const double f2, const double s0, const double zetp, const double16 dh, const int x0, const int n0,
  const double rmax2, const double winv, double* acc)
{
  const double a00 = dh.s0 * dh.s0 + dh.s1 * dh.s1 + dh.s2 * dh.s2, q = exp(-2.0 * zetp * a00);
  const double wx = f1 * dh.s3 + f2 * dh.s6, wy = f1 * dh.s4 + f2 * dh.s7, wz = f1 * dh.s5 + f2 * dh.s8;
#if WINDOW
  const float w2 = (float)(2.0 * a00 * winv); /* second difference of t along the row */
#endif
  double cx[LP1], ex = 0.0, rx = 0.0;
  float wt = 0.0f, wd = 0.0f; /* window: t along the row and its difference (quadratic in u0) */
  int lx, ly, j, u0, k = 0, have = 0;
  UNROLL_AUTO for (lx = 0; lx <= lp; ++lx) {
    double sy = 0.0, py = 1.0;
    UNROLL_AUTO for (ly = 0; ly <= lp - lx; ++ly) {
      sy += grid_run(c + k, lp - lx - ly, f2) * py;
      k += lp - lx - ly + 1;
      py *= f1;
    }
    cx[lx] = sy;
  }
  if (rg.y - rg.x < n0) { /* a point has at most one image in the range */
    u0 = grid_image(x0, rg.x, n0);
    UNROLL_AUTO for (j = 0; j < TX; ++j) {
      if (0 != j && ++u0 == rg.x + n0) {
        u0 = rg.x;
        have = 0;
      }
      if (x0 + j < n0 && u0 <= rg.y) {
        const double f0 = u0 - s0;
        double p = cx[lp];
        if (0 == have) {
          const double dx = f0 * dh.s0 + wx, dy = f0 * dh.s1 + wy, dz = f0 * dh.s2 + wz;
          const double r2 = dx * dx + dy * dy + dz * dz, dr2 = 2.0 * (dx * dh.s0 + dy * dh.s1 + dz * dh.s2) + a00;
          ex = exp(-zetp * r2);
          rx = exp(-zetp * dr2);
#if WINDOW
          wt = (float)((rmax2 - r2) * winv);
          wd = (float)(-dr2 * winv);
#endif
          have = 1;
        }
        else {
          ex *= rx;
          rx *= q;
#if WINDOW
          wt += wd;
          wd -= w2;
#endif
        }
        UNROLL_AUTO for (lx = lp - 1; 0 <= lx; --lx) p = p * f0 + cx[lx];
#if WINDOW
        p *= (double)grid_window(wt);
#endif
        acc[j] += ex * p;
      }
      else have = 0;
    }
  }
  else { /* several images per point: the Gaussian per image */
    UNROLL_AUTO for (j = 0; j < TX; ++j) {
      if (x0 + j < n0) {
        for (u0 = grid_image(x0 + j, rg.x, n0); u0 <= rg.y; u0 += n0) {
          const double f0 = u0 - s0, dx = f0 * dh.s0 + wx, dy = f0 * dh.s1 + wy, dz = f0 * dh.s2 + wz;
          const double d2 = dx * dx + dy * dy + dz * dz;
          double p = cx[lp];
          UNROLL_AUTO for (lx = lp - 1; 0 <= lx; --lx) p = p * f0 + cx[lx];
#if WINDOW
          p *= (double)grid_window((float)((rmax2 - d2) * winv));
#endif
          acc[j] += exp(-zetp * d2) * p;
        }
      }
    }
  }
}


#if 0 == GRID_ORTHO
#  define GRID_CASE(N) \
    case N: \
      grid_row_gen(N, c, rg, u1 - s1, u2 - s2, s0, zetp, dh, x0, n0, rmax2, winv, acc); \
      break;
#elif FP32
#  define GRID_CASE(N) \
    case N: \
      if (0 == tail) grid_row64(N, c, eyz, rg, dy, dz, h0, gx, s0, x0, n0, bt, rmax2, winv, acc); \
      else grid_row32(N, c, eyz, rg, dy, dz, h0, gx, s0, x0, n0, bt, rmax2, winv, acc); \
      break;
#else
#  define GRID_CASE(N) \
    case N: \
      grid_row64(N, c, eyz, rg, dy, dz, h0, gx, s0, x0, n0, bt, rmax2, winv, acc); \
      break;
#endif


/* Position of a tile of the level (tiles ordered x fastest) */
static inline void grid_tile_origin(const int tile, const int n0, const int n1, int* x0, int* y0, int* z0)
{
  const int ntx = (n0 + TX - 1) / TX, nty = (n1 + TY - 1) / TY;
  *x0 = (tile % ntx) * TX;
  *y0 = ((tile / ntx) % nty) * TY;
  *z0 = (tile / (ntx * nty)) * TZ;
}


/**
 * A work-group takes a unit: a tile and a chunk of its tasks. A work-item owns a row of TX points (no atomics),
 * and the order of the chunk fixes the sum. A tile of a single chunk writes the grid, else its chunks write
 * partial tiles which grid_collocate_reduce sums in chunk order (deterministic).
 */
__kernel __attribute__((reqd_work_group_size(1, TY, TZ))) void grid_collocate_tile(
  const int n0, const int n1, const int n2, const double16 dh, const double16 dhi, const int unit_base,
  __global const int* restrict unit_tile, __global const int* restrict unit_start,
  __global const int* restrict unit_slot, __global const int* restrict tile_task,
  CONSTANT const double* restrict geo, CONSTANT const int* restrict box,
  CONSTANT const int* restrict cinfo, CONSTANT const double* restrict coef,
  __global double* restrict grid, const int grid_offset, CONSTANT const double* restrict gauss,
  CONSTANT const int* restrict gofs, __global double* restrict partial)
{
  const int unit = unit_base + (int)get_group_id(0), ly = (int)get_local_id(1), lz = (int)get_local_id(2);
  const int slot = unit_slot[unit];
  int x0, y0, z0, g1, g2;
  double acc[TX];
  int i, j;
  grid_tile_origin(unit_tile[unit], n0, n1, &x0, &y0, &z0);
  g1 = y0 + ly;
  g2 = z0 + lz;
  for (j = 0; j < TX; ++j) acc[j] = 0.0;
  if (g1 < n1 && g2 < n2) {
    const int end = unit_start[unit + 1];
    for (i = unit_start[unit]; i < end; ++i) {
      const int tail = (0 > tile_task[i]); /* uniform per work-group: the whole tile is in the tail */
      const int t = (0 == tail ? tile_task[i] : ~tile_task[i]);
      CONSTANT const double* const gt = geo + GRID_OPENCL_NG * t;
      CONSTANT const int* const bt = box + GRID_OPENCL_NB * t;
      CONSTANT const double* const c = coef + cinfo[2 * t];
      const int lp = cinfo[2 * t + 1]; /* -1: zero coefficients in this call */
      const double s0 = gt[0], s1 = gt[1], s2 = gt[2], zetp = gt[3], rmax = gt[6];
      const double rmax2 = rmax * rmax, winv = gt[7];
      int u0, u1, u2;
      if (0 <= lp && (0 == GRID_ORTHO || bt[3] - bt[0] < n0)) { /* rows (orthorhombic: along x at most one image) */
#if GRID_ORTHO
        CONSTANT const double* const gx = gauss + gofs[t]; /* table of the Gaussian */
        CONSTANT const double* const gy = gx + (bt[3] - bt[0] + 1);
        CONSTANT const double* const gz = gy + (bt[4] - bt[1] + 1);
        const double h0 = dh.s0;
#endif
        for (u2 = grid_image(g2, bt[2], n2); u2 <= bt[5]; u2 += n2) {
          for (u1 = grid_image(g1, bt[1], n1); u1 <= bt[4]; u1 += n1) {
            const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
            if (rg.x <= rg.y) { /* lp is uniform per task: a case per lp unrolls the row */
#if GRID_ORTHO
              const double dz = (u2 - s2) * dh.s8, dy = (u1 - s1) * dh.s4;
              const double eyz = gy[u1 - bt[1]] * gz[u2 - bt[2]];
#endif
              switch (lp) {
                GRID_CASE(0)
#if 1 < LP1
                GRID_CASE(1)
#endif
#if 2 < LP1
                GRID_CASE(2)
#endif
#if 3 < LP1
                GRID_CASE(3)
#endif
#if 4 < LP1
                GRID_CASE(4)
#endif
#if 5 < LP1
                GRID_CASE(5)
#endif
#if 6 < LP1
                GRID_CASE(6)
#endif
#if 7 < LP1
                GRID_CASE(7)
#endif
#if 8 < LP1
                GRID_CASE(8)
#endif
#if 9 < LP1
                GRID_CASE(9)
#endif
#if 10 < LP1
                GRID_CASE(10)
#endif
#if 11 < LP1
                GRID_CASE(11)
#endif
#if 12 < LP1
                GRID_CASE(12)
#endif
#if 13 < LP1
                GRID_CASE(13)
#endif
#if 14 < LP1
                GRID_CASE(14)
#endif
#if 15 < LP1
                GRID_CASE(15)
#endif
#if 16 < LP1
#  error "lp > 15"
#endif
              }
            }
          }
        }
      }
      else if (0 <= lp) {
        UNROLL_AUTO for (j = 0; j < TX; ++j) {
          const int g0 = x0 + j;
          if (g0 < n0) {
            for (u2 = grid_image(g2, bt[2], n2); u2 <= bt[5]; u2 += n2) {
              const double f2 = u2 - s2;
              for (u1 = grid_image(g1, bt[1], n1); u1 <= bt[4]; u1 += n1) {
                const double f1 = u1 - s1;
                const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
                for (u0 = grid_image(g0, bt[0], n0); u0 <= bt[3]; u0 += n0) {
                  const double f0 = u0 - s0;
                  const double dx = f0 * dh.s0, dy = f1 * dh.s4, dz = f2 * dh.s8; /* orthorhombic */
                  const double r2 = dx * dx + dy * dy + dz * dz;
#if WINDOW
                  if (rg.x <= u0 && u0 <= rg.y) {
                    const float wt = (float)((rmax2 - r2) * winv);
                    acc[j] += grid_point(c, lp, zetp, r2, dx, dy, dz) * (double)grid_window(wt);
                  }
#else
                  if (rg.x <= u0 && u0 <= rg.y) acc[j] += grid_point(c, lp, zetp, r2, dx, dy, dz);
#endif
                }
              }
            }
          }
        }
      }
    }
    if (0 > slot) {
      UNROLL_AUTO for (j = 0; j < TX; ++j) {
        if (x0 + j < n0) grid[grid_offset + ((size_t)g2 * n1 + g1) * n0 + x0 + j] = acc[j];
      }
    }
    else {
      __global double* const p = partial + ((size_t)slot * TZ + lz) * TY * TX + ly * TX;
      UNROLL_AUTO for (j = 0; j < TX; ++j) p[j] = acc[j];
    }
  }
}


/* Sums the partial tiles of a tile in chunk order */
__kernel __attribute__((reqd_work_group_size(1, TY, TZ))) void grid_collocate_reduce(
  const int n0, const int n1, const int n2, const int red_base, __global const int* restrict red_tile,
  __global const int* restrict red_slot, __global const int* restrict red_n, __global const double* restrict partial,
  __global double* restrict grid, const int grid_offset)
{
  const int r = red_base + (int)get_group_id(0), ly = (int)get_local_id(1), lz = (int)get_local_id(2);
  const int first = red_slot[r], n = red_n[r];
  int x0, y0, z0, j, c;
  grid_tile_origin(red_tile[r], n0, n1, &x0, &y0, &z0);
  if (y0 + ly < n1 && z0 + lz < n2) {
    UNROLL_AUTO for (j = 0; j < TX; ++j) {
      if (x0 + j < n0) {
        double sum = 0.0;
        for (c = 0; c < n; ++c) sum += partial[((size_t)(first + c) * TZ + lz) * TY * TX + ly * TX + j];
        grid[grid_offset + ((size_t)(z0 + lz) * n1 + y0 + ly) * n0 + x0 + j] = sum;
      }
    }
  }
}
