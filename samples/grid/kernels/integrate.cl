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
#include "grid_cab_cxyz.h"
#include "grid_process_vab.h"
#include "grid_blocks.h"


/* (ly, lz) pairs per work-item */
#define NSLOT ((LP1 * (LP1 + 1) / 2 + WG - 1) / WG)


static inline double cab_get(const cab_store* cab, const orbital a, const orbital b)
{
  return cab->data[idx(b) * cab->n1 + idx(a)];
}


static inline int grid_wrap(const int u, const int n)
{
  const int r = u % n;
  return (0 > r ? (r + n) : r);
}


/* Start of the run (lx, ly) in the coefficient layout (lx, ly, lz ascending) */
static inline int grid_start(const int lp, const int lx, const int ly)
{
  const int a = lp + 1, b = lp - lx + 1;
  return (a * (a + 1) * (a + 2) - b * (b + 1) * (b + 2)) / 6 + ly * b - ly * (ly - 1) / 2;
}


/* One work-group per task: rows (or points) are spread over the work-items, and every
 * work-item owns a pair (ly, lz) and sums the rows in a fixed order (deterministic). */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void grid_integrate_coef(
  const int n0, const int n1, const int n2, const double16 dh, const int task_base,
  __global const int* restrict level_task, __global const double* restrict geo,
  __global const int* restrict box, __global const int* restrict cinfo,
  __global const double* restrict grid, const int grid_offset, __global double* restrict coef)
{
  __local double mom[WG * LP1], ytab[WG * LP1], ztab[WG * LP1];
  const int lid = (int)get_local_id(0);
  const int t = level_task[task_base + (int)get_group_id(0)];
  __global const double* const gt = geo + GRID_OPENCL_NG * t;
  __global const int* const bt = box + GRID_OPENCL_NB * t;
  const int lp = cinfo[2 * t + 1];
  const double s0 = gt[0], s1 = gt[1], s2 = gt[2], zetp = gt[3], r2max = gt[4];
  const int nx = (0 != GRID_ORTHO ? 1 : (bt[3] - bt[0] + 1)), ny = bt[4] - bt[1] + 1;
  const int nitems = nx * ny * (bt[5] - bt[2] + 1);
  __local double* const m = mom + lid * LP1;
  __local double* const ty = ytab + lid * LP1;
  __local double* const tz = ztab + lid * LP1;
  double acc[NSLOT * LP1];
  int qy[NSLOT], qz[NSLOT], r0, k, q;
  for (q = 0; q < NSLOT; ++q) { /* pairs lid, lid + WG, ... */
    qy[q] = 0;
    qz[q] = lid + q * WG;
    while (qy[q] <= lp && qz[q] > lp - qy[q]) {
      qz[q] -= lp - qy[q] + 1;
      ++qy[q];
    }
  }
  for (k = 0; k < NSLOT * LP1; ++k) acc[k] = 0.0;
  for (r0 = 0; r0 < nitems; r0 += WG) {
    const int r = r0 + lid;
    for (k = 0; k <= lp; ++k) m[k] = ty[k] = tz[k] = 0.0;
    if (r < nitems && 0 < r2max) {
#if GRID_ORTHO
      const int u1 = bt[1] + r % ny, u2 = bt[2] + r / ny;
      const double h0 = dh.s0, dy = (u1 - s1) * dh.s4, dz = (u2 - s2) * dh.s8, ryz = dy * dy + dz * dz;
      if (ryz < r2max) {
        __global const double* const row = grid + grid_offset + ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0;
        const double q = exp(-2.0 * zetp * h0 * h0), d0 = (bt[0] - s0) * h0;
        double ex = exp(-zetp * d0 * d0), rx = exp(-zetp * (2.0 * d0 * h0 + h0 * h0)), py, pz = 1.0;
        int u0, g = grid_wrap(bt[0], n0);
        for (u0 = bt[0]; u0 <= bt[3]; ++u0) { /* the Gaussian along the row by recurrence */
          const double dx = (u0 - s0) * h0;
          if (0 == ((u0 - bt[0]) & 31) && u0 != bt[0]) { /* restart: the recurrence drifts on long rows */
            ex = exp(-zetp * dx * dx);
            rx = exp(-zetp * (2.0 * dx * h0 + h0 * h0));
          }
          if (dx * dx + ryz < r2max) {
            double p = row[g] * ex;
            for (k = 0; k <= lp; ++k) {
              m[k] += p;
              p *= dx;
            }
          }
          ex *= rx;
          rx *= q;
          if (++g == n0) g = 0;
        }
        py = exp(-zetp * ryz);
        for (k = 0; k <= lp; ++k) {
          ty[k] = py;
          tz[k] = pz;
          py *= dy;
          pz *= dz;
        }
      }
#else
      const int u0 = bt[0] + r % nx, u1 = bt[1] + (r / nx) % ny, u2 = bt[2] + r / (nx * ny);
      const double f0 = u0 - s0, f1 = u1 - s1, f2 = u2 - s2;
      const double dx = f0 * dh.s0 + f1 * dh.s3 + f2 * dh.s6;
      const double dy = f0 * dh.s1 + f1 * dh.s4 + f2 * dh.s7;
      const double dz = f0 * dh.s2 + f1 * dh.s5 + f2 * dh.s8;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < r2max) {
        const size_t g = ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0 + grid_wrap(u0, n0);
        double p = grid[grid_offset + g] * exp(-zetp * r2), py = 1.0, pz = 1.0;
        for (k = 0; k <= lp; ++k) {
          m[k] = p;
          ty[k] = py;
          tz[k] = pz;
          p *= dx;
          py *= dy;
          pz *= dz;
        }
      }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (q = 0; q < NSLOT; ++q) {
      if (qy[q] <= lp) {
        int rr;
        for (rr = 0; rr < WG; ++rr) {
          const double tyz = ytab[rr * LP1 + qy[q]] * ztab[rr * LP1 + qz[q]];
          for (k = 0; k <= lp - qy[q] - qz[q]; ++k) acc[q * LP1 + k] += mom[rr * LP1 + k] * tyz;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (q = 0; q < NSLOT; ++q) {
    if (qy[q] <= lp) {
      __global double* const c = coef + cinfo[2 * t];
      for (k = 0; k <= lp - qy[q] - qz[q]; ++k) c[grid_start(lp, k, qy[q]) + qz[q]] = acc[q * LP1 + k];
    }
  }
}


/* One work-item per segment (tasks of one block and set pair): it owns that part of the
 * block, and its forces and virial are merged by the host in segment order. */
__kernel void grid_integrate_hab(const int nsegs, __global const int* restrict seg_start,
  __global const int* restrict seg_task, __global const double* restrict pab_blocks, const int pab_offset,
  __global const double* restrict sphi, __global const int* restrict tint,
  __global const double* restrict tdbl, __global const int* restrict cinfo,
  __global const double* restrict coef, __global double* restrict hab_blocks, const int hab_offset,
  __global double* restrict fv)
{
  const int iseg = (int)get_global_id(0);
  if (iseg < nsegs) {
    const process_ldiffs ld = process_get_ldiffs(0 != FORCES, 0 != VIRIAL, 0 != TAU);
    double f[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, va[9], vb[9];
    double pab[NPAB * NPAB], hab[NPAB * NPAB], w[NPAB], cabd[NCA * NCB];
    double alpha[3 * LMAX1A * LMAX1B * LP1], cxyz[LP1 * LP1 * LP1], tx[LP1], txy[LP1 * LP1];
    int i, j, k;
    for (k = 0; k < 9; ++k) va[k] = vb[k] = 0.0;
    for (k = seg_start[iseg]; k < seg_start[iseg + 1]; ++k) {
      const int t = seg_task[k];
      __global const int* const ti = tint + GRID_OPENCL_NI * t;
      __global const double* const td = tdbl + GRID_OPENCL_ND * t;
      __global const double* const c = coef + cinfo[2 * t];
      const int la_max = ti[0], la_min = ti[1], lb_max = ti[2], lb_min = ti[3];
      const int la_max_local = la_max + ld.la_max_diff, lb_max_local = lb_max + ld.lb_max_diff;
      const int la_min_local = imax(0, la_min + ld.la_min_diff);
      const int lb_min_local = imax(0, lb_min + ld.lb_min_diff);
      const int ncoa = ncoset(la_max), ncob = ncoset(lb_max), n1 = ncoset(la_max_local);
      const int lp = la_max_local + lb_max_local, lp1 = lp + 1;
      const double zeta = td[9], zetb = td[10], prefactor = td[11] / (0 != ti[13] ? 1.0 : 2.0);
      double ra[3], rb[3], rp[3], rab[3];
      cab_store cab;
      int la, lb, ax, ay, bx, by, lx, ly, lz, n = 0;
      for (i = 0; i < 3; ++i) {
        ra[i] = td[i];
        rb[i] = td[3 + i];
        rp[i] = td[6 + i];
        rab[i] = td[12 + i];
      }
      for (i = 0; i < lp1 * lp1 * lp1; ++i) cxyz[i] = 0.0;
      for (lx = 0; lx <= lp; ++lx) {
        for (ly = 0; ly <= lp - lx; ++ly) {
          for (lz = 0; lz <= lp - lx - ly; ++lz) cxyz[(lz * lp1 + ly) * lp1 + lx] = c[n++];
        }
      }
      for (i = 0; i < n1 * ncoset(lb_max_local); ++i) cabd[i] = 0.0;
      grid_cab_alpha(la_max_local, lb_max_local, ra, rb, rp, alpha);
      grid_cxyz_to_cab(la_max_local, la_min_local, lb_max_local, lb_min_local, prefactor, alpha, cxyz, cabd, tx, txy);
      cab.n1 = n1;
      cab.data = cabd;
      if (0 != FORCES || 0 != VIRIAL) grid_load_pab(pab_blocks + pab_offset, sphi, ti, pab, w);
      for (i = 0; i < ncoa * ncob; ++i) hab[i] = 0.0;
      for (la = la_min; la <= la_max; ++la) {
        for (ax = 0; ax <= la; ++ax) {
          for (ay = 0; ay <= la - ax; ++ay) {
            const orbital a = {{ax, ay, la - ax - ay}};
            for (lb = lb_min; lb <= lb_max; ++lb) {
              for (bx = 0; bx <= lb; ++bx) {
                for (by = 0; by <= lb - bx; ++by) {
                  const orbital b = {{bx, by, lb - bx - by}};
                  hab[idx(b) * ncoa + idx(a)] += get_hab(a, b, zeta, zetb, &cab, 0 != TAU);
                  if (0 != FORCES) {
                    const double pabval = pab[idx(b) * ncoa + idx(a)];
                    for (i = 0; i < 3; ++i) {
                      f[i] += pabval * get_force_a(a, b, i, zeta, zetb, &cab, 0 != TAU);
                      f[3 + i] += pabval * get_force_b(a, b, i, zeta, zetb, rab, &cab, 0 != TAU);
                    }
                  }
                  if (0 != VIRIAL) {
                    const double pabval = pab[idx(b) * ncoa + idx(a)];
                    for (i = 0; i < 3; ++i) {
                      for (j = 0; j < 3; ++j) {
                        va[3 * i + j] += pabval * get_virial_a(a, b, i, j, zeta, zetb, &cab, 0 != TAU);
                        vb[3 * i + j] += pabval * get_virial_b(a, b, i, j, zeta, zetb, rab, &cab, 0 != TAU);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      grid_store_hab(hab_blocks + hab_offset, sphi, ti, hab, w);
    }
    for (k = 0; k < 6; ++k) fv[24 * iseg + k] = f[k];
    for (k = 0; k < 9; ++k) {
      fv[24 * iseg + 6 + k] = va[k];
      fv[24 * iseg + 15 + k] = vb[k];
    }
  }
}
