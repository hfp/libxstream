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
#include "grid_cab_cxyz.h"
#include "grid_process_vab.h"
#include "grid_blocks.h"


/* coefficients of a task at most, and of the partial sums of a lane */
#define NCOEF (LP1 * (LP1 + 1) * (LP1 + 2) / 6)
/* items per lane before a task gets more lanes: a point of a general cell is a fraction of a row */
#define ITEMS (0 != GRID_ORTHO ? ROWS : (8 * ROWS))

#define GRID_ICASE(N) \
  case N: \
    for (r = lane - myoff; r < nitems && 0 < gt[4]; r += myw) { \
      grid_item(r, N, n0, n1, n2, dh, dhi, gt[0], gt[1], gt[2], gt[3], gt[6], gauss + gofs[mytask], bt, \
        grid + grid_offset, mine); \
    } \
    break;


static inline double cab_get(const cab_store* cab, const orbital a, const orbital b)
{
  return cab->data[idx(b) * cab->n1 + idx(a)];
}


static inline int grid_wrap(const int u, const int n)
{
  const int r = u % n;
  return (0 > r ? (r + n) : r);
}


/* Width of a task (lanes): the smallest power of two with at most ITEMS items per lane */
static inline int grid_width(const int nitems)
{
  int w = 1;
  while (w < ENGINE && ITEMS * w < nitems) w <<= 1;
  return w;
}


/* Adds one item (a row, or a point of a general cell) to the partial C_xyz of a lane. Inlined with a constant
 * lp (switch), the loops unroll and the arrays stay in registers: a runtime bound puts them on the stack. */
static inline void grid_item(const int r, const int lp, const int n0, const int n1, const int n2,
  const double16 dh, const double16 dhi, const double s0, const double s1, const double s2, const double zetp,
  const double rmax, __global const double* restrict gx, __global const int* restrict bt, __global const double* restrict grid,
  __local double* restrict part)
{
  const int ny = bt[4] - bt[1] + 1;
  double m[LP1], ty[LP1], tz[LP1];
  int k, lx, ly, lz, valid = 0;
#if GRID_ORTHO
  const int u1 = bt[1] + r % ny, u2 = bt[2] + r / ny;
  const double h0 = dh.s0, dy = (u1 - s1) * dh.s4, dz = (u2 - s2) * dh.s8;
  const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
  if (rg.x <= rg.y) {
    __global const double* const row = grid + ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0;
    __global const double* const gy = gx + (bt[3] - bt[0] + 1);
    __global const double* const gz = gy + (bt[4] - bt[1] + 1);
    double py = gy[u1 - bt[1]] * gz[u2 - bt[2]], pz = 1.0;
    int u0, g = grid_wrap(rg.x, n0);
    double dx = (rg.x - s0) * h0;
    UNROLL_AUTO for (k = 0; k <= lp; ++k) m[k] = 0.0;
    for (u0 = rg.x; u0 <= rg.y; ++u0) { /* the range of the row: no mask */
      double p = row[g] * gx[u0 - bt[0]];
      UNROLL_AUTO for (k = 0; k <= lp; ++k) {
        m[k] += p;
        p *= dx;
      }
      dx += h0;
      if (++g == n0) g = 0;
    }
    UNROLL_AUTO for (k = 0; k <= lp; ++k) {
      ty[k] = py;
      tz[k] = pz;
      py *= dy;
      pz *= dz;
    }
    valid = 1;
  }
#else
  const int nx = bt[3] - bt[0] + 1;
  const int u0 = bt[0] + r % nx, u1 = bt[1] + (r / nx) % ny, u2 = bt[2] + r / (nx * ny);
  const double f0 = u0 - s0, f1 = u1 - s1, f2 = u2 - s2;
  const double dx = f0 * dh.s0 + f1 * dh.s3 + f2 * dh.s6;
  const double dy = f0 * dh.s1 + f1 * dh.s4 + f2 * dh.s7;
  const double dz = f0 * dh.s2 + f1 * dh.s5 + f2 * dh.s8;
  const double r2 = dx * dx + dy * dy + dz * dz;
  const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
  if (rg.x <= u0 && u0 <= rg.y) {
    const size_t g = ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0 + grid_wrap(u0, n0);
    double p = grid[g] * exp(-zetp * r2), py = 1.0, pz = 1.0;
    UNROLL_AUTO for (k = 0; k <= lp; ++k) {
      m[k] = p;
      ty[k] = py;
      tz[k] = pz;
      p *= dx;
      py *= dy;
      pz *= dz;
    }
    valid = 1;
  }
#endif
  if (0 != valid) {
    k = 0;
    UNROLL_AUTO for (lx = 0; lx <= lp; ++lx) {
      UNROLL_AUTO for (ly = 0; ly <= lp - lx; ++ly) {
        const double mxy = m[lx] * ty[ly];
        UNROLL_AUTO for (lz = 0; lz <= lp - lx - ly; ++lz) part[k++] += mxy * tz[lz];
      }
    }
  }
}


/**
 * A fixed number of engines (work-groups of ENGINE lanes) take chunks of tasks from a queue.
 * Each task gets a power-of-two width from its size alone, and is placed at the next offset
 * aligned to that width; a full engine runs its pack. A task's lanes split its items in a
 * fixed way and their partial sums are combined in lane order, so the result does not depend
 * on which engine took the task: the queue decides who, never how (deterministic).
 */
__kernel __attribute__((reqd_work_group_size(ENGINE, 1, 1))) void grid_integrate_coef(
  const int n0, const int n1, const int n2, const double16 dh, const double16 dhi, const int task_base,
  const int ntasks, __global const int* restrict level_task, __global const double* restrict geo,
  __global const int* restrict box, __global const int* restrict cinfo,
  __global const double* restrict grid, const int grid_offset, __global double* restrict coef,
  __global int* restrict queue, __global const double* restrict gauss, __global const int* restrict gofs)
{
  __local double part[ENGINE * NCOEF];
  __local int chunk_task[ENGINE], chunk_width[ENGINE], chunk_base;
  const int lane = (int)get_local_id(0);
  __local double* const mine = part + lane * NCOEF;
  int offset = 0, mytask = -1, myoff = 0, myw = 0, k;
  for (k = 0; k < NCOEF; ++k) mine[k] = 0.0;
  for (;;) {
    int j, n;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (0 == lane) chunk_base = atomic_add(queue, ENGINE);
    barrier(CLK_LOCAL_MEM_FENCE);
    n = max(0, min(ENGINE, ntasks - chunk_base));
    if (lane < n) {
      const int t = level_task[task_base + chunk_base + lane];
      __global const int* const bt = box + GRID_OPENCL_NB * t;
      const int nx = (0 != GRID_ORTHO ? 1 : (bt[3] - bt[0] + 1));
      chunk_task[lane] = t;
      chunk_width[lane] = grid_width(nx * (bt[4] - bt[1] + 1) * (bt[5] - bt[2] + 1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j = 0; j <= n; ++j) { /* j == n: flush the last pack once the queue is empty */
      const int w = (j < n ? chunk_width[j] : 1);
      const int at = (offset + w - 1) & ~(w - 1);
      if (j < n ? (ENGINE < at + w) : (0 == n && 0 < offset)) { /* run the pack: full, or the queue is empty */
        if (0 <= mytask) {
          __global const double* const gt = geo + GRID_OPENCL_NG * mytask;
          __global const int* const bt = box + GRID_OPENCL_NB * mytask;
          const int lp = cinfo[2 * mytask + 1];
          const int nx = (0 != GRID_ORTHO ? 1 : (bt[3] - bt[0] + 1));
          const int nitems = nx * (bt[4] - bt[1] + 1) * (bt[5] - bt[2] + 1);
          int r;
          switch (lp) { /* uniform for the lanes of a task, and mostly for a pack (neighbouring tasks) */
            GRID_ICASE(0)
#if 1 < LP1
            GRID_ICASE(1)
#endif
#if 2 < LP1
            GRID_ICASE(2)
#endif
#if 3 < LP1
            GRID_ICASE(3)
#endif
#if 4 < LP1
            GRID_ICASE(4)
#endif
#if 5 < LP1
            GRID_ICASE(5)
#endif
#if 6 < LP1
            GRID_ICASE(6)
#endif
#if 7 < LP1
            GRID_ICASE(7)
#endif
#if 8 < LP1
            GRID_ICASE(8)
#endif
#if 9 < LP1
            GRID_ICASE(9)
#endif
#if 10 < LP1
            GRID_ICASE(10)
#endif
#if 11 < LP1
            GRID_ICASE(11)
#endif
#if 12 < LP1
            GRID_ICASE(12)
#endif
#if 13 < LP1
            GRID_ICASE(13)
#endif
#if 14 < LP1
            GRID_ICASE(14)
#endif
#if 15 < LP1
            GRID_ICASE(15)
#endif
#if 16 < LP1
#  error "lp > 15"
#endif
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (0 <= mytask) {
          const int lp = cinfo[2 * mytask + 1], nc = (lp + 1) * (lp + 2) * (lp + 3) / 6;
          __global double* const c = coef + cinfo[2 * mytask];
          for (k = lane - myoff; k < nc; k += myw) {
            double sum = 0.0;
            int l;
            for (l = 0; l < myw; ++l) sum += part[(myoff + l) * NCOEF + k];
            c[k] = sum;
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (k = 0; k < NCOEF; ++k) mine[k] = 0.0;
        mytask = -1;
        offset = 0;
      }
      if (j < n) {
        const int a = (offset + w - 1) & ~(w - 1);
        if (a <= lane && lane < a + w) {
          mytask = chunk_task[j];
          myoff = a;
          myw = w;
        }
        offset = a + w;
      }
    }
    if (0 >= n) break;
  }
}


/* One work-item per task: cab, the primitive hab, forces and virial into the task's scratch */
__kernel void grid_integrate_task(const int ntasks, __global const double* restrict pab_blocks, const int pab_offset,
  __global const double* restrict sphi, __global const int* restrict tint, __global const double* restrict tdbl,
  __global const int* restrict cinfo, __global const double* restrict coef, __global const int* restrict hoff,
  __global double* restrict hab_task, __global double* restrict fv_task)
{
  const int t = (int)get_global_id(0);
  if (t < ntasks) {
    const process_ldiffs ld = process_get_ldiffs(0 != FORCES, 0 != VIRIAL, 0 != TAU);
    __global const int* const ti = tint + GRID_OPENCL_NI * t;
    __global const double* const td = tdbl + GRID_OPENCL_ND * t;
    __global const double* const c = coef + cinfo[2 * t];
    __global double* const hab = hab_task + hoff[t];
    const int la_max = ti[0], la_min = ti[1], lb_max = ti[2], lb_min = ti[3];
    const int la_max_local = la_max + ld.la_max_diff, lb_max_local = lb_max + ld.lb_max_diff;
    const int la_min_local = imax(0, la_min + ld.la_min_diff);
    const int lb_min_local = imax(0, lb_min + ld.lb_min_diff);
    const int ncoa = ncoset(la_max), ncob = ncoset(lb_max), n1 = ncoset(la_max_local);
    const int lp = la_max_local + lb_max_local, lp1 = lp + 1;
    const double zeta = td[9], zetb = td[10], prefactor = td[11] / (0 != ti[13] ? 1.0 : 2.0);
    double pab[NPAB * NPAB], w[NPAB], cabd[NCA * NCB], f[6], va[9], vb[9];
    double alpha[3 * LMAX1A * LMAX1B * LP1], cxyz[LP1 * LP1 * LP1], tx[LP1], txy[LP1 * LP1];
    double ra[3], rb[3], rp[3], rab[3];
    cab_store cab;
    int i, j, la, lb, ax, ay, bx, by, lx, ly, lz, n = 0;
    for (i = 0; i < 6; ++i) f[i] = 0.0;
    for (i = 0; i < 9; ++i) va[i] = vb[i] = 0.0;
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
    if (0 != FORCES || 0 != VIRIAL) {
      __global double* const fv = fv_task + 24 * (size_t)t;
      for (i = 0; i < 6; ++i) fv[i] = f[i];
      for (i = 0; i < 9; ++i) {
        fv[6 + i] = va[i];
        fv[15 + i] = vb[i];
      }
    }
  }
}


/* One work-item per segment (tasks of one block and set pair): it owns that part of the
 * block, adds the tasks in order, and sums their forces and virial in order as well. */
__kernel void grid_integrate_hab(const int nsegs, __global const int* restrict seg_start,
  __global const int* restrict seg_task, __global const double* restrict sphi, __global const int* restrict tint,
  __global const int* restrict hoff, __global const double* restrict hab_task,
  __global const double* restrict fv_task, __global double* restrict hab_blocks, const int hab_offset,
  __global double* restrict fv)
{
  const int iseg = (int)get_global_id(0);
  if (iseg < nsegs) {
    double sum[24], w[NPAB];
    int i, k;
    for (i = 0; i < 24; ++i) sum[i] = 0.0;
    for (k = seg_start[iseg]; k < seg_start[iseg + 1]; ++k) {
      const int t = seg_task[k];
      grid_store_hab(hab_blocks + hab_offset, sphi, tint + GRID_OPENCL_NI * t, hab_task + hoff[t], w);
      if (0 != FORCES || 0 != VIRIAL) {
        for (i = 0; i < 24; ++i) sum[i] += fv_task[24 * (size_t)t + i];
      }
    }
    for (i = 0; i < 24; ++i) fv[24 * iseg + i] = sum[i];
  }
}
