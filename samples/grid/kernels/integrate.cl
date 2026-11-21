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
/* window terms of forces and virial: moments weighted by w'(r) / r, in single precision (values below eps) */
#define WU (0 != WINDOW && (0 != FORCES || 0 != VIRIAL))

#define GRID_ICASE(N) \
  case N: \
    for (r = lane - myoff; r < nitems && 0 < gt[4]; r += myw) { \
      grid_item(r, N, n0, n1, n2, dh, dhi, gt[0], gt[1], gt[2], gt[3], gt[6], gt[7], gauss + gofs[mytask], bt, \
        grid + grid_offset, mine, mineu); \
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


/* Width of a task (lanes): the smallest power of two with at most ROWS rows per lane */
static inline int grid_width(const int nitems)
{
  int w = 1;
  while (w < ENGINE && ROWS * w < nitems) w <<= 1;
  return w;
}


/* Adds a row to the partial C_xyz of a lane (general cells: in lattice coordinates). Inlined with a constant
 * lp (switch), the loops unroll and the arrays stay in registers: a runtime bound puts them on the stack. */
static inline void grid_item(const int r, const int lp, const int n0, const int n1, const int n2,
  const double16 dh, const double16 dhi, const double s0, const double s1, const double s2, const double zetp,
  const double rmax, const double winv, global const double* restrict gx, global const int* restrict bt,
  global const double* restrict grid, local double* restrict part, local float* restrict partu)
{
  const int ny = bt[4] - bt[1] + 1;
  double m[LP1], ty[LP1], tz[LP1];
#if WU
  float mu[LP1], tyf[LP1], tzf[LP1];
#endif
  int k, lx, ly, lz, valid = 0;
#if GRID_ORTHO
  const int u1 = bt[1] + r % ny, u2 = bt[2] + r / ny;
  const double h0 = dh.s0, dy = (u1 - s1) * dh.s4, dz = (u2 - s2) * dh.s8;
  const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
  if (rg.x <= rg.y) {
    global const double* const row = grid + ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0;
    global const double* const gy = gx + (bt[3] - bt[0] + 1);
    global const double* const gz = gy + (bt[4] - bt[1] + 1);
    double py = gy[u1 - bt[1]] * gz[u2 - bt[2]], pz = 1.0;
    int u0, g = grid_wrap(rg.x, n0);
    double dx = (rg.x - s0) * h0;
#if WINDOW
    const float wc = (float)((rmax * rmax - dy * dy - dz * dz) * winv), wv = (float)winv; /* t = wc - dx^2 wv */
#endif
    UNROLL_AUTO for (k = 0; k <= lp; ++k) m[k] = 0.0;
#if WU
    UNROLL_AUTO for (k = 0; k <= lp; ++k) mu[k] = 0.0;
#endif
    for (u0 = rg.x; u0 <= rg.y; ++u0) { /* the range of the row: no mask */
      double p = row[g] * gx[u0 - bt[0]];
#if WU
      float pu; /* weight w'(r) / r */
      const float dxf = (float)dx;
#endif
#if WINDOW
      { const float dxw = (float)dx, wt = wc - dxw * dxw * wv;
#  if WU
        pu = (float)p * (-2.0f * wv * grid_window_d(wt));
#  endif
        p *= (double)grid_window(wt);
      }
#endif
      UNROLL_AUTO for (k = 0; k <= lp; ++k) {
        m[k] += p;
        p *= dx;
#if WU
        mu[k] += pu;
        pu *= dxf;
#endif
      }
      dx += h0;
      if (++g == n0) g = 0;
    }
    UNROLL_AUTO for (k = 0; k <= lp; ++k) {
      ty[k] = py;
      tz[k] = pz;
#if WU
      tyf[k] = (float)py;
      tzf[k] = (float)pz;
#endif
      py *= dy;
      pz *= dz;
    }
    valid = 1;
  }
#else
  const int u1 = bt[1] + r % ny, u2 = bt[2] + r / ny;
  const int2 rg = grid_row_range(u1, u2, s0, s1, s2, rmax, bt[6], bt[7], bt[8], dh, dhi);
  (void)gx; /* no tables: the Gaussian of a general cell does not factor along the axes */
  if (rg.x <= rg.y) { /* lattice moments along the row, the Gaussian by recurrence (|d|^2 is quadratic in u0) */
    global const double* const row = grid + ((size_t)grid_wrap(u2, n2) * n1 + grid_wrap(u1, n1)) * n0;
    const double f1 = u1 - s1, f2 = u2 - s2, a00 = dh.s0 * dh.s0 + dh.s1 * dh.s1 + dh.s2 * dh.s2;
    const double wx = f1 * dh.s3 + f2 * dh.s6, wy = f1 * dh.s4 + f2 * dh.s7, wz = f1 * dh.s5 + f2 * dh.s8;
    const double q = exp(-2.0 * zetp * a00);
    double ex = 0.0, rx = 0.0, py = 1.0, pz = 1.0;
#if WINDOW
    const float w2 = (float)(2.0 * a00 * winv), wv = (float)winv; /* second difference of t along the row */
    float wt = 0.0f, wd = 0.0f; /* window: t along the row and its difference (quadratic in u0) */
#endif
    int u0, g = grid_wrap(rg.x, n0);
    UNROLL_AUTO for (k = 0; k <= lp; ++k) m[k] = 0.0;
#if WU
    UNROLL_AUTO for (k = 0; k <= lp; ++k) mu[k] = 0.0;
#endif
    for (u0 = rg.x; u0 <= rg.y; ++u0) {
      const double f0 = u0 - s0;
      double p;
#if WU
      float pu; /* weight w'(r) / r */
      const float f0f = (float)f0;
#endif
      if (0 == ((u0 - rg.x) & 31)) { /* start, and restart: the recurrence drifts on long rows */
        const double dx = f0 * dh.s0 + wx, dy = f0 * dh.s1 + wy, dz = f0 * dh.s2 + wz;
        const double r2 = dx * dx + dy * dy + dz * dz, dr2 = 2.0 * (dx * dh.s0 + dy * dh.s1 + dz * dh.s2) + a00;
        ex = exp(-zetp * r2);
        rx = exp(-zetp * dr2);
#if WINDOW
        wt = (float)((rmax * rmax - r2) * winv);
        wd = (float)(-dr2 * winv);
#endif
      }
      p = row[g] * ex;
#if WINDOW
#  if WU
      pu = (float)p * (-2.0f * wv * grid_window_d(wt));
#  endif
      p *= (double)grid_window(wt);
      wt += wd;
      wd -= w2;
#endif
      UNROLL_AUTO for (k = 0; k <= lp; ++k) {
        m[k] += p;
        p *= f0;
#if WU
        mu[k] += pu;
        pu *= f0f;
#endif
      }
      ex *= rx;
      rx *= q;
      if (++g == n0) g = 0;
    }
    UNROLL_AUTO for (k = 0; k <= lp; ++k) {
      ty[k] = py;
      tz[k] = pz;
#if WU
      tyf[k] = (float)py;
      tzf[k] = (float)pz;
#endif
      py *= f1;
      pz *= f2;
    }
    valid = 1;
  }
#endif
  if (0 != valid) {
    k = 0;
    UNROLL_AUTO for (lx = 0; lx <= lp; ++lx) {
      UNROLL_AUTO for (ly = 0; ly <= lp - lx; ++ly) {
        const double mxy = m[lx] * ty[ly];
#if WU
        const float uxy = mu[lx] * tyf[ly];
#endif
        UNROLL_AUTO for (lz = 0; lz <= lp - lx - ly; ++lz) {
#if WU
          partu[k] += uxy * tzf[lz];
#endif
          part[k++] += mxy * tz[lz];
        }
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
kernel __attribute__((reqd_work_group_size(ENGINE, 1, 1))) REQD_SG void grid_integrate_coef(
  const int n0, const int n1, const int n2, const double16 dh, const double16 dhi, const int task_base,
  const int ntasks, global const int* restrict level_task, global const double* restrict geo,
  global const int* restrict box, global const int* restrict cinfo,
  global const double* restrict grid, const int grid_offset, global double* restrict coef,
  global int* restrict queue, global const double* restrict gauss, global const int* restrict gofs)
{
  local double part[ENGINE * NCOEF];
#if WU
  local float part_u[ENGINE * NCOEF];
  local float* const mineu = part_u + (int)get_local_id(0) * NCOEF;
#else
  local float* const mineu = NULL;
#endif
  local int chunk_task[ENGINE], chunk_width[ENGINE], chunk_base;
  const int lane = (int)get_local_id(0);
  local double* const mine = part + lane * NCOEF;
  int offset = 0, mytask = -1, myoff = 0, myw = 0, k;
  for (k = 0; k < NCOEF; ++k) mine[k] = 0.0;
#if WU
  for (k = 0; k < NCOEF; ++k) mineu[k] = 0.0f;
#endif
  for (;;) {
    int j, n;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (0 == lane) chunk_base = atomic_add(queue, ENGINE);
    barrier(CLK_LOCAL_MEM_FENCE);
    n = max(0, min(ENGINE, ntasks - chunk_base));
    if (lane < n) {
      const int t = level_task[task_base + chunk_base + lane];
      global const int* const bt = box + GRID_OPENCL_NB * t;
      chunk_task[lane] = t;
      chunk_width[lane] = grid_width((bt[4] - bt[1] + 1) * (bt[5] - bt[2] + 1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j = 0; j <= n; ++j) { /* j == n: flush the last pack once the queue is empty */
      const int w = (j < n ? chunk_width[j] : 1);
      const int at = (offset + w - 1) & ~(w - 1);
      if (j < n ? (ENGINE < at + w) : (0 == n && 0 < offset)) { /* run the pack: full, or the queue is empty */
        if (0 <= mytask) {
          global const double* const gt = geo + GRID_OPENCL_NG * mytask;
          global const int* const bt = box + GRID_OPENCL_NB * mytask;
          const int lp = cinfo[2 * mytask + 1];
              const int nitems = (bt[4] - bt[1] + 1) * (bt[5] - bt[2] + 1);
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
          global double* const c = coef + cinfo[2 * mytask];
          for (k = lane - myoff; k < nc; k += myw) {
            double sum = 0.0;
            int l;
            for (l = 0; l < myw; ++l) sum += part[(myoff + l) * NCOEF + k];
            c[k] = sum;
#if WU
            sum = 0.0;
            for (l = 0; l < myw; ++l) sum += part_u[(myoff + l) * NCOEF + k];
            c[nc + k] = sum;
#endif
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (k = 0; k < NCOEF; ++k) mine[k] = 0.0;
#if WU
        for (k = 0; k < NCOEF; ++k) mineu[k] = 0.0f;
#endif
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


/* Cartesian moments (cube of stride lp1) from the triangular moments of a task (general cells: lattice moments) */
static inline void grid_moments(const int lp, const int lp1, global const double* restrict c, const double16 dh,
  double* cxyz)
{
  int i, lx, ly, lz, n = 0;
  for (i = 0; i < lp1 * lp1 * lp1; ++i) cxyz[i] = 0.0;
#if GRID_ORTHO
  (void)dh;
  for (lx = 0; lx <= lp; ++lx) {
    for (ly = 0; ly <= lp - lx; ++ly) {
      for (lz = 0; lz <= lp - lx - ly; ++lz) cxyz[(lz * lp1 + ly) * lp1 + lx] = c[n++];
    }
  }
#else
  { /* general cells: Cartesian moments from the lattice moments (transpose of the change of variables) */
    double latt[LP1 * LP1 * LP1], px[LP1 * LP1], pxy[LP1 * LP1], pxyz[LP1 * LP1], tmp[LP1 * LP1];
    for (lx = 0; lx <= lp; ++lx) {
      for (ly = 0; ly <= lp - lx; ++ly) {
        for (lz = 0; lz <= lp - lx - ly; ++lz) latt[(lx * LP1 + ly) * LP1 + lz] = c[n++];
      }
    }
    grid_lattice(lp, lp1, dh, cxyz, latt, 0, px, pxy, pxyz, tmp);
  }
#endif
}


/* One work-item per task: cab, the primitive hab, forces and virial into the task's scratch */
kernel REQD_SG void grid_integrate_task(const int ntasks, global const double* restrict pab_blocks,
  const int pab_offset, global const double* restrict sphi, global const int* restrict tint,
  global const double* restrict tdbl, global const int* restrict cinfo, global const double* restrict coef,
  global const int* restrict hoff, global double* restrict hab_task, global double* restrict fv_task,
  global const double16* restrict ldh)
{
  const int t = (int)get_global_id(0);
  if (t < ntasks) {
    const process_ldiffs ld = process_get_ldiffs(0 != FORCES, 0 != VIRIAL, 0 != TAU);
    global const int* const ti = tint + GRID_OPENCL_NI * t;
    global const double* const td = tdbl + GRID_OPENCL_ND * t;
    global const double* const c = coef + cinfo[2 * t];
    global double* const hab = hab_task + hoff[t];
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
    int i, j, la, lb, ax, ay, bx, by, lx, ly, lz;
    for (i = 0; i < 6; ++i) f[i] = 0.0;
    for (i = 0; i < 9; ++i) va[i] = vb[i] = 0.0;
    for (i = 0; i < 3; ++i) {
      ra[i] = td[i];
      rb[i] = td[3 + i];
      rp[i] = td[6 + i];
      rab[i] = td[12 + i];
    }
    grid_moments(lp, lp1, c, ldh[ti[14]], cxyz);
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
#if WU
    { /**
       * Window terms: w(|p - rp|) moves with rp = (zeta ra + zetb rb) / zetp, and with the strain. With the moments
       * of V g w'(r) / r, a moment shifted by e_k gives sum V rho w'(r) / r d_k (forces: times -zeta / zetp and
       * -zetb / zetp), shifted by e_i + e_j it gives the virial's -sum V rho w'(r) / r d_i d_j.
       */
      const process_ldiffs lh = process_get_ldiffs(0, 0, 0 != TAU);
      const int la_max_h = la_max + lh.la_max_diff, lb_max_h = lb_max + lh.lb_max_diff;
      const int la_min_h = imax(0, la_min + lh.la_min_diff), lb_min_h = imax(0, lb_min + lh.lb_min_diff);
      const int lh1 = la_max_h + lb_max_h + 1, zero = (0 == FORCES ? 3 : 0), nshift = (0 == VIRIAL ? 3 : 9);
      double ck[LP1 * LP1 * LP1], cabk[NCA * NCB];
      cab_store cabw;
      int q;
      grid_moments(lp, lp1, c + ncoset(lp), ldh[ti[14]], cxyz); /* cab is built: cxyz takes the moments of w' */
      grid_cab_alpha(la_max_h, lb_max_h, ra, rb, rp, alpha);
      cabw.n1 = ncoset(la_max_h);
      cabw.data = cabk;
      for (q = zero; q < nshift; ++q) { /* 0-2: e_k, 3-8: e_i + e_j (i <= j) */
        const int si = (3 > q ? q : (6 > q ? 0 : (8 > q ? 1 : 2)));
        const int sj = (3 > q ? -1 : (6 > q ? (q - 3) : (8 > q ? (q - 5) : 2)));
        const int ex = (0 == si) + (0 == sj), ey = (1 == si) + (1 == sj), ez = (2 == si) + (2 == sj);
        double g = 0.0;
        for (lx = 0; lx < lh1; ++lx) {
          for (ly = 0; ly < lh1 - lx; ++ly) {
            for (lz = 0; lz < lh1 - lx - ly; ++lz) {
              ck[(lz * lh1 + ly) * lh1 + lx] = cxyz[((lz + ez) * lp1 + ly + ey) * lp1 + lx + ex];
            }
          }
        }
        for (i = 0; i < cabw.n1 * ncoset(lb_max_h); ++i) cabk[i] = 0.0;
        grid_cxyz_to_cab(la_max_h, la_min_h, lb_max_h, lb_min_h, prefactor, alpha, ck, cabk, tx, txy);
        for (la = la_min; la <= la_max; ++la) {
          for (ax = 0; ax <= la; ++ax) {
            for (ay = 0; ay <= la - ax; ++ay) {
              const orbital a = {{ax, ay, la - ax - ay}};
              for (lb = lb_min; lb <= lb_max; ++lb) {
                for (bx = 0; bx <= lb; ++bx) {
                  for (by = 0; by <= lb - bx; ++by) {
                    const orbital b = {{bx, by, lb - bx - by}};
                    g += pab[idx(b) * ncoa + idx(a)] * get_hab(a, b, zeta, zetb, &cabw, 0 != TAU);
                  }
                }
              }
            }
          }
        }
        if (3 > q) {
          f[q] -= zeta / (zeta + zetb) * g;
          f[3 + q] -= zetb / (zeta + zetb) * g;
        }
        else {
          va[3 * si + sj] -= g;
          if (si != sj) va[3 * sj + si] -= g;
        }
      }
    }
#endif
    if (0 != FORCES || 0 != VIRIAL) {
      global double* const fv = fv_task + 24 * (size_t)t;
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
kernel void grid_integrate_hab(const int nsegs, global const int* restrict seg_start,
  global const int* restrict seg_task, global const double* restrict sphi, global const int* restrict tint,
  global const int* restrict hoff, global const double* restrict hab_task,
  global const double* restrict fv_task, global double* restrict hab_blocks, const int hab_offset,
  global double* restrict fv)
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
