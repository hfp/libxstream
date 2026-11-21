/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef GRID_BLOCKS_H
#define GRID_BLOCKS_H

/* Primitive pab[jco][ico] of a task: sphib^T P sphia, with the block layout of grid_opencl.c */
static inline void grid_load_pab(global const double* restrict blocks, global const double* restrict sphi,
  global const int* restrict ti, double* pab, double* w)
{
  const int ncoa = ncoset(ti[0]), ncob = ncoset(ti[2]);
  int i, s, t;
  for (i = 0; i < ncob * ncoa; ++i) pab[i] = 0.0;
  for (t = 0; t < ti[8]; ++t) {
    int ico, jco;
    for (ico = 0; ico < ncoa; ++ico) {
      double sum = 0.0;
      for (s = 0; s < ti[7]; ++s) sum += blocks[ti[4] + s * ti[5] + t * ti[6]] * sphi[ti[9] + s * ti[10] + ico];
      w[ico] = sum;
    }
    for (jco = 0; jco < ncob; ++jco) {
      const double b = sphi[ti[11] + t * ti[12] + jco];
      for (ico = 0; ico < ncoa; ++ico) pab[jco * ncoa + ico] += b * w[ico];
    }
  }
}

/* Adds the primitive hab[jco][ico] of a task to its block: sphib hab sphia^T */
static inline void grid_store_hab(global double* restrict blocks, global const double* restrict sphi,
  global const int* restrict ti, global const double* restrict hab, double* w)
{
  const int ncoa = ncoset(ti[0]), ncob = ncoset(ti[2]);
  int s, t;
  for (t = 0; t < ti[8]; ++t) {
    int ico, jco;
    for (ico = 0; ico < ncoa; ++ico) {
      double sum = 0.0;
      for (jco = 0; jco < ncob; ++jco) sum += sphi[ti[11] + t * ti[12] + jco] * hab[jco * ncoa + ico];
      w[ico] = sum;
    }
    for (s = 0; s < ti[7]; ++s) {
      double sum = 0.0;
      for (ico = 0; ico < ncoa; ++ico) sum += w[ico] * sphi[ti[9] + s * ti[10] + ico];
      blocks[ti[4] + s * ti[5] + t * ti[6]] += sum;
    }
  }
}

/**
 * Range [lo, hi] along x of the row (u1, u2) of a task (empty if lo > hi), i.e., which points the task covers.
 * Orthorhombic cells (default): points within rmax of the cell holding rp, REF's nested bounds verbatim (cell c).
 * General cells or CENTRED: points within rmax of rp, REF's quadratic of the general case verbatim.
 */
static inline int2 grid_row_range(const int u1, const int u2, const double s0, const double s1, const double s2,
  const double rmax, const int cx, const int cy, const int cz, const double16 dh, const double16 dhi)
{
  int2 r = (int2)(1, 0);
#if 0 == GRID_ORTHO || 0 != CENTRED
  const double v0 = (0 - s0) * dh.s0 + (u1 - s1) * dh.s3 + (u2 - s2) * dh.s6;
  const double v1 = (0 - s0) * dh.s1 + (u1 - s1) * dh.s4 + (u2 - s2) * dh.s7;
  const double v2 = (0 - s0) * dh.s2 + (u1 - s1) * dh.s5 + (u2 - s2) * dh.s8;
  double a = 0.0, b = 0.0, c = 0.0, d;
  a += dh.s0 * dh.s0;
  b += 2.0 * v0 * dh.s0;
  c += v0 * v0;
  a += dh.s1 * dh.s1;
  b += 2.0 * v1 * dh.s1;
  c += v1 * v1;
  a += dh.s2 * dh.s2;
  b += 2.0 * v2 * dh.s2;
  c += v2 * v2;
  d = b * b - 4.0 * a * (c - rmax * rmax);
  if (0.0 < d) {
    const double sqrt_d = sqrt(d), inv_2a = 1.0 / (2.0 * a);
    r.x = (int)ceil((-b - sqrt_d) * inv_2a);
    r.y = (int)floor((-b + sqrt_d) * inv_2a);
  }
#else
  const int k = u2 - cz, j = u1 - cy;
  const int kd = (0 >= k ? k : (1 - k)); /* pairs of opposing points share the distance to the cell */
  const double kr = kd * dh.s8, kremain = rmax * rmax - kr * kr;
  const int jstart = (int)ceil(-1e-8 - sqrt(fmax(0.0, kremain)) * dhi.s4);
  if (jstart <= j && j <= 1 - jstart) {
    const int jd = (0 >= j ? j : (1 - j));
    const double jr = jd * dh.s4, jremain = kremain - jr * jr;
    const int istart = (int)ceil(-1e-8 - sqrt(fmax(0.0, jremain)) * dhi.s0);
    r.x = cx + istart;
    r.y = cx + 1 - istart;
  }
#endif
  return r;
}


/**
 * Smooth window (WINDOW) of t = (rmax^2 - r^2) / (rmax^2 - radius^2): 1 within the radius of the task, 0 at rmax,
 * C2 between (quintic in r^2: no sqrt). Single precision: only values below eps (the margin) get a weight below 1.
 */
static inline float grid_window(const float t)
{
  const float u = fmin(fmax(t, 0.0f), 1.0f);
  return u * u * u * (10.0f + u * (6.0f * u - 15.0f));
}


/* Derivative dw/dt of the window (0 outside the margin): w'(r) / r = -2 dw/dt / (rmax^2 - radius^2) */
static inline float grid_window_d(const float t)
{
  const float u = fmin(fmax(t, 0.0f), 1.0f), v = 1.0f - u;
  return 30.0f * u * u * v * v;
}


/**
 * Homogeneous polynomial of degree d in f0, f1, f2 times the linear form l0 f0 + l1 f1 + l2 f2. The term
 * f0^i f1^j f2^k (i = d - j - k) is stored at [j * LP1 + k]: the linear forms of the lattice vectors have no
 * constant, hence their products are homogeneous (LP1^2 entries instead of a cube).
 */
static inline void grid_poly_lin(const int d, const double* p, const double l0, const double l1, const double l2,
  double* q)
{
  int i, j;
  for (j = 0; j <= d + 1; ++j) {
    int k;
    for (k = 0; k <= d + 1 - j; ++k) q[j * LP1 + k] = 0.0;
  }
  for (i = 0; i <= d; ++i) { /* i, then j ascending: each term of q sums its three contributions in a fixed order */
    for (j = 0; j <= d - i; ++j) {
      const int k = d - i - j;
      const double v = p[j * LP1 + k];
      q[j * LP1 + k] += v * l0;
      q[(j + 1) * LP1 + k] += v * l1;
      q[j * LP1 + k + 1] += v * l2;
    }
  }
}


/* Copies a homogeneous polynomial of degree d (layout of grid_poly_lin) */
static inline void grid_poly_copy(const int d, const double* p, double* q)
{
  int j, k;
  for (j = 0; j <= d; ++j) {
    for (k = 0; k <= d - j; ++k) q[j * LP1 + k] = p[j * LP1 + k];
  }
}


/**
 * Change of variables between Cartesian d and lattice coordinates f (d = f0 a0 + f1 a1 + f2 a2, rows of dh) of a
 * polynomial of degree lp. cart is a cube [c][b][a] of stride lp1 (d_x^a d_y^b d_z^c), latt a cube of stride LP1.
 * to_lattice: latt = sum cart_abc L_x^a L_y^b L_z^c (coefficients); else cart_abc = <L_x^a L_y^b L_z^c, latt>
 * (moments, the transpose). x, xy, xyz, tmp: scratch of LP1^2 (homogeneous polynomials, see grid_poly_lin).
 */
static inline void grid_lattice(const int lp, const int lp1, const double16 dh, double* cart, double* latt,
  const int to_lattice, double* x, double* xy, double* xyz, double* tmp)
{
  int a, b, c, i;
  if (0 != to_lattice) {
    for (i = 0; i < LP1 * LP1 * LP1; ++i) latt[i] = 0.0;
  }
  x[0] = 1.0;
  for (a = 0; a <= lp; ++a) {
    grid_poly_copy(a, x, xy);
    for (b = 0; b <= lp - a; ++b) {
      grid_poly_copy(a + b, xy, xyz);
      for (c = 0; c <= lp - a - b; ++c) {
        const int d = a + b + c, ic = (c * lp1 + b) * lp1 + a;
        int ii, jj;
        double sum = 0.0;
        for (ii = 0; ii <= d; ++ii) { /* the terms of degree d, in the order of the cube */
          for (jj = 0; jj <= d - ii; ++jj) {
            const int kk = d - ii - jj, n = (ii * LP1 + jj) * LP1 + kk;
            const double v = xyz[jj * LP1 + kk];
            if (0 != to_lattice) latt[n] += cart[ic] * v;
            else sum += v * latt[n];
          }
        }
        if (0 == to_lattice) cart[ic] = sum;
        if (c < lp - a - b) { /* times d_z = f0 a0z + f1 a1z + f2 a2z */
          grid_poly_lin(d, xyz, dh.s2, dh.s5, dh.s8, tmp);
          grid_poly_copy(d + 1, tmp, xyz);
        }
      }
      if (b < lp - a) {
        grid_poly_lin(a + b, xy, dh.s1, dh.s4, dh.s7, tmp);
        grid_poly_copy(a + b + 1, tmp, xy);
      }
    }
    if (a < lp) {
      grid_poly_lin(a, x, dh.s0, dh.s3, dh.s6, tmp);
      grid_poly_copy(a + 1, tmp, x);
    }
  }
}


/* Gaussian of a task along each axis of its box (orthorhombic): the rows need no exp */
kernel void grid_gauss(const int ntasks, global const double* restrict geo, global const int* restrict box,
  global const double* restrict tdbl, global const int* restrict gofs, global double* restrict gauss)
{
  const int t = (int)get_group_id(0); /* work-group per task, work-item per entry */
  if (t < ntasks) {
    global const double* const gt = geo + GRID_OPENCL_NG * t;
    global const int* const bt = box + GRID_OPENCL_NB * t;
    global const double* const td = tdbl + GRID_OPENCL_ND * t;
    global double* const g = gauss + gofs[t];
    const int n0 = bt[3] - bt[0] + 1, n01 = n0 + bt[4] - bt[1] + 1, n = n01 + bt[5] - bt[2] + 1;
    int o;
    for (o = (int)get_local_id(0); o < n; o += (int)get_local_size(0)) {
      const int axis = (o < n0 ? 0 : (o < n01 ? 1 : 2));
      const int u = bt[axis] + o - (0 == axis ? 0 : (1 == axis ? n0 : n01));
      const double d = (u - gt[axis]) * td[15 + axis];
      g[o] = exp(-gt[3] * (d * d));
    }
  }
}

#endif /*GRID_BLOCKS_H*/
