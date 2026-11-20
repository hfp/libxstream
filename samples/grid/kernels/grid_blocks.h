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
static inline void grid_load_pab(__global const double* restrict blocks, __global const double* restrict sphi,
  __global const int* restrict ti, double* pab, double* w)
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
static inline void grid_store_hab(__global double* restrict blocks, __global const double* restrict sphi,
  __global const int* restrict ti, __global const double* restrict hab, double* w)
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


/* Gaussian of a task along each axis of its box (orthorhombic): the rows need no exp */
__kernel void grid_gauss(const int ntasks, __global const double* restrict geo, __global const int* restrict box,
  __global const double* restrict tdbl, __global const int* restrict gofs, __global double* restrict gauss)
{
  const int t = (int)get_global_id(0);
  if (t < ntasks) {
    __global const double* const gt = geo + GRID_OPENCL_NG * t;
    __global const int* const bt = box + GRID_OPENCL_NB * t;
    __global const double* const td = tdbl + GRID_OPENCL_ND * t;
    __global double* const g = gauss + gofs[t];
    int axis, u, o = 0;
    for (axis = 0; axis < 3; ++axis) {
      for (u = bt[axis]; u <= bt[3 + axis]; ++u) {
        const double d = (u - gt[axis]) * td[15 + axis];
        g[o++] = exp(-gt[3] * (d * d));
      }
    }
  }
}

#endif /*GRID_BLOCKS_H*/
