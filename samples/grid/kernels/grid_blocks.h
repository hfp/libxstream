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
