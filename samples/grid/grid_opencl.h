/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef GRID_OPENCL_H
#define GRID_OPENCL_H

/* per-task records shared by grid_opencl.c and the kernels */
#define GRID_OPENCL_NI 14
#define GRID_OPENCL_ND 15
#define GRID_OPENCL_NG 7
#define GRID_OPENCL_NB 6

/* CP2K's grid_prepare_pab.h expects the includer to define the store */
typedef struct {
  int n1;
  double* data;
} cab_store;

/* first image of the unwrapped range [lo, ...] that maps to grid index g */
static inline int grid_image(const int g, const int lo, const int n)
{
  int u = g; /* no integer division: boxes start within a few periods of the grid */
  while (u < lo) u += n;
  while (lo <= u - n) u -= n;
  return u;
}

#endif /*GRID_OPENCL_H*/
