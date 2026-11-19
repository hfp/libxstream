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

/* per-task records shared by grid_opencl.c and kernels/collocate.cl */
#define GRID_OPENCL_NI 13
#define GRID_OPENCL_ND 12
#define GRID_OPENCL_NG 5
#define GRID_OPENCL_NB 6

/* CP2K's grid_prepare_pab.h expects the includer to define the store */
typedef struct {
  int n1;
  double* data;
} cab_store;

#endif /*GRID_OPENCL_H*/
