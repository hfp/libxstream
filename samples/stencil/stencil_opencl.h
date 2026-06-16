/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_OPENCL_H
#define STENCIL_OPENCL_H

#include <libxstream/libxstream.h>
#include <libxstream/libxstream_opencl.h>

#define STENCIL_BLK 32
#define STENCIL_RADIUS 4
#define STENCIL_WIDTH (2 * STENCIL_RADIUS + 1)
#define STENCIL_NDIGITS_A 2
#define STENCIL_NDIGITS_X 3
#define STENCIL_K_PAD 32
#define STENCIL_N_TOTAL (STENCIL_BLK * STENCIL_BLK)
#define STENCIL_N_PAD STENCIL_N_TOTAL
#define STENCIL_XMX_M 8
#define STENCIL_XMX_N 16
#define STENCIL_M_TILES (STENCIL_BLK / STENCIL_XMX_M)
#define STENCIL_N_STRIPS (STENCIL_N_TOTAL / STENCIL_XMX_N)
#define STENCIL_SG 16


typedef struct {
  cl_kernel preprocess_x;
  cl_kernel stencil_apply;
  cl_kernel stencil_apply_tti;
  cl_mem dk[3];
  libxstream_stream_t* stream;
  int nblocks[3];
  int grid_size[3];
  int verbosity;
} stencil_context_t;


int stencil_init(stencil_context_t* ctx, int verbosity);
int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz);
int stencil_precompute_operators(stencil_context_t* ctx,
                                 const double* fd_weights, int radius);
int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_in, void* y_out, void* vel,
                            int nterms);
void stencil_finalize(stencil_context_t* ctx);

#endif /*STENCIL_OPENCL_H*/
