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

#include <libxstream/libxstream_opencl.h>
#include <libxs/libxs_reg.h>

#define STENCIL_BLK 32
#define STENCIL_RADIUS 4
#define STENCIL_WIDTH (2 * STENCIL_RADIUS + 1)
#define STENCIL_NDIGITS_A 2
#define STENCIL_NDIGITS_X 3
#define STENCIL_ALIGN16(VALUE) (((VALUE) + 15) & ~15)
#define STENCIL_K_BASE (STENCIL_BLK + 2 * STENCIL_RADIUS)
#define STENCIL_K_PAD STENCIL_ALIGN16(STENCIL_K_BASE)
#define STENCIL_N_TOTAL (STENCIL_BLK * STENCIL_BLK)
#define STENCIL_N_PAD STENCIL_N_TOTAL
#define STENCIL_XMX_M 8
#define STENCIL_XMX_N 16
#define STENCIL_M_TILES (STENCIL_BLK / STENCIL_XMX_M)
#define STENCIL_N_STRIPS (STENCIL_N_TOTAL / STENCIL_XMX_N)
#define STENCIL_STRIPS_PER_WG 2
#define STENCIL_N_STRIP_GROUPS (STENCIL_N_STRIPS / STENCIL_STRIPS_PER_WG)
#define STENCIL_K_PAD_I8 64
#define STENCIL_SG 16


typedef enum {
  STENCIL_DIRECT    = 0,
  STENCIL_STAGED_R1 = 1,
  STENCIL_STAGED_R2 = 2,
  STENCIL_STAGED_FIT = 3
} stencil_method_t;

typedef struct {
  int method;
  int k_steps;
  int r_per_step;
  int strips_per_wg;
  int sg;
  int grf256;
  int trim;
  int nterms;
  int lu;
  int fp32;
  int int8;
  int bf16s;
  int blocked;
} stencil_opencl_key_t;

typedef struct {
  cl_kernel stencil_apply;
  cl_kernel stencil_apply_tti;
  cl_kernel stencil_apply_direct;
} stencil_kernels_t;

typedef struct {
  void* dk[3];
  void* dk_scale;
  void* coeff;
  libxstream_stream_t* stream;
  int nblocks[3];
  int grid_size[3];
  stencil_method_t method;
  int k_steps;
  int r_per_step;
  int strips_per_wg;
  int sg;
  int grf256;
  int trim;
  int nterms;
  int lu;
  int fp32;
  int int8;
  int bf16s;
  int blocked;
  int dpas;
  int verbosity;
} stencil_context_t;


int stencil_init(stencil_context_t* ctx, int verbosity, int method_override);
int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz);
int stencil_precompute_operators(stencil_context_t* ctx,
                                 const double* fd_weights, int radius);
int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_cur, void* p_old, void* p_new,
                            void* vel, float dt2, int nterms);
void stencil_finalize(stencil_context_t* ctx);

size_t stencil_blocked_size(int nbx, int nby, int nbz);
void stencil_pack_blocked(float* dst, const float* src,
                          int nx, int ny, int nz,
                          int nbx, int nby, int nbz);
void stencil_pack_bf16s(unsigned short* dst, const float* src, size_t n);

#endif /*STENCIL_OPENCL_H*/
