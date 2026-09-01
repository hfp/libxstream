/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_OPENCL_H
#define STENCIL_OPENCL_H

#if defined(STENCIL_CPU) && (0 < STENCIL_CPU)
# include "stencil_hostmem.h"
#else
# include <libxstream/libxstream_opencl.h>
#endif
#include <libxs/libxs_reg.h>

#define STENCIL_BLK 32
#define STENCIL_RADIUS 4
#define STENCIL_WIDTH (2 * STENCIL_RADIUS + 1)
#define STENCIL_NDIGITS_A_DEFAULT 1
#define STENCIL_NDIGITS_A_MAX 3
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
#define STENCIL_I8_EXP_MARGIN 1
#define STENCIL_SG 16
#define STENCIL_PML_ETA_PEAK 0.0199


typedef enum {
  STENCIL_DIRECT     = 0,
  STENCIL_COMPACT_R1 = 1,
  STENCIL_COMPACT_R2 = 2,
  STENCIL_COMPACT_FIT = 3
} stencil_method_t;

typedef enum {
  STENCIL_I8_OP_IMPLICIT = 0,
  STENCIL_I8_OP_BANDED   = 1
} stencil_i8_op_t;

typedef struct {
  int method;
  int k_steps;
  int r_per_step;
  int strips_per_wg;
  int sg;
  int nterms;
  int grid_key;
  int fp32_wg;
  int r_gather;
  int ndigits_x;
  int i8_op;
  signed char grf256;
  signed char trim;
  signed char lu;
  signed char bf16;
  signed char int8;
  signed char bf16s;
  signed char fp16;
  signed char blocked;
  signed char layout;
  signed char pml;
  signed char fp32_block_io;
  signed char fp32_sblock;
  signed char ndigits_a;
} stencil_opencl_key_t;

#if !defined(STENCIL_CPU) || (0 >= STENCIL_CPU)
typedef struct {
  cl_kernel stencil_apply;
  cl_kernel stencil_apply_tti;
  cl_kernel stencil_apply_direct;
} stencil_kernels_t;
#endif

typedef struct {
  void* dk[3];
  void* dk_scale;
  void* exp_buf[2];
  int exp_phase;
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
  int bf16;
  int int8;
  int i8_op;
  int r_gather;
  int ndigits_a;
  int ndigits_x;
  int bf16s;
  int fp16;
  int blocked;
  int layout;
  int halo[3];
  int pml;
  int hint;
  void* eta;
  void* phi;
  int verbosity;
} stencil_context_t;


int stencil_init(stencil_context_t* ctx, int verbosity, int method_override);
int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz);
int stencil_precompute_operators(stencil_context_t* ctx,
                                 const double* fd_weights, int radius);
/**
 * Advances one time step. p_cur holds the current wavefield, p_old the previous
 * one on entry and the next one on exit: the update is in-place, which is why
 * the kernels take a single buffer for both roles.
 */
int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_cur, void* p_old,
                            void* vel, float dt2, float dh, int nterms);
void stencil_finalize(stencil_context_t* ctx);

int stencil_seed_exp_buf(stencil_context_t* ctx, const float* p_host,
                         int nx, int ny, int nz);
size_t stencil_blocked_size(int nbx, int nby, int nbz);
void stencil_pack_blocked(float* dst, const float* src,
                          int nx, int ny, int nz,
                          int nbx, int nby, int nbz);
/* ndigits: 0 = single IEEE FP16 value, 1 or 2 = that many Dekker BF16 limbs. */
void stencil_pack_bf16s(unsigned short* dst, const float* src, size_t n,
                        int ndigits);
void stencil_pack_bf16s_blocked(unsigned short* dst, const float* src,
                                int nx, int ny, int nz,
                                int nbx, int nby, int nbz, int ndigits);
void stencil_pack_bf16s_zyx(unsigned short* dst, const float* src,
                            int nx, int ny, int nz,
                            int hx, int hy, int hz, int ndigits);
void stencil_unpack_bf16s(float* dst, const unsigned short* src, size_t n,
                          int ndigits);

/**
 * Runs the FP32 device kernel on the host: stencil_cpu.c translates
 * kernels/stencil_fp32.cl with the ordinary C compiler. The kernel is
 * specialized at build time, so a grid or term count that disagrees with the
 * compiled-in configuration returns EXIT_FAILURE rather than wrong results.
 */
int stencil_cpu_apply_direct(const float* p_grid, float* p_old,
                             const float* vel, const float* coeff, float dt2,
                             int nx, int ny, int nz, int nterms);

#endif /*STENCIL_OPENCL_H*/
