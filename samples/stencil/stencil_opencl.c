/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_opencl.h"
#include "stencil_kernels.h"
#include <libxs/libxs_math.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_timer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_BF16)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_BF16)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_FP32)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_FP32)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_INT8)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_INT8)"
#endif


static int stencil_method_params(stencil_method_t method, int* k_steps, int* r_per_step)
{
  int result = EXIT_SUCCESS;
  switch (method) {
    case STENCIL_DIRECT:
      *k_steps = 1; *r_per_step = STENCIL_RADIUS;
      break;
    case STENCIL_COMPACT_R1:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    case STENCIL_COMPACT_R2:
      *k_steps = 2; *r_per_step = (STENCIL_RADIUS + 1) / 2;
      break;
    case STENCIL_COMPACT_FIT:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    default:
      result = EXIT_FAILURE;
      break;
  }
  return result;
}


static double stencil_fd_weight(const double* fd_weights, int radius, int dist)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    result = fd_weights[dist + radius];
  }
  return result;
}


static double stencil_compact_weight(int radius, int dist, double inv_h2)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    if (1 == radius) {
      result = (0 == dist) ? -2.0 : 1.0;
    }
    else if (2 == radius) {
      if (0 == dist) result = -5.0 / 2.0;
      else if (1 == dist || -1 == dist) result = 4.0 / 3.0;
      else result = -1.0 / 12.0;
    }
  }
  result *= inv_h2;
  return result;
}


static void stencil_store_bf16_digits(cl_ushort* dst, int stride,
                                      int ndigits, float value)
{
  libxs_bf16_t digits[4];
  int digit;
  libxs_dekker_bf16((double)value, ndigits, digits);
  for (digit = 0; digit < ndigits; ++digit) {
    dst[(long)digit * stride] = digits[digit];
  }
}


void stencil_pack_bf16s(cl_ushort* dst, const float* src, size_t n)
{
  size_t i;
  for (i = 0; i < n; ++i) {
    libxs_bf16_t digits[2];
    libxs_dekker_bf16((double)src[i], 2, digits);
    dst[i] = digits[0];
    dst[i + n] = digits[1];
  }
}


int stencil_seed_exp_buf(stencil_context_t* ctx, const float* p_host,
                         int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;
  const int nbx = ctx->nblocks[0], nby = ctx->nblocks[1], nbz = ctx->nblocks[2];
  const int total_blocks = nbx * nby * nbz;
  const int nstrips = STENCIL_N_STRIPS;
  const int n_exp = total_blocks * ctx->nterms * nstrips;
  const size_t exp_size = (size_t)n_exp * sizeof(int);
  int* exp_host = NULL;
  int ix, iy, iz, ei;
  int global_max_exp = 0;

  if (EXIT_SUCCESS == result && NULL != ctx->exp_buf[0]) {
    for (iz = 0; iz < nz; ++iz) {
      for (iy = 0; iy < ny; ++iy) {
        for (ix = 0; ix < nx; ++ix) {
          unsigned int bits;
          int e;
          memcpy(&bits, &p_host[(long)iz * ny * nx + (long)iy * nx + ix], sizeof(bits));
          e = (int)((bits >> 23) & 0xFFu);
          if (e > global_max_exp) global_max_exp = e;
        }
      }
    }
    exp_host = (int*)malloc(exp_size);
    if (NULL != exp_host) {
      for (ei = 0; ei < n_exp; ++ei) {
        exp_host[ei] = global_max_exp + STENCIL_I8_EXP_MARGIN;
      }
      result = libxstream_mem_copy_h2d(exp_host, ctx->exp_buf[0], exp_size, ctx->stream);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(exp_host, ctx->exp_buf[1], exp_size, ctx->stream);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_stream_sync(ctx->stream);
      }
      free(exp_host);
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  return result;
}


size_t stencil_blocked_size(int nbx, int nby, int nbz)
{
  return (size_t)nbx * nby * nbz * STENCIL_BLK * STENCIL_BLK * STENCIL_BLK
    * sizeof(float);
}


void stencil_pack_blocked(float* dst, const float* src,
                          int nx, int ny, int nz,
                          int nbx, int nby, int nbz)
{
  const int blk = STENCIL_BLK;
  int bz, by, bx, lz, ly, lx;
  for (bz = 0; bz < nbz; ++bz) {
    for (by = 0; by < nby; ++by) {
      for (bx = 0; bx < nbx; ++bx) {
        const long tile_base = ((long)bz * nby * nbx + (long)by * nbx + bx)
                             * (long)(blk * blk * blk);
        for (lz = 0; lz < blk; ++lz) {
          const int gz = bz * blk + lz;
          for (ly = 0; ly < blk; ++ly) {
            const int gy = by * blk + ly;
            for (lx = 0; lx < blk; ++lx) {
              const int gx = bx * blk + lx;
              float val = 0.0f;
              if (gx < nx && gy < ny && gz < nz) {
                val = src[(long)gz * ny * nx + (long)gy * nx + gx];
              }
              dst[tile_base + (long)lz * blk * blk + (long)ly * blk + lx] = val;
            }
          }
        }
      }
    }
  }
}


static int stencil_valid_strips_per_wg(int value)
{
  int result = value;
  if (1 > result) result = STENCIL_STRIPS_PER_WG;
  if (1 > result) result = 1;
  if (0 != (STENCIL_N_STRIPS % result)) result = 1;
  return result;
}


static const stencil_kernels_t* stencil_get_kernels(stencil_context_t* ctx)
{
  static libxs_registry_t* kernel_registry /*= NULL*/;
  static libxs_lock_t compile_lock /*= LIBXS_LOCK_INITIALIZER*/;
  static char base_flags[256];
  static int base_ready /*= 0*/;

  const libxstream_opencl_config_t* config = &libxstream_opencl_config;
  const libxstream_opencl_device_t* devinfo = &config->device;
  stencil_opencl_key_t key;
  stencil_kernels_t* kptr;

  if (0 == LIBXS_ATOMIC_LOAD(&base_ready, LIBXS_ATOMIC_SEQ_CST)) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, config->lock_main);
    if (0 == base_ready) {
      LIBXS_MEMZERO(base_flags);
      { const char* cmem = (EXIT_SUCCESS != libxstream_opencl_use_cmem_size(
            devinfo, STENCIL_WIDTH * sizeof(float))) ? "global" : "constant";
        LIBXS_SNPRINTF(base_flags, sizeof(base_flags),
          "-cl-fast-relaxed-math -cl-denorms-are-zero"
          " -DBLK=%d -DNDIGITS_A=%d -DNDIGITS_X=%d -DGPU=1 -DCONSTANT=%s",
          STENCIL_BLK, STENCIL_NDIGITS_A, STENCIL_NDIGITS_X, cmem);
      }
      kernel_registry = libxs_registry_create();
      LIBXS_ATOMIC_STORE(&base_ready, 1, LIBXS_ATOMIC_SEQ_CST);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, config->lock_main);
  }

  LIBXS_MEMZERO(&key);
  key.method = (int)ctx->method;
  key.k_steps = ctx->k_steps;
  key.r_per_step = ctx->r_per_step;
  key.strips_per_wg = ctx->strips_per_wg;
  key.sg = ctx->sg;
  key.grf256 = ctx->grf256;
  key.trim = ctx->trim;
  key.nterms = ctx->nterms;
  key.lu = ctx->lu;
  key.fp32 = ctx->fp32;
  key.int8 = ctx->int8;
  key.bf16s = ctx->bf16s;
  key.blocked = ctx->blocked;
  key.pml = ctx->pml;

  kptr = (stencil_kernels_t*)libxs_registry_get(
    kernel_registry, &key, sizeof(key), libxs_registry_lock(kernel_registry));

  if (NULL == kptr || NULL == kptr->stencil_apply) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, &compile_lock);
    kptr = (stencil_kernels_t*)libxs_registry_get(
      kernel_registry, &key, sizeof(key), libxs_registry_lock(kernel_registry));

    if (NULL == kptr || NULL == kptr->stencil_apply) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      char flags[LIBXSTREAM_BUFFERSIZE];
      const char* options = NULL;
      stencil_kernels_t knl;
      cl_program program = NULL;
      int ok = EXIT_SUCCESS;

      LIBXS_MEMZERO(&knl);

      { const int intel_level = (int)devinfo->intel;
        const int nv_level = (int)devinfo->nv;
        LIBXS_SNPRINTF(flags, sizeof(flags),
          "%s -DRADIUS=%d -DK_STEPS=%d -DR_PER_STEP=%d -DSTRIPS_PER_WG=%d"
          " -DSG=%d -DINTEL=%d -DNV=%d -DMETHOD=%d -DTRIM=%d -DNTERMS=%d -DLU=%d"
          " -DSTENCIL_FP32=%d -DSTENCIL_INT8=%d -DSTENCIL_BF16S=%d -DSTENCIL_BLOCKED=%d"
          " -DSTENCIL_PML=%d %s",
          base_flags, (0 == key.method) ? STENCIL_RADIUS : key.r_per_step,
          key.k_steps, key.r_per_step,
          key.strips_per_wg, key.sg, intel_level, nv_level, key.method, key.trim, key.nterms,
          key.lu, key.fp32, key.int8, key.bf16s, key.blocked, ctx->pml,
          (0 != key.fp32) ? ""
            : ((intel_level >= 2) ? "-DUSE_BF16_EXT=1" : "-DUSE_BF16=1"));
      }

      if (0 != key.grf256 && 0 != devinfo->intel && 0 == devinfo->biggrf) {
        options = "-cl-intel-256-GRF-per-thread";
      }

      if (EXIT_SUCCESS == ok) {
        const char* source;
        if (1 == key.fp32) source = OPENCL_KERNELS_SOURCE_STENCIL_FP32;
        else if (0 != key.int8) source = OPENCL_KERNELS_SOURCE_STENCIL_INT8;
        else source = OPENCL_KERNELS_SOURCE_STENCIL_BF16;
        ok = libxstream_opencl_program(0 /*source_kind*/,
          source, "stencil", flags,
          options, NULL /*try*/, NULL /*try_ok*/, NULL /*exts*/, 0,
          &program);
      }
      if (EXIT_SUCCESS == ok && 1 == key.fp32) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_direct", &knl.stencil_apply_direct);
      }
      else if (EXIT_SUCCESS == ok && 0 != key.int8) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_int8", &knl.stencil_apply);
      }
      else if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply", &knl.stencil_apply);
      }
      if (EXIT_SUCCESS == ok && 0 == key.fp32) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_tti", &knl.stencil_apply_tti);
      }

      if (EXIT_SUCCESS == ok) {
        kptr = (stencil_kernels_t*)libxs_registry_set(
          kernel_registry, &key, sizeof(key), &knl, sizeof(knl),
          libxs_registry_lock(kernel_registry));
      }

      if (2 <= ctx->verbosity || 0 > ctx->verbosity) {
        const libxs_timer_tick_t t1 = libxs_timer_tick();
        fprintf(stderr, "%s ACC/STENCIL: method=%d k=%d r=%d strips=%d sg=%d grf256=%d -> ",
          EXIT_SUCCESS == ok ? "INFO" : "ERROR",
          key.method, key.k_steps, key.r_per_step, key.strips_per_wg,
          key.sg, key.grf256);
        if (EXIT_SUCCESS == ok) {
          fprintf(stderr, "%.1f ms\n", 1E3 * libxs_timer_duration(t0, t1));
        }
        else {
          fprintf(stderr, "FAILED!\n");
        }
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, &compile_lock);
  }
  return kptr;
}


int stencil_init(stencil_context_t* ctx, int verbosity, int method_override)
{
  int result = EXIT_SUCCESS;
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  const char *const strips_env = getenv("STENCIL_STRIPS_PER_WG");
  const char *const blocked_env = getenv("STENCIL_BLOCKED");
  const char *const method_env = getenv("STENCIL_METHOD");
  const char *const bf16s_env = getenv("STENCIL_BF16S");
  const char *const int8_env = getenv("STENCIL_INT8");
  const char *const fp32_env = getenv("STENCIL_FP32");
  const char *const grf_env = getenv("STENCIL_GRF256");
  const char *const trim_env = getenv("STENCIL_TRIM");
  const char *const sg_env = getenv("STENCIL_SG");
  const char *const lu_env = getenv("STENCIL_LU");
  int method_val;

  LIBXS_MEMZERO(ctx);
  ctx->verbosity = verbosity;

  if (0 <= method_override) {
    method_val = method_override;
  }
  else {
    method_val = (NULL == method_env) ? 0 : atoi(method_env);
  }
  if (method_val < 0 || method_val > 3) method_val = 0;
  ctx->method = (stencil_method_t)method_val;

  if (EXIT_SUCCESS == result) {
    result = stencil_method_params(ctx->method, &ctx->k_steps, &ctx->r_per_step);
  }

  ctx->sg = (NULL == sg_env)
    ? ((0 < (int)devinfo->wgsize[2]) ? (int)devinfo->wgsize[2] : STENCIL_SG)
    : atoi(sg_env);

  ctx->grf256 = (NULL == grf_env) ? 0 : atoi(grf_env);

  ctx->lu = (NULL == lu_env) ? 0 : atoi(lu_env);
  ctx->fp32 = (NULL == fp32_env) ? 0 : atoi(fp32_env);
  ctx->int8 = (NULL != int8_env) ? atoi(int8_env) : 0;

  ctx->strips_per_wg = stencil_valid_strips_per_wg(
    (NULL != strips_env) ? atoi(strips_env)
    : ((0 != ctx->int8) ? 1 : STENCIL_STRIPS_PER_WG));

  ctx->trim = (NULL == trim_env)
    ? ((STENCIL_RADIUS >= 4) ? 1 : 0)
    : atoi(trim_env);
  if (ctx->trim < 0) ctx->trim = 0;
  ctx->bf16s = (NULL == bf16s_env) ? 0 : atoi(bf16s_env);
  ctx->blocked = (NULL == blocked_env) ? 0 : atoi(blocked_env);
  { const char *const pml_env = getenv("STENCIL_PML");
    ctx->pml = (NULL == pml_env) ? 0 : atoi(pml_env);
  }

  ctx->nterms = 3;
  ctx->dpas = (0 != ctx->fp32) ? 0 : ((devinfo->intel >= 2) ? 1 : 0);
  if (0 == ctx->dpas && NULL == fp32_env) ctx->fp32 = 2;

  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_create(&ctx->stream, "stencil", 0);
  }
  return result;
}


int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;

  if (NULL == ctx) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    ctx->grid_size[0] = nx;
    ctx->grid_size[1] = ny;
    ctx->grid_size[2] = nz;
    ctx->nblocks[0] = (nx + STENCIL_BLK - 1) / STENCIL_BLK;
    ctx->nblocks[1] = (ny + STENCIL_BLK - 1) / STENCIL_BLK;
    ctx->nblocks[2] = (nz + STENCIL_BLK - 1) / STENCIL_BLK;
  }
  if (EXIT_SUCCESS == result && 0 != ctx->int8) {
    const int total_blocks = ctx->nblocks[0] * ctx->nblocks[1] * ctx->nblocks[2];
    const int nstrips = STENCIL_N_STRIPS;
    const size_t exp_size = (size_t)total_blocks * ctx->nterms * nstrips * sizeof(int);
    int eb;
    for (eb = 0; eb < 2 && EXIT_SUCCESS == result; ++eb) {
      result = libxstream_mem_dev_allocate_hint(
        &ctx->exp_buf[eb], exp_size, libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_zero(ctx->exp_buf[eb], 0, exp_size, ctx->stream);
      }
    }
    ctx->exp_phase = 0;
    if (EXIT_SUCCESS == result) {
      result = libxstream_stream_sync(ctx->stream);
    }
  }

  if (EXIT_SUCCESS == result && 0 != ctx->pml) {
    const size_t grid_n = (size_t)nx * ny * nz;
    const size_t grid_bytes = grid_n * sizeof(float);
    const int pml_width = 20;
    float* eta_host = NULL;
    result = libxstream_mem_host_allocate((void**)&eta_host, grid_bytes, ctx->stream);
    if (EXIT_SUCCESS == result) {
      size_t idx;
      int ix, iy, iz;
      for (idx = 0; idx < grid_n; ++idx) eta_host[idx] = 0.0f;
      for (iz = 0; iz < nz; ++iz) {
        for (iy = 0; iy < ny; ++iy) {
          for (ix = 0; ix < nx; ++ix) {
            int d = nx;
            int dist;
            if (ix < pml_width) { dist = pml_width - ix; if (dist < d) d = dist; }
            if (ix >= nx - pml_width) { dist = ix - (nx - pml_width - 1); if (dist < d) d = dist; }
            if (iy < pml_width) { dist = pml_width - iy; if (dist < d) d = dist; }
            if (iy >= ny - pml_width) { dist = iy - (ny - pml_width - 1); if (dist < d) d = dist; }
            if (iz < pml_width) { dist = pml_width - iz; if (dist < d) d = dist; }
            if (iz >= nz - pml_width) { dist = iz - (nz - pml_width - 1); if (dist < d) d = dist; }
            if (d < nx) {
              float r = (float)d / (float)pml_width;
              eta_host[(long)iz * ny * nx + (long)iy * nx + ix] = 0.05f * r * r;
            }
          }
        }
      }
      result = libxstream_mem_dev_allocate_hint(&ctx->eta, grid_bytes,
        libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(eta_host, ctx->eta, grid_bytes, ctx->stream);
      }
      libxstream_mem_host_deallocate(eta_host, ctx->stream);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_dev_allocate_hint(&ctx->phi, grid_bytes,
        libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_zero(ctx->phi, 0, grid_bytes, ctx->stream);
      }
    }
  }

  return result;
}


int stencil_precompute_operators(stencil_context_t* ctx,
                                 const double* fd_weights, int radius)
{
  int result = EXIT_SUCCESS;
  const int blk = STENCIL_BLK;
  const int kpad = STENCIL_K_PAD;
  const int nda = STENCIL_NDIGITS_A;
  const int k_steps = ctx->k_steps;
  const int r_step = ctx->r_per_step;
  const int d_rows = blk;
  const int d_band = STENCIL_WIDTH;
  const int use_fp32 = ctx->fp32;
  const size_t d_size_bf16 = (size_t)nda * d_rows * kpad * sizeof(cl_ushort);
  const size_t d_size_fp32 = (size_t)d_rows * d_band * sizeof(float);
  const size_t d_size = (0 != use_fp32) ? d_size_fp32 : d_size_bf16;
  const double inv_h2 = -72.0 * fd_weights[radius] / 205.0;
  void* d_host = NULL;
  int dim;

  if (NULL == ctx || NULL == fd_weights || radius != STENCIL_RADIUS) {
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result && 0 != use_fp32) {
    float* d_fp32 = (float*)calloc((size_t)d_rows * d_band, sizeof(float));
    if (NULL == d_fp32) result = EXIT_FAILURE;

    if (EXIT_SUCCESS == result) {
      if (1 == k_steps) {
        int row, r;
        for (row = 0; row < blk; ++row) {
          for (r = 0; r < d_band; ++r) {
            const int dist = r - radius;
            d_fp32[row * d_band + r] = (float)stencil_fd_weight(fd_weights, radius, dist);
          }
        }
      }
      else {
        int row, r;
        for (row = 0; row < d_rows; ++row) {
          for (r = 0; r < d_band; ++r) {
            const int dist = r - r_step;
            d_fp32[row * d_band + r] = (float)stencil_compact_weight(r_step, dist, inv_h2);
          }
        }
      }
      d_host = d_fp32;
    }
  }
  else if (EXIT_SUCCESS == result) {
    cl_ushort* d_bf16 = (cl_ushort*)calloc((size_t)nda * d_rows * kpad, sizeof(cl_ushort));
    double d_mat[STENCIL_K_PAD * STENCIL_K_PAD];
    int row, col, dist;

    if (NULL == d_bf16) result = EXIT_FAILURE;

    if (EXIT_SUCCESS == result) {
      for (row = 0; row < kpad; ++row) {
        for (col = 0; col < kpad; ++col) {
          d_mat[row * kpad + col] = 0.0;
        }
      }
      if (1 == k_steps) {
        for (row = 0; row < blk; ++row) {
          for (col = 0; col < kpad; ++col) {
            dist = col - radius - row;
            d_mat[row * kpad + col] = stencil_fd_weight(fd_weights, radius, dist);
          }
        }
      }
      else {
        for (row = 0; row < d_rows; ++row) {
          for (col = 0; col < kpad; ++col) {
            dist = col - r_step - row;
            d_mat[row * kpad + col] = stencil_compact_weight(r_step, dist, inv_h2);
          }
        }
      }
      for (row = 0; row < d_rows; ++row) {
        for (col = 0; col < kpad; ++col) {
          stencil_store_bf16_digits(
            d_bf16 + row * kpad + col, d_rows * kpad, nda,
            (float)d_mat[row * kpad + col]);
        }
      }
      d_host = d_bf16;
    }
  }

  if (1 == use_fp32) {
    const size_t coeff_size = (size_t)3 * STENCIL_WIDTH * sizeof(float);
    float coeff_host[3 * (2 * STENCIL_RADIUS + 1)];
    int r, d;
    for (d = 0; d < 3; ++d) {
      for (r = 0; r < STENCIL_WIDTH; ++r) {
        coeff_host[d * STENCIL_WIDTH + r] = (float)fd_weights[r];
      }
    }
    result = libxstream_mem_dev_allocate_hint(
      (void**)&ctx->coeff, coeff_size, libxstream_opencl_mem_hint_compress);
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_copy_h2d(coeff_host, ctx->coeff, coeff_size,
                                       ctx->stream);
    }
  }
  else if (0 != ctx->int8) {
    const int kpad_i8 = STENCIL_K_PAD_I8;
    const size_t d_i8_size = (size_t)nda * d_rows * kpad_i8 * sizeof(char);
    const size_t scale_size = (size_t)nda * d_rows * sizeof(float);
    signed char* d_i8 = (signed char*)calloc((size_t)nda * d_rows * kpad_i8, sizeof(char));
    float* d_scale = (float*)calloc((size_t)nda * d_rows, sizeof(float));
    if (NULL == d_i8 || NULL == d_scale) result = EXIT_FAILURE;

    if (EXIT_SUCCESS == result) {
      double d_mat_i8[STENCIL_K_PAD * STENCIL_K_PAD];
      int row, col, sa;

      for (row = 0; row < kpad; ++row) {
        for (col = 0; col < kpad; ++col) {
          d_mat_i8[row * kpad + col] = 0.0;
        }
      }
      if (1 == k_steps) {
        for (row = 0; row < blk; ++row) {
          for (col = 0; col < kpad; ++col) {
            int dist = col - radius - row;
            d_mat_i8[row * kpad + col] = stencil_fd_weight(fd_weights, radius, dist);
          }
        }
      }
      else {
        for (row = 0; row < d_rows; ++row) {
          for (col = 0; col < kpad; ++col) {
            int dist = col - r_step - row;
            d_mat_i8[row * kpad + col] = stencil_compact_weight(r_step, dist, inv_h2);
          }
        }
      }

      for (row = 0; row < d_rows; ++row) {
        double row_max = 0.0;
        int max_exp_row;
        for (col = 0; col < kpad; ++col) {
          double av = d_mat_i8[row * kpad + col];
          if (av < 0) av = -av;
          if (av > row_max) row_max = av;
        }
        if (row_max > 0.0) {
          unsigned int bits;
          float fmax = (float)row_max;
          memcpy(&bits, &fmax, sizeof(bits));
          max_exp_row = (int)((bits >> 23) & 0xFFu);
        }
        else {
          max_exp_row = 0;
        }
        for (sa = 0; sa < nda; ++sa) {
          d_scale[sa * d_rows + row] = (float)(row_max > 0.0
            ? ldexp(1.0, max_exp_row - 127 - 23 + 7 * sa) : 0.0);
          for (col = 0; col < kpad_i8; ++col) {
            signed char digit = 0;
            if (col < kpad) {
              float val = (float)d_mat_i8[row * kpad + col];
              unsigned int vbits;
              int e, sign_bit, shift;
              unsigned int mantissa;
              memcpy(&vbits, &val, sizeof(vbits));
              e = (int)((vbits >> 23) & 0xFFu);
              sign_bit = (int)(vbits >> 31);
              mantissa = (0 != e) ? ((vbits & 0x7FFFFFu) | 0x800000u) : 0;
              shift = max_exp_row - e;
              if (shift > 0 && shift < 32) mantissa >>= shift;
              else if (shift >= 32) mantissa = 0;
              { int high = 23 - 7 * sa;
                int low = (high - 6 > 0) ? (high - 6) : 0;
                int width = high - low + 1;
                if (width > 0 && high >= 0) {
                  digit = (signed char)((mantissa >> low) & ((1U << width) - 1U));
                }
                if (sign_bit) digit = (signed char)(-digit);
              }
            }
            d_i8[(long)sa * d_rows * kpad_i8 + (long)row * kpad_i8 + col] = digit;
          }
        }
      }
    }

    for (dim = 0; dim < 3 && EXIT_SUCCESS == result; ++dim) {
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk[dim], d_i8_size, libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(d_i8, ctx->dk[dim], d_i8_size, ctx->stream);
      }
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk_scale, scale_size, libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(d_scale, ctx->dk_scale, scale_size, ctx->stream);
      }
    }

    free(d_scale);
    free(d_i8);
  }
  else {
    for (dim = 0; dim < 3 && EXIT_SUCCESS == result; ++dim) {
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk[dim], d_size, libxstream_opencl_mem_hint_compress);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(d_host, ctx->dk[dim], d_size,
                                         ctx->stream);
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_sync(ctx->stream);
  }

  free(d_host);
  return result;
}


int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_cur, void* p_old, void* p_new,
                            void* vel, float dt2, int nterms)
{
  int result = EXIT_SUCCESS;
  const stencil_kernels_t* knl = stencil_get_kernels(ctx);
  const int total_blocks = ctx->nblocks[0] * ctx->nblocks[1] * ctx->nblocks[2];
  const libxstream_opencl_stream_t* str =
    (const libxstream_opencl_stream_t*)ctx->stream;
  const int nx = ctx->grid_size[0];
  const int ny = ctx->grid_size[1];
  const int nz = ctx->grid_size[2];
  const int nbx = ctx->nblocks[0];
  const int nby = ctx->nblocks[1];
  size_t global_apply[3], local_apply[3];

  if (NULL == knl) result = EXIT_FAILURE;

  if (EXIT_SUCCESS == result && 1 == ctx->fp32 && NULL != knl->stencil_apply_direct) {
    size_t global_direct[3], local_direct[3];
    cl_int i = 0;
    local_direct[0] = 32;
    local_direct[1] = 8;
    local_direct[2] = 1;
    global_direct[0] = ((size_t)nx + 31) & ~(size_t)31;
    global_direct[1] = ((size_t)ny + 7) & ~(size_t)7;
    global_direct[2] = (size_t)((nz + STENCIL_BLK - 1) / STENCIL_BLK);
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_cur));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_old));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_new));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, vel));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, ctx->coeff));
    if (0 != ctx->pml) {
      const float h = 10.0f;
      const float hdx_2 = 0.5f / h;
      const float hdy_2 = 0.5f / h;
      const float hdz_2 = 0.5f / h;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, ctx->eta));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, ctx->phi));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(float), &hdx_2));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(float), &hdy_2));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(float), &hdz_2));
    }
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(float), &dt2));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &nx));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &ny));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &nz));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->stencil_apply_direct,
      3, NULL, global_direct, local_direct, 0, NULL, NULL));
  }
  else if (EXIT_SUCCESS == result) {
    global_apply[0] = (size_t)ctx->nblocks[0] * ctx->sg;
    global_apply[1] = (size_t)ctx->nblocks[1] * STENCIL_M_TILES;
    global_apply[2] = (size_t)ctx->nblocks[2] * (STENCIL_N_STRIPS / ctx->strips_per_wg);
    local_apply[0] = (size_t)ctx->sg;
    local_apply[1] = STENCIL_M_TILES;
    local_apply[2] = 1;

    { const int nterms_iso = (nterms <= 3) ? nterms : 3;
      cl_int i = 0;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[0]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[1]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[2]));
      if (0 != ctx->int8) {
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk_scale));
      }
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_cur));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_old));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_new));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, vel));
      if (0 != ctx->int8) {
        const int rd = ctx->exp_phase;
        const int wr = 1 - rd;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->exp_buf[rd]));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->exp_buf[wr]));
        ctx->exp_phase = wr;
      }
      if (0 != ctx->pml) {
        const float h = 10.0f;
        const float hdx_2 = 0.5f / h;
        const float hdy_2 = 0.5f / h;
        const float hdz_2 = 0.5f / h;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->eta));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->phi));
        if (0 == ctx->int8) {
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_cur));
        }
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(float), &hdx_2));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(float), &hdy_2));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(float), &hdz_2));
      }
      if (0 == ctx->int8) {
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nterms_iso));
      }
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(float), &dt2));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nx));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &ny));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nz));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nbx));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nby));

      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->stencil_apply,
        3, NULL, global_apply, local_apply, 0, NULL, NULL));
    }

    if (nterms > 3 && 0 == ctx->fp32 && EXIT_SUCCESS == result) {
      static const int cross_pairs[6][2] = {
        {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}
      };
      size_t global_tti[3], local_tti[3];
      int pair;
      global_tti[0] = (size_t)total_blocks * ctx->sg;
      global_tti[1] = STENCIL_M_TILES;
      global_tti[2] = STENCIL_N_STRIPS;
      local_tti[0] = (size_t)ctx->sg;
      local_tti[1] = STENCIL_M_TILES;
      local_tti[2] = 1;
      for (pair = 0; pair < 6 && EXIT_SUCCESS == result; ++pair) {
        const int dim_i = cross_pairs[pair][0];
        const int dim_j = cross_pairs[pair][1];
        cl_int i = 0;
        int ys = STENCIL_N_TOTAL;
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_tti, i++, ctx->dk[dim_i]));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_tti, i++, ctx->dk[dim_j]));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_tti, i++, p_cur));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_tti, i++, p_new));
        CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_tti, i++, vel));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &ys));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &dim_j));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &nx));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &ny));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &nz));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &nbx));
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply_tti, i++, sizeof(int), &nby));

        CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->stencil_apply_tti,
          3, NULL, global_tti, local_tti, 0, NULL, NULL));
      }
    }
  } /* else if (EXIT_SUCCESS == result) */

  return result;
}


void stencil_finalize(stencil_context_t* ctx)
{
  int dim;
  if (NULL != ctx) {
    for (dim = 0; dim < 3; ++dim) {
      if (NULL != ctx->dk[dim]) libxstream_mem_dev_deallocate_hint(ctx->dk[dim]);
    }
    if (NULL != ctx->dk_scale) libxstream_mem_dev_deallocate_hint(ctx->dk_scale);
    if (NULL != ctx->exp_buf[0]) libxstream_mem_dev_deallocate_hint(ctx->exp_buf[0]);
    if (NULL != ctx->exp_buf[1]) libxstream_mem_dev_deallocate_hint(ctx->exp_buf[1]);
    if (NULL != ctx->coeff) libxstream_mem_dev_deallocate_hint(ctx->coeff);
    if (NULL != ctx->eta) libxstream_mem_dev_deallocate_hint(ctx->eta);
    if (NULL != ctx->phi) libxstream_mem_dev_deallocate_hint(ctx->phi);
    if (NULL != ctx->stream) libxstream_stream_destroy(ctx->stream);
    LIBXS_MEMZERO(ctx);
  }
}
