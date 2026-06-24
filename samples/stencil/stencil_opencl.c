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


static int stencil_method_params(stencil_method_t method, int* k_steps, int* r_per_step)
{
  int result = EXIT_SUCCESS;
  switch (method) {
    case STENCIL_DIRECT:
      *k_steps = 1; *r_per_step = STENCIL_RADIUS;
      break;
    case STENCIL_STAGED_R1:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    case STENCIL_STAGED_R2:
      *k_steps = 2; *r_per_step = (STENCIL_RADIUS + 1) / 2;
      break;
    case STENCIL_STAGED_FIT:
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
      { const char* cmem = (EXIT_SUCCESS != libxstream_opencl_use_cmem(
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
  key.bf16s = ctx->bf16s;
  key.blocked = ctx->blocked;

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
        LIBXS_SNPRINTF(flags, sizeof(flags),
          "%s -DRADIUS=%d -DK_STEPS=%d -DR_PER_STEP=%d -DSTRIPS_PER_WG=%d"
          " -DSG=%d -DINTEL=%d -DMETHOD=%d -DTRIM=%d -DNTERMS=%d -DLU=%d"
          " -DSTENCIL_FP32=%d -DSTENCIL_BF16S=%d -DSTENCIL_BLOCKED=%d %s",
          base_flags, (0 == key.method) ? STENCIL_RADIUS : key.r_per_step,
          key.k_steps, key.r_per_step,
          key.strips_per_wg, key.sg, intel_level, key.method, key.trim, key.nterms,
          key.lu, key.fp32, key.bf16s, key.blocked,
          (0 != key.fp32) ? ""
            : ((intel_level >= 2) ? "-DUSE_BF16_EXT=1" : "-DUSE_BF16=1"));
      }

      if (0 != key.grf256 && 0 != devinfo->intel && 0 == devinfo->biggrf) {
        options = "-cl-intel-256-GRF-per-thread";
      }

      if (EXIT_SUCCESS == ok) {
        const char* source = (2 == key.fp32)
          ? OPENCL_KERNELS_SOURCE_STENCIL_FP32
          : OPENCL_KERNELS_SOURCE_STENCIL_BF16;
        ok = libxstream_opencl_program(0 /*source_kind*/,
          source, "stencil", flags,
          options, NULL /*try*/, NULL /*try_ok*/, NULL /*exts*/, 0,
          &program);
      }
      if (EXIT_SUCCESS == ok && 2 == key.fp32) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_direct", &knl.stencil_apply_direct);
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
  const char* method_env = getenv("STENCIL_METHOD");
  const char* sg_env = getenv("STENCIL_SG");
  const char* strips_env = getenv("STENCIL_STRIPS_PER_WG");
  const char* grf_env = getenv("STENCIL_GRF256");
  const char* trim_env = getenv("STENCIL_TRIM");
  const char* lu_env = getenv("STENCIL_LU");
  const char* fp32_env = getenv("STENCIL_FP32");
  const char* bf16s_env = getenv("STENCIL_BF16S");
  const char* blocked_env = getenv("STENCIL_BLOCKED");
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

  ctx->strips_per_wg = stencil_valid_strips_per_wg(
    (NULL == strips_env) ? STENCIL_STRIPS_PER_WG : atoi(strips_env));

  ctx->grf256 = (NULL == grf_env) ? 0 : atoi(grf_env);

  ctx->trim = (NULL == trim_env)
    ? ((STENCIL_RADIUS >= 4) ? 1 : 0)
    : atoi(trim_env);
  if (ctx->trim < 0) ctx->trim = 0;

  ctx->lu = (NULL == lu_env) ? 0 : atoi(lu_env);
  ctx->fp32 = (NULL == fp32_env) ? 0 : atoi(fp32_env);
  ctx->bf16s = (NULL == bf16s_env) ? 0 : atoi(bf16s_env);
  ctx->blocked = (NULL == blocked_env) ? 0 : atoi(blocked_env);

  ctx->nterms = 3;
  ctx->dpas = (0 != ctx->fp32) ? 0 : ((devinfo->intel >= 2) ? 1 : 0);

  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_create(&ctx->stream, "stencil", 0);
  }
  return result;
}


int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;

  if (NULL == ctx) return EXIT_FAILURE;
  ctx->grid_size[0] = nx;
  ctx->grid_size[1] = ny;
  ctx->grid_size[2] = nz;
  ctx->nblocks[0] = (nx + STENCIL_BLK - 1) / STENCIL_BLK;
  ctx->nblocks[1] = (ny + STENCIL_BLK - 1) / STENCIL_BLK;
  ctx->nblocks[2] = (nz + STENCIL_BLK - 1) / STENCIL_BLK;

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
    return EXIT_FAILURE;
  }

  if (0 != use_fp32) {
    float* d_fp32 = (float*)calloc((size_t)d_rows * d_band, sizeof(float));
    if (NULL == d_fp32) return EXIT_FAILURE;

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
  else {
    cl_ushort* d_bf16 = (cl_ushort*)calloc((size_t)nda * d_rows * kpad, sizeof(cl_ushort));
    double d_mat[STENCIL_K_PAD * STENCIL_K_PAD];
    int row, col, dist;

    if (NULL == d_bf16) return EXIT_FAILURE;

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

  if (2 == use_fp32) {
    const size_t coeff_size = (size_t)STENCIL_WIDTH * sizeof(float);
    float coeff_host[2 * STENCIL_RADIUS + 1];
    int r;
    for (r = 0; r < STENCIL_WIDTH; ++r) {
      coeff_host[r] = (float)fd_weights[r];
    }
    result = libxstream_mem_dev_allocate_hint(
      (void**)&ctx->coeff, coeff_size, libxstream_opencl_mem_hint_compress);
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_copy_h2d(coeff_host, ctx->coeff, coeff_size,
                                       ctx->stream);
    }
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

  if (NULL == knl) return EXIT_FAILURE;

  if (2 == ctx->fp32 && NULL != knl->stencil_apply_direct) {
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
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(float), &dt2));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &nx));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &ny));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply_direct, i++, sizeof(int), &nz));
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->stencil_apply_direct,
      3, NULL, global_direct, local_direct, 0, NULL, NULL));
    return result;
  }

  global_apply[0] = (size_t)ctx->nblocks[0] * ctx->sg;
  global_apply[1] = (size_t)ctx->nblocks[1] * STENCIL_M_TILES;
  global_apply[2] = (size_t)ctx->nblocks[2] * (STENCIL_N_STRIPS / ctx->strips_per_wg);
  local_apply[0] = (size_t)ctx->sg;
  local_apply[1] = STENCIL_M_TILES;
  local_apply[2] = 1;

  { cl_int i = 0;
    const int nterms_iso = (nterms <= 3) ? nterms : 3;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[0]));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[1]));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[2]));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_cur));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_old));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, p_new));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, vel));
    CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &nterms_iso));
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

  return result;
}


void stencil_finalize(stencil_context_t* ctx)
{
  int dim;
  if (NULL == ctx) return;
  for (dim = 0; dim < 3; ++dim) {
    if (NULL != ctx->dk[dim]) libxstream_mem_dev_deallocate_hint(ctx->dk[dim]);
  }
  if (NULL != ctx->coeff) libxstream_mem_dev_deallocate_hint(ctx->coeff);
  if (NULL != ctx->stream) libxstream_stream_destroy(ctx->stream);
  LIBXS_MEMZERO(ctx);
}
