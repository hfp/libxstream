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
#include <libxs/libxs_mem.h>
#include <libxs/libxs_timer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_BF16)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_BF16)"
#endif


static int stencil_method_params(stencil_method_t method, int* k_steps, int* r_per_step)
{
  int result = EXIT_SUCCESS;
  switch (method) {
    case STENCIL_SPARSE:
      *k_steps = 1; *r_per_step = STENCIL_RADIUS;
      break;
    case STENCIL_DENSE:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    case STENCIL_HYBRID:
      *k_steps = 2; *r_per_step = (STENCIL_RADIUS + 1) / 2;
      break;
    case STENCIL_BEST:
      *k_steps = STENCIL_RADIUS; *r_per_step = 1;
      break;
    default:
      result = EXIT_FAILURE;
      break;
  }
  return result;
}


static const stencil_kernels_t* stencil_get_kernels(stencil_context_t* ctx)
{
  static libxs_registry_t* kernel_registry /*= NULL*/;
  static libxs_lock_t compile_lock /*= LIBXS_LOCK_INITIALIZER*/;
  static char base_flags[LIBXSTREAM_BUFFERSIZE];
  static int base_ready /*= 0*/;

  const libxstream_opencl_config_t* config = &libxstream_opencl_config;
  const libxstream_opencl_device_t* devinfo = &config->device;
  stencil_opencl_key_t key;
  stencil_kernels_t* kptr;

  if (0 == LIBXS_ATOMIC_LOAD(&base_ready, LIBXS_ATOMIC_SEQ_CST)) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, config->lock_main);
    if (0 == base_ready) {
      LIBXS_MEMZERO(base_flags);
      LIBXS_SNPRINTF(base_flags, sizeof(base_flags),
        "-cl-fast-relaxed-math -cl-denorms-are-zero"
        " -DBLK=%d -DNDIGITS_A=%d -DNDIGITS_X=%d -DGPU=1",
        STENCIL_BLK, STENCIL_NDIGITS_A, STENCIL_NDIGITS_X);
      kernel_registry = libxs_registry_create();
      LIBXS_ATOMIC_STORE(&base_ready, 1, LIBXS_ATOMIC_SEQ_CST);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, config->lock_main);
  }

  LIBXS_MEMZERO(&key);
  key.method = (int)ctx->method;
  key.k_steps = ctx->k_steps;
  key.r_per_step = ctx->r_per_step;
  key.sg = ctx->sg;
  key.grf256 = ctx->grf256;

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
          "%s -DRADIUS=%d -DK_STEPS=%d -DR_PER_STEP=%d -DSG=%d -DINTEL=%d"
          " -DMETHOD=%d %s",
          base_flags, key.r_per_step, key.k_steps, key.r_per_step,
          key.sg, intel_level, key.method,
          (intel_level >= 2) ? "-DUSE_BF16_EXT=1" : "-DUSE_BF16=1");
      }

      if (0 != key.grf256 && 0 != devinfo->intel && 0 == devinfo->biggrf) {
        options = "-cl-intel-256-GRF-per-thread";
      }

      if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_program(0 /*source_kind*/,
          OPENCL_KERNELS_SOURCE_STENCIL_BF16, "stencil", flags,
          options, NULL /*try*/, NULL /*try_ok*/, NULL /*exts*/, 0,
          &program);
      }
      if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_kernel_query(program, "preprocess_x", &knl.preprocess_x);
      }
      if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply", &knl.stencil_apply);
      }
      if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_tti", &knl.stencil_apply_tti);
      }

      if (EXIT_SUCCESS == ok) {
        kptr = (stencil_kernels_t*)libxs_registry_set(
          kernel_registry, &key, sizeof(key), &knl, sizeof(knl),
          libxs_registry_lock(kernel_registry));
      }

      if (2 <= ctx->verbosity || 0 > ctx->verbosity) {
        const libxs_timer_tick_t t1 = libxs_timer_tick();
        fprintf(stderr, "%s ACC/STENCIL: method=%d k=%d r=%d sg=%d grf256=%d -> ",
          EXIT_SUCCESS == ok ? "INFO" : "ERROR",
          key.method, key.k_steps, key.r_per_step, key.sg, key.grf256);
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
  const char* grf_env = getenv("STENCIL_GRF256");
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

  ctx->grf256 = (NULL == grf_env)
    ? devinfo->biggrf
    : atoi(grf_env);

  ctx->dpas = (devinfo->intel >= 2) ? 1 : 0;

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

  if (ctx->k_steps > 1 && EXIT_SUCCESS == result) {
    const size_t grid_bytes = (size_t)nx * ny * nz * sizeof(float);
    result = libxstream_mem_allocate(&ctx->cascade_a, grid_bytes);
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_allocate(&ctx->cascade_b, grid_bytes);
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
  const size_t d_size = (size_t)nda * blk * kpad * sizeof(cl_ushort);
  cl_ushort* d_host = NULL;
  int dim;

  if (NULL == ctx || NULL == fd_weights || radius != STENCIL_RADIUS) {
    return EXIT_FAILURE;
  }

  d_host = (cl_ushort*)calloc((size_t)nda * blk * kpad, sizeof(cl_ushort));
  if (NULL == d_host) return EXIT_FAILURE;

  {
    const double* weights;
    double sub_w[2 * STENCIL_RADIUS + 1];
    int eff_radius;
    int row, col, s;

    if (1 == k_steps) {
      weights = fd_weights;
      eff_radius = radius;
    }
    else {
      int i;
      LIBXS_MEMZERO(sub_w);
      sub_w[r_step] = -2.0;
      for (i = 1; i <= r_step; ++i) {
        sub_w[r_step - i] = 1.0;
        sub_w[r_step + i] = 1.0;
      }
      weights = sub_w;
      eff_radius = r_step;
    }

    for (row = 0; row < blk; ++row) {
      for (col = 0; col < blk; ++col) {
        const int dist = col - row;
        float val = 0.0f;
        float residual;
        unsigned int bits;

        if (dist >= -eff_radius && dist <= eff_radius) {
          val = (float)weights[dist + eff_radius];
        }

        residual = val;
        for (s = 0; s < nda; ++s) {
          unsigned int rounded, bf32;
          cl_ushort bf;
          float bf_f;
          memcpy(&bits, &residual, sizeof(bits));
          rounded = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
          bf = (cl_ushort)(rounded >> 16);
          bf32 = (unsigned int)bf << 16;
          memcpy(&bf_f, &bf32, sizeof(bf_f));
          d_host[s * blk * kpad + row * kpad + col] = bf;
          residual -= bf_f;
        }
      }
    }
  }

  for (dim = 0; dim < 3 && EXIT_SUCCESS == result; ++dim) {
    result = libxstream_mem_allocate((void**)&ctx->dk[dim], d_size);
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_copy_h2d(d_host, ctx->dk[dim], d_size,
                                       ctx->stream);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_sync(ctx->stream);
  }

  free(d_host);
  return result;
}


int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_in, void* y_out, void* vel,
                            int nterms)
{
  int result = EXIT_SUCCESS;
  const stencil_kernels_t* knl = stencil_get_kernels(ctx);
  const int total_blocks = ctx->nblocks[0] * ctx->nblocks[1] * ctx->nblocks[2];
  const int k_steps = ctx->k_steps;
  const size_t x_size = (size_t)STENCIL_NDIGITS_X * STENCIL_K_PAD
                       * STENCIL_N_PAD * sizeof(cl_ushort);
  const libxstream_opencl_stream_t* str =
    (const libxstream_opencl_stream_t*)ctx->stream;
  void* xk = NULL;
  int dim;

  if (NULL == knl) return EXIT_FAILURE;

  result = libxstream_mem_allocate(&xk, x_size);

  for (dim = 0; dim < nterms && EXIT_SUCCESS == result; ++dim) {
    const int super_m = STENCIL_BLK + 2 * ctx->r_per_step;
    const int super_n = STENCIL_BLK + 2 * ctx->r_per_step;
    const int super_p = STENCIL_BLK + 2 * ctx->r_per_step;
    int kstep;

    for (kstep = 0; kstep < k_steps && EXIT_SUCCESS == result; ++kstep) {
      /* mode: 0=write intermediate, 1=write Y=v^2*res, 2=accumulate Y+=res */
      const int is_final = (kstep == k_steps - 1) ? 1 : 0;
      const int mode = (0 == is_final) ? 0 : ((0 == dim) ? 1 : 2);
      void* src;
      void* dst;
      size_t global_pre[3], local_pre[3];
      size_t global_apply[2], local_apply[2];
      cl_int i;

      if (1 == k_steps) {
        src = p_in;
        dst = y_out;
      }
      else if (0 == kstep) {
        src = p_in;
        dst = ctx->cascade_a;
      }
      else if (0 != is_final) {
        src = (0 == (kstep & 1)) ? ctx->cascade_a : ctx->cascade_b;
        dst = y_out;
      }
      else {
        src = (0 == (kstep & 1)) ? ctx->cascade_a : ctx->cascade_b;
        dst = (0 == (kstep & 1)) ? ctx->cascade_b : ctx->cascade_a;
      }

      global_pre[0] = STENCIL_BLK;
      global_pre[1] = STENCIL_BLK;
      global_pre[2] = STENCIL_BLK;
      local_pre[0] = STENCIL_BLK;
      local_pre[1] = 1;
      local_pre[2] = 1;

      i = 0;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->preprocess_x, i++, src));
      CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &dim));
      { int stride = (0 == dim) ? 1 : (1 == dim) ? super_m : super_m * super_n;
        CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &stride));
      }
      CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &super_m));
      CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &super_n));
      CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &super_p));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->preprocess_x, i++, xk));
      { int doff = 0;
        CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &doff));
      }
      { int nd = STENCIL_NDIGITS_X;
        CL_CHECK(result, clSetKernelArg(knl->preprocess_x, i++, sizeof(int), &nd));
      }

      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->preprocess_x,
        3, NULL, global_pre, ctx->dpas ? local_pre : NULL, 0, NULL, NULL));

      global_apply[0] = (size_t)total_blocks * ctx->sg;
      global_apply[1] = STENCIL_M_TILES * STENCIL_N_STRIPS;
      local_apply[0] = (size_t)ctx->sg;
      local_apply[1] = global_apply[1];

      i = 0;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, ctx->dk[dim]));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, xk));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, dst));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply, i++, vel));
      CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &mode));
      { int ys = STENCIL_N_TOTAL;
        CL_CHECK(result, clSetKernelArg(knl->stencil_apply, i++, sizeof(int), &ys));
      }

      CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, knl->stencil_apply,
        2, NULL, global_apply, ctx->dpas ? local_apply : NULL, 0, NULL, NULL));
    }
  }

  if (NULL != xk) libxstream_mem_deallocate(xk);
  return result;
}


void stencil_finalize(stencil_context_t* ctx)
{
  int dim;
  if (NULL == ctx) return;
  for (dim = 0; dim < 3; ++dim) {
    if (NULL != ctx->dk[dim]) libxstream_mem_deallocate(ctx->dk[dim]);
  }
  if (NULL != ctx->cascade_b) libxstream_mem_deallocate(ctx->cascade_b);
  if (NULL != ctx->cascade_a) libxstream_mem_deallocate(ctx->cascade_a);
  if (NULL != ctx->stream) libxstream_stream_destroy(ctx->stream);
  LIBXS_MEMZERO(ctx);
}
