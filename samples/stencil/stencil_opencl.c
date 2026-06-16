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

#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_BF16)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_BF16)"
#endif


static int stencil_build_kernels(stencil_context_t* ctx)
{
  int result = EXIT_SUCCESS;
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  const int sg = (0 < devinfo->sg_sizes[0]) ? devinfo->sg_sizes[0] : STENCIL_SG;
  char flags[512];
  cl_program program = NULL;

  snprintf(flags, sizeof(flags),
    "-DBLK=%d -DRADIUS=%d -DNDIGITS_A=%d -DNDIGITS_X=%d "
    "-DSG=%d -DINTEL=%d -DGPU=1 -DUSE_BF16_EXT=1",
    STENCIL_BLK, STENCIL_RADIUS,
    STENCIL_NDIGITS_A, STENCIL_NDIGITS_X,
    sg, (devinfo->intel >= 2) ? devinfo->intel : 2);

  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_program(
      &program, OPENCL_KERNELS_SOURCE_STENCIL_BF16, flags, NULL);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel_query(
      program, "preprocess_x", &ctx->preprocess_x);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel_query(
      program, "stencil_apply", &ctx->stencil_apply);
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_kernel_query(
      program, "stencil_apply_tti", &ctx->stencil_apply_tti);
  }
  return result;
}


int stencil_init(stencil_context_t* ctx, int verbosity)
{
  int result = EXIT_SUCCESS;

  LIBXS_MEMZERO(ctx);
  ctx->verbosity = verbosity;

  if (EXIT_SUCCESS == result) {
    result = stencil_build_kernels(ctx);
  }
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
  const size_t d_size = (size_t)nda * blk * kpad * sizeof(cl_ushort);
  cl_ushort* d_host = NULL;
  int dim;

  if (NULL == ctx || NULL == fd_weights || radius != STENCIL_RADIUS) {
    return EXIT_FAILURE;
  }

  d_host = (cl_ushort*)calloc((size_t)nda * blk * kpad, sizeof(cl_ushort));
  if (NULL == d_host) return EXIT_FAILURE;

  {
    int row, col, s;
    for (row = 0; row < blk; ++row) {
      for (col = 0; col < blk; ++col) {
        const int dist = col - row;
        float val = 0.0f;
        float residual;

        if (dist >= -radius && dist <= radius) {
          val = (float)fd_weights[dist + radius];
        }

        residual = val;
        for (s = 0; s < nda; ++s) {
          const unsigned int bits = *(const unsigned int*)&residual;
          const unsigned int rounded =
            (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
          const cl_ushort bf = (cl_ushort)(rounded >> 16);
          const unsigned int bf32 = (unsigned int)bf << 16;
          const float bf_f = *(const float*)&bf32;
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
  const int total_blocks = ctx->nblocks[0] * ctx->nblocks[1] * ctx->nblocks[2];
  const size_t x_size = (size_t)STENCIL_NDIGITS_X * STENCIL_K_PAD
                       * STENCIL_N_PAD * sizeof(cl_ushort);
  const libxstream_opencl_stream_t* str =
    (const libxstream_opencl_stream_t*)ctx->stream;
  void* xk = NULL;
  int dim;

  result = libxstream_mem_allocate(&xk, x_size);

  for (dim = 0; dim < nterms && EXIT_SUCCESS == result; ++dim) {
    const int super_m = STENCIL_BLK + 2 * STENCIL_RADIUS;
    const int super_n = STENCIL_BLK + 2 * STENCIL_RADIUS;
    const int super_p = STENCIL_BLK + 2 * STENCIL_RADIUS;
    const int is_first = (0 == dim) ? 1 : 0;
    size_t global_pre[3], local_pre[3];
    size_t global_apply[2], local_apply[2];
    cl_int i;

    global_pre[0] = STENCIL_BLK;
    global_pre[1] = STENCIL_BLK;
    global_pre[2] = STENCIL_BLK;
    local_pre[0] = STENCIL_BLK;
    local_pre[1] = 1;
    local_pre[2] = 1;

    i = 0;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->preprocess_x, i++, p_in));
    CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &dim));
    { int stride = (0 == dim) ? 1 : (1 == dim) ? super_m : super_m * super_n;
      CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &stride));
    }
    CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &super_m));
    CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &super_n));
    CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &super_p));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->preprocess_x, i++, xk));
    { int doff = 0;
      CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &doff));
    }
    { int nd = STENCIL_NDIGITS_X;
      CL_CHECK(result, clSetKernelArg(ctx->preprocess_x, i++, sizeof(int), &nd));
    }

    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->preprocess_x,
      3, NULL, global_pre, local_pre, 0, NULL, NULL));

    local_apply[0] = STENCIL_SG;
    local_apply[1] = STENCIL_M_TILES * STENCIL_N_STRIPS;
    global_apply[0] = (size_t)total_blocks * STENCIL_SG;
    global_apply[1] = local_apply[1];

    i = 0;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->stencil_apply, i++, ctx->dk[dim]));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->stencil_apply, i++, xk));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->stencil_apply, i++, y_out));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->stencil_apply, i++, vel));
    CL_CHECK(result, clSetKernelArg(ctx->stencil_apply, i++, sizeof(int), &is_first));
    { int ys = STENCIL_N_TOTAL;
      CL_CHECK(result, clSetKernelArg(ctx->stencil_apply, i++, sizeof(int), &ys));
    }

    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, ctx->stencil_apply,
      2, NULL, global_apply, local_apply, 0, NULL, NULL));
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
  if (NULL != ctx->stencil_apply_tti) clReleaseKernel(ctx->stencil_apply_tti);
  if (NULL != ctx->stencil_apply) clReleaseKernel(ctx->stencil_apply);
  if (NULL != ctx->preprocess_x) clReleaseKernel(ctx->preprocess_x);
  if (NULL != ctx->stream) libxstream_stream_destroy(ctx->stream);
  LIBXS_MEMZERO(ctx);
}
