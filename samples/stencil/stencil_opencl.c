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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STENCIL_FP32_WG_X_DEFAULT 32
#define STENCIL_FP32_WG_Y_DEFAULT 8
#define STENCIL_FP32_SBLOCK_DEFAULT 2

#define STENCIL_KEY_GRF256(FLAGS) ((int)((FLAGS) & 1u))
#define STENCIL_KEY_TRIM(FLAGS) ((int)(((FLAGS) >> 1) & 255u))
#define STENCIL_KEY_LU(FLAGS) ((int)(((FLAGS) >> 9) & 15u))
#define STENCIL_KEY_BF16(FLAGS) ((int)(((FLAGS) >> 13) & 3u))
#define STENCIL_KEY_INT8(FLAGS) ((int)(((FLAGS) >> 15) & 3u))
#define STENCIL_KEY_BF16S(FLAGS) ((int)(((FLAGS) >> 17) & 3u))
#define STENCIL_KEY_BLOCKED(FLAGS) ((int)(((FLAGS) >> 19) & 3u))
#define STENCIL_KEY_LAYOUT(FLAGS) ((int)(((FLAGS) >> 21) & 3u))
#define STENCIL_KEY_PML(FLAGS) ((int)(((FLAGS) >> 23) & 1u))
#define STENCIL_KEY_FP32_BLOCK_IO(FLAGS) ((int)(((FLAGS) >> 24) & 1u))
#define STENCIL_KEY_FP32_SBLOCK(FLAGS) ((int)(((FLAGS) >> 25) & 3u))
#define STENCIL_KEY_NDIGITS_A(FLAGS) ((int)(((FLAGS) >> 27) & 3u))
#define STENCIL_KEY_FP32_WGX(KEY) ((int)((unsigned int)(KEY).fp32_wg >> 16))
#define STENCIL_KEY_FP32_WGY(KEY) ((int)((KEY).fp32_wg & 65535))

#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_BF16)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_BF16)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_FP32)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_FP32)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_STENCIL_INT8)
# error "OpenCL kernel source not found (stencil_kernels.h must define OPENCL_KERNELS_SOURCE_STENCIL_INT8)"
#endif


typedef struct {
  double tmax;
  int nq;
} stencil_fit_data_t;


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
    case STENCIL_COMPACT_FIT: { 
      const char* rfit_env = getenv("STENCIL_RADIUS_FIT");
      const int rfit = (NULL != rfit_env) ? atoi(rfit_env) : 3;
      *r_per_step = (rfit >= 1 && rfit <= STENCIL_RADIUS) ? rfit : 3;
      *k_steps = (STENCIL_RADIUS + *r_per_step - 1) / *r_per_step;
    } break;
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


static double stencil_ricker_weight(double t, double tpeak)
{
  const double u = t / tpeak;
  return u * u * exp(1.0 - u * u);
}


static double stencil_fit_error_r2(double alpha, double t)
{
  const double S = -(2.0 - 6.0 * alpha)
    + 2.0 * (1.0 - 4.0 * alpha) * cos(t) + 2.0 * alpha * cos(2.0 * t);
  return S + t * t;
}


static double stencil_fit_maxerr_r2(double alpha, double tmax, int nq)
{
  double mx = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * tmax / nq;
    const double e = fabs(stencil_fit_error_r2(alpha, t));
    if (e > mx) mx = e;
  }
  return mx;
}


static double stencil_fit_error_r3(double alpha, double beta, double t)
{
  const double a0 = -(2.0 - 6.0 * beta - 16.0 * alpha);
  const double a1 = 1.0 - 4.0 * beta - 9.0 * alpha;
  const double S = a0 + 2.0 * a1 * cos(t)
    + 2.0 * beta * cos(2.0 * t) + 2.0 * alpha * cos(3.0 * t);
  return S + t * t;
}


static double stencil_fit_maxerr_r3(double alpha, double beta,
                                    double tmax, int nq)
{
  double mx = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * tmax / nq;
    const double e = fabs(stencil_fit_error_r3(alpha, beta, t));
    if (e > mx) mx = e;
  }
  return mx;
}


static double stencil_gss_maxerr_r2(double alpha, const void* data)
{
  const stencil_fit_data_t* d = (const stencil_fit_data_t*)data;
  return stencil_fit_maxerr_r2(alpha, d->tmax, d->nq);
}


static double stencil_fit_optimal_beta(double alpha, double tmax, int nq)
{
  const double dt = tmax / nq;
  double num = 0.0, den = 0.0;
  int q;
  for (q = 0; q < nq; ++q) {
    const double t = (q + 0.5) * dt;
    const double a0 = -(2.0 - 16.0 * alpha);
    const double a1 = 1.0 - 9.0 * alpha;
    const double S0 = a0 + 2.0 * a1 * cos(t) + 2.0 * alpha * cos(3.0 * t);
    const double Bb = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
    const double rhs = S0 + t * t;
    num += rhs * Bb;
    den += Bb * Bb;
  }
  return (den > 1e-30) ? -num / den : -3.0 / 20.0;
}


static double stencil_gss_maxerr_r3(double alpha, const void* data)
{
  const stencil_fit_data_t* d = (const stencil_fit_data_t*)data;
  const double beta = stencil_fit_optimal_beta(alpha, d->tmax, d->nq);
  return stencil_fit_maxerr_r3(alpha, beta, d->tmax, d->nq);
}


static void stencil_fit_coeffs(int radius, double ppw, int fit_method,
                               double* coeffs)
{
  const double pi = 3.14159265358979323846;
  const double tmax = 2.0 * pi / ppw;
  const double tpeak = 2.0 * pi / (ppw * 0.6);
  const int nq = 1024;
  const double dt = tmax / nq;
  int q;

  if (2 == radius) {
    double alpha;
    if (2 == fit_method) {
      stencil_fit_data_t gss_data;
      double xmin;
      gss_data.tmax = tmax;
      gss_data.nq = nq;
      libxs_gss_min(stencil_gss_maxerr_r2, &gss_data,
        -0.5, 0.5, &xmin, 100, LIBXS_GSS_EVAL_ENDPOINTS, 1e-12, NULL);
      alpha = xmin;
    }
    else {
      double num = 0.0, den = 0.0;
      for (q = 0; q < nq; ++q) {
        const double t = (q + 0.5) * dt;
        const double w = (1 == fit_method)
          ? stencil_ricker_weight(t, tpeak) : 1.0;
        const double A = -2.0 + 2.0 * cos(t);
        const double B = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
        num += w * (A + t * t) * B;
        den += w * B * B;
      }
      alpha = (den > 1e-30) ? -num / den : -1.0 / 12.0;
    }
    coeffs[2] = -(2.0 - 6.0 * alpha);
    coeffs[1] = 1.0 - 4.0 * alpha;
    coeffs[0] = alpha;
  }
  else if (3 == radius) {
    double alpha, beta;
    if (2 == fit_method) {
      stencil_fit_data_t gss_data;
      double xmin;
      gss_data.tmax = tmax;
      gss_data.nq = nq;
      libxs_gss_min(stencil_gss_maxerr_r3, &gss_data,
        -0.2, 0.2, &xmin, 100, LIBXS_GSS_EVAL_ENDPOINTS, 1e-12, NULL);
      alpha = xmin;
      beta = stencil_fit_optimal_beta(alpha, tmax, nq);
    }
    else {
      double m00 = 0.0, m01 = 0.0, m11 = 0.0;
      double v0 = 0.0, v1 = 0.0, det;
      for (q = 0; q < nq; ++q) {
        const double t = (q + 0.5) * dt;
        const double w = (1 == fit_method)
          ? stencil_ricker_weight(t, tpeak) : 1.0;
        const double A = -2.0 + 2.0 * cos(t);
        const double Ba = 16.0 - 18.0 * cos(t) + 2.0 * cos(3.0 * t);
        const double Bb = 6.0 - 8.0 * cos(t) + 2.0 * cos(2.0 * t);
        const double rhs = A + t * t;
        m00 += w * Ba * Ba;
        m01 += w * Ba * Bb;
        m11 += w * Bb * Bb;
        v0 += w * rhs * Ba;
        v1 += w * rhs * Bb;
      }
      det = m00 * m11 - m01 * m01;
      if (det * det < 1e-30) {
        alpha = 1.0 / 90.0;
        beta = -3.0 / 20.0;
      }
      else {
        alpha = -(m11 * v0 - m01 * v1) / det;
        beta = -(m00 * v1 - m01 * v0) / det;
      }
    }
    coeffs[3] = -(2.0 - 6.0 * beta - 16.0 * alpha);
    coeffs[2] = 1.0 - 4.0 * beta - 9.0 * alpha;
    coeffs[1] = beta;
    coeffs[0] = alpha;
  }
}


static double stencil_fit_weight(int radius, int dist, double inv_h2,
                                 double ppw, int fit_method)
{
  double result = 0.0;
  if (dist >= -radius && dist <= radius) {
    if (3 == radius) {
      double c[4];
      stencil_fit_coeffs(3, ppw, fit_method, c);
      if (0 == dist) result = c[3];
      else if (1 == dist || -1 == dist) result = c[2];
      else if (2 == dist || -2 == dist) result = c[1];
      else result = c[0];
    }
    else if (2 == radius) {
      double c[3];
      stencil_fit_coeffs(2, ppw, fit_method, c);
      if (0 == dist) result = c[2];
      else if (1 == dist || -1 == dist) result = c[1];
      else result = c[0];
    }
    else if (1 == radius) {
      result = (0 == dist) ? -2.0 : 1.0;
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


static void stencil_store_bf16s_value(cl_ushort* dst, size_t idx,
                                      size_t stride, float value)
{
  const libxs_bf16_t hi = libxs_round_bf16_f32(value);
  dst[idx] = hi;
  dst[idx + stride] = libxs_round_bf16_f32(value - libxs_bf16_to_f32(hi));
}


void stencil_pack_bf16s(cl_ushort* dst, const float* src, size_t n)
{
  size_t i;
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < n; ++i) {
    stencil_store_bf16s_value(dst, i, n, src[i]);
  }
}


void stencil_pack_bf16s_blocked(cl_ushort* dst, const float* src,
                                int nx, int ny, int nz,
                                int nbx, int nby, int nbz)
{
  const int blk = STENCIL_BLK;
  const size_t stride = (size_t)nbx * nby * nbz * blk * blk * blk;
  int bz, by, bx, lz, ly, lx;
  memset(dst, 0, 2 * stride * sizeof(cl_ushort));
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3)
#endif
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
              const long dst_idx = tile_base + (long)lz * blk * blk + (long)ly * blk + lx;
              if (gx < nx && gy < ny && gz < nz) {
                const float val = src[(long)gz * ny * nx + (long)gy * nx + gx];
                stencil_store_bf16s_value(dst, (size_t)dst_idx, stride, val);
              }
            }
          }
        }
      }
    }
  }
}


void stencil_pack_bf16s_zyx(cl_ushort* dst, const float* src,
                            int nx, int ny, int nz,
                            int hx, int hy, int hz)
{
  const int pnx = nx + 2 * hx, pny = ny + 2 * hy, pnz = nz + 2 * hz;
  const size_t stride = (size_t)pnx * pny * pnz;
  int ix, iy, iz;
  memset(dst, 0, 2 * stride * sizeof(cl_ushort));
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3)
#endif
  for (ix = 0; ix < nx; ++ix) {
    for (iy = 0; iy < ny; ++iy) {
      for (iz = 0; iz < nz; ++iz) {
        const long src_idx = (long)iz * ny * nx + (long)iy * nx + ix;
        const long dst_idx = (long)(ix + hx) * pny * pnz
          + (long)(iy + hy) * pnz + (iz + hz);
        stencil_store_bf16s_value(dst, (size_t)dst_idx, stride, src[src_idx]);
      }
    }
  }
}


void stencil_unpack_bf16s(float* dst, const cl_ushort* src, size_t n)
{
  size_t i;
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < n; ++i) {
    dst[i] = (float)(libxs_bf16_to_f64(src[i])
      + libxs_bf16_to_f64(src[i + n]));
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
#if defined(_OPENMP)
#   pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3) reduction(max:global_max_exp)
#endif
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
#if defined(_OPENMP)
#     pragma omp parallel for
#endif
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
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3)
#endif
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


static void stencil_fp32_wg_dims(int* wg_x, int* wg_y)
{
  const char *const wgx_env = getenv("STENCIL_FP32_WG_X");
  const char *const wgy_env = getenv("STENCIL_FP32_WG_Y");
  int x = (NULL == wgx_env) ? STENCIL_FP32_WG_X_DEFAULT : atoi(wgx_env);
  int y = (NULL == wgy_env) ? STENCIL_FP32_WG_Y_DEFAULT : atoi(wgy_env);
  if (1 > x || 0 != (x & (x - 1)) || 64 < x) x = STENCIL_FP32_WG_X_DEFAULT;
  if (1 > y || 0 != (y & (y - 1)) || 32 < y) y = STENCIL_FP32_WG_Y_DEFAULT;
  if (256 != x * y) {
    x = STENCIL_FP32_WG_X_DEFAULT;
    y = STENCIL_FP32_WG_Y_DEFAULT;
  }
  *wg_x = x;
  *wg_y = y;
}


static int stencil_fp32_sblock(const stencil_context_t* ctx)
{
  const char *const env = getenv("STENCIL_FP32_SBLOCK");
  int result;
  if (NULL != env) {
    result = atoi(env);
    if (2 != result) result = 1;
  }
  else {
    result = (NULL != ctx && 0 != ctx->pml && 2 == ctx->layout)
      ? 1 : STENCIL_FP32_SBLOCK_DEFAULT;
  }
  return result;
}


static unsigned int stencil_key_flags(const stencil_context_t* ctx)
{
  const char *const fp32_block_io_env = getenv("STENCIL_FP32_BLOCK_IO");
  const int fp32_sblock = stencil_fp32_sblock(ctx);
  unsigned int result = 0u;
  result |= (unsigned int)(ctx->grf256 & 1);
  result |= (unsigned int)(ctx->trim & 255) << 1;
  result |= (unsigned int)(ctx->lu & 15) << 9;
  result |= (unsigned int)(ctx->bf16 & 3) << 13;
  result |= (unsigned int)(ctx->int8 & 3) << 15;
  result |= (unsigned int)(ctx->bf16s & 3) << 17;
  result |= (unsigned int)(ctx->blocked & 3) << 19;
  result |= (unsigned int)(ctx->layout & 3) << 21;
  result |= (unsigned int)((0 != ctx->pml) ? 1 : 0) << 23;
  result |= (unsigned int)((1 == fp32_sblock && NULL != fp32_block_io_env && 0 != atoi(fp32_block_io_env)) ? 1 : 0) << 24;
  result |= (unsigned int)fp32_sblock << 25;
  result |= (unsigned int)(ctx->ndigits_a & 3) << 27;
  return result;
}


static const stencil_kernels_t* stencil_get_kernels(stencil_context_t* ctx)
{
  static libxs_registry_t* kernel_registry /*= NULL*/;
  static libxs_lock_t compile_lock /*= LIBXS_LOCK_INITIALIZER*/;
  static char base_flags[256] = { 0 };
  static int base_ready /*= 0*/;

  const libxstream_opencl_config_t* config = &libxstream_opencl_config;
  const libxstream_opencl_device_t* devinfo = &config->device;
  stencil_opencl_key_t key;
  stencil_kernels_t* kptr;

  if (0 == LIBXS_ATOMIC_LOAD(&base_ready, LIBXS_ATOMIC_SEQ_CST)) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, config->lock_main);
    if (0 == base_ready) {
      const char* cmem = (EXIT_SUCCESS != libxstream_opencl_use_cmem_size(
          devinfo, STENCIL_WIDTH * sizeof(float))) ? "global" : "constant";
      LIBXS_SNPRINTF(base_flags, sizeof(base_flags),
        "-DBLK=%d -DNDIGITS_X=%d -DGPU=1 -DCONSTANT=%s",
        STENCIL_BLK, STENCIL_NDIGITS_X, cmem);
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
  key.nterms = ctx->nterms;
  key.grid_key = ctx->grid_size[0] ^ (ctx->grid_size[1] * 65599)
               ^ (ctx->grid_size[2] * 8191) ^ (ctx->halo[0] * 131)
               ^ (ctx->halo[1] * 257) ^ (ctx->halo[2] * 521);
  key.fp32_wg = 0;
  key.flags = stencil_key_flags(ctx);
  if (0 != ctx->fp32) {
    int fp32_wgx, fp32_wgy;
    stencil_fp32_wg_dims(&fp32_wgx, &fp32_wgy);
    key.fp32_wg = (fp32_wgx << 16) | fp32_wgy;
  }

  kptr = (stencil_kernels_t*)libxs_registry_get(
    kernel_registry, &key, sizeof(key), libxs_registry_lock(kernel_registry));

  if (NULL == kptr || (NULL == kptr->stencil_apply && NULL == kptr->stencil_apply_direct)) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, &compile_lock);
    kptr = (stencil_kernels_t*)libxs_registry_get(
      kernel_registry, &key, sizeof(key), libxs_registry_lock(kernel_registry));

    if (NULL == kptr || (NULL == kptr->stencil_apply && NULL == kptr->stencil_apply_direct)) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      char flags[LIBXSTREAM_BUFFERSIZE];
      char options[256];
      cl_program program = NULL;
      stencil_kernels_t knl;
      int ok = EXIT_SUCCESS;

      { const int intel_level = (int)devinfo->intel;
        const int nv_level = (int)devinfo->nv;
        LIBXS_SNPRINTF(flags, sizeof(flags),
          "%s -DRADIUS=%d -DK_STEPS=%d -DR_PER_STEP=%d -DSTRIPS_PER_WG=%d"
          " -DSG=%d -DINTEL=%d -DNV=%d -DMETHOD=%d -DTRIM=%d -DNTERMS=%d -DLU=%d"
          " -DNDIGITS_A=%d -DNSLICES_A=%d"
          " -DSTENCIL_BF16=%d -DSTENCIL_INT8=%d -DSTENCIL_BF16S=%d -DSTENCIL_BLOCKED=%d"
          " -DSTENCIL_LAYOUT=%d -DSTENCIL_PML=%d %s",
          base_flags, (0 == key.method) ? STENCIL_RADIUS : key.r_per_step,
          key.k_steps, key.r_per_step,
          key.strips_per_wg, key.sg, intel_level, nv_level, key.method,
          STENCIL_KEY_TRIM(key.flags), key.nterms, STENCIL_KEY_LU(key.flags),
          ctx->ndigits_a, ctx->ndigits_a,
          STENCIL_KEY_BF16(key.flags), STENCIL_KEY_INT8(key.flags),
          STENCIL_KEY_BF16S(key.flags), STENCIL_KEY_BLOCKED(key.flags),
          STENCIL_KEY_LAYOUT(key.flags), STENCIL_KEY_PML(key.flags),
          (0 != ctx->fp32) ? ""
            : ((intel_level >= 2) ? "-DUSE_BF16_EXT=1" : "-DUSE_BF16=1"));
      }

      { const int nx = ctx->grid_size[0], ny = ctx->grid_size[1], nz = ctx->grid_size[2];
        char stride_flags[512];
        if (2 == STENCIL_KEY_LAYOUT(key.flags)) {
          const int lx = ctx->halo[0], ly = ctx->halo[1], lz = ctx->halo[2];
          const long p_n = (long)(nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
          const int p_sx = (nz + 2 * lz) * (ny + 2 * ly);
          const int p_sy = (nz + 2 * lz);
          const int v_sx = nz * ny;
          const int v_sy = nz;
          const int e_sx = (nz + 2) * (ny + 2);
          const int e_sy = (nz + 2);
          const int radius = (0 == key.method) ? STENCIL_RADIUS : key.r_per_step;
          int padded = (lx >= radius && ly >= radius && lz >= radius) ? 1 : 0;
          if (0 != padded && 1 == ctx->fp32) {
            int wg_x, wg_y;
            int max_fast, max_med;
            stencil_fp32_wg_dims(&wg_x, &wg_y);
            max_fast = ((nz + wg_x - 1) / wg_x) * wg_x - 1 + radius;
            max_med = ((ny + wg_y - 1) / wg_y) * wg_y - 1 + radius;
            if (max_fast >= nz + lz || max_med >= ny + ly) {
              padded = 0;
            }
          }
          else if (0 != padded && 0 == ctx->fp32) {
            const int kpad = (0 != ctx->int8) ? STENCIL_K_PAD_I8 : STENCIL_K_PAD;
            const int max_gather = kpad - 1 - radius;
            const int nbx = ctx->nblocks[0], nby = ctx->nblocks[1], nbz = ctx->nblocks[2];
            if ((nbx - 1) * STENCIL_BLK + max_gather >= nx + lx
             || (nby - 1) * STENCIL_BLK + max_gather >= ny + ly
             || (nbz - 1) * STENCIL_BLK + max_gather >= nz + lz)
            {
              padded = 0;
            }
          }
          LIBXS_SNPRINTF(stride_flags, sizeof(stride_flags),
            " -DSTENCIL_P_N=%ld -DSTENCIL_NX=%d -DSTENCIL_NY=%d -DSTENCIL_NZ=%d"
            " -DSTENCIL_P_SX=%d -DSTENCIL_P_SY=%d"
            " -DSTENCIL_P_LX=%d -DSTENCIL_P_LY=%d -DSTENCIL_P_LZ=%d"
            " -DSTENCIL_V_SX=%d -DSTENCIL_V_SY=%d"
            " -DSTENCIL_V_LX=0 -DSTENCIL_V_LY=0 -DSTENCIL_V_LZ=0"
            " -DSTENCIL_E_SX=%d -DSTENCIL_E_SY=%d"
            " -DSTENCIL_E_LX=1 -DSTENCIL_E_LY=1 -DSTENCIL_E_LZ=1"
            " -DSTENCIL_PADDED=%d",
            p_n, nx, ny, nz, p_sx, p_sy, lx, ly, lz, v_sx, v_sy, e_sx, e_sy,
            padded);
        }
        else {
          const long p_n = (0 != STENCIL_KEY_BLOCKED(key.flags))
            ? (long)ctx->nblocks[0] * ctx->nblocks[1] * ctx->nblocks[2]
              * STENCIL_BLK * STENCIL_BLK * STENCIL_BLK
            : (long)nx * ny * nz;
          LIBXS_SNPRINTF(stride_flags, sizeof(stride_flags),
            " -DSTENCIL_P_N=%ld -DSTENCIL_NX=%d -DSTENCIL_NY=%d -DSTENCIL_NZ=%d",
            p_n, nx, ny, nz);
        }
        strncat(flags, stride_flags, sizeof(flags) - strlen(flags) - 1);
      }

      if (1 == ctx->fp32) {
        char fp32_flags[128];
        LIBXS_SNPRINTF(fp32_flags, sizeof(fp32_flags),
          " -DWG_X=%d -DWG_Y=%d -DFP32_DISABLE_BLOCK_IO=%d -DFP32_SBLOCK=%d",
          STENCIL_KEY_FP32_WGX(key), STENCIL_KEY_FP32_WGY(key),
          (0 == STENCIL_KEY_FP32_BLOCK_IO(key.flags)) ? 1 : 0,
          STENCIL_KEY_FP32_SBLOCK(key.flags));
        strncat(flags, fp32_flags, sizeof(flags) - strlen(flags) - 1);
      }

      LIBXS_SNPRINTF(options, sizeof(options), "-cl-fast-relaxed-math -cl-denorms-are-zero%s",
        (0 != STENCIL_KEY_GRF256(key.flags) && 0 != devinfo->intel && 0 == devinfo->biggrf)
          ? " -cl-intel-256-GRF-per-thread" : "");

      if (EXIT_SUCCESS == ok) {
        const char* source;
        if (1 == ctx->fp32) source = OPENCL_KERNELS_SOURCE_STENCIL_FP32;
        else if (0 != STENCIL_KEY_INT8(key.flags)) source = OPENCL_KERNELS_SOURCE_STENCIL_INT8;
        else source = OPENCL_KERNELS_SOURCE_STENCIL_BF16;
        ok = libxstream_opencl_program(0 /*source_kind*/,
          source, "stencil", flags,
          options, NULL /*try*/, NULL /*try_ok*/, NULL /*exts*/, 0,
          &program);
      }
      LIBXS_MEMZERO(&knl);
      if (EXIT_SUCCESS == ok && 1 == ctx->fp32) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_direct", &knl.stencil_apply_direct);
      }
      else if (EXIT_SUCCESS == ok && 0 != STENCIL_KEY_INT8(key.flags)) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply_int8", &knl.stencil_apply);
      }
      else if (EXIT_SUCCESS == ok) {
        ok = libxstream_opencl_kernel_query(program, "stencil_apply", &knl.stencil_apply);
      }
      if (EXIT_SUCCESS == ok && 0 == ctx->fp32) {
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
          key.sg, STENCIL_KEY_GRF256(key.flags));
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
  libxstream_init_config_t cfg;
  const libxstream_opencl_device_t* devinfo;
  const char *strips_env, *blocked_env, *layout_env, *method_env;
  const char *bf16s_env, *bf16_env, *int8_env;
  const char *grf_env, *trim_env, *sg_env, *lu_env;
  int result, method_val;

  libxstream_init_config_default(&cfg);
  result = libxstream_init_config(&cfg);
  devinfo = &libxstream_opencl_config.device;
  strips_env = getenv("STENCIL_STRIPS_PER_WG");
  blocked_env = getenv("STENCIL_BLOCKED");
  layout_env = getenv("STENCIL_LAYOUT");
  method_env = getenv("STENCIL_METHOD");
  bf16s_env = getenv("STENCIL_BF16S");
  bf16_env = getenv("STENCIL_BF16");
  int8_env = getenv("STENCIL_INT8");
  grf_env = getenv("STENCIL_GRF256");
  trim_env = getenv("STENCIL_TRIM");
  sg_env = getenv("STENCIL_SG");
  lu_env = getenv("STENCIL_LU");

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
  ctx->int8 = (NULL != int8_env) ? atoi(int8_env) : 0;

  ctx->bf16s = (NULL == bf16s_env) ? 0 : atoi(bf16s_env);

  { const int bf16_val = (NULL != bf16_env) ? atoi(bf16_env) : 0;
    const int has_dpas = (devinfo->intel >= 2) ? 1 : 0;
    if (0 != bf16_val) {
      ctx->bf16 = has_dpas ? bf16_val : 2;
      ctx->fp32 = 0;
    }
    else if (0 != ctx->int8) {
      if (!has_dpas) ctx->int8 = 2;
      ctx->bf16 = 0;
      ctx->fp32 = 0;
    }
    else {
      ctx->bf16 = 0;
      ctx->fp32 = 1;
      ctx->int8 = 0;
    }
  }

  { const char *const nda_env = getenv("STENCIL_NDIGITS_A");
    int nda = (NULL != nda_env) ? atoi(nda_env) : STENCIL_NDIGITS_A_DEFAULT;
    if (nda < 1) nda = 1;
    if (nda > STENCIL_NDIGITS_A_MAX) nda = STENCIL_NDIGITS_A_MAX;
    ctx->ndigits_a = nda;
  }

  ctx->strips_per_wg = stencil_valid_strips_per_wg(
    (NULL != strips_env) ? atoi(strips_env)
    : ((0 != ctx->int8) ? 1 : STENCIL_STRIPS_PER_WG));

  ctx->trim = (NULL == trim_env)
    ? ((STENCIL_RADIUS >= 4) ? 1 : 0)
    : atoi(trim_env);
  if (ctx->trim < 0) ctx->trim = 0;
  ctx->blocked = (NULL == blocked_env) ? 0 : atoi(blocked_env);
  ctx->layout = (NULL != layout_env) ? atoi(layout_env)
    : ((0 != ctx->blocked) ? 1 : 0);
  { const char *const halo_env = getenv("STENCIL_HALO");
    const int halo_val = (NULL != halo_env) ? atoi(halo_env) : 0;
    ctx->halo[0] = ctx->halo[1] = ctx->halo[2] = halo_val;
  }
  { const char *const pml_env = getenv("STENCIL_PML");
    ctx->pml = (NULL == pml_env) ? 0 : atoi(pml_env);
  }
  { const char *const hint_env = getenv("STENCIL_HINT");
    ctx->hint = (NULL != hint_env) ? atoi(hint_env) : 0;
  }

  ctx->nterms = 3;

  if (EXIT_SUCCESS == result) {
    result = libxstream_stream_create(&ctx->stream, "stencil", 0);
  }
  return result;
}


int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;
  libxstream_opencl_mem_hint_t mem_hint;

  if (NULL == ctx) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 2 == ctx->layout && 0 == ctx->fp32) {
    ctx->fp32 = 1;
    ctx->bf16 = 0;
    ctx->int8 = 0;
  }
  mem_hint = (NULL != ctx && 0 != ctx->hint)
    ? libxstream_opencl_mem_hint_atomics : libxstream_opencl_mem_hint_compress;
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
        &ctx->exp_buf[eb], exp_size, mem_hint);
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
    const size_t eta_n = (size_t)(nx + 2) * (ny + 2) * (nz + 2);
    const size_t eta_bytes = eta_n * sizeof(float);
    const int pml_width = 20;
    float* eta_host = NULL;
    result = libxstream_mem_host_allocate((void**)&eta_host, eta_bytes, ctx->stream);
    if (EXIT_SUCCESS == result) {
      size_t idx;
      int ix, iy, iz;
      for (idx = 0; idx < eta_n; ++idx) eta_host[idx] = 0.0f;
      for (iz = 0; iz < nz; ++iz) {
        for (iy = 0; iy < ny; ++iy) {
          for (ix = 0; ix < nx; ++ix) {
            int d = nx;
            int dist;
            long ei;
            if (ix < pml_width) { dist = pml_width - ix; if (dist < d) d = dist; }
            if (ix >= nx - pml_width) { dist = ix - (nx - pml_width - 1); if (dist < d) d = dist; }
            if (iy < pml_width) { dist = pml_width - iy; if (dist < d) d = dist; }
            if (iy >= ny - pml_width) { dist = iy - (ny - pml_width - 1); if (dist < d) d = dist; }
            if (iz < pml_width) { dist = pml_width - iz; if (dist < d) d = dist; }
            if (iz >= nz - pml_width) { dist = iz - (nz - pml_width - 1); if (dist < d) d = dist; }
            if (d < nx) {
              float r = (float)d / (float)pml_width;
              if (2 == ctx->layout) {
                ei = (long)(ix + 1) * (nz + 2) * (ny + 2) + (long)(iy + 1) * (nz + 2) + (iz + 1);
              }
              else {
                ei = (long)(iz + 1) * (ny + 2) * (nx + 2) + (long)(iy + 1) * (nx + 2) + (ix + 1);
              }
              eta_host[ei] = 0.05f * r * r;
            }
          }
        }
      }
      result = libxstream_mem_dev_allocate_hint(&ctx->eta, eta_bytes, mem_hint);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(eta_host, ctx->eta, eta_bytes, ctx->stream);
      }
      libxstream_mem_host_deallocate(eta_host, ctx->stream);
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_dev_allocate_hint(&ctx->phi, grid_bytes, mem_hint);
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
  const int nda = ctx->ndigits_a;
  const int k_steps = ctx->k_steps;
  const int r_step = ctx->r_per_step;
  const int d_rows = blk;
  const int d_band = STENCIL_WIDTH;
  const libxstream_opencl_mem_hint_t mem_hint = (0 != ctx->hint)
    ? libxstream_opencl_mem_hint_atomics : libxstream_opencl_mem_hint_compress;
  const int use_fp32 = ctx->fp32;
  const int use_float_d = (0 != ctx->fp32 || 2 <= ctx->bf16) ? 1 : 0;
  const size_t d_size_bf16 = (size_t)nda * d_rows * kpad * sizeof(cl_ushort);
  const size_t d_size_fp32 = (size_t)d_rows * d_band * sizeof(float);
  const size_t d_size = (0 != use_float_d) ? d_size_fp32 : d_size_bf16;
  const double inv_h2 = -72.0 * fd_weights[radius] / 205.0;
  void* d_host = NULL;
  int dim;

  if (NULL == ctx || NULL == fd_weights || radius != STENCIL_RADIUS) {
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result && 0 != use_float_d) {
    float* d_fp32 = (float*)calloc((size_t)d_rows * d_band, sizeof(float));
    if (NULL == d_fp32) result = EXIT_FAILURE;

    if (EXIT_SUCCESS == result) {
      const int use_fit = (STENCIL_COMPACT_FIT == ctx->method);
      const char *const ppw_env = use_fit ? getenv("STENCIL_PPW") : NULL;
      const char *const fit_env = use_fit ? getenv("STENCIL_FIT") : NULL;
      const double ppw = (NULL != ppw_env) ? atof(ppw_env) : 8.0;
      const int fit_method = (NULL != fit_env) ? atoi(fit_env) : 2;
      int row, r;
      if (1 == k_steps) {
        for (row = 0; row < blk; ++row) {
          for (r = 0; r < d_band; ++r) {
            const int dist = r - radius;
            d_fp32[row * d_band + r] = (float)stencil_fd_weight(fd_weights, radius, dist);
          }
        }
      }
      else if (use_fit) {
        for (row = 0; row < d_rows; ++row) {
          for (r = 0; r < d_band; ++r) {
            const int dist = r - r_step;
            d_fp32[row * d_band + r] = (float)stencil_fit_weight(r_step, dist, inv_h2,
                                                                   ppw, fit_method);
          }
        }
      }
      else {
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
      const int use_fit = (STENCIL_COMPACT_FIT == ctx->method);
      const char *const ppw_env = use_fit ? getenv("STENCIL_PPW") : NULL;
      const char *const fit_env = use_fit ? getenv("STENCIL_FIT") : NULL;
      const double ppw = (NULL != ppw_env) ? atof(ppw_env) : 8.0;
      const int fit_method = (NULL != fit_env) ? atoi(fit_env) : 2;
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
      else if (use_fit) {
        for (row = 0; row < d_rows; ++row) {
          for (col = 0; col < kpad; ++col) {
            dist = col - r_step - row;
            d_mat[row * kpad + col] = stencil_fit_weight(r_step, dist, inv_h2,
                                                          ppw, fit_method);
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
    const int r_eff = (1 == k_steps) ? radius : r_step;
    const int width_eff = 2 * r_eff + 1;
    const size_t coeff_size = (size_t)3 * width_eff * sizeof(float);
    float coeff_host[3 * (2 * STENCIL_RADIUS + 1)];
    const int use_fit = (STENCIL_COMPACT_FIT == ctx->method);
    const char *const ppw_env = use_fit ? getenv("STENCIL_PPW") : NULL;
    const char *const fit_env = use_fit ? getenv("STENCIL_FIT") : NULL;
    const double ppw = (NULL != ppw_env) ? atof(ppw_env) : 8.0;
    const int fit_method = (NULL != fit_env) ? atoi(fit_env) : 2;
    int r, d;
    for (d = 0; d < 3; ++d) {
      for (r = 0; r < width_eff; ++r) {
        if (1 == k_steps) {
          coeff_host[d * width_eff + r] = (float)fd_weights[r];
        }
        else if (use_fit) {
          coeff_host[d * width_eff + r] =
            (float)stencil_fit_weight(r_step, r - r_eff, inv_h2,
                                      ppw, fit_method);
        }
        else {
          coeff_host[d * width_eff + r] =
            (float)stencil_compact_weight(r_step, r - r_eff, inv_h2);
        }
      }
    }
    result = libxstream_mem_dev_allocate_hint(
      (void**)&ctx->coeff, coeff_size, mem_hint);
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
      const int use_fit = (STENCIL_COMPACT_FIT == ctx->method);
      const char *const ppw_env = use_fit ? getenv("STENCIL_PPW") : NULL;
      const char *const fit_env = use_fit ? getenv("STENCIL_FIT") : NULL;
      const double ppw = (NULL != ppw_env) ? atof(ppw_env) : 8.0;
      const int fit_method = (NULL != fit_env) ? atoi(fit_env) : 2;
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
      else if (use_fit) {
        for (row = 0; row < d_rows; ++row) {
          for (col = 0; col < kpad; ++col) {
            int dist = col - r_step - row;
            d_mat_i8[row * kpad + col] = stencil_fit_weight(r_step, dist, inv_h2,
                                                             ppw, fit_method);
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
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk[dim], d_i8_size, mem_hint);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(d_i8, ctx->dk[dim], d_i8_size, ctx->stream);
      }
    }
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk_scale, scale_size, mem_hint);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_copy_h2d(d_scale, ctx->dk_scale, scale_size, ctx->stream);
      }
    }

    free(d_scale);
    free(d_i8);
  }
  else {
    for (dim = 0; dim < 3 && EXIT_SUCCESS == result; ++dim) {
      result = libxstream_mem_dev_allocate_hint((void**)&ctx->dk[dim], d_size, mem_hint);
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
                            void* vel, float dt2, float dh, int nterms)
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
    int fp32_wgx, fp32_wgy;
    cl_int i = 0;
    stencil_fp32_wg_dims(&fp32_wgx, &fp32_wgy);
    local_direct[0] = (size_t)fp32_wgx;
    local_direct[1] = (size_t)fp32_wgy;
    local_direct[2] = 1;
    if (2 == ctx->layout) {
      global_direct[0] = ((size_t)(nz + fp32_wgx - 1) / fp32_wgx) * fp32_wgx;
      global_direct[1] = ((size_t)(ny + fp32_wgy - 1) / fp32_wgy) * fp32_wgy;
      global_direct[2] = (size_t)((nx + STENCIL_BLK - 1) / STENCIL_BLK);
    }
    else {
      global_direct[0] = ((size_t)(nx + fp32_wgx - 1) / fp32_wgx) * fp32_wgx;
      global_direct[1] = ((size_t)(ny + fp32_wgy - 1) / fp32_wgy) * fp32_wgy;
      global_direct[2] = (size_t)((nz + STENCIL_BLK - 1) / STENCIL_BLK);
    }
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_cur));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_old));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, p_new));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, vel));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(knl->stencil_apply_direct, i++, ctx->coeff));
    if (0 != ctx->pml) {
      const float hdx_2 = 0.5f / dh;
      const float hdy_2 = 0.5f / dh;
      const float hdz_2 = 0.5f / dh;
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
        const float hdx_2 = 0.5f / dh;
        const float hdy_2 = 0.5f / dh;
        const float hdz_2 = 0.5f / dh;
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
