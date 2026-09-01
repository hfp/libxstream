/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_opencl.h"
#include <libxs/libxs_macros.h>
#include <libxs/libxs_math.h>

#include <stdlib.h>
#include <string.h>


/**
 * ndigits selects the storage format: 0 = single IEEE FP16 value,
 * 1 or 2 = that many Dekker BF16 limbs (limb k at idx + k * stride).
 */
static void stencil_store_bf16s_value(unsigned short* dst, size_t idx,
                                      size_t stride, int ndigits, float value)
{
  if (0 == ndigits) {
    dst[idx] = libxs_round_f16_f32(value);
  }
  else {
    const libxs_bf16_t hi = libxs_round_bf16_f32(value);
    dst[idx] = hi;
    if (1 < ndigits) {
      dst[idx + stride] = libxs_round_bf16_f32(value - libxs_bf16_to_f32(hi));
    }
  }
}


void stencil_pack_bf16s(unsigned short* dst, const float* src, size_t n, int ndigits)
{
  size_t i;
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < n; ++i) {
    stencil_store_bf16s_value(dst, i, n, ndigits, src[i]);
  }
}


void stencil_pack_bf16s_blocked(unsigned short* dst, const float* src,
                                int nx, int ny, int nz,
                                int nbx, int nby, int nbz, int ndigits)
{
  const int blk = STENCIL_BLK;
  const size_t stride = (size_t)nbx * nby * nbz * blk * blk * blk;
  const int nlimbs = (0 < ndigits) ? ndigits : 1;
  int bz, by, bx, lz, ly, lx;
  memset(dst, 0, (size_t)nlimbs * stride * sizeof(unsigned short));
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3) private(by, bx, lz, ly, lx)
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
                stencil_store_bf16s_value(dst, (size_t)dst_idx, stride, ndigits, val);
              }
            }
          }
        }
      }
    }
  }
}


void stencil_pack_bf16s_zyx(unsigned short* dst, const float* src,
                            int nx, int ny, int nz,
                            int hx, int hy, int hz, int ndigits)
{
  const int pnx = nx + 2 * hx, pny = ny + 2 * hy, pnz = nz + 2 * hz;
  const size_t stride = (size_t)pnx * pny * pnz;
  const int nlimbs = (0 < ndigits) ? ndigits : 1;
  int ix, iy, iz;
  memset(dst, 0, (size_t)nlimbs * stride * sizeof(unsigned short));
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3) private(iy, iz)
#endif
  for (ix = 0; ix < nx; ++ix) {
    for (iy = 0; iy < ny; ++iy) {
      for (iz = 0; iz < nz; ++iz) {
        const long src_idx = (long)iz * ny * nx + (long)iy * nx + ix;
        const long dst_idx = (long)(ix + hx) * pny * pnz
          + (long)(iy + hy) * pnz + (iz + hz);
        stencil_store_bf16s_value(dst, (size_t)dst_idx, stride, ndigits, src[src_idx]);
      }
    }
  }
}


void stencil_unpack_bf16s(float* dst, const unsigned short* src, size_t n, int ndigits)
{
  size_t i;
  if (0 == ndigits) {
#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (i = 0; i < n; ++i) {
      dst[i] = libxs_f16_to_f32(src[i]);
    }
  }
  else {
#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (i = 0; i < n; ++i) {
      double acc = libxs_bf16_to_f64(src[i]);
      if (1 < ndigits) acc += libxs_bf16_to_f64(src[i + n]);
      dst[i] = (float)acc;
    }
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
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(3) private(by, bx, lz, ly, lx)
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
