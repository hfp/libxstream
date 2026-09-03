/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_pml.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


static void stencil_pml_profile(float* profile, int n, int ndamp, float scale)
{
  const float inv = (ndamp > 0) ? scale / (float)(ndamp * ndamp) : 0.0f;
  int i;
  for (i = 0; i < n; ++i) profile[i] = 0.0f;
  for (i = 1; i <= ndamp && i <= n; ++i) {
    const float q = (float)(i * i) * inv;
    profile[ndamp - i] = q;
    if (n - i >= 0 && n - i < n) profile[n - i] = q;
  }
}


int stencil_pml_setup(stencil_context_t* ctx, int nx, int ny, int nz,
                      libxstream_opencl_mem_hint_t mem_hint)
{
  int result = EXIT_SUCCESS;
  /**
   * A caller that models its own absorbing boundary passes eta and phi in, and
   * then owns them; the profile below is only built when it does not.
   */
  if (0 != ctx->pml && NULL == ctx->eta && NULL == ctx->phi) {
    const size_t grid_n = (size_t)nx * ny * nz;
    const size_t grid_bytes = grid_n * sizeof(float);
    const size_t eta_n = (size_t)(nx + 2) * (ny + 2) * (nz + 2);
    const size_t eta_bytes = eta_n * sizeof(float);
    const int pml_width = STENCIL_PML_W;
    const char *const eta_env = getenv("STENCIL_PML_ETA");
    const float eta_scale = (NULL != eta_env)
      ? (float)atof(eta_env) : (float)STENCIL_PML_ETA_PEAK;
    float* eta_host = NULL;
    float* etax = (float*)calloc((size_t)nx, sizeof(float));
    float* etay = (float*)calloc((size_t)ny, sizeof(float));
    float* etaz = (float*)calloc((size_t)nz, sizeof(float));
    result = libxstream_mem_host_allocate((void**)&eta_host, eta_bytes, ctx->stream);
    if (NULL == etax || NULL == etay || NULL == etaz) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      size_t idx;
      int ix, iy, iz;
      stencil_pml_profile(etax, nx, pml_width, eta_scale);
      stencil_pml_profile(etay, ny, pml_width, eta_scale);
      stencil_pml_profile(etaz, nz, pml_width, eta_scale);
      for (idx = 0; idx < eta_n; ++idx) eta_host[idx] = 0.0f;
      for (iz = 0; iz < nz; ++iz) {
        for (iy = 0; iy < ny; ++iy) {
          for (ix = 0; ix < nx; ++ix) {
            const float e = etax[ix] + etay[iy] + etaz[iz];
            long ei;
            if (e > 0.0f) {
              if (2 == ctx->layout) {
                ei = (long)(ix + 1) * (nz + 2) * (ny + 2) + (long)(iy + 1) * (nz + 2) + (iz + 1);
              }
              else {
                ei = (long)(iz + 1) * (ny + 2) * (nx + 2) + (long)(iy + 1) * (nx + 2) + (ix + 1);
              }
              eta_host[ei] = e;
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
    free(etaz);
    free(etay);
    free(etax);
    if (EXIT_SUCCESS == result) {
      result = libxstream_mem_dev_allocate_hint(&ctx->phi, grid_bytes, mem_hint);
      if (EXIT_SUCCESS == result) {
        result = libxstream_mem_zero(ctx->phi, 0, grid_bytes, ctx->stream);
      }
    }
    if (EXIT_SUCCESS == result) ctx->pml_owned = 1;
  }
  return result;
}
