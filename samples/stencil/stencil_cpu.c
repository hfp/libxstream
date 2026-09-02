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
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Build-time configuration of the host kernel. The JIT specializes the device
 * kernel at every launch, whereas a host translation unit is specialized once,
 * so the launcher rejects a grid that disagrees with what was compiled in.
 * Each knob accepts a -D override.
 */
#if !defined(INTEL)
# define INTEL 0 /* generic path: no DPAS, no 2D block I/O, no sub-groups */
#endif
#if !defined(STENCIL_CPU_LANES)
# define STENCIL_CPU_LANES 1
#endif
#if !defined(STENCIL_CPU_TEAM)
# define STENCIL_CPU_TEAM 0
#endif
#if (0 != STENCIL_CPU_LANES) && (0 != STENCIL_CPU_TEAM)
# error STENCIL_CPU_LANES and STENCIL_CPU_TEAM are alternative work-group models
#endif
#if !defined(STENCIL_LAYOUT)
# define STENCIL_LAYOUT 0
#elif (0 != STENCIL_LAYOUT)
# error the host launcher maps the XYZ layout only
#endif
#if !defined(STENCIL_PML)
# define STENCIL_PML 0
#elif (0 != STENCIL_PML)
# error the host kernel has no PML path: it would change the launcher signature
#endif

/**
 * With neither lane loops nor a team, a 1x1 work-group is the only shape that
 * keeps the kernel correct: it removes the cross-lane dependency the barriers
 * stand for. The other two models keep the on-device work-group shape.
 */
#if (0 == STENCIL_CPU_LANES) && (0 == STENCIL_CPU_TEAM)
# if !defined(WG_X)
#   define WG_X 1
# endif
# if !defined(WG_Y)
#   define WG_Y 1
# endif
#elif (0 != STENCIL_CPU_LANES)
/* Span the fast axis: no halo along it, contiguous staging. Capacity only. */
# if !defined(WG_X)
#   define WG_X 256
# endif
# if !defined(WG_Y)
#   define WG_Y 8
# endif
#endif

/* STENCIL_PADDED stays off so the coordinate clamping remains live. */

/**
 * The kernel derives its indexing from STENCIL_NX/NY/NZ, which the JIT knows at
 * build time and a host build does not. Bind them to the kernel's own runtime
 * extents so that one host binary serves any grid. Pinning all three through
 * CPUDEF trades that generality for constant-folded index arithmetic.
 */
#if defined(STENCIL_NX) || defined(STENCIL_NY) || defined(STENCIL_NZ)
# if !defined(STENCIL_NX) || !defined(STENCIL_NY) || !defined(STENCIL_NZ)
#   error pin all three grid extents or none of them
# endif
# define STENCIL_CPU_PINNED 1
#else
# define STENCIL_CPU_PINNED 0
# define STENCIL_NX nx
# define STENCIL_NY ny
# define STENCIL_NZ nz
#endif

/* The kernel sources derive STENCIL_WIDTH from RADIUS, asserted equal below. */
#undef STENCIL_WIDTH

#include "stencil_cpu.h"
#include "kernels/stencil_fp32.cl"

/**
 * The OpenCL keywords are meaningful only while the kernel source is being
 * translated, and some of them collide with OpenMP clause names. GCC expands
 * macros in a pragma line, so an empty "private" silently drops the clause
 * instead of failing. Retire them before the launcher, which is plain C.
 */
#undef global
#undef local
#undef private
#undef constant
#undef kernel
#undef barrier

#if (STENCIL_BLK != BLK)
# error BLK disagrees with STENCIL_BLK
#endif
#if (STENCIL_RADIUS != RADIUS)
# error RADIUS disagrees with STENCIL_RADIUS
#endif

/* Publish the coordinates of one work-item, the way a launch would. */
#define STENCIL_CPU_COORD(G0, G1, G2, L0, L1, W0, W1) do { \
  stencil_cpu_gid[0] = (G0); \
  stencil_cpu_gid[1] = (G1); \
  stencil_cpu_gid[2] = (G2); \
  stencil_cpu_lid[0] = (L0); \
  stencil_cpu_lid[1] = (L1); \
  stencil_cpu_lid[2] = 0; \
  stencil_cpu_lsz[0] = (W0); \
  stencil_cpu_lsz[1] = (W1); \
  stencil_cpu_lsz[2] = 1; \
} while (0)


int stencil_cpu_apply_direct(const float* p_grid, float* p_old,
                             const float* vel, const float* coeff, float dt2,
                             int nx, int ny, int nz, int nterms)
{
  int result = EXIT_SUCCESS;

  if (NTERMS != nterms
#if (0 != STENCIL_CPU_PINNED)
    || STENCIL_NX != nx || STENCIL_NY != ny || STENCIL_NZ != nz
#endif
    )
  {
    result = EXIT_FAILURE;
  }
  else {
    /**
     * One flat loop over the work-groups rather than a collapsed nest: the
     * group coordinates stay body-local, hence private without a clause.
     */
    /* Tile as wide as the fast axis allows, capped by the compile-time room. */
    const int width_f = (nx < WG_X) ? nx : WG_X;
    const int width_m = (ny < WG_Y) ? ny : WG_Y;
    const int ng0 = DIVUP(nx, width_f);
    const int ng1 = DIVUP(ny, width_m);
    const int ngroups = ng0 * ng1 * DIVUP(nz, BLK);
    int g;
#if (0 == STENCIL_CPU_TEAM)
# if defined(_OPENMP)
#   pragma omp parallel for
# endif
    for (g = 0; g < ngroups; ++g) {
      STENCIL_CPU_COORD(g % ng0, (g / ng0) % ng1, g / (ng0 * ng1), 0, 0,
        width_f, width_m);
      stencil_apply_direct(p_grid, p_old, vel, coeff, dt2, nx, ny, nz);
    }
#else
    for (g = 0; g < ngroups; ++g) {
#     pragma omp parallel num_threads(WG_X * WG_Y)
      { const int tid = omp_get_thread_num();
        STENCIL_CPU_COORD(g % ng0, (g / ng0) % ng1, g / (ng0 * ng1),
          tid % width_f, tid / width_f, width_f, width_m);
        stencil_apply_direct(p_grid, p_old, vel, coeff, dt2, nx, ny, nz);
      }
    }
#endif
  }

  return result;
}


#if defined(STENCIL_CPU) && (0 < STENCIL_CPU)

int stencil_host_allocate(void** ptr, size_t nbytes)
{
  int result = EXIT_SUCCESS;
  if (NULL == ptr) {
    result = EXIT_FAILURE;
  }
  else {
    *ptr = malloc(nbytes);
    if (NULL == *ptr) result = EXIT_FAILURE;
  }
  return result;
}


int stencil_host_deallocate(void* ptr)
{
  free(ptr);
  return EXIT_SUCCESS;
}


/* A transfer between two names for the same memory has nothing to move. */
int stencil_host_copy(void* dst, const void* src, size_t nbytes)
{
  int result = EXIT_SUCCESS;
  if (NULL == dst || NULL == src) {
    result = EXIT_FAILURE;
  }
  else if (dst != src) {
    memcpy(dst, src, nbytes);
  }
  return result;
}


int stencil_host_zero(void* ptr, size_t offset, size_t nbytes)
{
  int result = EXIT_SUCCESS;
  if (NULL == ptr) {
    result = EXIT_FAILURE;
  }
  else {
    memset((char*)ptr + offset, 0, nbytes);
  }
  return result;
}


/**
 * The host path carries the FP32 direct kernel and nothing else, so every
 * request for another operator, precision or layout is refused here instead of
 * being ignored on the way to a kernel that cannot honor it.
 */
int stencil_init(stencil_context_t* ctx, int verbosity, int method_override)
{
  static const char *const unsupported[] = {
    "STENCIL_BF16", "STENCIL_BF16S", "STENCIL_FP16S", "STENCIL_INT8",
    "STENCIL_BLOCKED", "STENCIL_LAYOUT", "STENCIL_METHOD", "STENCIL_PML"
  };
  const int nunsupported = (int)(sizeof(unsupported) / sizeof(*unsupported));
  int result = EXIT_SUCCESS;
  int i;

  for (i = 0; i < nunsupported; ++i) {
    const char *const value = getenv(unsupported[i]);
    if (NULL != value && 0 != atoi(value)) {
      fprintf(stderr, "ERROR: %s is not available with OCL=0\n", unsupported[i]);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && STENCIL_DIRECT < method_override) {
    fprintf(stderr, "ERROR: the host kernel implements the direct method only\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->verbosity = verbosity;
    ctx->method = STENCIL_DIRECT;
    ctx->k_steps = 1;
    ctx->r_per_step = STENCIL_RADIUS;
    ctx->strips_per_wg = 1;
    ctx->ndigits_a = 1;
    ctx->ndigits_x = STENCIL_NDIGITS_X;
    ctx->sg = 1;
    ctx->fp32 = 1;
  }
  return result;
}


int stencil_configure(stencil_context_t* ctx, int nx, int ny, int nz)
{
  int result = EXIT_SUCCESS;
#if (0 != STENCIL_CPU_PINNED)
  if (STENCIL_NX != nx || STENCIL_NY != ny || STENCIL_NZ != nz) {
    fprintf(stderr, "ERROR: the host kernel was pinned to %dx%dx%d;"
      " drop the extents from CPUDEF or set them to %dx%dx%d\n",
      STENCIL_NX, STENCIL_NY, STENCIL_NZ, nx, ny, nz);
    result = EXIT_FAILURE;
  }
  else
#endif
  if (NTERMS != ctx->nterms) {
    fprintf(stderr, "ERROR: the host kernel was built for %d terms;"
      " rebuild with CPUDEF=\"-DNTERMS=%d\"\n", NTERMS, ctx->nterms);
    result = EXIT_FAILURE;
  }
  else {
    ctx->grid_size[0] = nx;
    ctx->grid_size[1] = ny;
    ctx->grid_size[2] = nz;
    ctx->nblocks[0] = DIVUP(nx, BLK);
    ctx->nblocks[1] = DIVUP(ny, BLK);
    ctx->nblocks[2] = DIVUP(nz, BLK);
  }
  return result;
}


int stencil_precompute_operators(stencil_context_t* ctx,
                                 const double* fd_weights, int radius)
{
  int result = EXIT_SUCCESS;
  if (STENCIL_RADIUS != radius) {
    fprintf(stderr, "ERROR: the host kernel was built for radius %d\n",
      STENCIL_RADIUS);
    result = EXIT_FAILURE;
  }
  else {
    const int width = 2 * radius + 1;
    result = stencil_host_allocate(&ctx->coeff,
      (size_t)3 * width * sizeof(float));
    if (EXIT_SUCCESS == result) {
      float *const coeff = (float*)ctx->coeff;
      int d, r;
      for (d = 0; d < 3; ++d) {
        for (r = 0; r < width; ++r) {
          coeff[d * width + r] = (float)fd_weights[r];
        }
      }
    }
  }
  return result;
}


int stencil_apply_laplacian(stencil_context_t* ctx,
                            void* p_cur, void* p_old,
                            void* vel, float dt2, float dh, int nterms)
{
  int result;
  /* Spacing is folded into the operator weights; only PML reads it again. */
  (void)dh;
  result = stencil_cpu_apply_direct((const float*)p_cur, (float*)p_old,
    (const float*)vel, (const float*)ctx->coeff, dt2,
    ctx->grid_size[0], ctx->grid_size[1], ctx->grid_size[2], nterms);
  return result;
}


int stencil_seed_exp_buf(stencil_context_t* ctx, const float* p_host,
                         int nx, int ny, int nz)
{
  /* Reached for INT8 only, which stencil_init already refused. */
  (void)ctx; (void)p_host; (void)nx; (void)ny; (void)nz;
  return EXIT_SUCCESS;
}


void stencil_finalize(stencil_context_t* ctx)
{
  if (NULL != ctx) {
    if (NULL != ctx->coeff) stencil_host_deallocate(ctx->coeff);
    memset(ctx, 0, sizeof(*ctx));
  }
}

#endif /*STENCIL_CPU*/
