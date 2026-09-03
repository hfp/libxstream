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
#include "stencil_weights.h"
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
#if !defined(LIBXSTREAM_CPU_TEAM)
# define LIBXSTREAM_CPU_TEAM 0
#endif
#if (0 != STENCIL_CPU_LANES) && (0 != LIBXSTREAM_CPU_TEAM)
# error STENCIL_CPU_LANES and LIBXSTREAM_CPU_TEAM are alternative work-group models
#endif
#if !defined(STENCIL_LAYOUT)
# define STENCIL_LAYOUT 0
#endif
#if !defined(STENCIL_PML)
# define STENCIL_PML 0
#endif

/**
 * With neither lane loops nor a team, a 1x1 work-group is the only shape that
 * keeps the kernel correct: it removes the cross-lane dependency the barriers
 * stand for. The other two models keep the on-device work-group shape.
 */
#if (0 == STENCIL_CPU_LANES) && (0 == LIBXSTREAM_CPU_TEAM)
# if !defined(WG_X)
#   define WG_X 1
# endif
# if !defined(WG_Y)
#   define WG_Y 1
# endif
#elif (0 != STENCIL_CPU_LANES)
/**
 * Span the fast axis: no halo along it, contiguous staging. WG_X is a capacity,
 * so it costs per-thread stack rather than generality, and a narrower grid gets
 * a narrower tile.
 */
# if !defined(WG_X)
#   define WG_X 1024
# endif
# if !defined(WG_Y)
#   define WG_Y 8
# endif
#endif

/**
 * The JIT specializes the device kernel per launch. A host build instead
 * compiles one instance per combination it may be asked for and selects between
 * them in stencil_configure; stencil_cpu_instance.h documents the encoding.
 *
 * Axes: the boundary treatment (a gather that leaves the grid is either clamped
 * or read out of the halo, which the grid decides, as on the device), the
 * wavefield storage format (STENCIL_BF16S, STENCIL_FP16S), and the operator
 * radius (STENCIL_METHOD) once STENCIL_CPU_COMPACT is built in.
 *
 * An explicit -DSTENCIL_PADDED forces the boundary treatment, and then only that
 * half of the set exists; the other half becomes a hole in the table.
 */
#if defined(STENCIL_PADDED)
# if (0 < STENCIL_PADDED)
#   define STENCIL_CPU_FORCE_PAD 1
# else
#   define STENCIL_CPU_FORCE_PAD 0
# endif
# undef STENCIL_PADDED
#endif
#if !defined(STENCIL_CPU_COMPACT)
# define STENCIL_CPU_COMPACT 0
#endif
#if (0 != STENCIL_CPU_COMPACT)
# define STENCIL_CPU_NINST 32
#else
# define STENCIL_CPU_NINST 8
#endif
#define STENCIL_CPU_CAT2(A, B) A##B
#define STENCIL_CPU_CAT(A, B) STENCIL_CPU_CAT2(A, B)

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

/* The kernel sources derive STENCIL_WIDTH from RADIUS, which varies per instance. */
#undef STENCIL_WIDTH

/**
 * Array geometry the JIT supplies as -D per launch and a host build cannot:
 * {sx, sy} strides and {lx, ly, lz} halo of the wavefield. Uniform for the whole
 * launch, hence not threadprivate. Only the Z-innermost layout indexes through
 * them, and STENCIL_LAYOUT_ZYX is not spelled out yet at this point.
 */
/* Wavefield element count, which the BF16 two-limb format offsets the low limb by. */
static long stencil_cpu_p_n;
#define STENCIL_P_N stencil_cpu_p_n

#if (2 == STENCIL_LAYOUT)
static long stencil_cpu_stride[6];
static int stencil_cpu_halo[3];

#define STENCIL_P_SX stencil_cpu_stride[0]
#define STENCIL_P_SY stencil_cpu_stride[1]
#define STENCIL_P_LX stencil_cpu_halo[0]
#define STENCIL_P_LY stencil_cpu_halo[1]
#define STENCIL_P_LZ stencil_cpu_halo[2]
/* The velocity carries no halo, hence a stride of its own. */
#define STENCIL_V_SX stencil_cpu_stride[2]
#define STENCIL_V_SY stencil_cpu_stride[3]
#define STENCIL_V_LX 0
#define STENCIL_V_LY 0
#define STENCIL_V_LZ 0
/* eta carries a halo of exactly one. */
#define STENCIL_E_SX stencil_cpu_stride[4]
#define STENCIL_E_SY stencil_cpu_stride[5]
#define STENCIL_E_LX 1
#define STENCIL_E_LY 1
#define STENCIL_E_LZ 1
#endif

/**
 * Signature shared by every instance: the storage format lives inside the load
 * and store macros, so it does not reach the argument list.
 */
#if (0 != STENCIL_PML)
typedef void (*stencil_cpu_kernel_t)(const float*, float*, const float*,
  const float*, const float*, float*, float, float, float, float, int, int, int);
#else
typedef void (*stencil_cpu_kernel_t)(const float*, float*, const float*,
  const float*, float, int, int, int);
#endif

#include "stencil_cpu_instances.h"

static const stencil_cpu_kernel_t stencil_cpu_kernels[STENCIL_CPU_NINST] = {
#define STENCIL_CPU_TABLE 1
#include "stencil_cpu_instances.h"
#undef STENCIL_CPU_TABLE
};

/* Selected in stencil_configure, hence uniform for the whole launch. */
static stencil_cpu_kernel_t stencil_cpu_kernel;

#if (STENCIL_BLK != BLK)
# error BLK disagrees with STENCIL_BLK
#endif
#if (STENCIL_LAYOUT_BLK == STENCIL_LAYOUT)
# error the host path has no blocked layout
#endif

#if !defined(STENCIL_CPU_PAGE)
# define STENCIL_CPU_PAGE 4096
#endif

#define STENCIL_CPU_COORD LIBXSTREAM_CPU_WORKITEM

#if (0 != STENCIL_PML)
# define STENCIL_CPU_APPLY(FN) FN(p_grid, p_old, vel, coeff, \
    eta, phi, hd_2, hd_2, hd_2, dt2, nx, ny, nz)
#else
# define STENCIL_CPU_APPLY(FN) FN(p_grid, p_old, vel, coeff, dt2, nx, ny, nz)
#endif

/* Indirect once per work-group, which is 200k points of work behind the call. */
#define STENCIL_CPU_LAUNCH() STENCIL_CPU_APPLY(kernel)


int stencil_cpu_apply_direct(const float* p_grid, float* p_old,
                             const float* vel, const float* coeff,
                             const float* eta, float* phi,
                             float dt2, float dh,
                             int nx, int ny, int nz, int nterms)
{
  /* The plain FP32 instance also serves a caller that skipped stencil_configure. */
  const stencil_cpu_kernel_t kernel = (NULL != stencil_cpu_kernel)
    ? stencil_cpu_kernel : stencil_cpu_kernels[0];
  int result = EXIT_SUCCESS;
#if (0 != STENCIL_PML)
  const float hd_2 = 0.25f / (dh * dh);
#else
  (void)eta; (void)phi; (void)dh;
#endif

  if (NULL == kernel || NTERMS != nterms
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
#if (STENCIL_LAYOUT_ZYX == STENCIL_LAYOUT)
    const int nfast = nz, nmed = ny, nslow = nx;
#else
    const int nfast = nx, nmed = ny, nslow = nz;
#endif
    const int width_f = (nfast < WG_X) ? nfast : WG_X;
    const int width_m = (nmed < WG_Y) ? nmed : WG_Y;
    const int ng0 = DIVUP(nfast, width_f);
    const int ng1 = DIVUP(nmed, width_m);
    const int ngroups = ng0 * ng1 * DIVUP(nslow, BLK);
    int g;
#if (0 == LIBXSTREAM_CPU_TEAM)
# if defined(_OPENMP)
#   pragma omp parallel for
# endif
    for (g = 0; g < ngroups; ++g) {
      STENCIL_CPU_COORD(g % ng0, (g / ng0) % ng1, g / (ng0 * ng1), 0, 0,
        width_f, width_m);
      STENCIL_CPU_LAUNCH();
    }
#else
    for (g = 0; g < ngroups; ++g) {
#     pragma omp parallel num_threads(WG_X * WG_Y)
      { const int tid = omp_get_thread_num();
        STENCIL_CPU_COORD(g % ng0, (g / ng0) % ng1, g / (ng0 * ng1),
          tid % width_f, tid / width_f, width_f, width_m);
        STENCIL_CPU_LAUNCH();
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
    if (NULL == *ptr) {
      result = EXIT_FAILURE;
    }
    else {
      /**
       * First touch decides NUMA placement, and a serial one puts every page on
       * one node. Touched here with the schedule the launcher walks the groups
       * with, so a thread later works on the pages it placed.
       */
      char *const mem = (char*)*ptr;
      const long npages = (long)(nbytes / STENCIL_CPU_PAGE);
      long i;
#if defined(_OPENMP)
#     pragma omp parallel for
#endif
      for (i = 0; i < npages; ++i) mem[i * STENCIL_CPU_PAGE] = 0;
    }
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
 * The host path carries the FP32 kernel and nothing else, so a request for a
 * DPAS kernel or for a layout the build does not have is refused here rather
 * than ignored on the way to a kernel that cannot honor it. The storage format
 * and the operator radius are instances, hence taken from the environment.
 */
int stencil_init(stencil_context_t* ctx, int verbosity, int method_override)
{
  static const char *const unsupported[] = {
    "STENCIL_BF16", "STENCIL_INT8", "STENCIL_BLOCKED", "STENCIL_LAYOUT"
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
  if (EXIT_SUCCESS == result) {
    const char *const bf16s_env = getenv("STENCIL_BF16S");
    const char *const fp16s_env = getenv("STENCIL_FP16S");
    const char *const halo_env = getenv("STENCIL_HALO");
    const char *const method_env = getenv("STENCIL_METHOD");
    /* A negative override means the caller has no opinion. */
    const int method_val = (0 <= method_override) ? method_override
      : ((NULL != method_env) ? atoi(method_env) : STENCIL_DIRECT);
    memset(ctx, 0, sizeof(*ctx));
    ctx->verbosity = verbosity;
    ctx->strips_per_wg = 1;
    ctx->ndigits_a = 1;
    ctx->ndigits_x = STENCIL_NDIGITS_X;
    ctx->sg = 1;
    ctx->fp32 = 1;
    ctx->fp16 = (NULL != fp16s_env && 0 != atoi(fp16s_env)) ? 1 : 0;
    ctx->bf16s = (0 == ctx->fp16 && NULL != bf16s_env) ? atoi(bf16s_env) : 0;
    if (2 < ctx->bf16s) ctx->bf16s = 2;
    { const int halo_val = (NULL != halo_env) ? atoi(halo_env) : 0;
      ctx->halo[0] = ctx->halo[1] = ctx->halo[2] = halo_val;
    }
    ctx->method = (stencil_method_t)method_val;
    result = stencil_method_params(ctx->method, &ctx->k_steps,
      &ctx->r_per_step);
#if defined(_OPENMP) && (201307 <= _OPENMP)
    /**
     * Unbound threads cost more than any parameter here, and they also undo the
     * first-touch placement, because a thread has to reach the pages it touched.
     * There is no portable way to set the ICVs from inside the process.
     */
    if (0 != verbosity && omp_proc_bind_false == omp_get_proc_bind()) {
      fprintf(stderr, "WARNING: threads are unbound"
        " - export OMP_PROC_BIND=spread OMP_PLACES=cores\n");
    }
#endif
  }
  return result;
}


/**
 * Bind the instance the grid and the environment ask for. Absent combinations
 * are a hole in the table rather than a silent substitution, because a kernel
 * built for another storage format would read the wavefield as the wrong type.
 */
static int stencil_cpu_select(const stencil_context_t* ctx, int padded)
{
  const int sto = (0 != ctx->fp16) ? 3
    : ((2 <= ctx->bf16s) ? 2 : ((1 == ctx->bf16s) ? 1 : 0));
  const int rad = (STENCIL_DIRECT == ctx->method) ? 0 : ctx->r_per_step;
  const int idx = rad * 8 + sto * 2 + (0 != padded ? 1 : 0);
  int result = EXIT_SUCCESS;
  if (STENCIL_CPU_NINST <= idx || NULL == stencil_cpu_kernels[idx]) {
    fprintf(stderr, "ERROR: the host kernel was not built for method %d with"
      " bf16s=%d fp16s=%d; rebuild with CPUDEF=\"%s\"\n", (int)ctx->method,
      ctx->bf16s, ctx->fp16, (0 != rad)
        ? "-DSTENCIL_CPU_COMPACT=1" : "-USTENCIL_PADDED");
    result = EXIT_FAILURE;
  }
  else {
    stencil_cpu_kernel = stencil_cpu_kernels[idx];
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
  if ((0 != ctx->pml) != (0 != STENCIL_PML)) {
    /* Dropping the damping silently would look like a converging run. */
    fprintf(stderr, "ERROR: PML is requested but the host kernel was built"
      " without it; rebuild with CPUDEF=\"-DSTENCIL_PML=1\"\n");
    result = EXIT_FAILURE;
  }
  else if (NTERMS != ctx->nterms) {
    fprintf(stderr, "ERROR: the host kernel was built for %d terms;"
      " rebuild with CPUDEF=\"-DNTERMS=%d\"\n", NTERMS, ctx->nterms);
    result = EXIT_FAILURE;
  }
  else {
    int padded = 0;
    ctx->grid_size[0] = nx;
    ctx->grid_size[1] = ny;
    ctx->grid_size[2] = nz;
    ctx->nblocks[0] = DIVUP(nx, BLK);
    ctx->nblocks[1] = DIVUP(ny, BLK);
    ctx->nblocks[2] = DIVUP(nz, BLK);
    stencil_cpu_p_n = (long)nx * ny * nz;
#if (STENCIL_LAYOUT_ZYX == STENCIL_LAYOUT)
    /* Same test the JIT applies to decide -DSTENCIL_PADDED for this grid. */
    { const int lx = ctx->halo[0], ly = ctx->halo[1], lz = ctx->halo[2];
      const int radius = ctx->r_per_step;
      const int width_f = (nz < WG_X) ? nz : WG_X;
      const int width_m = (ny < WG_Y) ? ny : WG_Y;
      const int max_fast = DIVUP(nz, width_f) * width_f - 1 + radius;
      const int max_med = DIVUP(ny, width_m) * width_m - 1 + radius;
      padded = (max_fast < nz + lz && max_med < ny + ly) ? 1 : 0;
      /* Geometry the JIT would have supplied as -D. */
      stencil_cpu_halo[0] = lx;
      stencil_cpu_halo[1] = ly;
      stencil_cpu_halo[2] = lz;
      stencil_cpu_stride[0] = (long)(nz + 2 * lz) * (ny + 2 * ly);
      stencil_cpu_stride[1] = (long)(nz + 2 * lz);
      stencil_cpu_stride[2] = (long)nz * ny;
      stencil_cpu_stride[3] = (long)nz;
      stencil_cpu_stride[4] = (long)(nz + 2) * (ny + 2);
      stencil_cpu_stride[5] = (long)(nz + 2);
      stencil_cpu_p_n = (long)(nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
#if defined(STENCIL_CPU_FORCE_PAD)
      if (STENCIL_CPU_FORCE_PAD != padded) {
        if (0 != STENCIL_CPU_FORCE_PAD) { /* the halo has to cover the gather */
          fprintf(stderr, "ERROR: %dx%dx%d with halo %dx%dx%d gathers past the"
            " halo; drop -DSTENCIL_PADDED to let the grid decide\n",
            nx, ny, nz, lx, ly, lz);
          result = EXIT_FAILURE;
        }
        else if (0 != ctx->verbosity) { /* correct, but not what the device does */
          fprintf(stderr, "WARNING: %dx%dx%d with halo %dx%dx%d clamps although"
            " the halo covers the gather; drop -DSTENCIL_PADDED=0 to match the"
            " device\n", nx, ny, nz, lx, ly, lz);
        }
        padded = STENCIL_CPU_FORCE_PAD;
      }
#endif
    }
#endif
    if (EXIT_SUCCESS == result) {
      result = stencil_cpu_select(ctx, padded);
    }
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
    /* Same weights the device gets, hence the shared generators. */
    const int r_step = ctx->r_per_step;
    const int r_eff = (1 == ctx->k_steps) ? radius : r_step;
    const int width = 2 * r_eff + 1;
    const int use_fit = (STENCIL_COMPACT_FIT == ctx->method);
    const char *const ppw_env = use_fit ? getenv("STENCIL_PPW") : NULL;
    const char *const fit_env = use_fit ? getenv("STENCIL_FIT") : NULL;
    const double ppw = (NULL != ppw_env) ? atof(ppw_env) : 8.0;
    const int fit_method = (NULL != fit_env) ? atoi(fit_env) : 2;
    const double inv_h2 = -72.0 * fd_weights[radius] / 205.0;
    result = stencil_host_allocate(&ctx->coeff,
      (size_t)3 * width * sizeof(float));
    if (EXIT_SUCCESS == result) {
      float *const coeff = (float*)ctx->coeff;
      int d, r;
      for (d = 0; d < 3; ++d) {
        for (r = 0; r < width; ++r) {
          if (1 == ctx->k_steps) {
            coeff[d * width + r] = (float)fd_weights[r];
          }
          else if (0 != use_fit) {
            coeff[d * width + r] = (float)stencil_fit_weight(r_step, r - r_eff,
              inv_h2, ppw, fit_method);
          }
          else {
            coeff[d * width + r] =
              (float)stencil_compact_weight(r_step, r - r_eff, inv_h2);
          }
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
  result = stencil_cpu_apply_direct((const float*)p_cur, (float*)p_old,
    (const float*)vel, (const float*)ctx->coeff,
    (const float*)ctx->eta, (float*)ctx->phi, dt2, dh,
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
