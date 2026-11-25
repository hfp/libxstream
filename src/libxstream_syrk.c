/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxstream/libxstream_syrk.h>
#include <libxstream/libxstream_blas.h>
#include <libxstream/libxstream_opencl.h>
#include "libxstream_syrk_kernels.h"
#include <libxs/libxs_reg.h>
#include <math.h>

#if !defined(OPENCL_KERNELS_SOURCE_LIBXSTREAM_SYRK)
# error "OpenCL kernel code not found (libxstream_syrk_kernels.h)!"
#endif
/* work-items of a work-group along the rows and columns, elements per work-item, and depth in k */
#if !defined(LIBXSTREAM_SYRK_BSX)
# define LIBXSTREAM_SYRK_BSX 16
#endif
#if !defined(LIBXSTREAM_SYRK_BSY)
# define LIBXSTREAM_SYRK_BSY 16
#endif
#if !defined(LIBXSTREAM_SYRK_RX)
# define LIBXSTREAM_SYRK_RX 4
#endif
#if !defined(LIBXSTREAM_SYRK_RY)
# define LIBXSTREAM_SYRK_RY 4
#endif
#if !defined(LIBXSTREAM_SYRK_BK)
# define LIBXSTREAM_SYRK_BK 16
#endif
/* single precision: elements per work-item (4x4 kept 128 GRF and matched 8x8 at 160), and depth in k */
#if !defined(LIBXSTREAM_SYRK_RX32)
# define LIBXSTREAM_SYRK_RX32 4
#endif
#if !defined(LIBXSTREAM_SYRK_RY32)
# define LIBXSTREAM_SYRK_RY32 4
#endif
#if !defined(LIBXSTREAM_SYRK_BK32)
# define LIBXSTREAM_SYRK_BK32 32
#endif
/* unrolling of the kernel (LU, see libxstream_common.h): -1 = no hints (they slowed F32 down) */
#if !defined(LIBXSTREAM_SYRK_LU)
# define LIBXSTREAM_SYRK_LU -1
#endif
/* code version: 0 = local memory, 1 = sub-groups (Intel: SG lanes, RX rows per lane, RY <= SG columns) */
#if !defined(LIBXSTREAM_SYRK_KV)
# define LIBXSTREAM_SYRK_KV 0
#endif
/**
 * Code version in single precision: 2 = sub-groups with 2D block reads (Intel's cl_intel_subgroup_2d_block_io,
 * trans='N', pitch a multiple of 16 bytes), else local memory. The kernel took 1.6-1.9x less time than
 * local memory and A resident led MKL (n=1088 k=54272); sub-groups without them (1) did not lead.
 */
#if !defined(LIBXSTREAM_SYRK_KV32)
# define LIBXSTREAM_SYRK_KV32 2
#endif
#if !defined(LIBXSTREAM_SYRK_SGRX)
# define LIBXSTREAM_SYRK_SGRX 2
#endif
#if !defined(LIBXSTREAM_SYRK_SGRX32)
# define LIBXSTREAM_SYRK_SGRX32 4
#endif
#if !defined(LIBXSTREAM_SYRK_SGRY)
# define LIBXSTREAM_SYRK_SGRY 16
#endif
/* lanes of a sub-group of the code version 1, 0: as the device prefers */
#if !defined(LIBXSTREAM_SYRK_SG)
# define LIBXSTREAM_SYRK_SG 16
#endif
/**
 * Code version 0: register tile of a work-item, 1 = contiguous (vector reads of local memory), 0 = strided.
 * Single precision gained a fifth to a half (n=1088 k=54272), double precision lost a few percent.
 */
#if !defined(LIBXSTREAM_SYRK_VT)
# define LIBXSTREAM_SYRK_VT 0
#endif
#if !defined(LIBXSTREAM_SYRK_VT32)
# define LIBXSTREAM_SYRK_VT32 1
#endif
/* code version 0: elements per lane of a staging load, 0 = 16 bytes, 1 = scalar */
#if !defined(LIBXSTREAM_SYRK_SV)
# define LIBXSTREAM_SYRK_SV 0
#endif
/* SIMD width of the code version 0 (Intel), 0: the compiler decides (it took SIMD32 with 256 GRF in double precision) */
#if !defined(LIBXSTREAM_SYRK_SIMD)
# define LIBXSTREAM_SYRK_SIMD 16
#endif
/* steps of k per 2D block read of the code version 2 (1 to 32) */
#if !defined(LIBXSTREAM_SYRK_B2DK)
# define LIBXSTREAM_SYRK_B2DK 8
#endif
/* sub-groups per work-group of the code version 1 (along the columns), 0: the runtime decides */
#if !defined(LIBXSTREAM_SYRK_NSG)
# define LIBXSTREAM_SYRK_NSG 4
#endif
/* split of k: work-groups per compute unit to aim for, and the least depth of a slice */
#if !defined(LIBXSTREAM_SYRK_WAVES)
# define LIBXSTREAM_SYRK_WAVES 4
#endif
#if !defined(LIBXSTREAM_SYRK_KMIN)
# define LIBXSTREAM_SYRK_KMIN 1024
#endif
/* panels in flight, as many as LIBXS submits before it waits */
#define LIBXSTREAM_SYRK_NSLOTS 2


/* What a kernel hard-codes (code version): a program per key is built once per process (LIBXS_REGKEY_MAXSIZE) */
typedef struct libxstream_syrk_key_t {
  int lda, ldb;
  short sg, nsg, simd, bsx, bsy, rx, ry, bk;
  signed char f64, kv, vt, sv, lu, trans, twok, hasd, upper;
} libxstream_syrk_key_t;

/* A panel in flight: its result on the device, then in host memory for the merge */
typedef struct libxstream_syrk_slot_t {
  libxs_syrk_panel_t panel;
  void *dev, *host;
  size_t size;
  libxstream_event_t* event;
  int f64, upper, busy;
} libxstream_syrk_slot_t;

struct libxstream_syrk_t {
  libxstream_stream_t* stream;
  libxstream_syrk_key_t key; /* of the kernels: the panel, and the sum of the slices of k */
  cl_kernel kernel, sum;
  int built; /* 0: not yet, 1: built, -1: failed (panels are declined) */
  void* part; /* the slices of k beyond the first */
  size_t npart;
  int ncu; /* compute units of the device */
  int blk2d; /* 2D block reads (cl_intel_subgroup_2d_block_io) */
  void *a, *b, *d;     /* device copies of the operands and weights */
  size_t na, nb, nd;
  const void *ha, *hb; /* what the copies hold */
  const void* weights;
  int keep, valid;
  /**
   * The device BLAS (libxstream_blas.h) computes SYRK if it serves the device (LIBXSTREAM_SYRK_BLAS=0
   * disables it): with weights, as holds A*diag(sqrt(d)) per update (scaled is zero if a weight is
   * negative), and hs the square roots in host memory until they are copied.
   */
  int blas, scaled;
  void *as, *s, *hs;
  size_t nas, ns, nhs;
  /* what computed the panels so far: 1 = device BLAS, 2 = kernel, 4 = host (declined), and its label */
  int used;
  char label[32];
  libxstream_syrk_slot_t slot[LIBXSTREAM_SYRK_NSLOTS];
};

/* programs of the code versions (libxstream_syrk_key_t), shared by all contexts; NULL if one failed */
static libxs_registry_t* libxstream_syrk_registry /*= NULL*/;
static libxs_lock_t libxstream_syrk_lock /*= LIBXS_LOCK_INITIALIZER*/;


/* Device memory of at least size bytes, reallocated if smaller. */
static int libxstream_syrk_reserve(void** mem, size_t* capacity, size_t size);
static int libxstream_syrk_reserve(void** mem, size_t* capacity, size_t size)
{
  int result = EXIT_SUCCESS;
  if (*capacity < size) {
    if (NULL != *mem) result = libxstream_mem_deallocate(*mem);
    *mem = NULL;
    *capacity = 0;
    if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(mem, size);
    if (EXIT_SUCCESS == result) *capacity = size;
  }
  return result;
}


/**
 * Kernel of a plan's code version (libxstream_syrk_key_t): its program is built once per process and
 * the context creates its kernels from it. NULL if the device cannot (e.g., no fp64).
 */
static cl_kernel libxstream_syrk_kernel(libxstream_syrk_t* ctx, const libxs_syrk_t* plan);
static cl_kernel libxstream_syrk_kernel(libxstream_syrk_t* ctx, const libxs_syrk_t* plan)
{
  const int f64 = (LIBXS_DATATYPE_F64 == plan->shape.datatype), twok = (0 != plan->shape.ldb);
  libxstream_syrk_key_t key;
  LIBXS_STATIC_ASSERT(sizeof(libxstream_syrk_key_t) <= LIBXS_REGKEY_MAXSIZE);
  LIBXS_MEMZERO(&key); /* padding is part of the key */
  key.f64 = (signed char)f64;
  key.lu = (signed char)LIBXSTREAM_SYRK_LU;
  key.trans = (signed char)('N' != plan->shape.trans);
  key.twok = (signed char)twok;
  key.hasd = (signed char)(NULL != ctx->weights);
  key.upper = (signed char)('U' == plan->shape.uplo);
  key.lda = plan->shape.lda;
  key.ldb = (0 != twok ? plan->shape.ldb : plan->shape.lda);
  { /* code version: 2 only if the 2D block reads take the surfaces (pitch of 16 bytes, width of 64), else local memory */
    const int want = (0 != f64 ? LIBXSTREAM_SYRK_KV : LIBXSTREAM_SYRK_KV32);
    const int intel = (0 != libxstream_opencl_config.device.intel);
    const int pitched = (0 == ((key.lda * 4) % 16) && 64 <= key.lda * 4 && 0 == ((key.ldb * 4) % 16) && 64 <= key.ldb * 4);
    key.sg = (short)(2 == want ? 16 : (0 < LIBXSTREAM_SYRK_SG ? LIBXSTREAM_SYRK_SG : libxstream_opencl_config.device.wgsize[2]));
    key.kv = (signed char)(2 == want ? (0 == f64 && 0 != ctx->blk2d && 0 == key.trans && 0 != pitched ? 2 : 0)
                                     : (0 != want && 0 != intel && 0 < key.sg ? 1 : 0));
  }
  key.nsg = (short)(0 != key.kv ? LIBXSTREAM_SYRK_NSG : 0);
  key.simd = (short)(0 == key.kv ? LIBXSTREAM_SYRK_SIMD : 0);
  key.vt = (signed char)(0 == key.kv ? (0 != f64 ? LIBXSTREAM_SYRK_VT : LIBXSTREAM_SYRK_VT32) : 0);
  key.sv = (signed char)(0 == key.kv ? (0 < LIBXSTREAM_SYRK_SV ? LIBXSTREAM_SYRK_SV : (0 != f64 ? 2 : 4)) : 1);
  key.bsx = (short)(0 != key.kv ? key.sg : LIBXSTREAM_SYRK_BSX);
  key.bsy = (short)(0 != key.kv ? LIBXS_MAX(key.nsg, 1) : LIBXSTREAM_SYRK_BSY);
  key.rx = (short)(0 != key.kv ? (0 != f64 ? LIBXSTREAM_SYRK_SGRX : LIBXSTREAM_SYRK_SGRX32)
                               : (0 != f64 ? LIBXSTREAM_SYRK_RX : LIBXSTREAM_SYRK_RX32));
  key.ry = (short)(0 != key.kv ? LIBXS_MIN(LIBXSTREAM_SYRK_SGRY, key.sg)
                               : (0 != f64 ? LIBXSTREAM_SYRK_RY : LIBXSTREAM_SYRK_RY32));
  key.bk = (short)(2 == key.kv ? LIBXSTREAM_SYRK_B2DK : (0 != f64 ? LIBXSTREAM_SYRK_BK : LIBXSTREAM_SYRK_BK32));
  if (0 == ctx->built || 0 != memcmp(&key, &ctx->key, sizeof(key))) {
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    cl_program* program;
    int result = EXIT_FAILURE;
    if (NULL != ctx->kernel) clReleaseKernel(ctx->kernel);
    if (NULL != ctx->sum) clReleaseKernel(ctx->sum);
    ctx->kernel = ctx->sum = NULL;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxstream_syrk_lock);
    if (NULL == libxstream_syrk_registry) libxstream_syrk_registry = libxs_registry_create();
    program = (NULL != libxstream_syrk_registry ? (cl_program*)libxs_registry_get(libxstream_syrk_registry, &key, sizeof(key),
                                                    libxs_registry_lock(libxstream_syrk_registry))
                                                : NULL);
    if (NULL == program && NULL != libxstream_syrk_registry) { /* the first use of this code version */
      const char* exts[] = {"cl_khr_fp64"};
      char flags[LIBXSTREAM_BUFFERSIZE];
      cl_program built = NULL;
      int status = EXIT_FAILURE;
      const int nchars = LIBXS_SNPRINTF(flags, sizeof(flags),
        "-DT=%s -DSYRK_SG=%i -DSYRK_SIMD=%i -DVT=%i -DSV=%i -DBSX=%i -DBSY=%i -DRX=%i -DRY=%i -DBK=%i -DLU=%i -DSG=%i -DINTEL=%i%s"
        " -DSYRK_TRANS=%i -DSYRK_TWOK=%i -DSYRK_HASD=%i -DSYRK_UPPER=%i -DSYRK_LDA=%i -DSYRK_LDB=%i",
        0 != f64 ? "double" : "float", (int)key.kv, (int)key.simd, (int)key.vt, (int)key.sv, (int)key.bsx, (int)key.bsy, (int)key.rx, (int)key.ry, (int)key.bk,
        (int)key.lu, (int)key.sg, (int)(0 != devinfo->intel), CL_DEVICE_TYPE_GPU == devinfo->type ? " -DGPU" : "",
        (int)key.trans, (int)key.twok, (int)key.hasd, (int)key.upper, key.lda, key.ldb);
      if (0 < nchars && (int)sizeof(flags) > nchars) {
        status = libxstream_opencl_program(0 /*source_kind*/, OPENCL_KERNELS_SOURCE_LIBXSTREAM_SYRK, "syrk", flags,
          NULL /*options*/, NULL /*try*/, NULL /*try_ok*/, exts, 0 != f64 ? 1 : 0, &built);
      }
      if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity || EXIT_SUCCESS != status) {
        fprintf(stderr, "%s LIBXSTREAM: SYRK kernel %s\n", EXIT_SUCCESS == status ? "INFO" : "ERROR", flags);
      }
      program = (cl_program*)libxs_registry_set(libxstream_syrk_registry, &key, sizeof(key), &built, sizeof(built),
        libxs_registry_lock(libxstream_syrk_registry));
    }
    if (NULL != program && NULL != *program) {
      result = libxstream_opencl_kernel_query(*program, "syrk_panel", &ctx->kernel);
      if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(*program, "syrk_sum", &ctx->sum);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxstream_syrk_lock);
    ctx->key = key;
    ctx->built = (EXIT_SUCCESS == result ? 1 : -1);
  }
  return (0 < ctx->built ? ctx->kernel : NULL);
}


/* Copies the operands (and weights) of an update to the device, unless they are resident. */
static int libxstream_syrk_upload(libxstream_syrk_t* ctx, const libxs_syrk_t* plan, const void* a, const void* b);
static int libxstream_syrk_upload(libxstream_syrk_t* ctx, const libxs_syrk_t* plan, const void* a, const void* b)
{
  const size_t es = LIBXS_TYPESIZE(plan->shape.datatype);
  const int notrans = ('N' == plan->shape.trans), twok = (0 != plan->shape.ldb);
  const size_t ncols = (size_t)(0 != notrans ? plan->shape.k : plan->shape.n);
  int result = EXIT_SUCCESS;
  if (0 == ctx->keep || 0 == ctx->valid || ctx->ha != a || ctx->hb != b) {
    const size_t sa = (size_t)plan->shape.lda * ncols * es, sb = (size_t)plan->shape.ldb * ncols * es;
    ctx->valid = 0;
    result = libxstream_syrk_reserve(&ctx->a, &ctx->na, LIBXS_MAX(sa, es));
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(a, ctx->a, sa, ctx->stream);
    if (EXIT_SUCCESS == result && 0 != twok) {
      result = libxstream_syrk_reserve(&ctx->b, &ctx->nb, LIBXS_MAX(sb, es));
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(b, ctx->b, sb, ctx->stream);
    }
    if (EXIT_SUCCESS == result) {
      ctx->ha = a;
      ctx->hb = b;
      ctx->valid = 1;
    }
  }
  if (EXIT_SUCCESS == result && NULL != ctx->weights) {
    const size_t sd = (size_t)plan->shape.k * es;
    result = libxstream_syrk_reserve(&ctx->d, &ctx->nd, LIBXS_MAX(sd, es));
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(ctx->weights, ctx->d, sd, ctx->stream);
    ctx->scaled = 0;
    if (EXIT_SUCCESS == result && 0 != ctx->blas && 0 == twok) { /* A*diag(sqrt(d)) for the device BLAS */
      const int f64 = (LIBXS_DATATYPE_F64 == plan->shape.datatype), k = plan->shape.k;
      int i, positive = 1;
      if (EXIT_SUCCESS == libxstream_stream_sync(ctx->stream) && ctx->nhs < sd) { /* hs may still be read */
        free(ctx->hs);
        ctx->hs = malloc(sd);
        ctx->nhs = (NULL != ctx->hs ? sd : 0);
      }
      for (i = 0; i < k && NULL != ctx->hs && 0 != positive; ++i) {
        const double w = (0 != f64 ? ((const double*)ctx->weights)[i] : (double)((const float*)ctx->weights)[i]);
        if (0 <= w) {
          if (0 != f64) ((double*)ctx->hs)[i] = sqrt(w);
          else ((float*)ctx->hs)[i] = (float)sqrt(w);
        }
        else positive = 0; /* the own kernel takes any weights */
      }
      if (NULL != ctx->hs && 0 != positive && EXIT_SUCCESS == libxstream_syrk_reserve(&ctx->s, &ctx->ns, LIBXS_MAX(sd, es))
        && EXIT_SUCCESS == libxstream_syrk_reserve(&ctx->as, &ctx->nas, ctx->na)
        && EXIT_SUCCESS == libxstream_mem_copy_h2d(ctx->hs, ctx->s, sd, ctx->stream)
        && EXIT_SUCCESS == libxstream_blas_dgmm(ctx->stream, plan->shape.datatype, 0 != notrans ? 'R' : 'L',
          0 != notrans ? plan->shape.n : k, 0 != notrans ? k : plan->shape.n, ctx->a, plan->shape.lda, ctx->s, 1, ctx->as,
          plan->shape.lda))
      {
        ctx->scaled = 1;
      }
    }
  }
  return result;
}


/**
 * A panel by the device BLAS into its compact buffer: SYRK of the block on the diagonal, which writes
 * its triangle only, and GEMM of the rows above (upper) or below (lower) it.
 */
static int libxstream_syrk_blas(libxstream_syrk_t* ctx, const libxs_syrk_t* plan, const libxs_syrk_panel_t* panel, void* out);
static int libxstream_syrk_blas(libxstream_syrk_t* ctx, const libxs_syrk_t* plan, const libxs_syrk_panel_t* panel, void* out)
{
  const libxs_data_t dt = plan->shape.datatype;
  const size_t es = LIBXS_TYPESIZE(dt);
  const int notrans = ('N' == plan->shape.trans), upper = ('U' == plan->shape.uplo);
  const int k = plan->shape.k, lda = plan->shape.lda, w = panel->w, m = panel->m;
  const int rect = m - w, r1 = (0 != upper ? 0 : (panel->c0 + w)); /* rows of the rectangle, and its first one */
  const char* const a = (const char*)(NULL != ctx->weights ? ctx->as : ctx->a);
  const size_t stride = (0 != notrans ? 1 : (size_t)lda) * es; /* from one row of op(A) to the next */
  char* const diag = (char*)out + (size_t)(panel->c0 - panel->r0) * es;
  int result = libxstream_blas_syrk(ctx->stream, dt, plan->shape.uplo, plan->shape.trans, w, k, panel->alpha,
    a + panel->c0 * stride, lda, NULL, 0, 0.0, diag, m);
  if (EXIT_SUCCESS == result && 0 < rect) {
    result = libxstream_blas_gemm(ctx->stream, dt, 0 != notrans ? 'N' : 'T', 0 != notrans ? 'T' : 'N', rect, w, k,
      panel->alpha, a + r1 * stride, lda, a + panel->c0 * stride, lda, 0.0,
      (char*)out + (size_t)(0 != upper ? 0 : w) * es, m);
  }
  return result;
}


/* Starts a panel: kernel, then the copy of its result to host memory (see libxs_syrk_device_t). */
static int libxstream_syrk_submit(void* vctx, const libxs_syrk_t* plan, const void* a, const void* b,
  const libxs_syrk_panel_t* panel);
static int libxstream_syrk_submit(void* vctx, const libxs_syrk_t* plan, const void* a, const void* b,
  const libxs_syrk_panel_t* panel)
{
  libxstream_syrk_t *const ctx = (libxstream_syrk_t*)vctx;
  const int f64 = (LIBXS_DATATYPE_F64 == plan->shape.datatype), twok = (0 != plan->shape.ldb);
  const size_t es = LIBXS_TYPESIZE(plan->shape.datatype), size = (size_t)panel->m * panel->w * es;
  const int blas = (0 != ctx->blas && 0 == twok);
  const cl_kernel kernel = libxstream_syrk_kernel(ctx, plan);
  int ticket = -1, s = 0;
  for (; s < LIBXSTREAM_SYRK_NSLOTS && 0 != ctx->slot[s].busy; ++s);
  if ((NULL != kernel || 0 != blas) && LIBXSTREAM_SYRK_NSLOTS > s && 0 < size) { /* else the host computes the panel */
    libxstream_syrk_slot_t *const slot = ctx->slot + s;
    const cl_int m = panel->m, w = panel->w, r0 = panel->r0, c0 = panel->c0, k = plan->shape.k;
    const cl_int lda = plan->shape.lda, ldb = (0 != twok ? plan->shape.ldb : plan->shape.lda);
    const cl_int trans = ('N' != plan->shape.trans), ctwok = twok, hasd = (NULL != ctx->weights);
    const cl_int upper = ('U' == plan->shape.uplo);
    const cl_double alpha64 = panel->alpha;
    const cl_float alpha32 = (cl_float)panel->alpha;
    /* a work-group: bsx x bsy work-items of rx x ry elements, or (kv) bsy sub-groups of sg lanes along the columns */
    const size_t bm = (size_t)ctx->key.bsx * ctx->key.rx;
    const size_t bn = (size_t)ctx->key.bsy * ctx->key.ry;
    const cl_int bk = (1 != ctx->key.kv ? ctx->key.bk : 1); /* slices of whole steps (2D block reads too) */
    size_t global[3], local[3];
    const size_t ngroups = LIBXS_UPDIV((size_t)panel->m, bm) * LIBXS_UPDIV((size_t)panel->w, bn);
    const size_t ntarget = (size_t)LIBXSTREAM_SYRK_WAVES * LIBXS_MAX(ctx->ncu, 1);
    /* slices of k for enough work-groups, each at least LIBXSTREAM_SYRK_KMIN deep */
    const cl_int nmost = LIBXS_MAX(k / LIBXSTREAM_SYRK_KMIN, 1);
    const cl_int nwant = (cl_int)LIBXS_MIN(LIBXS_UPDIV(ntarget, ngroups), (size_t)nmost);
    const cl_int kslice = (1 < nwant ? (cl_int)LIBXS_UP(LIBXS_UPDIV(k, nwant), bk) : LIBXS_MAX(k, 1));
    const cl_int nslices = LIBXS_MAX((cl_int)LIBXS_UPDIV(k, kslice), 1), nelems = m * w, nparts = nslices - 1;
    int result = EXIT_SUCCESS, blas_done = 0;
    local[0] = (size_t)ctx->key.bsx;
    local[1] = (size_t)ctx->key.bsy;
    local[2] = 1;
    global[0] = LIBXS_UPDIV((size_t)panel->m, bm) * local[0];
    global[1] = LIBXS_UPDIV((size_t)panel->w, bn) * local[1]; /* kv: a column block per sub-group */
    global[2] = (size_t)nslices;
    if (0 == panel->c0) result = libxstream_syrk_upload(ctx, plan, a, b); /* the first panel of an update */
    if (EXIT_SUCCESS == result && slot->size < size) {
      if (NULL != slot->host) result = libxstream_mem_host_deallocate(slot->host, ctx->stream);
      slot->host = NULL;
      slot->size = 0;
      if (EXIT_SUCCESS == result && NULL != slot->dev) result = libxstream_mem_deallocate(slot->dev);
      slot->dev = NULL;
      if (EXIT_SUCCESS == result) result = libxstream_mem_allocate(&slot->dev, size);
      if (EXIT_SUCCESS == result) result = libxstream_mem_host_allocate(&slot->host, size, ctx->stream);
      if (EXIT_SUCCESS == result) slot->size = size;
    }
    if (EXIT_SUCCESS == result && NULL == slot->event) result = libxstream_event_create(&slot->event);
    if (EXIT_SUCCESS == result && 0 != blas && (NULL == ctx->weights || 0 != ctx->scaled)
      && EXIT_SUCCESS == libxstream_syrk_blas(ctx, plan, panel, slot->dev))
    {
      blas_done = 1;
    }
    else if (NULL == kernel) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result && 0 == blas_done) result = libxstream_opencl_set_kernel_ptr(kernel, 0, slot->dev);
    if (EXIT_SUCCESS == result && 0 == blas_done) result = libxstream_opencl_set_kernel_ptr(kernel, 1, ctx->a);
    if (EXIT_SUCCESS == result && 0 == blas_done) {
      result = libxstream_opencl_set_kernel_ptr(kernel, 2, 0 != twok ? ctx->b : ctx->a);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done) {
      result = libxstream_opencl_set_kernel_ptr(kernel, 3, NULL != ctx->weights ? ctx->d : ctx->a);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done) {
      cl_int status = clSetKernelArg(kernel, 4, sizeof(cl_int), &m);
      status |= clSetKernelArg(kernel, 5, sizeof(cl_int), &w);
      status |= clSetKernelArg(kernel, 6, sizeof(cl_int), &r0);
      status |= clSetKernelArg(kernel, 7, sizeof(cl_int), &c0);
      status |= clSetKernelArg(kernel, 8, sizeof(cl_int), &k);
      status |= clSetKernelArg(kernel, 9, sizeof(cl_int), &lda);
      status |= clSetKernelArg(kernel, 10, sizeof(cl_int), &ldb);
      status |= clSetKernelArg(kernel, 11, sizeof(cl_int), &trans);
      status |= clSetKernelArg(kernel, 12, sizeof(cl_int), &ctwok);
      status |= clSetKernelArg(kernel, 13, sizeof(cl_int), &hasd);
      if (0 != f64) status |= clSetKernelArg(kernel, 14, sizeof(cl_double), &alpha64);
      else status |= clSetKernelArg(kernel, 14, sizeof(cl_float), &alpha32);
      status |= clSetKernelArg(kernel, 15, sizeof(cl_int), &upper);
      status |= clSetKernelArg(kernel, 17, sizeof(cl_int), &kslice);
      if (CL_SUCCESS != status) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 0 == blas_done && 0 < nparts) {
      result = libxstream_syrk_reserve(&ctx->part, &ctx->npart, (size_t)nparts * size);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done) {
      result = libxstream_opencl_set_kernel_ptr(kernel, 16, 0 < nparts ? ctx->part : slot->dev);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done) {
      result = libxstream_opencl_launch_work(ctx->stream, kernel, 3, NULL, global,
        (0 == ctx->key.kv || 0 != ctx->key.nsg) ? local : NULL, 0, NULL, NULL,
        (size_t)2 * (1 + twok) * panel->m * panel->w * plan->shape.k, size * nslices);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done && 0 < nparts) { /* the slices of k into the result */
      const cl_kernel sum = ctx->sum;
      const size_t nsum = LIBXS_UP((size_t)nelems, 64);
      cl_int status;
      result = libxstream_opencl_set_kernel_ptr(sum, 0, slot->dev);
      if (EXIT_SUCCESS == result) result = libxstream_opencl_set_kernel_ptr(sum, 1, ctx->part);
      status = clSetKernelArg(sum, 2, sizeof(cl_int), &nelems);
      status |= clSetKernelArg(sum, 3, sizeof(cl_int), &nparts);
      if (EXIT_SUCCESS == result && CL_SUCCESS != status) result = EXIT_FAILURE;
      if (EXIT_SUCCESS == result) {
        result = libxstream_opencl_launch_work(ctx->stream, sum, 1, NULL, &nsum, NULL, 0, NULL, NULL,
          (size_t)nelems * nparts, size * nslices);
      }
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(slot->dev, slot->host, size, ctx->stream);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(slot->event, ctx->stream);
    if (EXIT_SUCCESS == result) {
      slot->panel = *panel;
      slot->f64 = f64;
      slot->upper = ('U' == plan->shape.uplo);
      slot->busy = 1;
      ctx->used |= (0 != blas_done ? 1 : 2);
      ticket = s;
    }
    else ctx->valid = 0; /* copy again next time */
  }
  if (0 > ticket) ctx->used |= 4;
  return ticket;
}


/* Completes a panel: out := result + beta*out within the triangle (beta=0 does not read out). */
static int libxstream_syrk_wait(void* vctx, int ticket);
static int libxstream_syrk_wait(void* vctx, int ticket)
{
  libxstream_syrk_t *const ctx = (libxstream_syrk_t*)vctx;
  int result = EXIT_FAILURE;
  if (0 <= ticket && LIBXSTREAM_SYRK_NSLOTS > ticket && 0 != ctx->slot[ticket].busy) {
    libxstream_syrk_slot_t *const slot = ctx->slot + ticket;
    const libxs_syrk_panel_t *const panel = &slot->panel;
    result = libxstream_event_sync(slot->event);
    if (EXIT_SUCCESS == result) {
      const double beta = panel->beta;
      int i, j;
      for (j = 0; j < panel->w; ++j) {
        const int gj = panel->c0 + j;
        const int begin = (0 != slot->upper ? 0 : (gj - panel->r0));
        const int end = (0 != slot->upper ? (gj - panel->r0 + 1) : panel->m);
        if (0 != slot->f64) {
          const double *const src = (const double*)slot->host + (size_t)j * panel->m;
          double *const dst = (double*)panel->out + (size_t)j * panel->ldo;
          if (0 == beta) {
            for (i = begin; i < end; ++i) dst[i] = src[i];
          }
          else {
            for (i = begin; i < end; ++i) dst[i] = src[i] + beta * dst[i];
          }
        }
        else {
          const float *const src = (const float*)slot->host + (size_t)j * panel->m;
          float *const dst = (float*)panel->out + (size_t)j * panel->ldo;
          const float fbeta = (float)beta;
          if (0 == beta) {
            for (i = begin; i < end; ++i) dst[i] = src[i];
          }
          else {
            for (i = begin; i < end; ++i) dst[i] = src[i] + fbeta * dst[i];
          }
        }
      }
    }
    slot->busy = 0;
  }
  return result;
}


LIBXSTREAM_API int libxstream_syrk_create(libxstream_syrk_t** ctx)
{
  int result = EXIT_FAILURE;
  if (NULL != ctx) {
    libxstream_syrk_t *const c = (libxstream_syrk_t*)malloc(sizeof(libxstream_syrk_t));
    *ctx = NULL; /* also if the context cannot be created */
    if (NULL != c) {
      LIBXS_MEMZERO(c);
      result = libxstream_stream_create(&c->stream, "syrk", LIBXSTREAM_STREAM_DEFAULT);
      if (EXIT_SUCCESS == result) {
        const char* const env = getenv("LIBXSTREAM_SYRK_BLAS");
        cl_uint ncu = 0;
        const char* const blk2d[] = {"cl_intel_subgroup_2d_block_io"};
        c->blas = ((NULL == env || 0 != atoi(env)) && NULL != libxstream_blas_query(c->stream));
        c->blk2d = (EXIT_SUCCESS == libxstream_opencl_device_ext(
                                      libxstream_opencl_config.devices[libxstream_opencl_config.device_id], blk2d, 1));
        if (CL_SUCCESS == clGetDeviceInfo(libxstream_opencl_config.devices[libxstream_opencl_config.device_id],
              CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ncu, NULL))
        {
          c->ncu = (int)ncu;
        }
      }
      if (EXIT_SUCCESS == result) *ctx = c;
      else free(c);
    }
  }
  return result;
}


LIBXSTREAM_API void libxstream_syrk_destroy(libxstream_syrk_t* ctx)
{
  if (NULL != ctx) {
    int s = 0;
    if (NULL != ctx->stream) libxstream_stream_sync(ctx->stream);
    for (; s < LIBXSTREAM_SYRK_NSLOTS; ++s) {
      if (NULL != ctx->slot[s].event) libxstream_event_destroy(ctx->slot[s].event);
      if (NULL != ctx->slot[s].dev) libxstream_mem_deallocate(ctx->slot[s].dev);
      if (NULL != ctx->slot[s].host) libxstream_mem_host_deallocate(ctx->slot[s].host, ctx->stream);
    }
    if (NULL != ctx->a) libxstream_mem_deallocate(ctx->a);
    if (NULL != ctx->b) libxstream_mem_deallocate(ctx->b);
    if (NULL != ctx->d) libxstream_mem_deallocate(ctx->d);
    if (NULL != ctx->as) libxstream_mem_deallocate(ctx->as);
    if (NULL != ctx->s) libxstream_mem_deallocate(ctx->s);
    if (NULL != ctx->part) libxstream_mem_deallocate(ctx->part);
    free(ctx->hs);
    if (NULL != ctx->kernel) clReleaseKernel(ctx->kernel);
    if (NULL != ctx->sum) clReleaseKernel(ctx->sum);
    if (NULL != ctx->stream) libxstream_stream_destroy(ctx->stream);
    free(ctx);
  }
}


LIBXSTREAM_API void libxstream_syrk_device(libxstream_syrk_t* ctx, libxs_syrk_device_t* device)
{
  if (NULL != device) {
    device->submit = libxstream_syrk_submit;
    device->wait = libxstream_syrk_wait;
    device->ctx = ctx;
  }
}


LIBXSTREAM_API const char* libxstream_syrk_backend(libxstream_syrk_t* ctx)
{
  const char* result = "none";
  if (NULL != ctx && 0 != ctx->used) {
    const char* const name = libxstream_blas_query(ctx->stream);
    const int used = ctx->used;
    LIBXS_SNPRINTF(ctx->label, sizeof(ctx->label), "%s%s%s%s%s", 0 != (1 & used) ? (NULL != name ? name : "BLAS") : "",
      (1 & used) && (6 & used) ? "+" : "", 0 != (2 & used) ? "kernel" : "", (2 & used) && (4 & used) ? "+" : "",
      0 != (4 & used) ? "host" : "");
    result = ctx->label;
  }
  return result;
}


LIBXSTREAM_API void libxstream_syrk_weights(libxstream_syrk_t* ctx, const void* d)
{
  if (NULL != ctx) ctx->weights = d;
}


LIBXSTREAM_API void libxstream_syrk_resident(libxstream_syrk_t* ctx, int keep)
{
  if (NULL != ctx) {
    ctx->keep = keep;
    ctx->valid = 0;
  }
}
