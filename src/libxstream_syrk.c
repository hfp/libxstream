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
/* single precision: elements per work-item, and depth in k (twice the elements per register) */
#if !defined(LIBXSTREAM_SYRK_RX32)
# define LIBXSTREAM_SYRK_RX32 8
#endif
#if !defined(LIBXSTREAM_SYRK_RY32)
# define LIBXSTREAM_SYRK_RY32 8
#endif
#if !defined(LIBXSTREAM_SYRK_BK32)
# define LIBXSTREAM_SYRK_BK32 32
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
  cl_kernel kernel[2], sum[2]; /* F64 and F32: the panel, and the sum of the slices of k */
  void* part; /* the slices of k beyond the first */
  size_t npart;
  int ncu; /* compute units of the device */
  int built[2];        /* 0: not yet, 1: built, -1: failed (panels are declined) */
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


/* Kernel of the datatype, built at its first use; NULL if the device cannot (e.g., no fp64). */
static cl_kernel libxstream_syrk_kernel(libxstream_syrk_t* ctx, int f64);
static cl_kernel libxstream_syrk_kernel(libxstream_syrk_t* ctx, int f64)
{
  const int which = (0 != f64 ? 0 : 1);
  if (0 == ctx->built[which]) {
    const char* exts[] = {"cl_khr_fp64"};
    char flags[LIBXSTREAM_BUFFERSIZE];
    cl_program program = NULL;
    int result = EXIT_FAILURE;
    const int nchars = LIBXS_SNPRINTF(flags, sizeof(flags), "-DT=%s -DBSX=%i -DBSY=%i -DRX=%i -DRY=%i -DBK=%i",
      0 != f64 ? "double" : "float", LIBXSTREAM_SYRK_BSX, LIBXSTREAM_SYRK_BSY,
      0 != f64 ? LIBXSTREAM_SYRK_RX : LIBXSTREAM_SYRK_RX32, 0 != f64 ? LIBXSTREAM_SYRK_RY : LIBXSTREAM_SYRK_RY32,
      0 != f64 ? LIBXSTREAM_SYRK_BK : LIBXSTREAM_SYRK_BK32);
    if (0 < nchars && (int)sizeof(flags) > nchars) {
      result = libxstream_opencl_program(0 /*source_kind*/, OPENCL_KERNELS_SOURCE_LIBXSTREAM_SYRK, "syrk", flags, NULL /*options*/,
        NULL /*try*/, NULL /*try_ok*/, exts, 0 != f64 ? 1 : 0, &program);
    }
    if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "syrk_panel", ctx->kernel + which);
    if (EXIT_SUCCESS == result) result = libxstream_opencl_kernel_query(program, "syrk_sum", ctx->sum + which);
    if (NULL != program) clReleaseProgram(program); /* the kernel keeps what it needs */
    ctx->built[which] = (EXIT_SUCCESS == result ? 1 : -1);
  }
  return (0 < ctx->built[which] ? ctx->kernel[which] : NULL);
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
  const cl_kernel kernel = libxstream_syrk_kernel(ctx, f64);
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
    const size_t bm = (size_t)LIBXSTREAM_SYRK_BSX * (0 != f64 ? LIBXSTREAM_SYRK_RX : LIBXSTREAM_SYRK_RX32);
    const size_t bn = (size_t)LIBXSTREAM_SYRK_BSY * (0 != f64 ? LIBXSTREAM_SYRK_RY : LIBXSTREAM_SYRK_RY32);
    const cl_int bk = (0 != f64 ? LIBXSTREAM_SYRK_BK : LIBXSTREAM_SYRK_BK32);
    const size_t local[3] = {LIBXSTREAM_SYRK_BSX, LIBXSTREAM_SYRK_BSY, 1};
    const size_t ngroups = LIBXS_UPDIV((size_t)panel->m, bm) * LIBXS_UPDIV((size_t)panel->w, bn);
    const size_t ntarget = (size_t)LIBXSTREAM_SYRK_WAVES * LIBXS_MAX(ctx->ncu, 1);
    /* slices of k for enough work-groups, each at least LIBXSTREAM_SYRK_KMIN deep */
    const cl_int nmost = LIBXS_MAX(k / LIBXSTREAM_SYRK_KMIN, 1);
    const cl_int nwant = (cl_int)LIBXS_MIN(LIBXS_UPDIV(ntarget, ngroups), (size_t)nmost);
    const cl_int kslice = (1 < nwant ? (cl_int)LIBXS_UP(LIBXS_UPDIV(k, nwant), bk) : LIBXS_MAX(k, 1));
    const cl_int nslices = LIBXS_MAX((cl_int)LIBXS_UPDIV(k, kslice), 1), nelems = m * w, nparts = nslices - 1;
    size_t global[3];
    int result = EXIT_SUCCESS, blas_done = 0;
    global[0] = LIBXS_UPDIV((size_t)panel->m, bm) * LIBXSTREAM_SYRK_BSX;
    global[1] = LIBXS_UPDIV((size_t)panel->w, bn) * LIBXSTREAM_SYRK_BSY;
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
      result = libxstream_opencl_launch_work(ctx->stream, kernel, 3, NULL, global, local, 0, NULL, NULL,
        (size_t)2 * (1 + twok) * panel->m * panel->w * plan->shape.k, size * nslices);
    }
    if (EXIT_SUCCESS == result && 0 == blas_done && 0 < nparts) { /* the slices of k into the result */
      const cl_kernel sum = ctx->sum[0 != f64 ? 0 : 1];
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
        c->blas = ((NULL == env || 0 != atoi(env)) && NULL != libxstream_blas_query(c->stream));
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
    if (NULL != ctx->kernel[0]) clReleaseKernel(ctx->kernel[0]);
    if (NULL != ctx->kernel[1]) clReleaseKernel(ctx->kernel[1]);
    if (NULL != ctx->sum[0]) clReleaseKernel(ctx->sum[0]);
    if (NULL != ctx->sum[1]) clReleaseKernel(ctx->sum[1]);
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
