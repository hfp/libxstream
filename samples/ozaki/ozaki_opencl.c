/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Embedded kernel source (generated at build time via acc_opencl.sh) */
#include "ozaki_kernels.h"

#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_INT8)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_INT8)"
#endif
#if !defined(OPENCL_KERNELS_SOURCE_OZAKI1_BF16)
# error "OpenCL kernel source not found (ozaki_kernels.h must define OPENCL_KERNELS_SOURCE_OZAKI1_BF16)"
#endif


#define CL_CHECK(CALL) do { \
  cl_int _err = (CALL); \
  if (CL_SUCCESS != _err) { \
    fprintf(stderr, "ERROR %s (%d) at %s:%d\n", cl_strerror(_err), (int)_err, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


/* Internal helpers */
static void ozaki_print_opt(FILE* stream, const char* name, int val);


int ozaki_init(ozaki_context_t* ctx, int bm, int bn, int bk,
               int use_double, int kind, int verbosity,
               int nslices, int batch_k,
               int ozflags, int oztrim)
{
  const libxstream_opencl_device_t* devinfo = &libxstream_opencl_config.device;
  cl_device_id device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  char build_options[256];
  char build_params[1024];
  size_t offset;
  const char* kernel_source;
  const char* env;
  int use_bf16, wg, sg, gpu, use_xmx, result;

  /* Derive bf16 flag from kind (3 = bf16, else int8) */
  use_bf16 = (3 == kind) ? 1 : 0;
  if (0 >= kind) kind = 1;

  gpu = (CL_DEVICE_TYPE_GPU == devinfo->type) ? 1 : 0;

  { char name[256] = "";
    libxstream_opencl_device_name(device, name, sizeof(name), NULL, 0, 1 /*cleanup*/);
    printf("Device: %s%s\n", name, gpu ? " (GPU)" : "");
  }

  /* If double requested, verify fp64 support */
  if (use_double) {
    const char* const fp64_ext[] = {"cl_khr_fp64"};
    if (EXIT_SUCCESS != libxstream_opencl_device_ext(device, fp64_ext, 1)) {
      fprintf(stderr, "WARN: device does not support cl_khr_fp64, falling back to float\n");
      use_double = 0;
    }
  }

  /* Detect hardware matrix multiply support (before choosing defaults) */
  { const char* const xmx_exts[] = {
      "cl_intel_subgroup_matrix_multiply_accumulate",
      "cl_intel_subgroup_2d_block_io"
    };
    env = getenv("OZAKI_XMX");
    if (NULL != env) {
      use_xmx = atoi(env);
    }
    else {
      use_xmx = (EXIT_SUCCESS == libxstream_opencl_device_ext(
                   device, xmx_exts, 2)) ? 1 : 0;
    }
  }

  /* Choose smart defaults: XMX-friendly when hardware is available.
   * XMX requires BK==32 (int8) or BK==16 (bf16), BM/BN divisible by 8. */
  if (0 >= bm) bm = 16;
  if (0 >= bn) bn = 16;
  if (0 >= bk) bk = use_xmx ? (use_bf16 ? 16 : 32) : 16;
  if (0 >= nslices) nslices = 8;
  if (0 >= batch_k) batch_k = 4;
  if (0 > ozflags) ozflags = OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE;

  /* Validate XMX constraints against final block sizes.
   * DPAS SG=16: XMX_N=16, so BN must be divisible by 16. */
  if (use_xmx && ((use_bf16 ? 16 : 32) != bk || 0 != (bm % 8) || 0 != (bn % 16))) {
    if (0 < verbosity) {
      fprintf(stderr, "INFO OZAKI: XMX disabled (BK=%d, BM=%d, BN=%d)\n",
              bk, bm, bn);
    }
    use_xmx = 0;
  }

  ctx->bm = bm;
  ctx->bn = bn;
  ctx->bk = bk;
  ctx->batch_k = batch_k;
  ctx->use_double = use_double;
  ctx->use_bf16 = use_bf16;
  ctx->use_xmx = use_xmx;
  ctx->nslices = nslices;
  ctx->kind = kind;
  ctx->ozflags = ozflags;
  ctx->oztrim  = oztrim;
  ctx->verbosity = verbosity;

  /* Environment-driven tuning */
  env = getenv("OZAKI_WG");
  wg = (NULL != env ? atoi(env) : 0);
  env = getenv("OZAKI_SG");
  sg = (NULL != env ? atoi(env) : (int)devinfo->wgsize[2]);

  /* 2D block I/O and SG=16 DPAS both require sub-group size 16 */
  if (use_xmx && 16 != sg) {
    if (0 < verbosity) {
      fprintf(stderr, "INFO OZAKI: SG forced to 16 for XMX\n");
    }
    sg = 16;
  }

  /* Assemble JIT build flags: compiler options and -D defines separately */
  LIBXS_SNPRINTF(build_options, sizeof(build_options),
    "-cl-fast-relaxed-math -cl-denorms-are-zero");

  { const char* constant_qual;

    env = getenv("OZAKI_CONSTANT");
    constant_qual = (NULL != env && 0 != atoi(env)) ? "constant" : "global";

    offset = 0;
    if (gpu) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        "-DGPU ");
    }

    offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
      "-DCONSTANT=%s -DBM=%d -DBN=%d -DBK=%d -DNSLICES=%d"
      " -DTRIANGULAR=%d -DSYMMETRIZE=%d -DTRIM=%d"
      " -DUSE_DOUBLE=%d",
      constant_qual, bm, bn, bk, nslices,
      (ozflags & OZAKI_TRIANGULAR) ? 1 : 0,
      (ozflags & OZAKI_SYMMETRIZE) ? 1 : 0,
      oztrim, use_double);

    if (!use_bf16) {
      const int mant_bits      = use_double ? 52 : 23;
      const int bias_plus_mant = use_double ? 1075 : 150;
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DMANT_BITS=%d -DBIAS_PLUS_MANT=%d", mant_bits, bias_plus_mant);
    }

    if (0 < wg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DWG=%d", wg);
    }
    if (0 < sg) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DSG=%d", sg);
    }
    if (use_xmx) {
      offset += (size_t)LIBXS_SNPRINTF(build_params + offset, sizeof(build_params) - offset,
        " -DUSE_XMX=1");
    }
  }
  ctx->sg = sg;

  if (0 < verbosity) {
    fprintf(stderr, "INFO OZAKI: build params: %s\n", build_params);
    fprintf(stderr, "INFO OZAKI: build options: %s\n", build_options);
  }

  /* JIT compile kernels via ACC */
  kernel_source = use_bf16
    ? OPENCL_KERNELS_SOURCE_OZAKI1_BF16
    : OPENCL_KERNELS_SOURCE_OZAKI1_INT8;

  result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
    "preprocess_a", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_preprocess_a);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build preprocess_a kernel\n");
    return EXIT_FAILURE;
  }

  result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
    "preprocess_b", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_preprocess_b);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build preprocess_b kernel\n");
    return EXIT_FAILURE;
  }

  result = libxstream_opencl_kernel(0 /*source*/, kernel_source,
    "dotprod", build_params, build_options,
    NULL, NULL, NULL, 0, &ctx->kern_dotprod);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: failed to build dotprod kernel\n");
    return EXIT_FAILURE;
  }

  /* Report compiled kernel info */
  if (0 < verbosity) {
    size_t wgs[3] = {0};
    if (CL_SUCCESS == clGetKernelWorkGroupInfo(ctx->kern_dotprod, device,
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgs), wgs, NULL))
    {
      fprintf(stderr, "INFO OZAKI: dotprod-%s compiled for WG=%ux%ux%u\n", use_bf16 ? "bf16" : "int8",
        LIBXS_CAST_UINT(wgs[0]), LIBXS_CAST_UINT(wgs[1]), LIBXS_CAST_UINT(wgs[2]));
    }
    fprintf(stderr, "INFO OZAKI: gpu=%d", gpu);
    ozaki_print_opt(stderr, "bf16", use_bf16);
    ozaki_print_opt(stderr, "fp", use_double ? 64 : 32);
    ozaki_print_opt(stderr, "xmx", use_xmx);
    ozaki_print_opt(stderr, "wg", wg);
    ozaki_print_opt(stderr, "sg", sg);
    ozaki_print_opt(stderr, "nslices", nslices);
    fprintf(stderr, "\n");
  }

  return EXIT_SUCCESS;
}


void ozaki_destroy(ozaki_context_t* ctx)
{
  if (NULL != ctx->kern_preprocess_a) clReleaseKernel(ctx->kern_preprocess_a);
  if (NULL != ctx->kern_preprocess_b) clReleaseKernel(ctx->kern_preprocess_b);
  if (NULL != ctx->kern_dotprod)      clReleaseKernel(ctx->kern_dotprod);
  LIBXS_MEMZERO(ctx);
}


int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream,
               char transa, char transb,
               int M, int N, int K,
               double alpha, const void* a, int lda,
                             const void* b, int ldb,
               double beta,        void* c, int ldc)
{
  const libxstream_opencl_stream_t* str = stream;
  const int BM = ctx->bm, BN = ctx->bn, BK = ctx->bk;
  /* Pad B column stride so surface width >= 64 bytes (2D block I/O).
   * bf16: 2 bytes/elem, min 32 elements.  int8: 1 byte/elem, min 64. */
  const int bn_min = ctx->use_bf16 ? 32 : 64;
  const int BN_PAD = ctx->use_xmx ? ((BN < bn_min) ? bn_min : BN) : BN;
  const int BATCH_K = ctx->batch_k;
  const int nslices = ctx->nslices;
  const int nblk_m = (M + BM - 1) / BM;
  const int nblk_n = (N + BN - 1) / BN;
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  const int nkb_total = (K + BK - 1) / BK;
  const int max_nkb = (BATCH_K < nkb_total) ? BATCH_K : nkb_total;
  const int n_batches = (0 < K) ? ((nkb_total + BATCH_K - 1) / BATCH_K) : 0;
  /* Device buffers for input matrices */
  void *d_a = NULL, *d_b = NULL, *d_c = NULL;
  /* Double-buffered preprocessing buffers (2 slots) */
  void *d_ak[2] = {NULL, NULL}, *d_expa[2] = {NULL, NULL};
  void *d_bk[2] = {NULL, NULL}, *d_expb[2] = {NULL, NULL};
  /* Helper streams: preprocess_a on stream_a, preprocess_b on stream_b */
  libxstream_stream_t *stream_a = NULL, *stream_b = NULL;
  /* Synchronization events */
  libxstream_event_t *evt_prep_a = NULL, *evt_prep_b = NULL;
  libxstream_event_t *evt_dotprod[2] = {NULL, NULL};
  size_t c_nbytes;
  int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;
  int batch;

  /* Create helper streams for overlapped preprocessing */
  if (EXIT_SUCCESS == result) result = libxstream_stream_create(&stream_a, "ozaki_a", -1);
  if (EXIT_SUCCESS == result) result = libxstream_stream_create(&stream_b, "ozaki_b", -1);
  /* Create synchronization events */
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&evt_prep_a);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&evt_prep_b);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&evt_dotprod[0]);
  if (EXIT_SUCCESS == result) result = libxstream_event_create(&evt_dotprod[1]);
  if (EXIT_SUCCESS != result) goto cleanup;

  /* Allocate device memory for A, B, C */
  { size_t a_cols = ta ? (size_t)M : (size_t)K;
    size_t b_cols = tb ? (size_t)K : (size_t)N;
    size_t a_nbytes = (size_t)lda * a_cols * elem_size;
    size_t b_nbytes = (size_t)ldb * b_cols * elem_size;
    c_nbytes = (size_t)ldc * (size_t)N * elem_size;

    if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_a, a_nbytes);
    if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_b, b_nbytes);
    if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_c, c_nbytes);
    if (EXIT_SUCCESS != result) goto cleanup;
    /* Overlapped H2D: A via stream_a, B via stream_b, C via main */
    if (EXIT_SUCCESS == result) result = libxstream_memcpy_h2d(a, d_a, a_nbytes, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_memcpy_h2d(b, d_b, b_nbytes, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_memcpy_h2d(c, d_c, c_nbytes, stream);
    if (EXIT_SUCCESS != result) goto cleanup;
  }

  /* Pre-allocate double-buffered preprocessing buffers (max batch size) */
  { const size_t slice_elem = ctx->use_bf16 ? 2 : 1; /* ushort vs char */
    const size_t ak_size = (size_t)max_nkb * nblk_m * BM * nslices * BK * slice_elem;
    const size_t bk_size = (size_t)max_nkb * nblk_n * BN_PAD * nslices * BK * slice_elem;
    int s;
    for (s = 0; s < 2 && s < n_batches; ++s) {
      if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_ak[s], ak_size);
      if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_bk[s], bk_size);
      if (!ctx->use_bf16) {
        const size_t expa_size = (size_t)max_nkb * nblk_m * BM * sizeof(cl_short);
        const size_t expb_size = (size_t)max_nkb * nblk_n * BN * sizeof(cl_short);
        if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_expa[s], expa_size);
        if (EXIT_SUCCESS == result) result = libxstream_memdev_allocate(&d_expb[s], expb_size);
      }
    }
    if (EXIT_SUCCESS != result) goto cleanup;
  }

  /* Double-buffered K-batch pipeline:
   *   stream_a  : preprocess_a  (parallel with preprocess_b)
   *   stream_b  : preprocess_b  (parallel with preprocess_a)
   *   stream    : dotprod + C transfers
   * Batch N dotprod overlaps with batch N+1 preprocessing. */
  for (batch = 0; batch < n_batches; ++batch) {
    const int cur = batch & 1;
    const int kb_batch = batch * BATCH_K * BK;
    const int batch_end = (kb_batch + BATCH_K * BK < K) ? (kb_batch + BATCH_K * BK) : K;
    const int nkb = (batch_end - kb_batch + BK - 1) / BK;
    const int first_batch = (0 == batch) ? 1 : 0;

    /* Ensure the dotprod that last used this buffer slot is done */
    if (2 <= batch) {
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_a, evt_dotprod[cur]);
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_b, evt_dotprod[cur]);
      if (EXIT_SUCCESS != result) goto cleanup;
    }

    /* Launch preprocess_a on stream_a */
    { const libxstream_opencl_stream_t* str_a = stream_a;
      size_t global_a[2], local_a[2];
      global_a[0] = (size_t)nblk_m * BM; global_a[1] = (size_t)nkb * BK;
      local_a[0] = BM; local_a[1] = BK;
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 0, d_a));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 1, sizeof(int), &M));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 2, sizeof(int), &K));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 3, sizeof(int), &lda));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 4, sizeof(int), &ta));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 5, sizeof(int), &kb_batch));
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 6, d_ak[cur]));
      if (ctx->use_bf16) {
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 7, sizeof(int), &nblk_m));
      }
      else {
        CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_a, 7, d_expa[cur]));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_a, 8, sizeof(int), &nblk_m));
      }
      CL_CHECK(clEnqueueNDRangeKernel(str_a->queue, ctx->kern_preprocess_a, 2,
                 NULL, global_a, local_a, 0, NULL, NULL));
    }

    /* Launch preprocess_b on stream_b (parallel with preprocess_a) */
    { const libxstream_opencl_stream_t* str_b = stream_b;
      size_t global_b[2], local_b[2];
      global_b[0] = (size_t)nblk_n * BN; global_b[1] = (size_t)nkb * BK;
      local_b[0] = BN; local_b[1] = BK;
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 0, d_b));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 1, sizeof(int), &N));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 2, sizeof(int), &K));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 3, sizeof(int), &ldb));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 4, sizeof(int), &tb));
      CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 5, sizeof(int), &kb_batch));
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 6, d_bk[cur]));
      if (ctx->use_bf16) {
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 7, sizeof(int), &nblk_n));
      }
      else {
        CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_preprocess_b, 7, d_expb[cur]));
        CL_CHECK(clSetKernelArg(ctx->kern_preprocess_b, 8, sizeof(int), &nblk_n));
      }
      CL_CHECK(clEnqueueNDRangeKernel(str_b->queue, ctx->kern_preprocess_b, 2,
                 NULL, global_b, local_b, 0, NULL, NULL));
    }

    /* Record preprocess completion events */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS != result) goto cleanup;

    /* Main stream waits for both preprocess results */
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);
    if (EXIT_SUCCESS != result) goto cleanup;

    /* Launch dotprod on main stream */
    { cl_int i = 0;
      size_t global_c[2], local_c[2];
      if (ctx->use_xmx) {
        const int ntm = BM / 8, ntn = BN / 16;  /* XMX_M=8, XMX_N=16 */
        local_c[0] = (size_t)ctx->sg;  /* SG=16 */
        local_c[1] = (size_t)(ntm * ntn);
        global_c[0] = (size_t)nblk_m * local_c[0];
        global_c[1] = (size_t)nblk_n * local_c[1];
      }
      else {
        local_c[0] = BM; local_c[1] = BN;
        global_c[0] = (size_t)nblk_m * BM;
        global_c[1] = (size_t)nblk_n * BN;
      }
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_ak[cur]));
      if (!ctx->use_bf16) {
        CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expa[cur]));
      }
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_bk[cur]));
      if (!ctx->use_bf16) {
        CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_expb[cur]));
      }
      CL_CHECK(libxstream_opencl_set_kernel_ptr(ctx->kern_dotprod, i++, d_c));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &M));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &N));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &ldc));
      if (ctx->use_double) {
        double dalpha = alpha, dbeta = beta;
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dalpha));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(double), &dbeta));
      }
      else {
        float falpha = (float)alpha, fbeta = (float)beta;
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &falpha));
        CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(float), &fbeta));
      }
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &first_batch));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nkb));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_m));
      CL_CHECK(clSetKernelArg(ctx->kern_dotprod, i++, sizeof(int), &nblk_n));
      CL_CHECK(clEnqueueNDRangeKernel(str->queue, ctx->kern_dotprod, 2,
                 NULL, global_c, local_c, 0, NULL, NULL));
    }

    /* Record dotprod completion for this buffer slot */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_dotprod[cur], stream);
    if (EXIT_SUCCESS != result) goto cleanup;
  }

  /* Read back result C */
  result = libxstream_memcpy_d2h(d_c, c, c_nbytes, stream);
  if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);

cleanup:
  /* Destroy double-buffered preprocessing buffers */
  { int s;
    for (s = 0; s < 2; ++s) {
      if (NULL != d_ak[s]) libxstream_memdev_deallocate(d_ak[s]);
      if (NULL != d_expa[s]) libxstream_memdev_deallocate(d_expa[s]);
      if (NULL != d_bk[s]) libxstream_memdev_deallocate(d_bk[s]);
      if (NULL != d_expb[s]) libxstream_memdev_deallocate(d_expb[s]);
    }
  }
  /* Destroy synchronization events */
  if (NULL != evt_prep_a) libxstream_event_destroy(evt_prep_a);
  if (NULL != evt_prep_b) libxstream_event_destroy(evt_prep_b);
  if (NULL != evt_dotprod[0]) libxstream_event_destroy(evt_dotprod[0]);
  if (NULL != evt_dotprod[1]) libxstream_event_destroy(evt_dotprod[1]);
  /* Destroy helper streams */
  if (NULL != stream_a) libxstream_stream_destroy(stream_a);
  if (NULL != stream_b) libxstream_stream_destroy(stream_b);
  /* Deallocate input/output matrices */
  if (NULL != d_a) libxstream_memdev_deallocate(d_a);
  if (NULL != d_b) libxstream_memdev_deallocate(d_b);
  if (NULL != d_c) libxstream_memdev_deallocate(d_c);

  return result;
}


static void ozaki_print_opt(FILE* stream, const char* name, int val) {
  if (0 != val) fprintf(stream, " %s=%d", name, val);
}
