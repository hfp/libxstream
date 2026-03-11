/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_OPENCL_H
#define OZAKI_OPENCL_H

#include "libxstream_opencl.h"

#if !defined(OZAKI_DEVPOOL) && 1
# define OZAKI_DEVPOOL
#endif


/* Ozaki flags */
typedef enum ozaki_flags_t {
  OZAKI_TRIANGULAR = 1,
  OZAKI_SYMMETRIZE = 2
} ozaki_flags_t;

/* Host-side preprocessing callback for one K-batch of A or B.
 * When non-NULL in the context, ozaki_gemm calls these instead of
 * the GPU preprocess kernels and skips the full-matrix H2D.
 *
 * matrix   : host pointer to source matrix (A or B)
 * ld       : leading dimension
 * trans    : 0 = not transposed, 1 = transposed
 * dim      : outer dimension: M (for A) or N (for B)
 * K        : inner dimension
 * kb_batch : K-offset (first K column of this batch)
 * nkb      : number of K-blocks in this batch
 * nblk     : number of row/column blocks
 * brc      : block size: BM (for A) or BN (for B)
 * bk       : block K dimension
 * nslices  : number of mantissa slices or CRT primes
 * kgroup   : K-grouping factor (1 unless CRT with oztrim>0)
 * use_xmx  : 1 if XMX layout (affects B-side only)
 * slices   : output flat slice buffer (GPU-compatible layout)
 * exp      : output flat exponent buffer (NULL for bf16 scheme)
 *
 * A-side slice layout : slices[panel][brc][nslices][bk]
 * B-side (no XMX)     : slices[panel][brc][nslices][bk]
 * B-side (XMX, int8)  : slices[panel][nslices][bk][bn_pad],
 *   where bn_pad = max(brc, 64)
 * B-side (XMX, bf16)  : slices[panel][nslices][bk][bn_pad],
 *   where bn_pad = max(brc, 32)
 * Exponent layout     : exp[panel][brc] (int16)
 * CRT exponent layout : exp[group][brc],
 *   where group = ki_group * nblk + blk_idx,
 *   panel = ki * nblk + blk_idx */
struct ozaki_context_t;
typedef void (*ozaki_host_preprocess_fn)(
    const void* matrix, int ld, int trans,
    int dim, int K, int kb_batch,
    int nkb, int nblk,
    int brc, int bk, int nslices,
    int kgroup, int use_xmx,
    void* slices, void* exp);

/* State for an Ozaki OpenCL session.
 * All tuning parameters are set by ozaki_init (0 = auto). */
typedef struct ozaki_context_t {
  cl_kernel kern_preprocess_a;
  cl_kernel kern_preprocess_b;
  cl_kernel kern_dotprod;
  int bm, bn, bk;  /* block dimensions (JIT-compiled into kernels) */
  int batch_k;     /* K sub-panels per kernel launch */
  int use_double;  /* 1: fp64, 0: fp32 */
  int use_bf16;    /* derived: 1 = bf16 Dekker slices, 0 = int8 mantissa slices */
  int use_xmx;     /* 1: hardware matrix multiply (DPAS/XMX) */
  int sg;          /* sub-group size used for compilation */
  int nslices;
  int kind;        /* 1: ozaki1 int8, 2: ozaki2 int8 (CRT), 3: ozaki1 bf16 */
  int ozflags;     /* bitmask: OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE */
  int oztrim;
  int kgroup;      /* K-grouping factor for kind==2: 2^oztrim, clamped to batch_k */
  int verbosity;   /* 0: quiet, 1: info, 2+: debug */
  libxs_hist_t* hist; /* kernel execution-time histogram (OZAKI_PROF) */
#if defined(OZAKI_DEVPOOL)
  void* devpool;   /* device memory pool (libxs_malloc-backed) */
#endif
  /* Main stream (set per ozaki_gemm call for pool realloc sync) */
  libxstream_stream_t *stream;
  /* Persistent helper streams for overlapped preprocessing */
  libxstream_stream_t *stream_a, *stream_b;
  /* Persistent synchronization events */
  libxstream_event_t *evt_prep_a, *evt_prep_b, *evt_dotprod[2];
  /* Optional host-side preprocessing (NULL = GPU kernels).
   * When non-NULL, ozaki_gemm calls these callbacks to fill host
   * staging buffers, then H2D-copies to device, skipping the GPU
   * preprocess kernels and the full-matrix H2D transfers. */
  ozaki_host_preprocess_fn host_preprocess_a;
  ozaki_host_preprocess_fn host_preprocess_b;
} ozaki_context_t;


/* Function prototypes (public API).
 * Pass 0 for bm/bn/bk/nslices/batch_k to use auto defaults.
 * Pass -1 for ozflags to use the default (TRIANGULAR | SYMMETRIZE);
 * 0 disables both flags.  Auto defaults choose XMX-friendly sizes
 * when hardware support is detected.
 * kind: 1 = ozaki1 int8 (default), 2 = ozaki2 int8 (CRT), 3 = ozaki1 bf16.
 * verbosity: 0 = quiet, 1 = info, 2+ = debug. */
int ozaki_init(ozaki_context_t* ctx, int bm, int bn, int bk,
               int use_double, int kind, int verbosity,
               int nslices, int batch_k,
               int ozflags, int oztrim);
void ozaki_destroy(ozaki_context_t* ctx);
/* ozaki_gemm enqueues the entire GEMM pipeline on stream and returns without
 * synchronizing — the caller must sync the stream before consuming the result.
 * Helper streams (ctx->stream_a/b) and events are kept persistent in the
 * context to avoid per-call creation overhead.  On the rare pool grow path
 * (larger problem size), the wrapped deallocator syncs all streams before
 * reallocating. */
int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream,
               char transa, char transb,
               int M, int N, int K,
               double alpha, const void* a, int lda,
                             const void* b, int ldb,
               double beta,        void* c, int ldc);

#endif /* OZAKI_OPENCL_H */
