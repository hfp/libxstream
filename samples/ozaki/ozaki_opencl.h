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

#if !defined(OZAKI_TINYTC_BM)
# define OZAKI_TINYTC_BM 256
#endif
#if !defined(OZAKI_TINYTC_BN)
# define OZAKI_TINYTC_BN 128
#endif

/* Device memory allocation macros (shared by ozaki_opencl.c and ozaki_gemm.c).
 * Callers must have a local `pool` variable (libxs_malloc_pool_t*)
 * when OZAKI_DEVPOOL is defined. */
#if defined(OZAKI_DEVPOOL)
# define OZAKI_DEV_ALLOC(PTR, SIZE) ( \
  (NULL != pool) \
    ? ((*(PTR) = libxs_malloc(pool, SIZE, LIBXS_MALLOC_NATIVE)) != NULL ? EXIT_SUCCESS : EXIT_FAILURE) \
    : libxstream_mem_allocate((void**)(PTR), SIZE))
# define OZAKI_DEV_FREE(PTR) do { \
  if (NULL != (PTR)) { \
    if (NULL != pool) libxs_free(PTR); else libxstream_mem_deallocate(PTR); \
  } \
} while (0)
#else
# define OZAKI_DEV_ALLOC(PTR, SIZE) \
  libxstream_mem_allocate((void**)(PTR), SIZE)
# define OZAKI_DEV_FREE(PTR) do { \
  if (NULL != (PTR)) libxstream_mem_deallocate(PTR); \
} while (0)
#endif


/* Ozaki flags */
typedef enum ozaki_flags_t {
  OZAKI_TRIANGULAR = 1,
  OZAKI_SYMMETRIZE = 2
} ozaki_flags_t;

/* Host-side preprocessing callback for A or B (GEMM single-shot model).
 * When non-NULL in the context, ozaki_gemm calls these instead of
 * the GPU preprocess kernels and skips the full-matrix H2D.
 *
 * matrix   : host pointer to source matrix (A or B)
 * ld       : leading dimension
 * trans    : 0 = not transposed, 1 = transposed
 * dim      : outer dimension: M (for A) or N (for B)
 * K        : inner dimension (unpadded)
 * K_pad    : padded K (multiple of 32, >= 64)
 * dim_pad  : padded outer dimension: M_pad (for A) or N_pad (for B)
 * ndecomp  : number of decomposition components (slices or CRT primes)
 * use_xmx  : 1 if XMX hardware (DPAS)
 * slices   : output int8 buffer, pre-zeroed, size ndecomp*dim_pad*K_pad (A)
 *            or ndecomp*K_pad*dim_pad (B)
 * exp      : output int32 exponent buffer, pre-zeroed, size dim
 *
 * A-side slice layout: slices[s * dim_pad * K_pad + row * K_pad + k]
 * B-side slice layout: slices[s * K_pad * dim_pad + k * dim_pad + col]
 * Exponent layout:     exp[i] -- 2^(max exponent) per row (A) or col (B),
 *                      stored as real_t (double or float matching matrix type) */
typedef void (*ozaki_host_preprocess_fn)(
    const void* matrix, int ld, int trans,
    int dim, int K, int K_pad, int dim_pad,
    int ndecomp, int use_xmx,
    void* slices, void* exp);

/* Per-side preprocessing cache: check fields + cached device buffers.
 * dim is the outer dimension (M for A, N for B). */
typedef struct ozaki_cache_side_t {
  const void* ptr;
  int dim, K, ld, trans;
  void* d_slices;
  void* d_exp;
  size_t slices_size, exp_size;
} ozaki_cache_side_t;

typedef struct ozaki_cache_t {
  libxs_lock_t lock;
  volatile LIBXS_ATOMIC_LOCKTYPE nusers;
  int flags; /* bitmask: 1=A, 2=B */
  ozaki_cache_side_t a, b;
} ozaki_cache_t;

/* State for an Ozaki OpenCL session.
 * All tuning parameters are set by ozaki_init (0 = auto). */
typedef struct ozaki_context_t {
  /* GEMM-mode kernels (tiled K-loop + fused accumulation) */
  cl_kernel kern_preprocess_a;
  cl_kernel kern_preprocess_b;
  cl_kernel kern_fused;
  cl_kernel kern_fused_bounds; /* bounds-checked variant for unaligned sizes */
  cl_kernel kern_scale_beta;
  /* Optional TinyTC SPIR-V kernel (loaded from .clx via OZAKI_TINYTC env) */
  cl_kernel kern_tinytc;
  cl_program prog_tinytc;
  /* CRT GEMM-mode kernels (Scheme-2 tiled path) */
  cl_kernel kern_crt_preprocess_a;
  cl_kernel kern_crt_preprocess_b;
  cl_kernel kern_crt_fused;
  cl_kernel kern_crt_scale_beta;
  int use_double;  /* 1: fp64, 0: fp32 */
  int use_xmx;     /* 1: hardware matrix multiply (DPAS/XMX) */
  int sg;          /* sub-group size used for compilation */
  int ndecomp;     /* number of decomposition components (slices or primes) */
  int kind;        /* 1: ozaki1 int8, 2: ozaki2 int8 (CRT) */
  int ozflags;     /* bitmask: OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE */
  int oztrim;      /* Scheme 1: diagonal trim (higher = less accurate, faster) */
  int verbosity;   /* 0: quiet, 1: info, 2+: debug */
  int profile;     /* 0: off, 1 (or negative): pre+gemm, 2: gemm, 3: pre-a, 4: pre-b */
  /* block sizes for preprocessing WGs */
  int bm_pre, bn_pre, bk_pre;
  /* output tile size (compiled into kernel) */
  int tm, tn;
  /* register tiling: sub-tiles per sub-group (compiled into kernel) */
  int rtm, rtn;
  int ku; /* K-loop unroll factor (compiled into kernel) */
  int rc; /* DPAS repeat count: 8 (default) or 4 (split) */
  int pb; /* CRT prime batching factor (compiled into kernel) */
  int biggrf; /* Ozaki-local 256-GRF decision */
  libxs_hist_t* hist; /* kernel execution-time histogram (OZAKI_PROF) */
#if defined(OZAKI_DEVPOOL)
  void* devpool;   /* device memory pool (libxs_malloc-backed) */
  /* Main stream (set per ozaki_gemm call for pool realloc sync) */
  libxstream_stream_t *stream;
#endif
  /* Persistent helper streams for overlapped preprocessing */
  libxstream_stream_t *stream_a, *stream_b;
  /* Persistent synchronization events */
  libxstream_event_t *evt_prep_a, *evt_prep_b;
  /* Optional host-side preprocessing (NULL = GPU kernels).
   * When non-NULL, ozaki_gemm calls these callbacks to fill host
   * staging buffers, then H2D-copies to device, skipping the GPU
   * preprocess kernels and the full-matrix H2D transfers. */
  ozaki_host_preprocess_fn host_preprocess_a;
  ozaki_host_preprocess_fn host_preprocess_b;
  /* Preprocessing cache (OZAKI_CACHE env, bitmask: 1=A, 2=B, 3=both). */
  ozaki_cache_t cache;
} ozaki_context_t;


/* Function prototypes (public API).
 * Pass 0 for tm/tn/ndecomp to use auto defaults.
 * Pass -1 for ozflags to use the default (TRIANGULAR | SYMMETRIZE);
 * 0 disables both flags.  Auto defaults choose XMX-friendly sizes
 * when hardware support is detected.
 * kind: 1 = ozaki1 int8 (default), 2 = ozaki2 int8 (CRT).
 * verbosity: 0 = quiet, 1 = info, 2+ = debug.
 * ozgroups (Scheme 2 only): K-grouping factor, 0/1 = disabled. */
int ozaki_init(ozaki_context_t* ctx, int tm, int tn,
               int use_double, int kind, int verbosity,
               int ndecomp, int ozflags, int oztrim,
               int ozgroups);
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
