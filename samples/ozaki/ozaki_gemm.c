/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"


/* Local helper functions (static) to manage kernel argument setup and launches.
 * These were kept local to avoid adding new translation units for the sample. */
static int ozaki_enqueue_preprocess(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern, void* d_src, void* d_slices, void* d_exp,
  int M, int K, int ld, int trans, int k_pad, int pad,
  int bm_pre, int bk_pre, cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled);
static int ozaki_enqueue_scale_beta(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern_scale, void* d_cg, int M, int N, int ldc, double beta);
static int ozaki_launch_tinytc(ozaki_context_t* ctx, libxstream_stream_t* stream,
  void* d_as, void* d_bs, void* d_scratch, void* d_cg,
  void* d_expa_g, void* d_expb_g,
  int M, int N, int k_pad, int m_pad, int n_pad,
  int nslices_g, int tm, int tn, int ldc, int cutoff, int sq,
  cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled);
static int ozaki_launch_fused(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern_g, void* d_as, void* d_bs, void* d_expa_g, void* d_expb_g,
  void* d_cg, int M, int N, int k_pad, int n_pad, int ldc, int m_pad,
  int tm, int tn, int ntm, int ntn, double alpha, int first_pair, int use_double,
  cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled);


int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream,
               char transa, char transb,
               int M, int N, int K,
               double alpha, const void* a, int lda,
                             const void* b, int ldb,
               double beta,        void* c, int ldc)
{
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);

  /* Persistent helper streams and events from context */
  libxstream_stream_t* const stream_a = ctx->stream_a;
  libxstream_stream_t* const stream_b = ctx->stream_b;
  libxstream_event_t* const evt_prep_a = ctx->evt_prep_a;
  libxstream_event_t* const evt_prep_b = ctx->evt_prep_b;
  size_t c_nbytes;
  const int ta = (transa != 'N' && transa != 'n') ? 1 : 0;
  const int tb = (transb != 'N' && transb != 'n') ? 1 : 0;
  int result = EXIT_SUCCESS;

#if defined(OZAKI_DEVPOOL)
  libxs_malloc_pool_t* const pool = (libxs_malloc_pool_t*)ctx->devpool;
  ctx->stream = stream; /* expose to deallocate wrapper */
#endif

  /* GEMM path (Scheme 1): full-split-then-tiled-GEMM.
   * Preprocesses entire K dimension up front into dense per-slice
   * int8 matrices, then runs a proper tiled GEMM per slice pair. */
  if (NULL != ctx->kern_fused && 0 < K) {
    const int nslices_g = ctx->ndecomp;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const int tm = ctx->tm, tn = ctx->tn;
    /* Pad K to multiple of bk_pre (=32) and ensure >= 64 for 2D block I/O */
    int k_pad = ((K + bk_pre - 1) / bk_pre) * bk_pre;
    const int m_pad = ((M + bm_pre - 1) / bm_pre) * bm_pre;
    int n_pad = ((N + bn_pre - 1) / bn_pre) * bn_pre;
    const int nblk_gm = (M + tm - 1) / tm;
    const int nblk_gn = (N + tn - 1) / tn;
    const int ntm = tm / (8 * ctx->rtm), ntn = tn / (16 * ctx->rtn);
    const int cutoff = 2 * (nslices_g - 1) - ctx->oztrim;
    /* Dense slice buffer sizes */
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL, *d_scratch = NULL;
    void *h_as = NULL, *h_expa = NULL, *h_bs = NULL, *h_expb = NULL;
    int first_pair;
    int n_profiled = 0;
    cl_event *evt_prof = NULL;
    int cache_hit_a = 0, cache_hit_b = 0;
    const int use_tinytc = (NULL != ctx->kern_tinytc
      && 0 == M % OZAKI_TINYTC_BM && 0 == N % OZAKI_TINYTC_BN);

    if (k_pad < 64) k_pad = 64;
    if (n_pad < 64) n_pad = 64;

    as_size   = (size_t)nslices_g * m_pad * k_pad;
    bs_size   = (size_t)nslices_g * k_pad * n_pad;
    expa_size = (size_t)nblk_gm * tm * elem_size;
    expb_size = (size_t)nblk_gn * tn * elem_size;
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    /* Preprocessing cache: reuse slices+exponents when matrix unchanged */
    LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 != (ctx->cache.flags & 1) && a == ctx->cache.a.ptr
        && M == ctx->cache.a.dim && K == ctx->cache.a.K
        && lda == ctx->cache.a.ld && ta == ctx->cache.a.trans
        && as_size == ctx->cache.a.slices_size && expa_size == ctx->cache.a.exp_size
        && NULL != ctx->cache.a.d_slices && NULL != ctx->cache.a.d_exp)
    {
      d_as = ctx->cache.a.d_slices; d_expa_g = ctx->cache.a.d_exp;
      cache_hit_a = 1;
    }
    if (0 != (ctx->cache.flags & 2) && b == ctx->cache.b.ptr
        && N == ctx->cache.b.dim && K == ctx->cache.b.K
        && ldb == ctx->cache.b.ld && tb == ctx->cache.b.trans
        && bs_size == ctx->cache.b.slices_size && expb_size == ctx->cache.b.exp_size
        && NULL != ctx->cache.b.d_slices && NULL != ctx->cache.b.d_exp)
    {
      d_bs = ctx->cache.b.d_slices; d_expb_g = ctx->cache.b.d_exp;
      cache_hit_b = 1;
    }
    if (0 != cache_hit_a || 0 != cache_hit_b) ++ctx->cache.nusers;
    LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);

    /* Allocate device memory (skip cached sides and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result && 0 != use_tinytc) {
      result = OZAKI_DEV_ALLOC(&d_scratch, (size_t)M * N * sizeof(cl_int));
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_as, as_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);
    }

    /* H2D transfers (skip cached and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    /* Zero exponent arrays and slice buffers (skip cached and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_zero(d_expb_g, 0, expb_size, stream_b);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs, 0, bs_size, stream_b);
    }

    /* Wait for H2D to finish before preprocessing */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

    /* Profiling: allocate event array (pre_a + pre_b + slice_pairs) */
    LIBXS_ASSERT(0 == ctx->profile || NULL != ctx->hist);
    if (NULL != ctx->hist) {
      evt_prof = (cl_event*)calloc((size_t)(nslices_g * nslices_g) + 2, sizeof(cl_event));
      if (EXIT_SUCCESS == result) result = libxstream_stream_set_profiling(stream);
    }

    /* Phase 1: Preprocess A (skip entirely on cache hit) */
    if (0 == cache_hit_a) {
      if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
          h_as = calloc(as_size, 1);
          h_expa = calloc(expa_size, 1);
        if (NULL != h_as && NULL != h_expa) {
          ctx->host_preprocess_a(a, lda, ta, M, K, k_pad, m_pad,
            nslices_g, ctx->use_xmx, h_as, h_expa);
          result = libxstream_mem_copy_h2d(h_as, d_as, as_size, stream_a);
          if (EXIT_SUCCESS == result) {
            result = libxstream_mem_copy_h2d(h_expa, d_expa_g, expa_size, stream_a);
          }
        }
        else result = EXIT_FAILURE;
      }
      else if (EXIT_SUCCESS == result) {
        result = ozaki_enqueue_preprocess(ctx, stream_a, ctx->kern_preprocess_a,
          d_ag, d_as, d_expa_g, M, K, lda, ta, k_pad, m_pad,
          bm_pre, bk_pre, evt_prof, 1, 3, &n_profiled);
      }
    } /* cache_hit_a */

    /* Phase 1: Preprocess B (skip entirely on cache hit) */
    if (0 == cache_hit_b) {
      if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
        h_bs = calloc(bs_size, 1);
        h_expb = calloc(expb_size, 1);
        if (NULL != h_bs && NULL != h_expb) {
          ctx->host_preprocess_b(b, ldb, tb, N, K, k_pad, n_pad,
            nslices_g, ctx->use_xmx, h_bs, h_expb);
          result = libxstream_mem_copy_h2d(h_bs, d_bs, bs_size, stream_b);
          if (EXIT_SUCCESS == result) {
            result = libxstream_mem_copy_h2d(h_expb, d_expb_g, expb_size, stream_b);
          }
        }
        else result = EXIT_FAILURE;
      }
      else if (EXIT_SUCCESS == result) {
        result = ozaki_enqueue_preprocess(ctx, stream_b, ctx->kern_preprocess_b,
          d_bg, d_bs, d_expb_g, N, K, ldb, tb, k_pad, n_pad,
          bn_pre, bk_pre, evt_prof, 1, 4, &n_profiled);
      }
    } /* cache_hit_b */

    /* Wait for both preprocessing to complete */
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Save preprocessed buffers to cache on miss (transfers ownership) */
    { const int prev_owned = (0 != cache_hit_a || 0 != cache_hit_b);
      LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      if (0 == cache_hit_a && 0 != (ctx->cache.flags & 1) && EXIT_SUCCESS == result) {
        if (NULL != ctx->cache.a.d_slices && as_size != ctx->cache.a.slices_size) {
          OZAKI_DEV_FREE(ctx->cache.a.d_slices); ctx->cache.a.d_slices = NULL;
        }
        if (NULL != ctx->cache.a.d_exp && expa_size != ctx->cache.a.exp_size) {
          OZAKI_DEV_FREE(ctx->cache.a.d_exp); ctx->cache.a.d_exp = NULL;
        }
        ctx->cache.a.ptr = a; ctx->cache.a.dim = M; ctx->cache.a.K = K;
        ctx->cache.a.ld = lda; ctx->cache.a.trans = ta;
        ctx->cache.a.d_slices = d_as; ctx->cache.a.d_exp = d_expa_g;
        ctx->cache.a.slices_size = as_size; ctx->cache.a.exp_size = expa_size;
        cache_hit_a = 1; /* ownership transferred; suppress cleanup free */
      }
      if (0 == cache_hit_b && 0 != (ctx->cache.flags & 2) && EXIT_SUCCESS == result) {
        if (NULL != ctx->cache.b.d_slices && bs_size != ctx->cache.b.slices_size) {
          OZAKI_DEV_FREE(ctx->cache.b.d_slices); ctx->cache.b.d_slices = NULL;
        }
        if (NULL != ctx->cache.b.d_exp && expb_size != ctx->cache.b.exp_size) {
          OZAKI_DEV_FREE(ctx->cache.b.d_exp); ctx->cache.b.d_exp = NULL;
        }
        ctx->cache.b.ptr = b; ctx->cache.b.dim = N; ctx->cache.b.K = K;
        ctx->cache.b.ld = ldb; ctx->cache.b.trans = tb;
        ctx->cache.b.d_slices = d_bs; ctx->cache.b.d_exp = d_expb_g;
        ctx->cache.b.slices_size = bs_size; ctx->cache.b.exp_size = expb_size;
        cache_hit_b = 1; /* ownership transferred; suppress cleanup free */
      }
      if (0 == prev_owned && (0 != cache_hit_a || 0 != cache_hit_b)) ++ctx->cache.nusers;
      LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);
    } /* prev_owned scope */

    /* Scale C by beta (skip for beta==0: first_pair overwrite handles it;
     * skip for beta==1: multiplying by 1 is a no-op) */
    if (EXIT_SUCCESS == result && 1.0 != beta && 0.0 != beta) {
      result = ozaki_enqueue_scale_beta(ctx, stream, ctx->kern_scale_beta, d_cg, M, N, ldc, beta);
    }

    /* Tiled GEMM phase: single kernel launch over all slice pairs */
    first_pair = (0.0 == beta) ? 1 : 0;
    { const int sq = ctx->ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE);
      if (0 != use_tinytc) {
        result = ozaki_launch_tinytc(ctx, stream, d_as, d_bs, d_scratch, d_cg,
          d_expa_g, d_expb_g, M, N, k_pad, m_pad, n_pad,
          nslices_g, tm, tn, ldc, cutoff, sq, evt_prof, 1, 2, &n_profiled);
      }
      else {
        cl_kernel kern_g = (0 != M % tm || 0 != N % tn) ? ctx->kern_fused_bounds : ctx->kern_fused;
        result = ozaki_launch_fused(ctx, stream, kern_g, d_as, d_bs, d_expa_g, d_expb_g,
          d_cg, M, N, k_pad, n_pad, ldc, m_pad, tm, tn, ntm, ntn,
          alpha, first_pair, ctx->use_double, evt_prof, 1, 2, &n_profiled);
      }
    }

    /* Collect profiling data */
    if (NULL != evt_prof) {
      int resprof = clWaitForEvents((cl_uint)n_profiled, evt_prof), pi;
      double total = 0;
      for (pi = 0; pi < n_profiled && EXIT_SUCCESS == resprof; ++pi) {
        total += libxstream_opencl_duration(evt_prof[pi], &resprof);
      }
      for (pi = 0; pi < n_profiled; ++pi) {
        if (NULL != evt_prof[pi]) clReleaseEvent(evt_prof[pi]);
      }
      if (EXIT_SUCCESS == resprof && 0 < total) {
        const double gflops = (2.0 * M * N * K) / (total * 1E9);
        libxs_hist_push(NULL, ctx->hist, &gflops);
      }
      free(evt_prof);
    }
    /* D2H result and cleanup */
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);

    OZAKI_DEV_FREE(d_ag); OZAKI_DEV_FREE(d_bg); OZAKI_DEV_FREE(d_cg);
    OZAKI_DEV_FREE(d_scratch);
    if (0 == cache_hit_a) { OZAKI_DEV_FREE(d_as); OZAKI_DEV_FREE(d_expa_g); }
    if (0 == cache_hit_b) { OZAKI_DEV_FREE(d_bs); OZAKI_DEV_FREE(d_expb_g); }
    if (0 != cache_hit_a || 0 != cache_hit_b) {
      LIBXS_ATOMIC_SUB_FETCH(&ctx->cache.nusers, 1, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  /* CRT GEMM path (Scheme 2): full-split-then-single-fused-GEMM.
   * Preprocesses entire K into dense per-prime CRT residue matrices,
   * then runs a single kernel per tile that loops over all primes
   * internally (full-K DPAS + Garner + Horner in one launch). */
  else if (NULL != ctx->kern_crt_fused && 0 < K) {
    const int nprimes_g = ctx->ndecomp;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const int tm = ctx->tm, tn = ctx->tn;
    int k_pad = ((K + bk_pre - 1) / bk_pre) * bk_pre;
    const int m_pad = ((M + bm_pre - 1) / bm_pre) * bm_pre;
    int n_pad = ((N + bn_pre - 1) / bn_pre) * bn_pre;
    const int nblk_gm = (M + tm - 1) / tm;
    const int nblk_gn = (N + tn - 1) / tn;
    const int ntm = tm / (8 * ctx->rtm), ntn = tn / (16 * ctx->rtn);
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    void *h_as = NULL, *h_expa = NULL, *h_bs = NULL, *h_expb = NULL;
    int first_tile;
    int n_profiled_c = 0;
    cl_event *evt_prof_c = NULL;
    int cache_hit_a = 0, cache_hit_b = 0;

    if (k_pad < 64) k_pad = 64;
    if (n_pad < 64) n_pad = 64;

    as_size   = (size_t)nprimes_g * m_pad * k_pad;
    bs_size   = (size_t)nprimes_g * k_pad * n_pad;
    expa_size = (size_t)nblk_gm * tm * sizeof(cl_int); /* pad to tile boundary */
    expb_size = (size_t)nblk_gn * tn * sizeof(cl_int);
    c_nbytes  = (size_t)ldc * (size_t)N * elem_size;

    /* Preprocessing cache: reuse slices+exponents when matrix unchanged */
    LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 != (ctx->cache.flags & 1) && a == ctx->cache.a.ptr
        && M == ctx->cache.a.dim && K == ctx->cache.a.K
        && lda == ctx->cache.a.ld && ta == ctx->cache.a.trans
        && as_size == ctx->cache.a.slices_size && expa_size == ctx->cache.a.exp_size
        && NULL != ctx->cache.a.d_slices && NULL != ctx->cache.a.d_exp)
    {
      d_as = ctx->cache.a.d_slices; d_expa_g = ctx->cache.a.d_exp;
      cache_hit_a = 1;
    }
    if (0 != (ctx->cache.flags & 2) && b == ctx->cache.b.ptr
        && N == ctx->cache.b.dim && K == ctx->cache.b.K
        && ldb == ctx->cache.b.ld && tb == ctx->cache.b.trans
        && bs_size == ctx->cache.b.slices_size && expb_size == ctx->cache.b.exp_size
        && NULL != ctx->cache.b.d_slices && NULL != ctx->cache.b.d_exp)
    {
      d_bs = ctx->cache.b.d_slices; d_expb_g = ctx->cache.b.d_exp;
      cache_hit_b = 1;
    }
    if (0 != cache_hit_a || 0 != cache_hit_b) ++ctx->cache.nusers;
    LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);

    /* Allocate device memory (skip cached sides and host-preprocessed sides) */
    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
    }
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_cg, c_nbytes);
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_as, as_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_bs, bs_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
      result = OZAKI_DEV_ALLOC(&d_expa_g, expa_size);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
      result = OZAKI_DEV_ALLOC(&d_expb_g, expb_size);
    }

    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_copy_h2d(a, d_ag,
        (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_copy_h2d(b, d_bg,
        (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);

    if (EXIT_SUCCESS == result && 0 == cache_hit_a && NULL == ctx->host_preprocess_a) {
      result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
    }
    if (EXIT_SUCCESS == result && 0 == cache_hit_b && NULL == ctx->host_preprocess_b) {
      result = libxstream_mem_zero(d_expb_g, 0, expb_size, stream_b);
      if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs, 0, bs_size, stream_b);
    }

    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

    /* Profiling: allocate event array (pre_a + pre_b + crt_gemm) */
    LIBXS_ASSERT(0 == ctx->profile || NULL != ctx->hist);
    if (NULL != ctx->hist) {
      evt_prof_c = (cl_event*)calloc(3, sizeof(cl_event));
      if (EXIT_SUCCESS == result) result = libxstream_stream_set_profiling(stream);
    }

    /* Preprocess A (skip entirely on cache hit) */
    if (0 == cache_hit_a) {
      if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_a) {
        h_as = calloc(as_size, 1);
        h_expa = calloc(expa_size, 1);
        if (NULL != h_as && NULL != h_expa) {
          ctx->host_preprocess_a(a, lda, ta, M, K, k_pad, m_pad,
            nprimes_g, ctx->use_xmx, h_as, h_expa);
          result = libxstream_mem_copy_h2d(h_as, d_as, as_size, stream_a);
          if (EXIT_SUCCESS == result) {
            result = libxstream_mem_copy_h2d(h_expa, d_expa_g, expa_size, stream_a);
          }
        }
        else result = EXIT_FAILURE;
      }
      else if (EXIT_SUCCESS == result) {
        const libxstream_opencl_stream_t* str_a = stream_a;
        size_t global_a[2], local_a[2];
        const int nblk_m_pre = (M + bm_pre - 1) / bm_pre;
        local_a[0] = bm_pre; local_a[1] = bk_pre;
        global_a[0] = (size_t)nblk_m_pre * bm_pre;
        global_a[1] = bk_pre; /* single WG in K: kernel loops internally */
        { cl_int i = 0;
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_ag));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &M));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &K));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &lda));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &ta));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_as));
          CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_preprocess_a, i++, d_expa_g));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &k_pad));
          CL_CHECK(result, clSetKernelArg(ctx->kern_crt_preprocess_a, i++, sizeof(int), &m_pad));
        }
        CL_CHECK(result, clEnqueueNDRangeKernel(str_a->queue, ctx->kern_crt_preprocess_a, 2,
          NULL, global_a, local_a, 0, NULL,
          (NULL != evt_prof_c && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
            ? (evt_prof_c + n_profiled_c) : NULL));
        if (EXIT_SUCCESS == result && NULL != evt_prof_c
          && (1 == ctx->profile || 3 == ctx->profile || 0 > ctx->profile))
        {
          ++n_profiled_c;
        }
      }
    } /* cache_hit_a */
    else if (EXIT_SUCCESS == result) {
      result = ozaki_enqueue_preprocess(ctx, stream_a, ctx->kern_crt_preprocess_a,
        d_ag, d_as, d_expa_g, M, K, lda, ta, k_pad, m_pad,
        bm_pre, bk_pre, evt_prof_c, 1, 3, &n_profiled_c);
    }

    /* Preprocess B (skip entirely on cache hit) */
    if (0 == cache_hit_b) {
      if (EXIT_SUCCESS == result && NULL != ctx->host_preprocess_b) {
        h_bs = calloc(bs_size, 1);
        h_expb = calloc(expb_size, 1);
        if (NULL != h_bs && NULL != h_expb) {
          ctx->host_preprocess_b(b, ldb, tb, N, K, k_pad, n_pad,
            nprimes_g, ctx->use_xmx, h_bs, h_expb);
          result = libxstream_mem_copy_h2d(h_bs, d_bs, bs_size, stream_b);
          if (EXIT_SUCCESS == result) {
            result = libxstream_mem_copy_h2d(h_expb, d_expb_g, expb_size, stream_b);
          }
        }
        else result = EXIT_FAILURE;
      }
      else if (EXIT_SUCCESS == result) {
        result = ozaki_enqueue_preprocess(ctx, stream_b, ctx->kern_crt_preprocess_b,
          d_bg, d_bs, d_expb_g, N, K, ldb, tb, k_pad, n_pad,
          bn_pre, bk_pre, evt_prof_c, 1, 4, &n_profiled_c);
      }
    } /* cache_hit_b */

    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
    if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
    if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

    /* Free host staging buffers (sync ensures H2D completed) */
    free(h_as); h_as = NULL; free(h_expa); h_expa = NULL;
    free(h_bs); h_bs = NULL; free(h_expb); h_expb = NULL;

    /* Save preprocessed buffers to cache on miss (transfers ownership) */
    { const int prev_owned = (0 != cache_hit_a || 0 != cache_hit_b);
      LIBXS_ATOMIC_ACQUIRE(&ctx->cache.lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      if (0 == cache_hit_a && 0 != (ctx->cache.flags & 1) && EXIT_SUCCESS == result) {
        if (NULL != ctx->cache.a.d_slices && as_size != ctx->cache.a.slices_size) {
          OZAKI_DEV_FREE(ctx->cache.a.d_slices); ctx->cache.a.d_slices = NULL;
        }
        if (NULL != ctx->cache.a.d_exp && expa_size != ctx->cache.a.exp_size) {
          OZAKI_DEV_FREE(ctx->cache.a.d_exp); ctx->cache.a.d_exp = NULL;
        }
        ctx->cache.a.ptr = a; ctx->cache.a.dim = M; ctx->cache.a.K = K;
        ctx->cache.a.ld = lda; ctx->cache.a.trans = ta;
        ctx->cache.a.d_slices = d_as; ctx->cache.a.d_exp = d_expa_g;
        ctx->cache.a.slices_size = as_size; ctx->cache.a.exp_size = expa_size;
        cache_hit_a = 1; /* ownership transferred; suppress cleanup free */
      }
      if (0 == cache_hit_b && 0 != (ctx->cache.flags & 2) && EXIT_SUCCESS == result) {
        if (NULL != ctx->cache.b.d_slices && bs_size != ctx->cache.b.slices_size) {
          OZAKI_DEV_FREE(ctx->cache.b.d_slices); ctx->cache.b.d_slices = NULL;
        }
        if (NULL != ctx->cache.b.d_exp && expb_size != ctx->cache.b.exp_size) {
          OZAKI_DEV_FREE(ctx->cache.b.d_exp); ctx->cache.b.d_exp = NULL;
        }
        ctx->cache.b.ptr = b; ctx->cache.b.dim = N; ctx->cache.b.K = K;
        ctx->cache.b.ld = ldb; ctx->cache.b.trans = tb;
        ctx->cache.b.d_slices = d_bs; ctx->cache.b.d_exp = d_expb_g;
        ctx->cache.b.slices_size = bs_size; ctx->cache.b.exp_size = expb_size;
        cache_hit_b = 1; /* ownership transferred; suppress cleanup free */
      }
      if (0 == prev_owned && (0 != cache_hit_a || 0 != cache_hit_b)) ++ctx->cache.nusers;
      LIBXS_ATOMIC_RELEASE(&ctx->cache.lock, LIBXS_ATOMIC_LOCKORDER);
    } /* prev_owned scope */

    /* Scale C by beta (skip for beta==0: first_tile overwrite handles it;
     * skip for beta==1: multiplying by 1 is a no-op) */
    if (EXIT_SUCCESS == result && 1.0 != beta && 0.0 != beta) {
      result = ozaki_enqueue_scale_beta(ctx, stream, ctx->kern_crt_scale_beta, d_cg, M, N, ldc, beta);
    }

    /* Single fused CRT GEMM: one launch per output tile covers all primes */
    first_tile = (0.0 == beta) ? 1 : 0;
    if (EXIT_SUCCESS == result) {
      cl_int i = 0;
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_as));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_bs));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_expa_g));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_expb_g));
      CL_CHECK(result, libxstream_opencl_set_kernel_ptr(ctx->kern_crt_fused, i++, d_cg));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &M));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &N));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &k_pad));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &n_pad));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &ldc));
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &m_pad));
      if (ctx->use_double) {
        double dalpha = alpha;
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(double), &dalpha));
      }
      else {
        float falpha = (float)alpha;
        CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(float), &falpha));
      }
      CL_CHECK(result, clSetKernelArg(ctx->kern_crt_fused, i++, sizeof(int), &first_tile));

      result = ozaki_launch_fused(ctx, stream, ctx->kern_crt_fused, d_as, d_bs, d_expa_g, d_expb_g,
        d_cg, M, N, k_pad, n_pad, ldc, m_pad, tm, tn, ntm, ntn,
        alpha, first_tile, ctx->use_double, evt_prof_c, 1, 2, &n_profiled_c);
    }

    /* Collect profiling data */
    if (NULL != evt_prof_c) {
      int resprof = clWaitForEvents((cl_uint)n_profiled_c, evt_prof_c), pi;
      double total = 0;
      for (pi = 0; pi < n_profiled_c && EXIT_SUCCESS == resprof; ++pi) {
        total += libxstream_opencl_duration(evt_prof_c[pi], &resprof);
      }
      for (pi = 0; pi < n_profiled_c; ++pi) {
        if (NULL != evt_prof_c[pi]) clReleaseEvent(evt_prof_c[pi]);
      }
      if (EXIT_SUCCESS == resprof && 0 < total) {
        const double gflops = (2.0 * M * N * K) / (total * 1E9);
        libxs_hist_push(NULL, ctx->hist, &gflops);
      }
      free(evt_prof_c);
    }
    if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);

    OZAKI_DEV_FREE(d_ag); OZAKI_DEV_FREE(d_bg); OZAKI_DEV_FREE(d_cg);
    if (0 == cache_hit_a) { OZAKI_DEV_FREE(d_as); OZAKI_DEV_FREE(d_expa_g); }
    if (0 == cache_hit_b) { OZAKI_DEV_FREE(d_bs); OZAKI_DEV_FREE(d_expb_g); }
    if (0 != cache_hit_a || 0 != cache_hit_b) {
      LIBXS_ATOMIC_SUB_FETCH(&ctx->cache.nusers, 1, LIBXS_ATOMIC_LOCKORDER);
    }
  }

  return result;
}


static int ozaki_enqueue_preprocess(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern, void* d_src, void* d_slices, void* d_exp,
  int M, int K, int ld, int trans, int k_pad, int pad,
  int bm_pre, int bk_pre, cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled)
{
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = stream;
  size_t global[2], local[2];
  const int nblk_m_pre = (M + bm_pre - 1) / bm_pre;
  local[0] = bm_pre; local[1] = bk_pre;
  global[0] = (size_t)nblk_m_pre * bm_pre;
  global[1] = bk_pre; /* single WG in K: kernel loops internally */
  { cl_int i = 0;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_src));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &K));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &ld));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &trans));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_slices));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_exp));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &k_pad));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &pad));
  }
  CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2,
    NULL, global, local, 0, NULL,
    (NULL != evt_prof && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
      ? (evt_prof + *n_profiled) : NULL));
  if (EXIT_SUCCESS == result && NULL != evt_prof
    && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
  {
    ++(*n_profiled);
  }
  return result;
}


static int ozaki_enqueue_scale_beta(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern_scale, void* d_cg, int M, int N, int ldc, double beta)
{
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = stream;
  size_t global_s[2], local_s[2];
  local_s[0] = (size_t)ctx->bm_pre; local_s[1] = 1;
  global_s[0] = (size_t)((M + ctx->bm_pre - 1) / ctx->bm_pre) * ctx->bm_pre;
  global_s[1] = (size_t)N;
  { cl_int i = 0;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_scale, i++, d_cg));
    CL_CHECK(result, clSetKernelArg(kern_scale, i++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern_scale, i++, sizeof(int), &N));
    CL_CHECK(result, clSetKernelArg(kern_scale, i++, sizeof(int), &ldc));
    if (ctx->use_double) {
      double dbeta = beta;
      CL_CHECK(result, clSetKernelArg(kern_scale, i++, sizeof(double), &dbeta));
    }
    else {
      float fbeta = (float)beta;
      CL_CHECK(result, clSetKernelArg(kern_scale, i++, sizeof(float), &fbeta));
    }
  }
  CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern_scale, 2,
    NULL, global_s, local_s, 0, NULL, NULL));
  return result;
}


static int ozaki_launch_tinytc(ozaki_context_t* ctx, libxstream_stream_t* stream,
  void* d_as, void* d_bs, void* d_scratch, void* d_cg,
  void* d_expa_g, void* d_expb_g,
  int M, int N, int k_pad, int m_pad, int n_pad,
  int nslices_g, int tm, int tn, int ldc, int cutoff, int sq,
  cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled)
{
  int result = EXIT_SUCCESS;
  cl_kernel kern = ctx->kern_tinytc;
  const libxstream_opencl_stream_t* str = stream;
  int nblk_tc_m = (M + OZAKI_TINYTC_BM - 1) / OZAKI_TINYTC_BM;
  int nblk_tc_n = (N + OZAKI_TINYTC_BN - 1) / OZAKI_TINYTC_BN;
  cl_uint tc_nargs = 0;
  cl_int i = 0;
  clGetKernelInfo(kern, CL_KERNEL_NUM_ARGS, sizeof(tc_nargs), &tc_nargs, NULL);
  if (tc_nargs <= 6) {
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_as));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_bs));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_scratch));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_cg));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_expa_g));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_expb_g));
  }
  else {
    cl_long a_s0 = k_pad, a_s1 = m_pad;
    cl_long a_st1 = k_pad, a_st2 = (cl_long)m_pad * k_pad;
    cl_long b_s0 = n_pad, b_s1 = k_pad;
    cl_long b_st1 = n_pad, b_st2 = (cl_long)k_pad * n_pad;
    cl_long sc_s0 = (cl_long)M, sc_s1 = (cl_long)N;
    cl_long sc_st1 = (cl_long)M;
    cl_long ca_s0 = m_pad, ca_s1 = n_pad, ca_st1 = (cl_long)ldc;
    cl_long ea_s0 = (cl_long)M, eb_s0 = (cl_long)N;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_as));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &a_s0));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &a_s1));
    if (tc_nargs > 22) {
      cl_long a_s2 = nslices_g;
      CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &a_s2));
    }
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &a_st1));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &a_st2));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_bs));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &b_s0));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &b_s1));
    if (tc_nargs > 22) {
      cl_long b_s2 = nslices_g;
      CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &b_s2));
    }
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &b_st1));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &b_st2));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_scratch));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &sc_s0));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &sc_s1));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &sc_st1));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_cg));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &ca_s0));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &ca_s1));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &ca_st1));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_expa_g));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &ea_s0));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, i++, d_expb_g));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_long), &eb_s0));
    if (tc_nargs > 22) {
      CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_int), &cutoff));
      CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(cl_int), &sq));
    }
  }
  { size_t cwgs[3] = {0, 0, 0};
    size_t local_g[2]; size_t global_g[2];
    cl_device_id device = libxstream_opencl_config.devices[
      libxstream_opencl_config.device_id];
    clGetKernelWorkGroupInfo(kern, device,
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(cwgs), cwgs, NULL);
    local_g[0] = cwgs[0]; local_g[1] = cwgs[1];
    global_g[0] = (size_t)nblk_tc_m * local_g[0];
    global_g[1] = (size_t)nblk_tc_n * local_g[1];
    CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern, 2,
        NULL, global_g, local_g, 0, NULL,
        (NULL != evt_prof && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
          ? (evt_prof + *n_profiled) : NULL));
    if (EXIT_SUCCESS == result && NULL != evt_prof
        && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
    {
      ++(*n_profiled);
    }
  }
  return result;
}


static int ozaki_launch_fused(ozaki_context_t* ctx, libxstream_stream_t* stream,
  cl_kernel kern_g, void* d_as, void* d_bs, void* d_expa_g, void* d_expb_g,
  void* d_cg, int M, int N, int k_pad, int n_pad, int ldc, int m_pad,
  int tm, int tn, int ntm, int ntn, double alpha, int first_pair, int use_double,
  cl_event* evt_prof, int prof_a, int prof_b, int* n_profiled)
{
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = stream;
  size_t local_g[2], global_g[2];
  local_g[0] = 16;
  local_g[1] = (size_t)(ntm * ntn);
  { const int nblk_gm = (M + tm - 1) / tm;
    const int nblk_gn = (N + tn - 1) / tn;
    global_g[0] = (size_t)nblk_gm * local_g[0];
    global_g[1] = (size_t)nblk_gn * local_g[1];
  }
  { cl_int i = 0;
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_g, i++, d_as));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_g, i++, d_bs));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_g, i++, d_expa_g));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_g, i++, d_expb_g));
    CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern_g, i++, d_cg));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &N));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &k_pad));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &n_pad));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &ldc));
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &m_pad));
    if (use_double) {
      double dalpha = alpha;
      CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(double), &dalpha));
    }
    else {
      float falpha = (float)alpha;
      CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(float), &falpha));
    }
    CL_CHECK(result, clSetKernelArg(kern_g, i++, sizeof(int), &first_pair));
  }
  CL_CHECK(result, clEnqueueNDRangeKernel(str->queue, kern_g, 2,
      NULL, global_g, local_g, 0, NULL,
      (NULL != evt_prof && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
        ? (evt_prof + *n_profiled) : NULL));
  if (EXIT_SUCCESS == result && NULL != evt_prof
      && (prof_a == ctx->profile || prof_b == ctx->profile || 0 > ctx->profile))
  {
    ++(*n_profiled);
  }
  return result;
}
