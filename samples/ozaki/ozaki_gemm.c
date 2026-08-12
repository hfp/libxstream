/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"
#include "ozaki_kernels.h"
#include <libxs/libxs_hash.h>
#include <libxs/libxs_mem.h>


/**
 * Local helper functions (static) to manage kernel argument setup and launches.
 * These were kept local to avoid adding new translation units for the sample.
 */
static void ozaki_cache_check(ozaki_context_t* ctx, const void* a, const void* b, int M, int N, int K, int lda, int ldb, int ta,
  int tb, size_t as_size, size_t bs_size, size_t expa_size, size_t expb_size, void** d_as, void** d_bs, void** d_expa_g,
  void** d_expb_g, int* cache_hit_a, int* cache_hit_b);
static void ozaki_cache_update(ozaki_context_t* ctx, int result, const void* a, const void* b, int M, int N, int K, int lda,
  int ldb, int ta, int tb, size_t as_size, size_t bs_size, size_t expa_size, size_t expb_size, void* d_as, void* d_bs,
  void* d_expa_g, void* d_expb_g, int prev_owned, int* cache_hit_a, int* cache_hit_b);
static int ozaki_set_ptr_base(cl_kernel kern, cl_int* i, const void* ptr, size_t elsize, int wide);
static int ozaki_enqueue_preprocess(ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern, void* d_src, void* d_slices,
  void* d_exp, int M, int K, int ld, int trans, int k_pad, int pad, int bm_pre, int bk_pre, void* d_occ, int kmajor);
static int ozaki_enqueue_scale_beta(
  ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern_scale, void* d_cg, int M, int N, int ldc, double beta);
static int ozaki_launch_fused(ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern_g, void* d_as, void* d_bs,
  void* d_expa_g, void* d_expb_g, void* d_cg, int M, int N, int k_pad, int n_pad, int ldc, int m_pad, int tm, int tn, int ntm,
  int ntn, double alpha, int first_pair, int use_double);
static cl_kernel ozaki_get_fused_kernel(ozaki_context_t* ctx, int cutoff, int bounds, int tm, int tn);
static cl_kernel ozaki_get_crt_kernel(ozaki_context_t* ctx, int bounds, int tm, int tn);


int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream, char transa, char transb, int M, int N, int K, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc, int dev)
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

  int use_scheme1;
  ctx->stream = stream;
  if (NULL != libxstream_opencl_config.pool_dev) {
    libxs_malloc_arg(libxstream_opencl_config.pool_dev, stream);
  }

  /**
   * Adaptive scheme selection (kind==3): stateless per-call comparison of the
   * two bottlenecks -- Scheme-1 pairs*K int8 MACs vs Scheme-2 P*K MACs plus
   * O(P^2) Garner reconstruction. Dividing by K:
   *   pairs < P + xover * P^2 / K
   * The reconstruction term vanishes as K grows (amortized) and dominates at
   * small K. The pair count uses the static cutoff 2*(nslices-1)-oztrim, which
   * depends only on (nslices, oztrim) and is knowable on every call with no
   * cross-call state -- required for correctness under LD_PRELOAD with mixed
   * matrix sizes. This is deliberately pessimistic for Scheme 1: occupancy may
   * trim further at run time (dynamic cutoff), so a chosen Scheme 1 can only be
   * faster than estimated, while Scheme 2 stays untrimmed (oztrim_crt forced to
   * 0 at init) and thus favored when accuracy matters. Trim is a Scheme-1-only
   * knob here; trimmed Scheme 2 needs explicit kind==2. kind==1 or kind==2
   * force that scheme.
   *
   * In fp32 the comparison does not apply: four slices yield at most 16 pairs
   * against nine moduli, a ratio of 1.8 rather than the 4 seen in fp64 (64 vs
   * 16), and Scheme 1 recovers that from its shorter K-loop and cheaper
   * epilogue. Measured on Xe DPAS, Scheme 1 leads for every shape tried, from
   * n=512 up to K=16384 where Scheme 2 comes closest (12135 vs 11350
   * GFLOPS/s), so counting GEMMs mispredicts here and Scheme 1 is selected
   * outright; fp32 CRT stays reachable through kind==2.
   */
  { const int sq = ctx->ozflags & (OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE);
    if (2 == ctx->kind) {
      use_scheme1 = 0;
    }
    else if (1 == ctx->kind || 0 == ctx->use_double) {
      use_scheme1 = 1;
    }
    else {
      const int co = 2 * (ctx->nslices - 1) - ctx->oztrim;
      const int pairs = ozaki_count_pairs(ctx->nslices, co, sq);
      const double p = ctx->nprimes;
      use_scheme1 = (0 < K && pairs < p + ctx->xover * p * p / K);
    }
    /* Scheme 2 is absent when its kernels were not built (no fp64). */
    if (0 == use_scheme1 && NULL == ctx->crt_registry) use_scheme1 = 1;
  }

  /**
   * GEMM path (Scheme 1): full-split-then-tiled-GEMM.
   * Preprocesses entire K dimension up front into dense per-slice
   * int8 matrices, then runs a proper tiled GEMM per slice pair.
   */
  if (0 != use_scheme1 && NULL != ctx->kernel_registry && 0 < K) {
    const int nslices_g = ctx->nslices;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const ozaki_tile_t tile = ozaki_tile_select(ctx, M, N, ctx->rtm, ctx->rtn);
    const int tm = tile.m, tn = tile.n;
    int m_pad = LIBXS_UP(M, bm_pre);
    int n_pad = LIBXS_UP(N, bn_pre);
    const int nblk_gm = LIBXS_UPDIV(M, tm);
    const int nblk_gn = LIBXS_UPDIV(N, tn);
    const int ntm = tm / (OZAKI_XMX_M(ctx) * ctx->rtm), ntn = tn / (OZAKI_XMX_N(ctx) * ctx->rtn);
    const int cutoff = 2 * (nslices_g - 1) - ctx->oztrim;
    /**
     * K-group: size buffers for min(K, maxk), not full K.
     * maxk=0 means no grouping (full K in one pass).
     */
    const int k_grp_size = (0 < ctx->maxk ? ctx->maxk : K);
    const int k_grp_max = K < k_grp_size ? K : k_grp_size;
    int k_grp_pad = LIBXS_UP(k_grp_max, bk_pre);
    const int n_kgroups = LIBXS_UPDIV(K, k_grp_size);
    size_t as_size, bs_size, expa_size, expb_size;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    void *d_occ_a = NULL, *d_occ_b = NULL;
    int first_pair;
    int cache_hit_a = 0, cache_hit_b = 0;
    const size_t occ_size = (size_t)nslices_g * sizeof(cl_int);
    int kg;

    if (k_grp_pad < 64) k_grp_pad = 64;
    if (n_pad < 64) n_pad = 64;
    /* Cover the whole tile grid: see the CRT path below for why. */
    if (n_pad < nblk_gn * tn) n_pad = nblk_gn * tn;
    if (m_pad < nblk_gm * tm) m_pad = nblk_gm * tm;

    as_size = (size_t)nslices_g * m_pad * k_grp_pad;
    bs_size = (size_t)nslices_g * k_grp_pad * n_pad;
    expa_size = (size_t)nblk_gm * tm * elem_size;
    expb_size = (size_t)nblk_gn * tn * elem_size;
    c_nbytes = (size_t)ldc * (size_t)N * elem_size;

    /* Preprocessing cache: skip when K-grouping is active or a/b/c are device pointers */
    if (0 == dev && n_kgroups <= 1) {
      ozaki_cache_check(ctx, a, b, M, N, K, lda, ldb, ta, tb, as_size, bs_size, expa_size, expb_size, &d_as, &d_bs, &d_expa_g,
        &d_expb_g, &cache_hit_a, &cache_hit_b);
    }

    /**
     * Allocate device memory (skip cached sides and host-preprocessed sides).
     * When dev != 0, a/b/c are already device pointers (e.g. from ozaki_gemm_complex).
     */
    if (0 != dev) {
      LIBXS_UNION_ASSIGN(void*, d_ag, const void*, a);
      LIBXS_UNION_ASSIGN(void*, d_bg, const void*, b);
      d_cg = c;
    }
    else {
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
      }
      if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
        result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
      }
      if (EXIT_SUCCESS == result) result = libxstream_mem_dev_allocate_hint((void**)&d_cg, c_nbytes, libxstream_opencl_mem_hint_atomics);
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
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_occ_a, occ_size);
    if (EXIT_SUCCESS == result) result = OZAKI_DEV_ALLOC(&d_occ_b, occ_size);

    /**
     * H2D transfers: full source matrices (once).
     * Skip when dev != 0: a/b/c are already on device.
     * Skip C when beta == 0: kernel does not read C_old (BLAS spec).
     */
    if (0 == dev) {
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = libxstream_mem_copy_h2d(a, d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
      }
      if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
        result = libxstream_mem_copy_h2d(b, d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
      }
      if (EXIT_SUCCESS == result && 0.0 != beta) {
        result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);
      }
    }

    /**
     * Scale C by beta (once, before K-group loop).
     * When beta == 0, zero d_cg so the fused kernel's tile-by-tile
     * read-modify-write (OZAKI_SCALE_FLUSH) starts from zero.
     */
    if (EXIT_SUCCESS == result && 1.0 != beta) {
      if (0.0 != beta) {
        result = ozaki_enqueue_scale_beta(ctx, stream, ctx->kern_scale_beta, d_cg, M, N, ldc, beta);
      }
      else {
        result = libxstream_mem_zero(d_cg, 0, c_nbytes, stream);
      }
    }
    first_pair = (0.0 == beta) ? 1 : 0;

    /* K-group loop: preprocess + GEMM per group */
    for (kg = 0; kg < n_kgroups && EXIT_SUCCESS == result; ++kg) {
      const int kb_grp = kg * k_grp_size;
      const int K_len = ((K - kb_grp) < k_grp_size) ? (K - kb_grp) : k_grp_size;
      int k_pad = LIBXS_UP(K_len, bk_pre);
      const size_t a_off = ta ? ((size_t)kb_grp * elem_size) : ((size_t)kb_grp * lda * elem_size);
      const size_t b_off = tb ? ((size_t)kb_grp * ldb * elem_size) : ((size_t)kb_grp * elem_size);
      if (k_pad < 64) k_pad = 64;

      /**
       * Ensure previous GEMM finished before helper streams zero/preprocess.
       * When dev != 0, a/b are device buffers produced by the caller on
       * stream (e.g. the 3M construct kernels), so the very first group must
       * wait too: the preprocess kernels below read them from stream_a and
       * stream_b, for which nothing else establishes the dependency.
       */
      if (kg > 0 || 0 != dev) {
        if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream);
        if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_a, evt_prep_a);
        if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_b, evt_prep_a);
      }

      /* Zero slice/exp/occupancy buffers */
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
        if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
        if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_occ_a, 0, occ_size, stream_a);
      }
      if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
        result = libxstream_mem_zero(d_expb_g, 0, expb_size, stream_b);
        if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs, 0, bs_size, stream_b);
        if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_occ_b, 0, occ_size, stream_b);
      }

      if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);

      /* Preprocess A for this K-group */
      if (0 == cache_hit_a && EXIT_SUCCESS == result) {
        result = ozaki_enqueue_preprocess(ctx, stream_a, ctx->kern_preprocess_a, (char*)d_ag + a_off, d_as, d_expa_g, M, K_len,
          lda, ta, k_pad, m_pad, bm_pre, bk_pre, d_occ_a, 0 /*kmajor*/);
      }
      /* Preprocess B for this K-group */
      if (0 == cache_hit_b && EXIT_SUCCESS == result) {
        result = ozaki_enqueue_preprocess(ctx, stream_b, ctx->kern_preprocess_b, (char*)d_bg + b_off, d_bs, d_expb_g, N, K_len,
          ldb, tb, k_pad, n_pad, bn_pre, bk_pre, d_occ_b, 0 /*kmajor*/);
      }

      /* Wait for preprocessing to complete */
      if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

      /**
       * Compute adaptive cutoff from occupancy data.
       * On cache hit: reuse last_cutoff (no D2H readback, no sync bubble).
       * On miss: read occupancy from GPU, compute eff_cutoff, save for next time.
       */
      { int eff_cutoff = cutoff;
        if (0 != cache_hit_a && 0 != cache_hit_b && 0 != ctx->cache.last_cutoff) {
          eff_cutoff = ctx->cache.last_cutoff;
        }
        else if (0 == cache_hit_a && 0 == cache_hit_b) {
          cl_int occ_a[20], occ_b[20]; /* max NSLICES = 16 (fp64), pad to 20 */
          int sma = -1, smb = -1, si;
          if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_occ_a, occ_a, occ_size, stream);
          if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_occ_b, occ_b, occ_size, stream);
          if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
          for (si = nslices_g - 1; si >= 0; --si) { if (0 != occ_a[si]) { sma = si; break; } }
          for (si = nslices_g - 1; si >= 0; --si) { if (0 != occ_b[si]) { smb = si; break; } }
          if (sma >= 0 && smb >= 0) { eff_cutoff = sma + smb < cutoff ? sma + smb : cutoff; }
          else eff_cutoff = -1;
          ctx->cache.last_cutoff = eff_cutoff;
        }
      /* Launch GEMM for this K-group */
      { const int bounds = (0 != M % tm || 0 != N % tn);
        { cl_kernel kern_g = ozaki_get_fused_kernel(ctx, eff_cutoff, bounds, tm, tn);
          if (NULL != kern_g) {
            result = ozaki_launch_fused(ctx, stream, kern_g, d_as, d_bs, d_expa_g, d_expb_g, d_cg, M, N, k_pad, n_pad, ldc, m_pad, tm,
              tn, ntm, ntn, alpha, first_pair, ctx->use_double);
          }
          else result = EXIT_FAILURE;
        }
      }
      } /* end adaptive cutoff scope */
      first_pair = 0; /* subsequent groups accumulate */
    } /* end K-group loop */

    /**
     * Save preprocessed buffers to cache (only for single-group case).
     * Skip when dev != 0: device pointers are not valid cache keys.
     */
    if (0 == dev && n_kgroups <= 1) {
      const int prev_owned = (0 != cache_hit_a || 0 != cache_hit_b);
      ozaki_cache_update(ctx, result, a, b, M, N, K, lda, ldb, ta, tb, as_size, bs_size, expa_size, expb_size, d_as, d_bs, d_expa_g,
        d_expb_g, prev_owned, &cache_hit_a, &cache_hit_b);
    }

    /**
     * D2H result and cleanup.
     * Skip when dev != 0: result is already in caller's device buffer.
     */
    if (0 == dev) {
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);
    }

    /**
     * Sync ALL streams before freeing device buffers to ensure transfers completed.
     * Device pool deallocator only syncs on grow path, not regular frees.
     * - Main stream uses d_cg (for D2H)
     * - stream_a uses d_ag (for preprocessing)
     * - stream_b uses d_bg (for preprocessing)
     * Without sync, freed buffers can be reallocated while DMA is still reading.
     */
    if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
    if (EXIT_SUCCESS == result && NULL != stream_a && 0 == cache_hit_a) {
      result = libxstream_stream_sync(stream_a);
    }
    if (EXIT_SUCCESS == result && NULL != stream_b && 0 == cache_hit_b) {
      result = libxstream_stream_sync(stream_b);
    }

    if (0 == dev) {
      OZAKI_DEV_FREE(d_ag);
      OZAKI_DEV_FREE(d_bg);
      if (NULL != d_cg) libxstream_mem_dev_deallocate_hint(d_cg);
    }
    OZAKI_DEV_FREE(d_occ_a);
    OZAKI_DEV_FREE(d_occ_b);
    if (0 == cache_hit_a) {
      OZAKI_DEV_FREE(d_as);
      OZAKI_DEV_FREE(d_expa_g);
    }
    if (0 == cache_hit_b) {
      OZAKI_DEV_FREE(d_bs);
      OZAKI_DEV_FREE(d_expb_g);
    }
    if (0 != cache_hit_a || 0 != cache_hit_b) {
      LIBXS_ATOMIC_SUB_FETCH(&ctx->cache.nusers, 1, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  /**
   * CRT GEMM path (Scheme 2): full-split-then-single-fused-GEMM.
   * Preprocesses entire K into dense per-prime CRT residue matrices,
   * then runs a single kernel per tile that loops over all primes
   * internally (full-K DPAS + Garner + Horner in one launch).
   */
  else if (NULL != ctx->crt_registry && 0 < K) {
    const int nprimes_g = ctx->nprimes;
    const int bk_pre = ctx->bk_pre;
    const int bm_pre = ctx->bm_pre;
    const int bn_pre = ctx->bn_pre;
    const ozaki_tile_t tile = ozaki_tile_select(ctx, M, N, ctx->crt_rtm, ctx->crt_rtn);
    const int tm = tile.m, tn = tile.n;
    int m_pad = LIBXS_UP(M, bm_pre);
    int n_pad;
    const int nblk_gm = LIBXS_UPDIV(M, tm);
    const int ntm = tm / (OZAKI_XMX_M(ctx) * ctx->crt_rtm), ntn = tn / (OZAKI_XMX_N(ctx) * ctx->crt_rtn);
    /**
     * K-group: size buffers for min(K, maxk), not full K.
     * maxk=0 means no grouping (full K in one pass).
     */
    const int k_grp_size = (0 < ctx->maxk ? ctx->maxk : K);
    const int k_grp_max = K < k_grp_size ? K : k_grp_size;
    const int ku_bk = ctx->ku * bk_pre;
    int k_grp_pad = LIBXS_UP(k_grp_max, ku_bk);
    const int n_kgroups = LIBXS_UPDIV(K, k_grp_size);
    /**
     * N-panel pipeline. Each panel owns a disjoint column block of B and C, so
     * C is read and written exactly once regardless of the panel count (a
     * K-split would instead re-read and re-write all of C per group), and panel
     * GEMMs are independent rather than chained through C accumulation.
     * Panelling is bypassed when it would not pay: n_panel == N gives one
     * panel and the code below degenerates to the original single-shot flow.
     * A is preprocessed once as a prologue; only B and C are panelled.
     *
     * Caching B is the one thing panelling cannot coexist with: a cached slice
     * buffer must hold all of B, whereas panels materialize one column block
     * per slot. The two are alternative ways to avoid the same work -- reuse
     * across calls versus overlap within a call. Caching keeps precedence
     * (an explicit OZAKI_CACHE request is honored unchanged); it is the
     * absence of a B-cache request -- the default -- that enables panelling.
     */
    const int cacheable_b = (0 != (ctx->cache.flags & 2));
    const int n_panel = (0 == dev && n_kgroups <= 1 && 0 == cacheable_b) ? ozaki_npanel(ctx, M, N, tm, tn) : N;
    const int npanels = LIBXS_UPDIV(N, n_panel);
    /* Per-panel B upload: only when a panel is a contiguous column block. */
    const int b_panel_h2d = (0 == dev && 1 < npanels && 0 == tb) ? 1 : 0;
    const int nslots = (1 < npanels) ? OZAKI_NSLOTS : 1;
    const int nblk_pn = LIBXS_UPDIV(n_panel, tn); /* tiles per panel (n_panel is a tn multiple) */
    size_t as_size, bs_size, expa_size, expb_size, bs_slot, expb_slot;
    void *d_as = NULL, *d_bs = NULL;
    void *d_expa_g = NULL, *d_expb_g = NULL;
    void *d_ag = NULL, *d_bg = NULL, *d_cg = NULL;
    int first_tile;
    int cache_hit_a = 0, cache_hit_b = 0;
    int kg;

    if (k_grp_pad < 64) k_grp_pad = 64;
    /**
     * Column extent of one panel's Bs plane. The tile grid spans nblk_pn * tn
     * columns, which exceeds the panel rounded to bn_pre whenever tn does not
     * divide it (e.g. panel 1024, tn=96 reaches 1056). Bs planes are strided by
     * k_pad * n_pad, so a last-tile column past n_pad reads the next prime's
     * plane and accumulates a foreign residue instead of the zero the padding
     * is meant to supply. Cover the whole grid.
     */
    n_pad = LIBXS_UP(n_panel, bn_pre);
    if (n_pad < 64) n_pad = 64;
    if (n_pad < nblk_pn * tn) n_pad = nblk_pn * tn;
    if (m_pad < nblk_gm * tm) m_pad = nblk_gm * tm;

    as_size = (size_t)nprimes_g * m_pad * k_grp_pad;
    bs_slot = (size_t)nprimes_g * k_grp_pad * n_pad;
    bs_size = bs_slot * nslots;
    expa_size = (size_t)nblk_gm * tm * sizeof(cl_int); /* pad to tile boundary */
    expb_slot = (size_t)nblk_pn * tn * sizeof(cl_int);
    expb_size = expb_slot * nslots;
    c_nbytes = (size_t)ldc * (size_t)N * elem_size;

    /**
     * Preprocessing cache: skip when K-grouping is active or a/b/c are device
     * pointers. Caching B excludes panelling (see cacheable_b), so the two
     * never both apply and no masking is needed here.
     */
    if (0 == dev && n_kgroups <= 1) {
      ozaki_cache_check(ctx, a, b, M, N, K, lda, ldb, ta, tb, as_size, bs_size, expa_size, expb_size, &d_as, &d_bs, &d_expa_g,
        &d_expb_g, &cache_hit_a, &cache_hit_b);
    }

    /**
     * Allocate device memory (skip cached sides and host-preprocessed sides).
     * When dev != 0, a/b/c are already device pointers (e.g. from ozaki_gemm_complex).
     */
    if (0 != dev) {
      LIBXS_UNION_ASSIGN(void*, d_ag, const void*, a);
      LIBXS_UNION_ASSIGN(void*, d_bg, const void*, b);
      d_cg = c;
    }
    else {
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = OZAKI_DEV_ALLOC(&d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size);
      }
      if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
        result = OZAKI_DEV_ALLOC(&d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size);
      }
      if (EXIT_SUCCESS == result) result = libxstream_mem_dev_allocate_hint((void**)&d_cg, c_nbytes, libxstream_opencl_mem_hint_atomics);
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

    /**
     * H2D transfers: full source matrices (once).
     * Skip when dev != 0: a/b/c are already on device.
     * Skip C when beta == 0: kernel does not read C_old (BLAS spec).
     *
     * B is uploaded per panel inside the loop when its columns are contiguous
     * (b_panel_h2d), so each upload overlaps the previous panel's GEMM instead
     * of stalling panel 0 behind all of B. With transb a panel is a row block
     * of a K-column matrix, i.e. strided, which one linear copy cannot express,
     * so it is uploaded whole up front in that case.
     */
    if (0 == dev) {
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = libxstream_mem_copy_h2d(a, d_ag, (size_t)lda * (ta ? (size_t)M : (size_t)K) * elem_size, stream_a);
      }
      if (EXIT_SUCCESS == result && 0 == cache_hit_b && 0 == b_panel_h2d) {
        result = libxstream_mem_copy_h2d(b, d_bg, (size_t)ldb * (tb ? (size_t)K : (size_t)N) * elem_size, stream_b);
      }
      if (EXIT_SUCCESS == result && 0.0 != beta) {
        result = libxstream_mem_copy_h2d(c, d_cg, c_nbytes, stream);
      }
    }

    /* Scale C when beta != 0 and beta != 1 (once, before K-group loop) */
    if (EXIT_SUCCESS == result && 1.0 != beta && 0.0 != beta) {
      result = ozaki_enqueue_scale_beta(ctx, stream, ctx->kern_crt_scale_beta, d_cg, M, N, ldc, beta);
    }
    first_tile = (0.0 == beta) ? 1 : 0;

    /* K-group loop: preprocess + CRT GEMM per group */
    for (kg = 0; kg < n_kgroups && EXIT_SUCCESS == result; ++kg) {
      const int kb_grp = kg * k_grp_size;
      const int K_len = ((K - kb_grp) < k_grp_size) ? (K - kb_grp) : k_grp_size;
      int k_pad = LIBXS_UP(K_len, ku_bk);
      const size_t a_off = ta ? ((size_t)kb_grp * elem_size) : ((size_t)kb_grp * lda * elem_size);
      const size_t b_off = tb ? ((size_t)kb_grp * ldb * elem_size) : ((size_t)kb_grp * elem_size);
      int pj;
      if (k_pad < 64) k_pad = 64;

      /**
       * Ensure previous GEMM finished before helper streams zero/preprocess.
       * When dev != 0, a/b are device buffers produced by the caller on
       * stream (e.g. the 3M construct kernels), so the very first group must
       * wait too: the preprocess kernels below read them from stream_a and
       * stream_b, for which nothing else establishes the dependency.
       */
      if (kg > 0 || 0 != dev) {
        if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream);
        if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_a, evt_prep_a);
        if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_b, evt_prep_a);
      }

      /* Prologue: zero and preprocess A once for the whole K-group. */
      if (EXIT_SUCCESS == result && 0 == cache_hit_a) {
        result = libxstream_mem_zero(d_expa_g, 0, expa_size, stream_a);
        if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_as, 0, as_size, stream_a);
        if (EXIT_SUCCESS == result) {
          result = ozaki_enqueue_preprocess(ctx, stream_a, ctx->kern_crt_preprocess_a, (char*)d_ag + a_off, d_as, d_expa_g, M, K_len,
            lda, ta, k_pad, m_pad, bm_pre, bk_pre, NULL /*no occ for CRT*/, 1 /*kmajor*/);
        }
      }
      if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_a, stream_a);
      if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_a);

      /**
       * Panel loop over N. Steady state overlaps three things: panel pj+1's
       * B preprocessing on stream_b, panel pj's GEMM on stream, and panel
       * pj-1's C download on stream_a. The slot events are what decouple them:
       * a panel reusing slot s waits only on the GEMM that consumed slot s
       * (OZAKI_NSLOTS panels back), not on the immediately preceding one.
       */
      for (pj = 0; pj < npanels && EXIT_SUCCESS == result; ++pj) {
        const int nb = pj * n_panel;
        const int N_len = ((N - nb) < n_panel) ? (N - nb) : n_panel;
        const int slot = (1 < nslots) ? (pj % nslots) : 0;
        void *const d_bs_s = (char*)d_bs + (size_t)slot * bs_slot;
        void *const d_expb_s = (char*)d_expb_g + (size_t)slot * expb_slot;
        /* B column block: N-major operand, so the panel offset depends on tb. */
        const size_t bp_off = tb ? ((size_t)nb * elem_size) : ((size_t)nb * ldb * elem_size);
        /* C column block: column-major, always ldc-strided. */
        const size_t cp_off = (size_t)nb * ldc * elem_size;

        /**
         * Reuse of this slot must wait for the GEMM that last read it. With
         * nslots slots that GEMM is nslots panels back, so the wait is a no-op
         * until the pipeline is full -- which is precisely the overlap.
         */
        if (pj >= nslots && EXIT_SUCCESS == result) {
          result = libxstream_stream_wait_event(stream_b, ctx->evt_slot[slot]);
        }

        /**
         * Upload this panel's B columns. Contiguous for the whole K extent, so
         * one copy covers the panel: column nb starts at nb*ldb and the panel
         * spans N_len columns. Enqueued on stream_b ahead of the preprocess
         * that consumes it, hence overlapped with the previous panel's GEMM.
         */
        if (EXIT_SUCCESS == result && 0 != b_panel_h2d) {
          result = libxstream_mem_copy_h2d((const char*)b + bp_off, (char*)d_bg + bp_off,
            (size_t)ldb * N_len * elem_size, stream_b);
        }

        /**
         * Zero and preprocess this panel's B slice into its slot. cache_hit_b
         * implies a single panel spanning all of B (panelling and a cached B
         * are mutually exclusive), so the hit skips the work as before.
         */
        if (EXIT_SUCCESS == result && 0 == cache_hit_b) {
          result = libxstream_mem_zero(d_expb_s, 0, expb_slot, stream_b);
          if (EXIT_SUCCESS == result) result = libxstream_mem_zero(d_bs_s, 0, bs_slot, stream_b);
          if (EXIT_SUCCESS == result) {
            result = ozaki_enqueue_preprocess(ctx, stream_b, ctx->kern_crt_preprocess_b, (char*)d_bg + b_off + bp_off, d_bs_s,
              d_expb_s, N_len, K_len, ldb, tb, k_pad, n_pad, bn_pre, bk_pre, NULL /*no occ for CRT*/, 0 /*kmajor*/);
          }
        }
        if (EXIT_SUCCESS == result) result = libxstream_event_record(evt_prep_b, stream_b);
        if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream, evt_prep_b);

        /**
         * GEMM for this panel. The kernel sees a standalone M x N_len problem:
         * C is offset to the panel's first column, and the exponent/slice
         * buffers are the panel's own, so no index translation is needed.
         * Bounds checking follows N_len, the extent this launch actually spans.
         */
        if (EXIT_SUCCESS == result) {
          const int bounds = (0 != M % tm || 0 != N_len % tn);
          cl_kernel crt_kern = ozaki_get_crt_kernel(ctx, bounds, tm, tn);
          if (NULL != crt_kern) {
            result = ozaki_launch_fused(ctx, stream, crt_kern, d_as, d_bs_s, d_expa_g, d_expb_s, (char*)d_cg + cp_off, M, N_len,
              k_pad, n_pad, ldc, m_pad, tm, tn, ntm, ntn, alpha, first_tile,
              ctx->use_double);
          }
          else result = EXIT_FAILURE;
        }
        /* Release the slot for a later panel, and gate this panel's D2H. */
        if (EXIT_SUCCESS == result) result = libxstream_event_record(ctx->evt_slot[slot], stream);
        /**
         * Download this panel's C block on stream_a so it overlaps the next
         * panel's GEMM. Only for the last K-group: earlier groups still have
         * pending accumulation into the same C columns.
         */
        if (EXIT_SUCCESS == result && 0 == dev && 1 < npanels && kg + 1 == n_kgroups) {
          result = libxstream_event_record(ctx->evt_panel, stream);
          if (EXIT_SUCCESS == result) result = libxstream_stream_wait_event(stream_a, ctx->evt_panel);
          if (EXIT_SUCCESS == result) {
            result = libxstream_mem_copy_d2h((char*)d_cg + cp_off, (char*)c + cp_off, (size_t)ldc * N_len * elem_size, stream_a);
          }
        }
      } /* end panel loop */
      first_tile = 0; /* subsequent groups accumulate */
    } /* end K-group loop */

    /**
     * Save preprocessed buffers to cache (only for single-group case).
     * Skip when dev != 0: device pointers are not valid cache keys.
     */
    if (0 == dev && n_kgroups <= 1) {
      const int prev_owned = (0 != cache_hit_a || 0 != cache_hit_b);
      ozaki_cache_update(ctx, result, a, b, M, N, K, lda, ldb, ta, tb, as_size, bs_size, expa_size, expb_size, d_as, d_bs, d_expa_g,
        d_expb_g, prev_owned, &cache_hit_a, &cache_hit_b);
    }

    /**
     * D2H result. Skip when dev != 0: result is already in caller's device
     * buffer. While panelling, each panel already downloaded its own column
     * block overlapped with the following panel's GEMM.
     */
    if (0 == dev && 1 >= npanels) {
      if (EXIT_SUCCESS == result) result = libxstream_mem_copy_d2h(d_cg, c, c_nbytes, stream);
    }

    /**
     * Sync ALL streams before freeing device buffers to ensure transfers completed.
     * Device pool deallocator only syncs on grow path, not regular frees.
     * - Main stream uses d_cg (for D2H)
     * - stream_a uses d_ag (for preprocessing) and the panelled C download
     * - stream_b uses d_bg (for preprocessing)
     * Without sync, freed buffers can be reallocated while DMA is still reading.
     */
    if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
    if (EXIT_SUCCESS == result && NULL != stream_a && (0 == cache_hit_a || 1 < npanels)) {
      result = libxstream_stream_sync(stream_a);
    }
    if (EXIT_SUCCESS == result && NULL != stream_b && 0 == cache_hit_b) {
      result = libxstream_stream_sync(stream_b);
    }

    if (0 == dev) {
      OZAKI_DEV_FREE(d_ag);
      OZAKI_DEV_FREE(d_bg);
      if (NULL != d_cg) libxstream_mem_dev_deallocate_hint(d_cg);
    }
    if (0 == cache_hit_a) {
      OZAKI_DEV_FREE(d_as);
      OZAKI_DEV_FREE(d_expa_g);
    }
    if (0 == cache_hit_b) {
      OZAKI_DEV_FREE(d_bs);
      OZAKI_DEV_FREE(d_expb_g);
    }
    if (0 != cache_hit_a || 0 != cache_hit_b) {
      LIBXS_ATOMIC_SUB_FETCH(&ctx->cache.nusers, 1, LIBXS_ATOMIC_LOCKORDER);
    }
  }

  /**
   * Invalidate cache entries whose pointer matches the output matrix C.
   * C was just written; if C's address is later passed as A or B,
   * stale preprocessed data from before the write would be used.
   */
  if (0 != ctx->cache.flags) {
    ozaki_invalidate_cache(ctx, c, c);
  }

  return result;
}


/**
 * Set a buffer argument as a (pointer, index) pair. Emitting both from one place
 * is what keeps them consistent: a call site that offsets the pointer but forgets
 * the index would read the panel from the wrong place, silently.
 *
 * With USM the offset travels in the pointer and the index is zero. Without USM
 * clSetKernelArg takes a cl_mem that cannot express an offset, so the pointer is
 * resolved to its registered base and the index carries the remainder -- which is
 * what lets panelling work with neither USM nor sub-buffers. The kernel applies
 * base unconditionally, so the two cases share one code path on the device.
 *
 * wide selects a long index for the slice buffers, whose element count exceeds
 * INT_MAX at a large K_pad/N_pad (nprimes * k_pad * n_pad), while the matrices
 * stay within int.
 */
static int ozaki_set_ptr_base(cl_kernel kern, cl_int* i, const void* ptr, size_t elsize, int wide)
{
  libxstream_opencl_info_memptr_t info;
  size_t offset = 0;
  int result = EXIT_SUCCESS;
  void* nc;
  LIBXS_UNION_ASSIGN(void*, nc, const void*, ptr);
  if (NULL != ptr) {
    result = libxstream_opencl_info_devptr(&info, ptr, elsize, NULL /*amount*/, &offset);
    /* info.memory is the base under registration, or the pointer itself under USM */
    if (EXIT_SUCCESS == result) LIBXS_ASSIGN(&nc, &info.memory);
  }
  CL_CHECK(result, libxstream_opencl_set_kernel_ptr(kern, (*i)++, nc));
  if (0 == wide) {
    const cl_int base = (cl_int)offset;
    assert((size_t)base == offset); /* matrices stay within int */
    CL_CHECK(result, clSetKernelArg(kern, (*i)++, sizeof(cl_int), &base));
  }
  else {
    const cl_long base = (cl_long)offset;
    CL_CHECK(result, clSetKernelArg(kern, (*i)++, sizeof(cl_long), &base));
  }
  return result;
}


/**
 * kmajor selects the work-group shape: 0 maps the M/N extent to dim 0 (the
 * layout-following mapping every preprocessor started with), 1 puts K there so
 * a sub-group's lanes walk the contiguous axis of the slice buffer. Only
 * preprocess_a_crt_dense wants the latter, and the shape is declared per kernel
 * via reqd_work_group_size, so the two must agree.
 */
static int ozaki_enqueue_preprocess(ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern, void* d_src, void* d_slices,
  void* d_exp, int M, int K, int ld, int trans, int k_pad, int pad, int bm_pre, int bk_pre, void* d_occ, int kmajor)
{
  int result = EXIT_SUCCESS;
  size_t global[2], local[2];
  const int nblk_m_pre = LIBXS_UPDIV(M, bm_pre);
  if (0 == kmajor) {
    local[0] = bm_pre;
    local[1] = bk_pre;
    global[0] = (size_t)nblk_m_pre * bm_pre;
    global[1] = bk_pre; /* single WG in K: kernel loops internally */
  }
  else {
    local[0] = bk_pre;
    local[1] = bm_pre;
    global[0] = bk_pre; /* single WG in K: kernel loops internally */
    global[1] = (size_t)nblk_m_pre * bm_pre;
  }
  {
    const size_t elsize = ctx->use_double ? sizeof(double) : sizeof(float);
    cl_int i = 0;
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern, &i, d_src, elsize, 0 /*int*/);
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &M));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &K));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &ld));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &trans));
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern, &i, d_slices, 1 /*char*/, 1 /*long*/);
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern, &i, d_exp, sizeof(cl_int), 0 /*int*/);
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &k_pad));
    CL_CHECK(result, clSetKernelArg(kern, i++, sizeof(int), &pad));
    if (NULL != d_occ) {
      if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern, &i, d_occ, sizeof(cl_int), 0 /*int*/);
    }
  }
  CL_CHECK(result, libxstream_opencl_launch(stream, kern, 2, NULL, global, local, 0, NULL, NULL));
  return result;
}


static int ozaki_enqueue_scale_beta(
  ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern_scale, void* d_cg, int M, int N, int ldc, double beta)
{
  int result = EXIT_SUCCESS;
  size_t global_s[2], local_s[2];
  local_s[0] = (size_t)ctx->bm_pre;
  local_s[1] = 1;
  global_s[0] = (size_t)LIBXS_UP(M, ctx->bm_pre);
  global_s[1] = (size_t)N;
  {
    const size_t elsize = ctx->use_double ? sizeof(double) : sizeof(float);
    cl_int i = 0;
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_scale, &i, d_cg, elsize, 0 /*int*/);
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
  CL_CHECK(result, libxstream_opencl_launch(stream, kern_scale, 2, NULL, global_s, local_s, 0, NULL, NULL));
  return result;
}


static cl_kernel ozaki_get_fused_kernel(ozaki_context_t* ctx, int cutoff, int bounds, int tm, int tn)
{
  ozaki_kernel_key_t key;
  ozaki_kernel_set_t* kset;
  memset(&key, 0, sizeof(key));
  key.cutoff = cutoff;
  key.bounds = bounds;
  key.tm = tm;
  key.tn = tn;
  kset = (ozaki_kernel_set_t*)libxs_registry_get(ctx->kernel_registry, &key,
    sizeof(key), libxs_registry_lock(ctx->kernel_registry));
  if (NULL == kset || NULL == kset->kern_fused) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->kernel_lock);
    kset = (ozaki_kernel_set_t*)libxs_registry_get(ctx->kernel_registry, &key,
      sizeof(key), libxs_registry_lock(ctx->kernel_registry));
    if (NULL == kset || NULL == kset->kern_fused) {
      char flags[1408];
      ozaki_kernel_set_t newset;
      cl_program program = NULL;
      int n;
      memset(&newset, 0, sizeof(newset));
      { char pname[64];
        LIBXS_SNPRINTF(pname, sizeof(pname), "oz1_c%d_%dx%d%s", cutoff, tm, tn, 0 != bounds ? "b" : "");
        n = LIBXS_SNPRINTF(flags, sizeof(flags), "%s -DBM=%d -DBN=%d -DOZAKI_CUTOFF=%d%s",
          ctx->base_flags, tm, tn, cutoff, 0 != bounds ? " -DOZAKI_BOUNDS=1" : "");
        LIBXS_UNUSED(n);
        if (EXIT_SUCCESS == libxstream_opencl_program(
              0, OPENCL_KERNELS_SOURCE_OZAKI1_INT8, pname, flags,
              ctx->base_options, NULL, NULL, NULL, 0, &program)) {
          libxstream_opencl_kernel_query(program, "gemm_fused", &newset.kern_fused);
        }
      }
      if (NULL != program) clReleaseProgram(program);
      if (NULL != newset.kern_fused) {
        kset = (ozaki_kernel_set_t*)libxs_registry_set(ctx->kernel_registry, &key,
          sizeof(key), &newset, sizeof(newset), libxs_registry_lock(ctx->kernel_registry));
      }
      if (0 > ctx->verbosity || 2 < ctx->verbosity) {
        fprintf(stderr, "INFO OZAKI: JIT cutoff=%d bounds=%d tile=%dx%d -> %s\n",
          cutoff, bounds, tm, tn, NULL != newset.kern_fused ? "OK" : "FAILED");
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->kernel_lock);
  }
  return (NULL != kset) ? kset->kern_fused : NULL;
}


static cl_kernel ozaki_get_crt_kernel(ozaki_context_t* ctx, int bounds, int tm, int tn)
{
  ozaki_crt_kernel_key_t key;
  ozaki_crt_kernel_set_t* kset;
  memset(&key, 0, sizeof(key));
  key.bounds = bounds;
  key.tm = tm;
  key.tn = tn;
  kset = (ozaki_crt_kernel_set_t*)libxs_registry_get(ctx->crt_registry, &key,
    sizeof(key), libxs_registry_lock(ctx->crt_registry));
  if (NULL == kset || NULL == kset->kern_fused) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->kernel_lock);
    kset = (ozaki_crt_kernel_set_t*)libxs_registry_get(ctx->crt_registry, &key,
      sizeof(key), libxs_registry_lock(ctx->crt_registry));
    if (NULL == kset || NULL == kset->kern_fused) {
      char flags[sizeof(ctx->crt_flags) + 64];
      ozaki_crt_kernel_set_t newset;
      cl_program program = NULL;
      memset(&newset, 0, sizeof(newset));
      { char pname[64];
        LIBXS_SNPRINTF(pname, sizeof(pname), "oz2_%dx%d%s", tm, tn, 0 != bounds ? "b" : "");
        LIBXS_SNPRINTF(flags, sizeof(flags), "%s -DBM=%d -DBN=%d%s",
          ctx->crt_flags, tm, tn, 0 != bounds ? " -DOZAKI_BOUNDS=1" : "");
        if (EXIT_SUCCESS == libxstream_opencl_program(
              0, OPENCL_KERNELS_SOURCE_OZAKI2_INT8, pname, flags,
              ctx->crt_options, NULL, NULL, NULL, 0, &program)) {
          libxstream_opencl_kernel_query(program, "gemm_crt_fused", &newset.kern_fused);
        }
      }
      if (NULL != program) clReleaseProgram(program);
      if (NULL != newset.kern_fused) {
        kset = (ozaki_crt_kernel_set_t*)libxs_registry_set(ctx->crt_registry, &key,
          sizeof(key), &newset, sizeof(newset), libxs_registry_lock(ctx->crt_registry));
      }
      if (0 > ctx->verbosity || 2 < ctx->verbosity) {
        fprintf(stderr, "INFO OZAKI: JIT crt bounds=%d tile=%dx%d -> %s\n",
          bounds, tm, tn, NULL != newset.kern_fused ? "OK" : "FAILED");
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->kernel_lock);
  }
  return (NULL != kset) ? kset->kern_fused : NULL;
}


static int ozaki_launch_fused(ozaki_context_t* ctx, libxstream_stream_t* stream, cl_kernel kern_g, void* d_as, void* d_bs,
  void* d_expa_g, void* d_expb_g, void* d_cg, int M, int N, int k_pad, int n_pad, int ldc, int m_pad, int tm, int tn, int ntm,
  int ntn, double alpha, int first_pair, int use_double)
{
  int result = EXIT_SUCCESS;
  size_t local_g[2], global_g[2];
  local_g[0] = (size_t)ctx->sg;
  local_g[1] = (size_t)(ntm * ntn);
  {
    const int nblk_gm = LIBXS_UPDIV(M, tm);
    const int nblk_gn = LIBXS_UPDIV(N, tn);
    global_g[0] = (size_t)nblk_gm * local_g[0];
    global_g[1] = (size_t)nblk_gn * local_g[1];
  }
  {
    const size_t elsize = use_double ? sizeof(double) : sizeof(float);
    cl_int i = 0;
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_g, &i, d_as, 1 /*char*/, 1 /*long*/);
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_g, &i, d_bs, 1 /*char*/, 1 /*long*/);
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_g, &i, d_expa_g, sizeof(cl_int), 0 /*int*/);
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_g, &i, d_expb_g, sizeof(cl_int), 0 /*int*/);
    if (EXIT_SUCCESS == result) result = ozaki_set_ptr_base(kern_g, &i, d_cg, elsize, 0 /*int*/);
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
  /**
   * Report the GEMM this launch realizes in the caller's precision, i.e. without
   * the npairs (or nprimes) INT8 products the decomposition needs to reach it.
   * Multiplying by that factor would state INT8 operation throughput instead --
   * a hardware-utilization figure that reads as impossible next to an FP peak,
   * and not the rate a caller of DGEMM is asking about.
   */
  CL_CHECK(result, libxstream_opencl_launch_work(stream, kern_g, 2, NULL, global_g, local_g, 0, NULL, NULL,
                     2 * (size_t)M * (size_t)N * (size_t)k_pad, 0 /*nbytes*/));
  return result;
}


unsigned int ozaki_cache_fingerprint(const void* ptr, size_t elem_size, int ncontig, int nld, int ld)
{
  const unsigned char* p = (const unsigned char*)ptr;
  const size_t stride = (size_t)ld * elem_size;
  const int rows = 0 < ncontig ? ncontig : 1;
  const int cols = 0 < nld ? nld : 1;
  unsigned int fp = 0;
  int pr[8], pc[8], i;
  pr[0] = 0;
  pc[0] = 0;
  pr[1] = 0;
  pc[1] = cols - 1;
  pr[2] = rows - 1;
  pc[2] = 0;
  pr[3] = rows - 1;
  pc[3] = cols - 1;
  pr[4] = rows / 2;
  pc[4] = cols / 2;
  pr[5] = rows / 3;
  pc[5] = cols / 3;
  pr[6] = rows - 1;
  pc[6] = cols / 2;
  pr[7] = 0;
  pc[7] = cols / 2;
  for (i = 0; i < 8; ++i) {
    const size_t offset = (size_t)pc[i] * stride + (size_t)pr[i] * elem_size;
    fp = libxs_hash(p + offset, (unsigned int)elem_size, fp);
  }
  return fp;
}


static void ozaki_cache_check(ozaki_context_t* ctx, const void* a, const void* b, int M, int N, int K, int lda, int ldb, int ta,
  int tb, size_t as_size, size_t bs_size, size_t expa_size, size_t expb_size, void** d_as, void** d_bs, void** d_expa_g,
  void** d_expb_g, int* cache_hit_a, int* cache_hit_b)
{
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->cache.lock);
  if (0 != (ctx->cache.flags & 1) && a == ctx->cache.a.ptr && M == ctx->cache.a.dim && K == ctx->cache.a.K &&
      lda == ctx->cache.a.ld && ta == ctx->cache.a.trans && as_size == ctx->cache.a.slices_size &&
      expa_size == ctx->cache.a.exp_size && NULL != ctx->cache.a.d_slices && NULL != ctx->cache.a.d_exp &&
      ctx->cache.a.fingerprint == ozaki_cache_fingerprint(a, elem_size, ta ? K : M, ta ? M : K, lda))
  {
    *d_as = ctx->cache.a.d_slices;
    *d_expa_g = ctx->cache.a.d_exp;
    *cache_hit_a = 1;
  }
  if (0 != (ctx->cache.flags & 2) && b == ctx->cache.b.ptr && N == ctx->cache.b.dim && K == ctx->cache.b.K &&
      ldb == ctx->cache.b.ld && tb == ctx->cache.b.trans && bs_size == ctx->cache.b.slices_size &&
      expb_size == ctx->cache.b.exp_size && NULL != ctx->cache.b.d_slices && NULL != ctx->cache.b.d_exp &&
      ctx->cache.b.fingerprint == ozaki_cache_fingerprint(b, elem_size, tb ? N : K, tb ? K : N, ldb))
  {
    *d_bs = ctx->cache.b.d_slices;
    *d_expb_g = ctx->cache.b.d_exp;
    *cache_hit_b = 1;
  }
  if (0 != *cache_hit_a || 0 != *cache_hit_b) ++ctx->cache.nusers;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->cache.lock);
}


static void ozaki_cache_update(ozaki_context_t* ctx, int result, const void* a, const void* b, int M, int N, int K, int lda,
  int ldb, int ta, int tb, size_t as_size, size_t bs_size, size_t expa_size, size_t expb_size, void* d_as, void* d_bs,
  void* d_expa_g, void* d_expb_g, int prev_owned, int* cache_hit_a, int* cache_hit_b)
{
  const size_t elem_size = ctx->use_double ? sizeof(double) : sizeof(float);
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &ctx->cache.lock);
  if ((0 == *cache_hit_a || 0 == *cache_hit_b) && EXIT_SUCCESS == result) {
    ctx->cache.last_cutoff = 0;
  }
  if (0 == *cache_hit_a && 0 != (ctx->cache.flags & 1) && EXIT_SUCCESS == result) {
    if (NULL != ctx->cache.a.d_slices) {
      OZAKI_DEV_FREE(ctx->cache.a.d_slices);
      ctx->cache.a.d_slices = NULL;
    }
    if (NULL != ctx->cache.a.d_exp) {
      OZAKI_DEV_FREE(ctx->cache.a.d_exp);
      ctx->cache.a.d_exp = NULL;
    }
    ctx->cache.a.ptr = a;
    ctx->cache.a.dim = M;
    ctx->cache.a.K = K;
    ctx->cache.a.ld = lda;
    ctx->cache.a.trans = ta;
    ctx->cache.a.d_slices = d_as;
    ctx->cache.a.d_exp = d_expa_g;
    ctx->cache.a.slices_size = as_size;
    ctx->cache.a.exp_size = expa_size;
    ctx->cache.a.fingerprint = ozaki_cache_fingerprint(a, elem_size, ta ? K : M, ta ? M : K, lda);
    *cache_hit_a = 1; /* ownership transferred; suppress cleanup free */
  }
  if (0 == *cache_hit_b && 0 != (ctx->cache.flags & 2) && EXIT_SUCCESS == result) {
    if (NULL != ctx->cache.b.d_slices) {
      OZAKI_DEV_FREE(ctx->cache.b.d_slices);
      ctx->cache.b.d_slices = NULL;
    }
    if (NULL != ctx->cache.b.d_exp) {
      OZAKI_DEV_FREE(ctx->cache.b.d_exp);
      ctx->cache.b.d_exp = NULL;
    }
    ctx->cache.b.ptr = b;
    ctx->cache.b.dim = N;
    ctx->cache.b.K = K;
    ctx->cache.b.ld = ldb;
    ctx->cache.b.trans = tb;
    ctx->cache.b.d_slices = d_bs;
    ctx->cache.b.d_exp = d_expb_g;
    ctx->cache.b.slices_size = bs_size;
    ctx->cache.b.exp_size = expb_size;
    ctx->cache.b.fingerprint = ozaki_cache_fingerprint(b, elem_size, tb ? N : K, tb ? K : N, ldb);
    *cache_hit_b = 1; /* ownership transferred; suppress cleanup free */
  }
  if (0 == prev_owned && (0 != *cache_hit_a || 0 != *cache_hit_b)) ++ctx->cache.nusers;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &ctx->cache.lock);
}
