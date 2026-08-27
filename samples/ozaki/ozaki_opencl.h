/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_OPENCL_H
#define OZAKI_OPENCL_H

#include <libxstream/libxstream_opencl.h>
#include <libxs/libxs_reg.h>

#if !defined(K_GRP_GPU)
# define K_GRP_GPU 32768
#endif

/**
 * Groups the hierarchical CRT's level-2 stage can reconstruct. The kernel writes
 * its group-product, Barrett and inverse tables out for this many groups and
 * restates the bound with #error, so the leaf group size must be chosen to keep
 * ceil(nprimes / HIER_GS) within it - see ozaki_hier_gs.
 */
#if !defined(OZAKI_HIER_NGROUPS_MAX)
# define OZAKI_HIER_NGROUPS_MAX 5
#endif

/**
 * Slots in the N-panel pipeline. Two suffice to cover one GEMM's worth of
 * upload+preprocess latency; more only helps if panel preprocessing time
 * varies enough to stall, at a proportional cost in B-slice memory.
 */
#if !defined(OZAKI_NSLOTS)
# define OZAKI_NSLOTS 2
#endif

/**
 * Device memory allocation macros (shared by ozaki_opencl.c and ozaki_gemm.c).
 * Uses libxstream's internal device pool when available.
 */
#define OZAKI_DEV_ALLOC(PTR, SIZE) \
    ((NULL != libxstream_opencl_config.pool_dev) \
      ? ((*(PTR) = libxs_malloc(libxstream_opencl_config.pool_dev, SIZE, LIBXS_MALLOC_NATIVE)) != NULL \
          ? EXIT_SUCCESS : EXIT_FAILURE) \
      : libxstream_mem_allocate((void**)(PTR), SIZE))
#define OZAKI_DEV_FREE(PTR) \
    do { \
      if (NULL != (PTR)) { \
        if (NULL != libxstream_opencl_config.pool_dev) libxs_free(PTR); \
        else libxstream_mem_deallocate(PTR); \
      } \
    } while (0)


/* Ozaki flags */
typedef enum ozaki_flags_t { OZAKI_TRIANGULAR = 1, OZAKI_SYMMETRIZE = 2 } ozaki_flags_t;

LIBXS_API_INLINE int ozaki_count_pairs(int nslices, int co, int flags)
{
  int sa, n = 0;
  for (sa = 0; sa < nslices && sa <= co; ++sa) {
    const int sb_start = (0 != (flags & OZAKI_TRIANGULAR)) ? sa : 0;
    const int sb_end = nslices < (co + 1 - sa) ? nslices : (co + 1 - sa);
    int sb;
    for (sb = sb_start; sb < sb_end; ++sb) {
      ++n;
      if (0 != (flags & OZAKI_SYMMETRIZE) && sa != sb) ++n;
    }
  }
  return n;
}

/**
 * Host-side preprocessing callback for A or B (GEMM single-shot model).
 * When non-NULL in the context, ozaki_gemm calls these instead of
 * the GPU preprocess kernels and skips the full-matrix H2D.
 *
 * matrix   : host pointer to source matrix (A or B)
 * ld       : leading dimension
 * Per-side preprocessing cache: check fields + cached device buffers.
 * dim is the outer dimension (M for A, N for B).
 */
typedef struct ozaki_cache_side_t {
  const void* ptr;
  int dim, K, ld, trans;
  void* d_slices;
  void* d_exp;
  size_t slices_size, exp_size;
  unsigned int fingerprint; /* content fingerprint to detect in-place modifications */
} ozaki_cache_side_t;

/**
 * Compute a lightweight fingerprint by sampling matrix elements at
 * deterministic positions. Catches in-place modifications that
 * pointer comparison alone cannot detect. The extents are passed
 * per layout rather than as a transpose flag, since the flag implies
 * opposite extents for A and B.
 */
unsigned int ozaki_cache_fingerprint(const void* ptr, size_t elem_size, int ncontig, int nld, int ld);

typedef struct ozaki_cache_t {
  libxs_lock_t lock;
  volatile LIBXS_ATOMIC_LOCKTYPE nusers;
  int flags; /* bitmask: 1=A, 2=B */
  int last_cutoff; /* occupancy-derived Scheme-1 cutoff cached across calls (reused on cache hit to skip D2H readback) */
  ozaki_cache_side_t a, b;
} ozaki_cache_t;

/**
 * Device scratch arena for the transient buffers of one call: the residue
 * planes, the device copy of C, and the operand uploads. They are carved from
 * one allocation that outlives the call instead of being created and destroyed
 * per call, which on a device without a memory pool is what dominates the wall
 * clock - 402 MB of create/destroy per call at n=4096, measured as 29 of 34 ms
 * on a GH200, where libxstream's pool does not exist because it is gated on USM.
 *
 * Per context, because that is what a BLAS interceptor can use: it holds one
 * context for the process and has no way to be handed memory by the
 * application. A caller that does own device memory can install it instead
 * (ozaki_scratch_set), in which case the arena is never grown or freed here.
 *
 * The arena is claimed for the duration of a call, so a second call on the same
 * context does not wait for it - it falls back to per-call allocation. That is
 * the same property a cuBLAS handle's workspace has, and it keeps the footprint
 * one arena rather than one per thread; a caller that wants concurrency without
 * the fallback should use a context per thread.
 */
typedef struct ozaki_scratch_t {
  void* ptr; /* arena base, NULL until first use */
  size_t size; /* capacity of ptr */
  size_t used; /* bump offset, meaningful only while claimed */
  size_t limit; /* upper bound for growth (OZAKI_ARENA), 0 disables the arena */
  int owned; /* 1: allocated here, grown on demand, freed at destroy; 0: caller's */
  /**
   * Claim nesting for one thread. The complex path carves its own buffers and
   * then calls ozaki_gemm for the embedded product, which claims again; without
   * this the inner claim would fail against the outer thread's own lock and fall
   * back to per-buffer allocation for the residue planes, which are the larger
   * half. Only the outermost claim resets the bump offset or grows the arena, so
   * a nested claim cannot dangle the pieces already handed out.
   */
  int depth;
  volatile LIBXS_ATOMIC_LOCKTYPE busy;
} ozaki_scratch_t;

/**
 * Ozaki-1 kernel specialization key: compile-time cutoff.
 * bounds: 0 = tile-aligned, 1 = bounds-checked variant.
 * tm/tn: output tile baked into the kernel (size-aware selection).
 * rtm/rtn: register tiling, likewise size-aware - it decides both the
 * work-group size and the tile granularity, so the two travel together.
 */
typedef struct ozaki_kernel_key_t {
  int cutoff;
  int bounds;
  int tm, tn;
  int rtm, rtn;
} ozaki_kernel_key_t;

/* Ozaki-1 kernel set: one entry per registry specialization. */
typedef struct ozaki_kernel_set_t {
  cl_kernel kern_fused;
} ozaki_kernel_set_t;

/**
 * Ozaki-2 kernel specialization key: output tile plus bounds variant.
 * Scheme 2 has no cutoff, so its registry is keyed by tile only.
 */
typedef struct ozaki_crt_kernel_key_t {
  int bounds;
  int tm, tn;
  int rtm, rtn;
} ozaki_crt_kernel_key_t;

/**
 * Ozaki-2 kernel set: one entry per registry specialization. kern_reduce is the
 * separate reconstruction pass (OZAKI_UNFUSE) and NULL when the epilogue is fused;
 * it shares the specialization because it must agree with the GEMM on the tile,
 * the register tiling and hence the blocked residue layout.
 */
typedef struct ozaki_crt_kernel_set_t {
  cl_kernel kern_fused;
  cl_kernel kern_reduce;
} ozaki_crt_kernel_set_t;

/**
 * State for an Ozaki OpenCL session.
 * All tuning parameters are set by ozaki_init (0 = auto).
 */
typedef struct ozaki_context_t {
  /* Ozaki-1: preprocessing + scale kernels (shared across specializations) */
  cl_kernel kern_preprocess_a;
  cl_kernel kern_preprocess_b;
  cl_kernel kern_scale_beta;
  /* Ozaki-1: registry of cutoff-specialized fused kernels */
  libxs_registry_t* kernel_registry;
  libxs_lock_t kernel_lock;
  char base_flags[1024]; /* base compile flags (without OZAKI_CUTOFF) */
  char base_options[128]; /* build options (e.g. -cl-intel-256-GRF-per-thread) */
  /* CRT GEMM-mode kernels (Scheme-2 tiled path) */
  cl_kernel kern_crt_preprocess_a;
  cl_kernel kern_crt_preprocess_b;
  cl_kernel kern_crt_scale_beta;
  /* Ozaki-2: registry of tile-specialized fused kernels */
  libxs_registry_t* crt_registry;
  char crt_flags[2048]; /* base compile flags (without BM/BN) */
  char crt_options[128]; /* build options for CRT kernels */
  int use_double; /* 1: fp64, 0: fp32 */
  int sg; /* sub-group size used for compilation */
  int ndecomp; /* number of decomposition components (slices or primes, per active kind) */
  int nslices; /* Ozaki-1: number of mantissa slices (compiled into Scheme-1 kernels) */
  int nprimes; /* Ozaki-2: number of CRT primes (compiled into Scheme-2 kernels) */
  int kind; /* resolved: 1 = ozaki1 int8, 2 = ozaki2 int8 (CRT), 3 = adaptive */
  int ozflags; /* bitmask: OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE */
  int oztrim; /* Precision levels to trim (~2 bits each). */
  int verbosity; /* 0: quiet, 1: info, 2+: debug */
  /* block sizes for preprocessing WGs */
  int bm_pre, bn_pre, bk_pre;
  /**
   * Output tile size. tm/tn hold the largest legal tile (the upper bound for
   * selection); tm_req/tn_req are explicit user overrides (0 = auto), in which
   * case selection is bypassed and tm/tn are used verbatim.
   */
  int tm, tn;
  int tm_req, tn_req;
  /* Device compute units: saturation target for size-aware tile selection. */
  int nunits;
  int tile_sat; /* divisor for nwg_min: nunits / tile_sat work-groups */
  /**
   * Preferred work-group size ceiling for tile selection (0 = use max_wgs).
   * Occupancy-driven and below the hardware bound; see ozaki_tile_select.
   */
  int wgs_max;
  size_t max_wgs; /* work-group size bound (halved under 256-GRF) */
  /**
   *  register tiling: sub-tiles per sub-group (compiled into kernel).
   * crt_rtm may differ from rtm when adaptive (kind=3) uses HIER+GRF128
   * for CRT while Scheme 1 uses GRF256.
   * crt_rtn likewise: the two schemes prefer opposite aspect ratios under MMA
   * (Scheme 1 peaks at RTN=2 and loses 26% at RTN=4; Scheme 2 keeps gaining up
   * to RTN=8), so the column tiling is per-scheme rather than shared.
   */
  int rtm, rtn, crt_rtm, crt_rtn;
  /**
   * The next step of register tiling, or the same pair when there is none. More
   * accumulators per work-item is the only way to spend a fixed work-group
   * budget on more output per thread, and it pays by a third to a half once the
   * problem is large enough to keep the device busy with the coarser tile
   * granularity it implies - but costs up to 2x below that. So it is a per-call
   * choice like the tile: see ozaki_rtile_select, which picks between this pair
   * and the base one and hands the result to the registry key.
   */
  int rtm_big, rtn_big, crt_rtm_big, crt_rtn_big;
  int ku; /* K-loop unroll factor (compiled into kernel) */
  int sb; /* Scheme-1 slice-block width for the pair loop (1 = unblocked) */
  int rc; /* DPAS repeat count: 8 (default) or 4 (split) */
  int nv_mma; /* NV MMA path enabled (m16n8k32, SG=32) */
  /**
   * NVIDIA GPU (devinfo->nv level, 0 = not NVIDIA). Kept because scheme selection
   * needs it: every Scheme-2 improvement of this port is NVIDIA-side, so the
   * crossover model no longer describes that device.
   */
  int nv;
  int u8; /* Sch.2: unsigned residues (moduli<=256) rather than signed i8 */
  /**
   * Warp-group MMA (Hopper): the fused CRT kernel carries markers that the host
   * splices into real wgmma instructions, since OpenCL C cannot express them.
   * A marker-only kernel accumulates nothing, so a failed splice must never be
   * used - ozaki_get_crt_kernel refuses the kernel instead.
   */
  int wgmma;
  /**
   * Warp-group MMA with A in registers (RS form) rather than both operands staged
   * in shared memory. Halves the tile's shared footprint and drops the A staging
   * and its share of the barrier; B keeps its descriptor because only A may come
   * from registers.
   */
  int wgmma_rs;
  /**
   * Unfused reconstruction: the GEMM stores residue bytes and a second kernel
   * reconstructs C. Trades one round trip through global memory for the
   * per-work-item group-value frame the fused epilogue has to keep live across
   * the prime loop, which is what makes that epilogue memory-bound.
   */
  int unfuse;
  int pb; /* CRT prime batching factor (compiled into kernel) */
  int hier; /* Hierarchical CRT: two-level Garner (compiled into kernel) */
  double xover; /* Scheme-1/2 crossover weight: reconstruction cost per Garner op vs int8 MAC */
  int maxk; /* max K per preprocessing pass (0 = no grouping) */
  /**
   * N-panel width for the pipelined Scheme-2 path (0 = auto, 1 = disabled).
   * Panels split the columns of B and C, so each panel owns a disjoint block
   * of C: unlike a K-split it adds no repeated read-modify-write of C, and
   * panel GEMMs are mutually independent. See ozaki_npanel.
   */
  int npanel;
  int biggrf; /* Ozaki-local 256-GRF decision */
  /* Main stream (set per ozaki_gemm call for pool realloc sync) */
  libxstream_stream_t* stream;
  /* Persistent helper streams for overlapped preprocessing */
  libxstream_stream_t *stream_a, *stream_b;
  /* Persistent synchronization events */
  libxstream_event_t *evt_prep_a, *evt_prep_b;
  /**
   * Per-slot events for the N-panel pipeline: evt_slot[i] is recorded after
   * the GEMM consuming slot i, so a later panel reusing that slot can wait on
   * it without serializing against the immediately preceding panel.
   */
  libxstream_event_t* evt_slot[OZAKI_NSLOTS];
  libxstream_event_t* evt_panel; /* GEMM completion, gates the panel D2H */
  /* Preprocessing cache (OZAKI_CACHE env, bitmask: 1=A, 2=B, 3=both). */
  ozaki_cache_t cache;
  ozaki_scratch_t scratch;
  /* Complex GEMM block-embedding kernels (construct A_hat, B_hat, finalize) */
  cl_kernel kern_zgemm_block_construct_a;
  cl_kernel kern_zgemm_block_construct_b_n;
  cl_kernel kern_zgemm_block_construct_b_t;
  cl_kernel kern_zgemm_block_finalize;
} ozaki_context_t;


/**
 * Function prototypes (public API).
 * Pass 0 for tm/tn/ndecomp to use auto defaults.
 * Pass -1 for ozflags to use the default (TRIANGULAR | SYMMETRIZE);
 * 0 disables both flags.  Auto defaults choose XMX-friendly sizes
 * when hardware support is detected.
 * kind: 1 = ozaki1 int8, 2 = ozaki2 int8 (CRT), anything else (pass 0) = no
 * request, i.e. adaptive per call. The default for an unspecified kind is
 * resolved here and not by the caller, so that two drivers cannot disagree
 * about what "unspecified" means; forcing a scheme stays the caller's right.
 * verbosity: 0 = quiet, 1 = info, 2+ = debug.
 * ozgroups (Scheme 2 only): K-grouping factor, 0/1 = disabled.
 */
int ozaki_init(ozaki_context_t* ctx, int tm, int tn, int use_double, int kind, int verbosity, int ndecomp, int ozflags, int oztrim,
  int ozgroups, int maxk);
void ozaki_destroy(ozaki_context_t* ctx);

/**
 * Device scratch (Scheme 2) - see ozaki_scratch_t. Both are optional: without
 * them the context grows its own arena on first use, which is what a BLAS
 * interceptor relies on.
 *
 * ozaki_scratch_size reports the bytes one call of the given shape can need. It
 * is an upper bound (no cached operand, unpanelled), so it never under-sizes;
 * the internal arena instead grows to the exact requirement of the call.
 *
 * ozaki_scratch_set installs caller-owned memory, or reverts to the internal
 * arena and releases it when dev_mem is NULL. Allocate it with
 * libxstream_mem_dev_allocate_hint(..., libxstream_opencl_mem_hint_atomics):
 * the device copy of C is carved from it and that is the hint it would
 * otherwise be allocated with. An arena too small for a given call is not an
 * error - that call falls back to allocating per buffer.
 */
size_t ozaki_scratch_size(const ozaki_context_t* ctx, char transa, char transb, int M, int N, int K, int lda, int ldb, int ldc);
int ozaki_scratch_set(ozaki_context_t* ctx, void* dev_mem, size_t nbytes);

/**
 * Arena carving, shared with the complex path rather than private to the real
 * one. Alignment is exposed because a caller sizing a claim has to round every
 * piece the same way ozaki_scratch_alloc will.
 */
#define OZAKI_SCRATCH_ALIGN 4096
int ozaki_scratch_claim(ozaki_context_t* ctx, size_t nbytes);
void ozaki_scratch_release(ozaki_context_t* ctx, int claimed);
int ozaki_scratch_alloc(ozaki_context_t* ctx, int claimed, void** ptr, size_t nbytes, int atomics);
void ozaki_scratch_free(const ozaki_context_t* ctx, void* ptr, int atomics);
/**
 * The arena as it currently stands (NULL when there is none yet), so a caller
 * can see what it is holding or reuse it. Reuse is limited by what the pointer
 * is: without USM it is a token resolved to a cl_mem plus an offset rather than
 * an address, and it belongs to the OpenCL context either way - handing it to
 * another runtime is the OZAKI_CUBLAS_XPTR experiment in ozaki_bench.c, which
 * faults by design. It is a device pointer for libxstream and for nothing else.
 */
void* ozaki_scratch_get(const ozaki_context_t* ctx, size_t* nbytes);
/* Selected output tile (BM x BN) for one GEMM call. */
typedef struct ozaki_tile_t {
  int m, n;
} ozaki_tile_t;

/**
 * Hardware sub-tile dimensions, mirroring XMX_M/XMX_N in ozaki_common.cl.
 * The MMA path transposes them (16x8 instead of 8x16), so any host-side
 * NTM/NTN arithmetic must derive from these rather than hard-code 8 and 16:
 * the kernel's reqd_work_group_size is SG x (NTM*NTN), and a mismatch means
 * sub-groups that never launch, i.e. silently unwritten parts of C.
 */
#define OZAKI_XMX_M(CTX) ((0 != (CTX)->nv_mma) ? 16 : 8)
#define OZAKI_XMX_N(CTX) ((0 != (CTX)->nv_mma) ? 8 : 16)

/**
 * Size-aware output tile selection: pick the largest legal tile that still
 * saturates the device, breaking ties on least padding waste. rtm/rtn are
 * passed explicitly because Scheme 1 and Scheme 2 may use different register
 * tiling (rtm vs crt_rtm, rtn vs crt_rtn), which changes both the tile
 * granularity and the resulting work-group size.
 */
ozaki_tile_t ozaki_tile_select(const ozaki_context_t* ctx, int M, int N, int rtm, int rtn);

/**
 * Register tiling for one call (crt: 0 = Scheme 1, 1 = Scheme 2), as an (m, n)
 * pair of sub-tiles per sub-group rather than an output tile. Takes the larger
 * of the context's two pairs when the problem still yields one tile per compute
 * unit at that granularity, since promoting halves the tile count; measured on
 * PVC as +14 to +36% from n=1024 up and up to 2x slower below n=768. The result
 * must be fed to ozaki_tile_select and to the kernel getter, which key their
 * specializations on it.
 */
ozaki_tile_t ozaki_rtile_select(const ozaki_context_t* ctx, int M, int N, int crt);

/**
 * Whether warp-group MMA is reachable on this device, answered by splicing and
 * building a throwaway kernel with the same marker, descriptor geometry (wbk) and
 * shared-memory footprint (lbytes) as the real one. width is the instruction's N
 * (64 or 128) and rs selects the operand form the real kernel will use, since the
 * two do not assemble alike. Build-only, hence no stream: see the definition for
 * why nothing cheaper is conclusive. EXIT_SUCCESS means the path may be enabled.
 */
int ozaki_wgmma_probe(const ozaki_context_t* ctx, int width, int wbk, size_t lbytes, int rs);

/**
 * N-panel width for the pipelined path, or N itself when panelling does not
 * apply (small N, too few tiles to keep the device busy, or npanel == 1).
 * Chosen jointly with the tile: the panel must stay a multiple of tn so tiles
 * never straddle a panel boundary, and the per-panel tile count must still
 * saturate the device, otherwise pipelining trades throughput for latency.
 */
int ozaki_npanel(const ozaki_context_t* ctx, int M, int N, int tm, int tn);
/**
 * ozaki_gemm enqueues the entire GEMM pipeline on stream and returns without
 * synchronizing - the caller must sync the stream before consuming the result.
 * Helper streams (ctx->stream_a/b) and events are kept persistent in the
 * context to avoid per-call creation overhead.  On the rare pool grow path
 * (larger problem size), the wrapped deallocator syncs all streams before
 * reallocating.
 */
int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream, char transa, char transb, int M, int N, int K, double alpha,
  const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc, int dev);

/**
 * Complex GEMM via block embedding - GPU-native version.
 * All complex matrices are in standard BLAS interleaved format.
 * alpha and beta each point to 2 consecutive real values [real, imag].
 * All intermediate buffers remain on device - no round-trips through host.
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
int ozaki_gemm_complex(ozaki_context_t* ctx, libxstream_stream_t* stream, char transa, char transb, int M, int N, int K,
  const double* alpha, const void* a, int lda, const void* b, int ldb, const double* beta, void* c, int ldc);

/**
 * Invalidate preprocessing cache entries for the given matrix pointers.
 * This function must be called when matrices are modified outside of
 * ozaki_gemm (e.g., by CPU-side BLAS operations) to prevent stale
 * cached data from being reused. Pass NULL for pointers that should
 * not be invalidated. Thread-safe.
 */
void ozaki_invalidate_cache(ozaki_context_t* ctx, const void* a, const void* b);

#endif /* OZAKI_OPENCL_H */
