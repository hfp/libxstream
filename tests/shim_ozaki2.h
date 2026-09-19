/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SHIM_OZAKI2_H
#define SHIM_OZAKI2_H

/**
 * Ozaki scheme 2 on the host through libxstream_cpu_begin.h: the shipped kernels,
 * driven the way samples/ozaki/ozaki_gemm.c drives them. A flavor is one translation
 * unit that states the build parameters before including this header, because the
 * precision and the reconstruction tables are compile-time and one unit holds one
 * of each.
 *
 * The generic path only (INTEL=0, NV=0), which is the one a device-less machine can
 * reach: the preprocessing, the scalar residue GEMM, the reconstruction fused into
 * its epilogue or as a pass of its own (OZAKI_UNFUSE), and the host's own tables. The work-group models follow from the kernels rather
 * than from a choice. Preprocessing shares a row's exponent across lanes through
 * local memory, so it runs as 1x1 work-groups (the flavor compiles BM_PRE, BN_PRE
 * and BK_PRE as 1, which only these kernels read); the GEMM and the reduction give
 * every lane its own column and share nothing, so their lanes run one after the
 * other within the work-group shape the host launches.
 *
 * Two kinds of operand, both held to a bound in the significand the flavor keeps.
 * The reconstruction rounds even where nothing else does: alignment scales every
 * element to the full significand, so the CRT recovers an integer of about twice
 * that width, and Horner forms it in floating point a group at a time. The device
 * does the same (one unit of 49 on a GPU for 3-bit operands), hence a result is not
 * bit-exact, and the bound is a few units of it. Small integers lose nothing in the
 * alignment, which leaves that rounding alone and makes their bound the tight one;
 * operands with spread exponents add what alignment truncates. A wrong index, sign,
 * or table entry is off by far more than either.
 *
 * Buffers the host leaves to the kernels to write in full are poisoned rather than
 * zeroed, so a byte a kernel fails to write corrupts the result instead of passing
 * as the zero it happens to be.
 */

#include <libxs/libxs_macros.h>

/**
 * The device part, which every flavor shares: the generic path with the tile
 * tests/kernels.c takes from the host's build strings. The preprocessing extents
 * are 1 because only a 1x1 work-group can run those kernels here; the padding
 * keeps the host's (SHIM_OZ2_BM_PRE, SHIM_OZ2_BK_PRE). A flavor states the rest:
 * precision, the tables for its modulus count, and the reconstruction mode.
 */
#define BK 32
#define KU 2
#define RC 8
#define SG 16
#define INTEL 0
#define NV 0
#define BM_PRE 1
#define BN_PRE 1
#define BK_PRE 1
#define PB 1
#define CONSTANT global
#define LU 0
#define BM 64
#define BN 64
#define RTM 2
#define RTN 2
#define OZAKI_BOUNDS 1

#include <libxstream/opencl/libxstream_cpu_begin.h>
#include "../samples/ozaki/kernels/ozaki2.cl"
#include <libxstream/opencl/libxstream_cpu_end.h>

/**
 * The host's own preprocessing extents, from which it derives the padding. Not the
 * kernel's BM_PRE and BK_PRE, which the flavor compiles as 1: the padding belongs
 * to the layout both producer and consumer agree on, not to a work-group shape.
 */
#define SHIM_OZ2_BM_PRE 16
#define SHIM_OZ2_BK_PRE 32
#define SHIM_OZ2_POISON 0xA5
#define SHIM_OZ2_UPDIV(N, UP) (((N) + (UP) - 1) / (UP))
#define SHIM_OZ2_UP(N, UP) (SHIM_OZ2_UPDIV(N, UP) * (UP))
/* Lanes of the work-group shape the host launches the GEMM and the reduction with. */
#define SHIM_OZ2_NTM (BM / (XMX_M * RTM))
#define SHIM_OZ2_NTN (BN / (XMX_N * RTN))

/**
 * Work-groups in turn and, within each, its lanes in turn: only right for a
 * kernel whose lanes share nothing, which is what the header comment establishes
 * for each kernel launched this way.
 */
#define SHIM_OZ2_LAUNCH(G0, G1, G2, L0, L1, CALL) do { \
  int shim_oz2_g0_, shim_oz2_g1_, shim_oz2_g2_, shim_oz2_l0_, shim_oz2_l1_; \
  LIBXSTREAM_CPU_GRID(G0, G1, G2); \
  for (shim_oz2_g2_ = 0; shim_oz2_g2_ < (G2); ++shim_oz2_g2_) { \
    for (shim_oz2_g1_ = 0; shim_oz2_g1_ < (G1); ++shim_oz2_g1_) { \
      for (shim_oz2_g0_ = 0; shim_oz2_g0_ < (G0); ++shim_oz2_g0_) { \
        for (shim_oz2_l1_ = 0; shim_oz2_l1_ < (L1); ++shim_oz2_l1_) { \
          for (shim_oz2_l0_ = 0; shim_oz2_l0_ < (L0); ++shim_oz2_l0_) { \
            LIBXSTREAM_CPU_WORKITEM(shim_oz2_g0_, shim_oz2_g1_, shim_oz2_g2_, \
              shim_oz2_l0_, shim_oz2_l1_, L0, L1); \
            CALL; \
          } \
        } \
      } \
    } \
  } \
} while (0)


typedef struct {
  int m, n, k;
  char transa, transb;
  real_t alpha, beta;
  /* work-groups along the reduction's third dimension, as the host spreads them */
  int nsplit;
  /* 0: small integers (lossless alignment), 1: spread exponents */
  int kind;
} shim_oz2_case_t;


static unsigned int shim_oz2_next(unsigned int* seed);
static void shim_oz2_fill(real_t z[], size_t n, int kind, unsigned int* seed);
static int shim_oz2_kpad(const char as[], const char bs[], int M, int N, int K, int k_pad, int m_pad, int n_pad);
static int shim_oz2_run(const shim_oz2_case_t* cs, unsigned int seed);
static int shim_ozaki2(const char flavor[]);


static unsigned int shim_oz2_next(unsigned int* seed)
{
  *seed = *seed * 1103515245U + 12345U;
  return *seed >> 16;
}


/**
 * Small integers in [-3, 3], or values in (-1, 1) at exponents 2^-4 to 2^4, both
 * exact in either precision: whatever the kernels lose is theirs.
 */
static void shim_oz2_fill(real_t z[], size_t n, int kind, unsigned int* seed)
{
  size_t i;
  for (i = 0; i < n; ++i) {
    if (0 == kind) {
      z[i] = (real_t)((int)(shim_oz2_next(seed) % 7) - 3);
    }
    else {
      const int mant = (int)(shim_oz2_next(seed) % 2001) - 1000;
      const int e = (int)(shim_oz2_next(seed) % 9) - 4;
      z[i] = (real_t)ldexp((double)mant / 1024.0, e);
    }
  }
}


/**
 * The preprocessing's contract past K: every residue there is zero, which is what
 * lets the host zero only the row and column padding. The product cannot show a
 * breach on one side while the other still holds its zeros there (a residue times
 * zero is zero), so the planes are read directly, through the kernels' own
 * indexing so that the layout is theirs.
 */
static int shim_oz2_kpad(const char as[], const char bs[], int M, int N, int K, int k_pad, int m_pad, int n_pad)
{
  int result = EXIT_SUCCESS, p, r, k;
  for (p = 0; p < NMODULI && EXIT_SUCCESS == result; ++p) {
    for (k = K; k < k_pad && EXIT_SUCCESS == result; ++k) {
      for (r = 0; r < M && EXIT_SUCCESS == result; ++r) {
        if (0 != as[((long)p * m_pad * k_pad + OZAKI_IDX_AS(r, k, k_pad)) * OZAKI_AS_ESZ]) {
          fprintf(stderr, "shim: ozaki2 A residue %d of row %d is not zero at k=%d >= K=%d\n", p, r, k, K);
          result = EXIT_FAILURE;
        }
      }
      for (r = 0; r < N && EXIT_SUCCESS == result; ++r) {
        if (0 != bs[((long)p * k_pad * n_pad + OZAKI_IDX_BS(k, r, n_pad, k_pad)) * OZAKI_BS_ESZ]) {
          fprintf(stderr, "shim: ozaki2 B residue %d of column %d is not zero at k=%d >= K=%d\n", p, r, k, K);
          result = EXIT_FAILURE;
        }
      }
    }
  }
  return result;
}


/**
 * One call of ozaki_gemm's scheme-2 path for a single K-group and panel: the
 * padding, the zeroing, and the launch geometry are the host's.
 */
static int shim_oz2_run(const shim_oz2_case_t* cs, unsigned int seed)
{
  const int M = cs->m, N = cs->n, K = cs->k;
  const int ta = ('N' != cs->transa), tb = ('N' != cs->transb);
  const int lda = (ta ? K : M) + 3, ldb = (tb ? N : K) + 3, ldc = M + 2;
  const int nblk_gm = SHIM_OZ2_UPDIV(M, BM), nblk_gn = SHIM_OZ2_UPDIV(N, BN);
  const int ku_bk = KU * SHIM_OZ2_BK_PRE;
  const int k_pad = LIBXS_MAX(SHIM_OZ2_UP(K, ku_bk), 64);
  const int m_pad = LIBXS_MAX(SHIM_OZ2_UP(M, SHIM_OZ2_BM_PRE), nblk_gm * BM);
  const int n_pad = LIBXS_MAX(LIBXS_MAX(SHIM_OZ2_UP(N, SHIM_OZ2_BM_PRE), 64), nblk_gn * BN);
  const int first = (0 == cs->beta);
  const size_t na = (size_t)lda * (ta ? M : K), nb = (size_t)ldb * (tb ? K : N), nc = (size_t)ldc * N;
  const size_t nas = (size_t)NMODULI * m_pad * k_pad, nbs = (size_t)NMODULI * k_pad * n_pad;
#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE)
  const size_t nres = (size_t)NMODULI * nblk_gm * BM * nblk_gn * BN;
#else /* reconstructed in the GEMM's epilogue: no residue planes in between */
  const size_t nres = 1;
#endif
  real_t *const a = (real_t*)malloc(sizeof(real_t) * na), *const b = (real_t*)malloc(sizeof(real_t) * nb);
  real_t *const c = (real_t*)malloc(sizeof(real_t) * nc), *const c0 = (real_t*)malloc(sizeof(real_t) * nc);
  char *const as = (char*)malloc(nas), *const bs = (char*)malloc(nbs);
  unsigned char* const res = (unsigned char*)malloc(nres);
  int *const expa = (int*)calloc((size_t)nblk_gm * BM, sizeof(int));
  int *const expb = (int*)calloc((size_t)nblk_gn * BN, sizeof(int));
  int result = EXIT_SUCCESS, i, j, k;
  if (NULL == a || NULL == b || NULL == c || NULL == c0 || NULL == as || NULL == bs || NULL == res
    || NULL == expa || NULL == expb)
  {
    fprintf(stderr, "shim: ozaki2 out of memory\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    shim_oz2_fill(a, na, cs->kind, &seed);
    shim_oz2_fill(b, nb, cs->kind, &seed);
    shim_oz2_fill(c0, nc, cs->kind, &seed);
    memcpy(c, c0, sizeof(real_t) * nc);
    /* the host zeroes a plane only where the preprocessing leaves padding unwritten */
    memset(as, m_pad > M ? 0 : SHIM_OZ2_POISON, nas);
    memset(bs, n_pad > N ? 0 : SHIM_OZ2_POISON, nbs);
    memset(res, SHIM_OZ2_POISON, nres);

    if (0 == first && 1 != cs->beta) {
      SHIM_OZ2_LAUNCH(M, N, 1, 1, 1, scale_beta(c, 0, M, N, ldc, cs->beta));
    }
    SHIM_OZ2_LAUNCH(1, M, 1, 1, 1,
      preprocess_a_crt_dense(a, 0, M, K, lda, ta, as, 0, expa, 0, k_pad, m_pad));
    SHIM_OZ2_LAUNCH(N, 1, 1, 1, 1,
      preprocess_b_crt_dense(b, 0, N, K, ldb, tb, bs, 0, expb, 0, k_pad, n_pad));
    result = shim_oz2_kpad(as, bs, M, N, K, k_pad, m_pad, n_pad);
  }
  if (EXIT_SUCCESS == result) {
#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE)
    SHIM_OZ2_LAUNCH(nblk_gm, nblk_gn, 1, SG, SHIM_OZ2_NTM * SHIM_OZ2_NTN,
      gemm_crt_fused(as, 0, bs, 0, expa, 0, expb, 0, c, 0, M, N, k_pad, n_pad, ldc, m_pad, cs->alpha, first, res, 0));
    SHIM_OZ2_LAUNCH(nblk_gm, nblk_gn, cs->nsplit, SG, SHIM_OZ2_NTM * SHIM_OZ2_NTN,
      gemm_crt_reduce(res, 0, expa, 0, expb, 0, c, 0, M, N, ldc, cs->alpha, first));
#else
    SHIM_OZ2_LAUNCH(nblk_gm, nblk_gn, 1, SG, SHIM_OZ2_NTM * SHIM_OZ2_NTN,
      gemm_crt_fused(as, 0, bs, 0, expa, 0, expb, 0, c, 0, M, N, k_pad, n_pad, ldc, m_pad, cs->alpha, first));
#endif

    for (j = 0; j < N && EXIT_SUCCESS == result; ++j) {
      for (i = 0; i < ldc && EXIT_SUCCESS == result; ++i) {
        const real_t got = c[i + (size_t)j * ldc];
        long double ref = 0, bound = 0;
        if (i < M) { /* a padding row must come back unchanged */
          long double s = 0, amax = 0, bmax = 0;
          for (k = 0; k < K; ++k) {
            const real_t x = ta ? a[k + (size_t)i * lda] : a[i + (size_t)k * lda];
            const real_t y = tb ? b[j + (size_t)k * ldb] : b[k + (size_t)j * ldb];
            s += (long double)x * y;
            if (amax < (x < 0 ? -x : x)) amax = (x < 0 ? -x : x);
            if (bmax < (y < 0 ? -y : y)) bmax = (y < 0 ? -y : y);
          }
          { const long double cij = (0 == first ? (long double)cs->beta * c0[i + (size_t)j * ldc] : 0);
            const long double aab = (cs->alpha < 0 ? -cs->alpha : cs->alpha);
            ref = (long double)cs->alpha * s + cij;
            /**
             * The reconstruction and the update round a few units of the terms
             * they combine; alignment keeps MANT_BITS below each row's and
             * column's largest exponent, so with spread exponents a product
             * loses up to a unit there too, K of them in a sum.
             */
            bound = (aab * (s < 0 ? -s : s) + (cij < 0 ? -cij : cij)) * (long double)ldexp(1.0, 2 - MANT_BITS);
            if (0 != cs->kind) bound += 4 * K * aab * amax * bmax * (long double)ldexp(1.0, -MANT_BITS);
          }
        }
        else ref = c0[i + (size_t)j * ldc];
        if ((M <= i) ? ((long double)got != ref)
          : (bound < ((long double)got - ref < 0 ? ref - (long double)got : (long double)got - ref)))
        {
          fprintf(stderr, "shim: ozaki2 %dx%dx%d %c%c kind=%d alpha=%g beta=%g nsplit=%d: c(%d,%d) = %.17g, expected %.17g\n",
            M, N, K, cs->transa, cs->transb, cs->kind, (double)cs->alpha, (double)cs->beta, cs->nsplit, i, j,
            (double)got, (double)ref);
          result = EXIT_FAILURE;
        }
      }
    }
  }
  free(a);
  free(b);
  free(c);
  free(c0);
  free(as);
  free(bs);
  free(res);
  free(expa);
  free(expb);
  return result;
}


/**
 * Partial tiles in every dimension; exact tiles, where the planes are poisoned,
 * once with K a padding multiple and once not, so that the kernel alone has to
 * write the zero columns; and a K past one padding step. All four transposes,
 * each alpha/beta form the host distinguishes (overwrite, accumulate, scale
 * first), and the reduction spread over one, all, and an uneven count of its
 * sub-tiles.
 */
static int shim_ozaki2(const char flavor[])
{
  static const int shapes[][3] = { { 70, 90, 100 }, { BM, BN, 64 }, { BM, BN, 100 }, { 17, 33, 300 } };
  static const char trans[][2] = { { 'N', 'N' }, { 'N', 'T' }, { 'T', 'N' }, { 'T', 'T' } };
  static const real_t ab[][2] = { { 1, 0 }, { 0.5, 1 }, { 2, -0.5 } };
  const int nsub = RTM * RTN, nsplits[] = { 1, RTM * RTN, (1 < RTM * RTN) ? (RTM * RTN - 1) : 1 };
  const int nshapes = (int)(sizeof(shapes) / sizeof(*shapes)), ntrans = (int)(sizeof(trans) / sizeof(*trans));
  int result = EXIT_SUCCESS, s, t, kind, n = 0;
  for (s = 0; s < nshapes && EXIT_SUCCESS == result; ++s) {
    for (t = 0; t < ntrans && EXIT_SUCCESS == result; ++t) {
      for (kind = 0; kind < 2 && EXIT_SUCCESS == result; ++kind) {
        shim_oz2_case_t cs;
        cs.m = shapes[s][0];
        cs.n = shapes[s][1];
        cs.k = shapes[s][2];
        cs.transa = trans[t][0];
        cs.transb = trans[t][1];
        cs.alpha = ab[(s + t) % 3][0];
        cs.beta = ab[(s + t) % 3][1];
        cs.nsplit = nsplits[(s + t + kind) % 3];
        cs.kind = kind;
        result = shim_oz2_run(&cs, (unsigned int)(1 + n));
        if (EXIT_SUCCESS == result) ++n;
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    printf("shim: ozaki2 %s, %d moduli, %d sub-tiles, %s: %d cases within bound\n", flavor, NMODULI, nsub,
#if defined(OZAKI_UNFUSE) && (OZAKI_UNFUSE)
      "unfused",
#else
      "fused",
#endif
      n);
  }
  return result;
}

#endif /*SHIM_OZAKI2_H*/
