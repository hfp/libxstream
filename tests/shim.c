/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/**
 * Runs an OpenCL kernel on the host through libxstream_cpu_begin.h, which the
 * kernel test cannot: that one proves a kernel compiles, this one what it
 * computes. The kernel is the shipped source, not a copy, so a host result that
 * disagrees with the reference is a finding about the kernel or about the shim.
 *
 * gemm3m.cl embeds a complex GEMM into one real GEMM of twice the size, and the
 * construction is all sign patterns and index swaps, which is where a defect
 * hides: every transpose and conjugate case is run. The operands are small
 * integers, so every product and every sum is exact in either precision and the
 * comparison needs no tolerance; a mismatch is a wrong index, not rounding.
 */
#if !defined(USE_DOUBLE)
# define USE_DOUBLE 1
#endif

/* the standard headers too, after the feature-test macros they must follow */
#include <libxs/libxs_macros.h>

#include <libxstream/opencl/libxstream_cpu_begin.h>
#include "../samples/ozaki/kernels/gemm3m.cl"
#include <libxstream/opencl/libxstream_cpu_end.h>

/**
 * Launch over a grid that is rounded up past the matrix, as a device rounds it
 * to the work-group size, so that the bound checks in the kernel are exercised.
 * One work-item per work-group, which is what a kernel without a barrier needs.
 */
#define SHIM_ROUNDUP 4
#define SHIM_LAUNCH(NI, NJ, CALL) do { \
  const int shim_ni_ = ((NI) + SHIM_ROUNDUP - 1) / SHIM_ROUNDUP * SHIM_ROUNDUP; \
  const int shim_nj_ = ((NJ) + SHIM_ROUNDUP - 1) / SHIM_ROUNDUP * SHIM_ROUNDUP; \
  int shim_i_, shim_j_; \
  LIBXSTREAM_CPU_GRID(shim_ni_, shim_nj_, 1); \
  for (shim_j_ = 0; shim_j_ < shim_nj_; ++shim_j_) { \
    for (shim_i_ = 0; shim_i_ < shim_ni_; ++shim_i_) { \
      LIBXSTREAM_CPU_WORKITEM(shim_i_, shim_j_, 0, 0, 0, 1, 1); \
      CALL; \
    } \
  } \
} while (0)

/* Shapes differ pairwise, so that a swapped extent reads the wrong element. */
#define SHIM_M 5
#define SHIM_N 3
#define SHIM_K 4
/* Leading dimensions exceed the rows, so that a missed ld is visible too. */
#define SHIM_PAD 2
#define SHIM_MAX (SHIM_M + SHIM_N + SHIM_K + SHIM_PAD)


static int shim_intrinsics(void);
static void shim_fill(real_t z[], int n, unsigned int* seed);
static int shim_case(char transa, char transb, const real_t alpha[2], const real_t beta[2]);


int main(void)
{
  static const char trans[] = { 'N', 'T', 'C' };
  static const real_t alpha[2] = { 2, -1 };
  static const real_t betas[2][2] = { { 1, 2 }, { 0, 0 } };
  int result = shim_intrinsics(), i, j, k, n = 0;
  for (i = 0; i < 3 && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < 3 && EXIT_SUCCESS == result; ++j) {
      for (k = 0; k < 2 && EXIT_SUCCESS == result; ++k) {
        result = shim_case(trans[i], trans[j], alpha, betas[k]);
        if (EXIT_SUCCESS == result) ++n;
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    printf("shim: gemm3m %d cases bit-exact (%s)\n", n, 8 == sizeof(real_t) ? "fp64" : "fp32");
  }
  return result;
}


/**
 * The built-ins the shim assembles itself rather than forwards, at the values
 * where an assembly goes wrong: mul_hi at the carry out of the low half, and clz
 * at either end of both widths.
 */
static int shim_intrinsics(void)
{
  const libxstream_cpu_ulong_t ones = ~(libxstream_cpu_ulong_t)0;
  const libxstream_cpu_ulong_t top = (libxstream_cpu_ulong_t)1 << 63;
  int result = EXIT_SUCCESS;
  if (0 != libxstream_cpu_mul_hi(ones >> 32, ones >> 32)
    || 1 != libxstream_cpu_mul_hi(top, 2)
    || 1 != libxstream_cpu_mul_hi(ones, 2)
    || (ones - 1) != libxstream_cpu_mul_hi(ones, ones)
    || 0x3FFFFFFF != libxstream_cpu_mul_hi(ones >> 1, ones >> 33))
  {
    fprintf(stderr, "shim: mul_hi is wrong\n");
    result = EXIT_FAILURE;
  }
  if (31 != libxstream_cpu_clz(1, 32) || 32 != libxstream_cpu_clz(0, 32)
    || 63 != libxstream_cpu_clz(1, 64) || 0 != libxstream_cpu_clz(top, 64))
  {
    fprintf(stderr, "shim: clz is wrong\n");
    result = EXIT_FAILURE;
  }
  return result;
}


/* Integers in [-3, 3]: exact in either precision, and signs of both kinds. */
static void shim_fill(real_t z[], int n, unsigned int* seed)
{
  int i;
  for (i = 0; i < n; ++i) {
    *seed = *seed * 1103515245U + 12345U;
    z[i] = (real_t)((int)((*seed >> 16) % 7) - 3);
  }
}


/**
 * One transpose/conjugate combination, as ozaki_gemm_complex drives it: embed,
 * multiply the real matrices, extract. The multiplication in the middle is a
 * plain loop, since what is under test is the embedding around it.
 */
static int shim_case(char transa, char transb, const real_t alpha[2], const real_t beta[2])
{
  const int ca = ('C' == transa), cb = ('C' == transb);
  const int ta = ('N' != transa), tb = ('N' != transb);
  const int ta_sign = (ta && !ca);
  const int a_rows = ta ? SHIM_K : SHIM_M, a_cols = ta ? SHIM_M : SHIM_K;
  const int b_rows = tb ? SHIM_N : SHIM_K, b_cols = tb ? SHIM_K : SHIM_N;
  const int lda = a_rows + SHIM_PAD, ldb = b_rows + SHIM_PAD, ldc = SHIM_M + SHIM_PAD;
  const int lda_hat = 2 * a_rows, ldb_hat = tb ? b_rows : 2 * b_rows, ldc_hat = 2 * SHIM_M;
  real_t a[2 * SHIM_MAX * SHIM_MAX], b[2 * SHIM_MAX * SHIM_MAX];
  real_t c[2 * SHIM_MAX * SHIM_MAX], ref[2 * SHIM_MAX * SHIM_MAX];
  real_t a_hat[4 * SHIM_MAX * SHIM_MAX], b_hat[4 * SHIM_MAX * SHIM_MAX], c_hat[4 * SHIM_MAX * SHIM_MAX];
  unsigned int seed = (unsigned int)(transa * 31 + transb);
  int result = EXIT_SUCCESS, i, j, k;
  shim_fill(a, 2 * lda * a_cols, &seed);
  shim_fill(b, 2 * ldb * b_cols, &seed);
  shim_fill(c, 2 * ldc * SHIM_N, &seed);
  memcpy(ref, c, sizeof(real_t) * 2 * ldc * SHIM_N);

  /* reference: alpha * op(A) * op(B) + beta * C in complex arithmetic */
  for (j = 0; j < SHIM_N; ++j) {
    for (i = 0; i < SHIM_M; ++i) {
      real_t sr = 0, si = 0, cr, ci;
      for (k = 0; k < SHIM_K; ++k) {
        const int ia = 2 * (ta ? (k + i * lda) : (i + k * lda));
        const int ib = 2 * (tb ? (j + k * ldb) : (k + j * ldb));
        const real_t xr = a[ia], xi = ca ? -a[ia + 1] : a[ia + 1];
        const real_t yr = b[ib], yi = cb ? -b[ib + 1] : b[ib + 1];
        sr += xr * yr - xi * yi;
        si += xr * yi + xi * yr;
      }
      cr = ref[2 * (i + j * ldc)];
      ci = ref[2 * (i + j * ldc) + 1];
      ref[2 * (i + j * ldc)] = alpha[0] * sr - alpha[1] * si + beta[0] * cr - beta[1] * ci;
      ref[2 * (i + j * ldc) + 1] = alpha[0] * si + alpha[1] * sr + beta[0] * ci + beta[1] * cr;
    }
  }

  SHIM_LAUNCH(a_rows, a_cols, zgemm_block_construct_a(a, a_hat, a_rows, a_cols, lda, ta_sign));
  if (0 != tb) {
    SHIM_LAUNCH(b_rows, b_cols, zgemm_block_construct_b_t(b, b_hat, b_rows, b_cols, ldb, cb));
  }
  else {
    SHIM_LAUNCH(b_rows, b_cols, zgemm_block_construct_b_n(b, b_hat, b_rows, b_cols, ldb, cb));
  }
  for (j = 0; j < SHIM_N; ++j) {
    for (i = 0; i < 2 * SHIM_M; ++i) {
      real_t s = 0;
      for (k = 0; k < 2 * SHIM_K; ++k) {
        s += (ta ? a_hat[k + i * lda_hat] : a_hat[i + k * lda_hat])
           * (tb ? b_hat[j + k * ldb_hat] : b_hat[k + j * ldb_hat]);
      }
      c_hat[i + j * ldc_hat] = s;
    }
  }
  SHIM_LAUNCH(SHIM_M, SHIM_N,
    zgemm_block_finalize(c, c_hat, SHIM_M, SHIM_N, ldc, alpha[0], alpha[1], beta[0], beta[1]));

  /* the padding rows are compared too: the kernel must not have written them */
  for (i = 0; i < 2 * ldc * SHIM_N && EXIT_SUCCESS == result; ++i) {
    if (c[i] != ref[i]) {
      fprintf(stderr, "shim: gemm3m %c%c beta=(%g,%g) differs at %i: %g != %g\n", transa, transb,
        (double)beta[0], (double)beta[1], i, (double)c[i], (double)ref[i]);
      result = EXIT_FAILURE;
    }
  }
  return result;
}
