/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../../../include/opencl/libxstream_common.h"
#include "ozaki_common.cl"

/* Ozaki Scheme 3 (bf16): Dekker-splitting GEMM via OpenCL.
 *
 * Three kernels:
 *   preprocess_a  - Dekker-split rows of A into bf16 slices
 *   preprocess_b  - Dekker-split cols of B into bf16 slices
 *   dotprod       - bf16 dot products over slice pairs, accumulate into C
 *
 * Unlike the int8 mantissa-slicing scheme (ozaki1_int8.cl), each bf16
 * slice carries its own sign and exponent.  This eliminates the shared
 * per-row/per-column exponent panels and the exponent-reconstruction
 * step during accumulation.  The dot-product result is already a
 * properly-scaled FP32 value.
 *
 * Compile-time parameters (supplied via -D):
 *   BM, BN, BK     - block dimensions (default 16)
 *   WG              - work-group size hint (0 to disable)
 *   SG              - sub-group size hint (0 to disable)
 *   NSLICES         - number of Dekker bf16 slices
 *   TRIANGULAR      - if 1, iterate upper triangle of slice pairs
 *   SYMMETRIZE      - if 1, compute mirror D(sb,sa) for off-diagonal pairs
 *   TRIM            - number of least-significant diagonals to skip
 *   USE_DOUBLE      - if 1, accumulate in double; otherwise float
 *   CONSTANT        - address-space qualifier for read-only buffers
 *   USE_BF16_EXT    - if 1, use cl_intel_bfloat16_conversions builtins
 *   USE_XMX         - if 1, use DPAS bf16 x bf16 -> f32
 *                     (requires BK == 16, BM divisible by 8, BN by 16)
 */

#if !defined(BM)
# define BM 16
#endif
#if !defined(BN)
# define BN 16
#endif
#if !defined(BK)
# define BK 16
#endif
#if !defined(NSLICES)
# define NSLICES 8
#endif
#if !defined(TRIANGULAR)
# define TRIANGULAR 1
#endif
#if !defined(SYMMETRIZE)
# define SYMMETRIZE 1
#endif
#if !defined(TRIM)
# define TRIM 0
#endif



/* XMX (hardware matrix multiply-accumulate) definitions */

#if defined(USE_XMX) && (0 < USE_XMX)
/* Extensions checked at init time; pragma enable is not required
 * and triggers warnings on some drivers. */
/*# pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable*/
/*# pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable*/

/* DPAS bf16 x bf16 -> f32 with SG=16:
 *   float8 intel_sub_group_bf16_bf16_matrix_mad_k16(
 *     short8 a, int8 b, float8 acc)
 *   A tile: 8 rows x K(=16) bf16
 *   B tile: K(=16) x N(=16) bf16 (VNNI-packed into int8)
 *   Result: 8 x 16 float
 *
 * 2D block I/O (cl_intel_subgroup_2d_block_io, SG=16):
 *   A: intel_sub_group_2d_block_read_16b_8r16x1c      -> ushort[8]
 *   B: intel_sub_group_2d_block_read_transform_16b_16r16x1c -> uint[8]
 *   Surface params: width/pitch in bytes (literal), height in rows.
 *   Width & pitch >= 64 bytes, pitch multiple of 16 bytes. */
# define XMX_M 8
# define XMX_N 16
# define NTM (BM / XMX_M)
# define NTN (BN / XMX_N)
/* Pad B column stride so surface width >= 64 bytes (= 32 bf16) */
# define BN_PAD ((BN) < 32 ? 32 : (BN))
/* A row stride in ushort elements: NSLICES * BK */
# define A_STRIDE (NSLICES * BK)
# if (16 != BK)
#   error "USE_XMX bf16 requires BK == 16"
# endif
# if (0 != (BM % XMX_M) || 0 != (BN % XMX_N))
#   error "USE_XMX requires BM divisible by XMX_M (8) and BN by XMX_N (16)"
# endif
# if (A_STRIDE * 2 < 64)
#   error "USE_XMX bf16 requires A surface width >= 64 bytes (NSLICES >= 2)"
# endif
/* B layout for XMX: bk[panel][s][kk][nj]  (K-major for VNNI load).
 * Rows padded to BN_PAD >= 32 for 2D block I/O surface constraints.
 * Default layout: bk[panel][nj][s][kk] (N-major for scalar access). */
# define BK_IDX(panel, nj, s, kk) \
    ((((long)(panel) * NSLICES + (s)) * BK + (kk)) * BN_PAD + (nj))
#else
# define BK_IDX(panel, nj, s, kk) \
    ((((long)(panel) * BN + (nj)) * NSLICES + (s)) * BK + (kk))
#endif


/**
 * preprocess_a: Dekker-split rows of A into bf16 slices.
 *
 * One work-group per (row-block, k-sub-block).
 * Work-group size: (BM, BK).
 *
 * Each work-item handles one element A[row,col] and produces NSLICES
 * bf16 values via successive rounding-to-bf16 and residual subtraction
 * (Dekker splitting).  No shared per-row exponent is needed — each
 * bf16 value carries its own sign and exponent.
 *
 * Inputs:
 *   a       - source matrix A (row-major or col-major depending on transa)
 *   M, K    - matrix dimensions
 *   lda     - leading dimension of A
 *   transa  - 0: not transposed, 1: transposed
 *   kb_offset - k-offset for this batch
 *
 * Outputs:
 *   ak      - bf16 slices (ushort) [panels][BM][NSLICES][BK]
 */
__attribute__((reqd_work_group_size(BM, BK, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_a(
  CONSTANT const real_t* restrict a,
  int M, int K, int lda, int transa,
  int kb_offset,
  global ushort* restrict ak,
  int nblk_m)
{
  const int ib_idx = (int)get_group_id(0);
  const int ki     = (int)get_group_id(1);
  const int mi     = (int)get_local_id(0);
  const int kk     = (int)get_local_id(1);

  const int ib  = ib_idx * BM;
  const int kb  = kb_offset + ki * BK;
  const int row = ib + mi;
  const int col = kb + kk;

  const int panel = ki * nblk_m + ib_idx;
#if defined(USE_XMX) && (0 < USE_XMX)
  const int stride = A_STRIDE;
#else
  const int stride = NSLICES * BK;
#endif
  SINT s;

  if (row < M && col < K) {
    const int idx = transa ? (row * lda + col) : (col * lda + row);
    const real_t val = a[idx];
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    double residual = (double)val;
#else
    float residual = (float)val;
#endif
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      const ushort bf = ROUND_TO_BF16((float)residual);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      residual -= (double)BF16_TO_F32(bf);
#else
      residual -= BF16_TO_F32(bf);
#endif
      ak[((long)panel * BM + mi) * stride + s * BK + kk] = bf;
    }
  }
  else {
    /* Zero or out-of-bounds: write zero slices */
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      ak[((long)panel * BM + mi) * stride + s * BK + kk] = 0;
    }
  }
}


/**
 * preprocess_b: Dekker-split columns of B into bf16 slices.
 *
 * One work-group per (col-block, k-sub-block).
 * Work-group size: (BN, BK).
 *
 * Output layout depends on USE_XMX:
 *   XMX:    bk[panel][NSLICES][BK][BN_PAD]  (K-major for VNNI)
 *   Scalar: bk[panel][BN][NSLICES][BK]      (N-major)
 */
__attribute__((reqd_work_group_size(BN, BK, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void preprocess_b(
  CONSTANT const real_t* restrict b,
  int N, int K, int ldb, int transb,
  int kb_offset,
  global ushort* restrict bk,
  int nblk_n)
{
  const int jb_idx = (int)get_group_id(0);
  const int ki     = (int)get_group_id(1);
  const int nj     = (int)get_local_id(0);
  const int kk     = (int)get_local_id(1);

  const int jb  = jb_idx * BN;
  const int kb  = kb_offset + ki * BK;
  const int col = jb + nj;
  const int row = kb + kk;

  const int panel = ki * nblk_n + jb_idx;
  SINT s;

  if (row < K && col < N) {
    const int idx = transb ? (row * ldb + col) : (col * ldb + row);
    const real_t val = b[idx];
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    double residual = (double)val;
#else
    float residual = (float)val;
#endif
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      const ushort bf = ROUND_TO_BF16((float)residual);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      residual -= (double)BF16_TO_F32(bf);
#else
      residual -= BF16_TO_F32(bf);
#endif
      bk[BK_IDX(panel, nj, s, kk)] = bf;
    }
  }
  else {
    /* Zero or out-of-bounds: write zero slices */
    UNROLL_FORCE(NSLICES) for (s = 0; s < NSLICES; ++s) {
      bk[BK_IDX(panel, nj, s, kk)] = 0;
    }
  }
}


#if defined(USE_XMX) && (0 < USE_XMX)
/**
 * dotprod (XMX path): DPAS bf16 x bf16 -> f32 for slice pairs.
 *
 * Uses cl_intel_subgroup_matrix_multiply_accumulate (bf16 DPAS)
 * with cl_intel_subgroup_2d_block_io for 16b data movement.
 * Required SG = 16.
 *
 * Each DPAS computes an XMX_M(8) x XMX_N(16) sub-tile of C.
 * Work-group: (SG, NTM * NTN, 1).
 * Each sub-group handles one sub-tile; WI sg_lid owns column sg_lid.
 *
 * DPAS for SG=16, bf16xbf16->f32:
 *   float8 intel_sub_group_bf16_bf16_matrix_mad_k16(
 *     short8 a, int8 b, float8 acc)
 *   a: short8  (1 bf16/WI * 16 WIs * 8 rows = 8x16 tile)
 *   b: int8    (2 bf16/uint VNNI * 8 * 16 WIs = 16x16 tile)
 *   result: float8 (8 row-values for this WI's column)
 *
 * 2D block I/O (cl_intel_subgroup_2d_block_io, SG=16):
 *   A: intel_sub_group_2d_block_read_16b_8r16x1c      -> ushort[8]
 *   B: intel_sub_group_2d_block_read_transform_16b_16r16x1c -> uint[8]
 *   Surface params: width/pitch in bytes, height in rows.
 *   Width & pitch >= 64 bytes, pitch multiple of 16 bytes.
 *   Coordinates: (byte_x, row_y).
 *
 * A layout: ak[panel][mi][s][kk]  (row stride = A_STRIDE ushorts).
 * B layout: bk[panel][s][kk][nj]  (K-major, padded to BN_PAD).
 *
 * No exponent panels — accumulation is a simple alpha * dot.
 *
 * Driver dispatch:
 *   local  = { SG(=16), NTM * NTN, 1 }
 *   global = { nblk_m * SG, nblk_n * NTM * NTN, 1 }
 */
__attribute__((reqd_work_group_size(SG, NTM * NTN, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void dotprod(
  CONSTANT const ushort* restrict ak,
  CONSTANT const ushort* restrict bk,
  global real_t* restrict c,
  int M, int N, int ldc,
  real_t alpha, real_t beta,
  int first_batch, int nkb,
  int nblk_m, int nblk_n)
{
  const int ib_idx  = (int)get_group_id(0);
  const int jb_idx  = (int)get_group_id(1);
  const int sg_lid  = (int)get_sub_group_local_id();
  const int sg_id   = (int)get_sub_group_id();
  const int tile_m  = sg_id / NTN;
  const int tile_n  = sg_id % NTN;
  const int mi_base = tile_m * XMX_M;
  const int nj_base = tile_n * XMX_N;
  const int col     = jb_idx * BN + nj_base + sg_lid;
  const int cutoff  = MAX(0, 2 * (NSLICES - 1) - TRIM);
  /* Width / pitch of A surface in bytes */
  const int a_wb    = A_STRIDE * 2;
  /* Width / pitch of B surface in bytes */
  const int b_wb    = BN_PAD * 2;
  real_t cval[XMX_M];
  int ki, m;
  SINT sa, sb;

  /* Load C: XMX_M row values for this WI's column.
   * All WIs must participate in DPAS; bounds checked per-element. */
  UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
    const int row_m = ib_idx * BM + mi_base + m;
    if (row_m < M && col < N) {
      cval[m] = c[col * ldc + row_m];
      if (first_batch) cval[m] *= beta;
    }
    else {
      cval[m] = ZERO;
    }
  }

  /* Loop over k-sub-panels */
  for (ki = 0; ki < nkb; ++ki) {
    const int a_panel = ki * nblk_m + ib_idx;
    const int b_panel = ki * nblk_n + jb_idx;

    /* Slice-pair loop */
    for (sa = 0; sa < NSLICES && sa <= cutoff; ++sa) {
#if (1 == TRIANGULAR)
      const SINT sb_start = sa;
#else
      const SINT sb_start = 0;
#endif
      const SINT sb_end = MIN(NSLICES, cutoff + 1 - sa);

      for (sb = sb_start; sb < sb_end; ++sb) {
        ushort8 a_raw; uint8 b_raw;
        float8 dot;

        /* Load A tile [8 x 16] of bf16 from ak[a_panel].
         * 2D surface: width = A_STRIDE * 2 bytes, height = BM rows.
         * Coord.x = sa * BK * 2 (byte offset to slice sa). */
        intel_sub_group_2d_block_read_16b_8r16x1c(
            (global void*)(ak + (long)a_panel * BM * A_STRIDE),
            a_wb, BM, a_wb,
            (int2)(sa * BK * 2, mi_base), (private ushort*)&a_raw);

        /* Load B tile [16 x 16] of bf16 with VNNI transform.
         * 2D surface over bk[b_panel][sb]: width = BN_PAD * 2 bytes.
         * Coord.x = nj_base * 2 (byte offset to column tile). */
        intel_sub_group_2d_block_read_transform_16b_16r16x1c(
            (global void*)(bk + ((long)b_panel * NSLICES + sb) * BK * BN_PAD),
            b_wb, BK, b_wb,
            (int2)(nj_base * 2, 0), (private uint*)&b_raw);

        /* DPAS: C[8x16] += A[8x16] * B[16x16]  (bf16 x bf16 -> f32) */
        dot = intel_sub_group_bf16_bf16_matrix_mad_k16(
                  as_short8(a_raw), as_int8(b_raw), (float8)(0.0f));

#if (1 == SYMMETRIZE)
        if (sa != sb) {
          ushort8 a_mir; uint8 b_mir;
          /* Mirror: swap slice indices (sb for A, sa for B) */
          intel_sub_group_2d_block_read_16b_8r16x1c(
              (global void*)(ak + (long)a_panel * BM * A_STRIDE),
              a_wb, BM, a_wb,
              (int2)(sb * BK * 2, mi_base), (private ushort*)&a_mir);
          intel_sub_group_2d_block_read_transform_16b_16r16x1c(
              (global void*)(bk + ((long)b_panel * NSLICES + sa) * BK * BN_PAD),
              b_wb, BK, b_wb,
              (int2)(nj_base * 2, 0), (private uint*)&b_mir);
          /* Chain DPAS: accumulate mirror result into same float8 */
          dot = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    as_short8(a_mir), as_int8(b_mir), dot);
        }
#endif

        /* Accumulate: bf16 dot result is already properly scaled float.
         * No exponent reconstruction needed — each bf16 slice carries
         * its own sign and exponent. */
        { union { float8 v; float a[8]; } dot_u;
          dot_u.v = dot;
          UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
            if (0.0f != dot_u.a[m]) {
              cval[m] += alpha * (real_t)dot_u.a[m];
            }
          }
        }
      }
    }
  }

  /* Write results: XMX_M rows of this WI's column */
  UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
    const int row_m = ib_idx * BM + mi_base + m;
    if (row_m < M && col < N) {
      c[col * ldc + row_m] = cval[m];
    }
  }
}

#else /* !USE_XMX: scalar dotprod */

/**
 * dotprod (scalar): bf16 dot products over slice pairs, accumulate into C.
 *
 * One work-group per (row-block, col-block) tile of C.
 * Work-group size: (BM, BN).
 *
 * Each work-item handles one element C[ib+mi, jb+nj] and iterates over
 * all k-sub-panels and slice pairs within the current K-batch.
 *
 * No exponent arrays — the bf16->f32 expansion in the dot product
 * produces correctly scaled results directly.
 */
__attribute__((reqd_work_group_size(BM, BN, 1)))
#if defined(SG) && (0 < SG)
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void dotprod(
  CONSTANT const ushort* restrict ak,
  CONSTANT const ushort* restrict bk,
  global real_t* restrict c,
  int M, int N, int ldc,
  real_t alpha, real_t beta,
  int first_batch, int nkb,
  int nblk_m, int nblk_n)
{
  const int ib_idx = (int)get_group_id(0);
  const int jb_idx = (int)get_group_id(1);
  const int mi     = (int)get_local_id(0);
  const int nj     = (int)get_local_id(1);

  const int row = ib_idx * BM + mi;
  const int col = jb_idx * BN + nj;
  const int cutoff = MAX(0, 2 * (NSLICES - 1) - TRIM);
  real_t cval;
  int ki;
  SINT sa, sb, kk;

  if (row >= M || col >= N) return;

  /* Beta scaling at first batch */
  cval = c[col * ldc + row];
  if (first_batch) {
    cval *= beta;
  }

  /* Loop over k-sub-panels */
  UNROLL_OUTER(1) for (ki = 0; ki < nkb; ++ki) {
    const int a_panel = ki * nblk_m + ib_idx;
    const int b_panel = ki * nblk_n + jb_idx;
    const long a_base = ((long)a_panel * BM + mi) * NSLICES;
    const long b_base = ((long)b_panel * BN + nj) * NSLICES;

    /* Slice-pair loop with optional triangular + symmetrize */
    UNROLL_AUTO for (sa = 0; sa < NSLICES && sa <= cutoff; ++sa) {
#if (1 == TRIANGULAR)
      const SINT sb_start = sa;
#else
      const SINT sb_start = 0;
#endif
      const SINT sb_end = MIN(NSLICES, cutoff + 1 - sa);

      UNROLL_AUTO for (sb = sb_start; sb < sb_end; ++sb) {
        float dot = 0.0f;

        UNROLL(BK) for (kk = 0; kk < BK; ++kk) {
          dot += BF16_TO_F32(ak[(a_base + sa) * BK + kk])
               * BF16_TO_F32(bk[(b_base + sb) * BK + kk]);
        }

#if (1 == SYMMETRIZE)
        if (sa != sb) {
          UNROLL(BK) for (kk = 0; kk < BK; ++kk) {
            dot += BF16_TO_F32(ak[(a_base + sb) * BK + kk])
                 * BF16_TO_F32(bk[(b_base + sa) * BK + kk]);
          }
        }
#endif

        if (0.0f != dot) {
          cval += alpha * (real_t)dot;
        }
      }
    }
  }

  c[col * ldc + row] = cval;
}

#endif /* USE_XMX */
