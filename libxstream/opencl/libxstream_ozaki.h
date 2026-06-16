/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_OPENCL_OZAKI_H
#define LIBXSTREAM_OPENCL_OZAKI_H

#include "libxstream_common.h"

/* Dekker splitting: iterative residual subtraction into BF16 digits.
 * Each digit carries its own sign and exponent -- no shared exponent
 * panel required.  The sum of all digits exactly reconstructs the
 * input (up to the trailing residual below BF16 precision^NDIGITS).
 *
 * NDIGITS: number of BF16 slices to produce (compile-time constant).
 * val:     input value (float or double).
 * dst:     output array of NDIGITS ushort BF16 encodings.
 */
#if defined(ROUND_TO_BF16) && defined(BF16_TO_F32)
inline void dekker_split_bf16(real_t val, int ndigits, global ushort* dst)
{
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  double residual = (double)val;
#else
  float residual = (float)val;
#endif
  int s;
  for (s = 0; s < ndigits; ++s) {
    const ushort bf = ROUND_TO_BF16((float)residual);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    residual -= (double)BF16_TO_F32(bf);
#else
    residual -= BF16_TO_F32(bf);
#endif
    dst[s] = bf;
  }
}

inline void dekker_split_bf16_private(real_t val, int ndigits,
                                      private ushort* dst)
{
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  double residual = (double)val;
#else
  float residual = (float)val;
#endif
  int s;
  for (s = 0; s < ndigits; ++s) {
    const ushort bf = ROUND_TO_BF16((float)residual);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
    residual -= (double)BF16_TO_F32(bf);
#else
    residual -= BF16_TO_F32(bf);
#endif
    dst[s] = bf;
  }
}
#endif


/* BF16 DPAS: bf16 x bf16 -> f32, one 8x16x16 tile.
 * Intel XMX (SG=16):
 *   A: 8 rows x 16 bf16  (short8 per WI -- one bf16/WI * 16 WIs * 8 rows)
 *   B: 16 rows x 16 bf16 (VNNI-packed into int8 -- 2 bf16/uint)
 *   C: 8 x 16 float      (float8 per WI -- 8 rows, sg_lid selects column)
 *
 * 2D block I/O for 16-bit elements:
 *   A: intel_sub_group_2d_block_read_16b_8r16x1c
 *   B: intel_sub_group_2d_block_read_transform_16b_16r16x1c (VNNI pack)
 *
 * Surface constraints: width/pitch >= 64 bytes, pitch 16-byte aligned. */
#if defined(INTEL) && (2 <= INTEL)

# define BF16_MAD_K16(A, B, ACC) \
    intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(A), as_int8(B), ACC)

# define BF16_DPAS(A_SURF, B_SURF, A_WB, A_HT, B_WB, B_HT, \
                          MI, NJ, A_KOFF, B_KOFF, ACC) \
    do { \
      ushort8 a_bf_; \
      uint8 b_bf_; \
      intel_sub_group_2d_block_read_16b_8r16x1c( \
        (global void*)(A_SURF), (A_WB), (A_HT), (A_WB), \
        (int2)((A_KOFF) * 2, (MI)), (private ushort*)&a_bf_); \
      intel_sub_group_2d_block_read_transform_16b_16r16x1c( \
        (global void*)(B_SURF), (B_WB), (B_HT), (B_WB), \
        (int2)((NJ) * 2, (B_KOFF)), (private uint*)&b_bf_); \
      (ACC) = BF16_MAD_K16(a_bf_, b_bf_, (ACC)); \
    } while (0)

# define BF16_DPAS_ONE(A, B, ACC) \
    (ACC) = BF16_MAD_K16(A, B, (ACC))

/* Load A tile (8 x 16 bf16) from 2D surface into register. */
# define BF16_LOAD_A(A_SURF, A_WB, A_HT, MI, KOFF, DST) \
    intel_sub_group_2d_block_read_16b_8r16x1c( \
      (global void*)(A_SURF), (A_WB), (A_HT), (A_WB), \
      (int2)((KOFF) * 2, (MI)), (private ushort*)(DST))

/* Load B tile (16 x 16 bf16, VNNI-transformed) from 2D surface. */
# define BF16_LOAD_B(B_SURF, B_WB, B_HT, NJ, KOFF, DST) \
    intel_sub_group_2d_block_read_transform_16b_16r16x1c( \
      (global void*)(B_SURF), (B_WB), (B_HT), (B_WB), \
      (int2)((NJ) * 2, (KOFF)), (private uint*)(DST))

/* Prefetch hints for bf16 tiles. */
# define BF16_PREFETCH_A(A_SURF, A_WB, A_HT, MI, KOFF) \
    intel_sub_group_2d_block_prefetch_16b_8r16x1c( \
      (global void*)(A_SURF), (A_WB), (A_HT), (A_WB), \
      (int2)((KOFF) * 2, (MI)))

# define BF16_PREFETCH_B(B_SURF, B_WB, B_HT, NJ, KOFF) \
    intel_sub_group_2d_block_prefetch_16b_16r16x1c( \
      (global void*)(B_SURF), (B_WB), (B_HT), (B_WB), \
      (int2)((NJ) * 2, (KOFF)))

#endif /* INTEL >= 2 */

#endif /*LIBXSTREAM_OPENCL_OZAKI_H*/
