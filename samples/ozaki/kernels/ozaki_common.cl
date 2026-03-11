/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_COMMON_CL
#define OZAKI_COMMON_CL

/* Shared primitives for all Ozaki kernel files.
 *
 * Provides:
 *   CONSTANT            - address-space qualifier (global or __constant)
 *   OZAKI_DPAS          - one DPAS step (2D block I/O + int8 MAD)
 *   ieee_decompose()    - IEEE-754 -> (sign, biased exponent, mantissa)
 *   ozaki_slice_digit() - extract a 7-bit signed digit from aligned mantissa
 */

#if !defined(CONSTANT)
# define CONSTANT global
#endif

/* One DPAS step: 2D block read + int8x8->int32 MAD.
 *
 *   int8 intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc)
 *   A tile: 8 rows x 32 cols  (read as ushort8 via 2D block read)
 *   B tile: 32 rows x 16 cols (read with VNNI transform via 2D block read)
 *   C tile: 8 x 16 int32      (int8 per WI — 8 rows, sg_lid selects column)
 *
 * 2D block I/O requires SG=16 and surface pitch >= 64 bytes. */
#define OZAKI_DPAS(AS, BS, K_PAD, N_PAD, MI, NJ, KOFF, M_HT, ACC) \
  do { \
    ushort8 a_blk_; \
    uint8 b_blk_; \
    intel_sub_group_2d_block_read_8b_8r32x1c( \
        (global void*)(AS), (K_PAD), (M_HT), (K_PAD), \
        (int2)((KOFF), (MI)), (private ushort*)&a_blk_); \
    intel_sub_group_2d_block_read_transform_8b_32r16x1c( \
        (global void*)(BS), (N_PAD), (K_PAD), (N_PAD), \
        (int2)((NJ), (KOFF)), (private uint*)&b_blk_); \
    (ACC) = intel_sub_group_i8_i8_matrix_mad_k32( \
                as_short8(a_blk_), as_int8(b_blk_), (ACC)); \
  } while (0)


/* Decompose an IEEE-754 value into sign, biased exponent, and implicit-1 mantissa.
 * Zero and subnormal inputs yield exp=0, mant=0.
 * real_t, uint_repr_t, EXP_MASK, and AS_UINT come from libxstream_common.h. */
inline void ieee_decompose(real_t val, int* sign, short* exp, uint_repr_t* mant)
{
  const uint_repr_t bits = AS_UINT(val);
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
  *sign = (int)(bits >> 63);
  *exp  = (short)((bits >> 52) & EXP_MASK);
  *mant = (bits & 0x000FFFFFFFFFFFFFUL) | 0x0010000000000000UL;
#else
  *sign = (int)(bits >> 31);
  *exp  = (short)((bits >> 23) & EXP_MASK);
  *mant = (bits & 0x007FFFFFU) | 0x00800000U;
#endif
  if (0 == *exp) {
    *mant = 0;
    *exp  = 0;
  }
}

/* Extract a 7-bit signed digit from an aligned mantissa for slice index S.
 * The mantissa ALIGNED is already right-shifted by (max_exp - elem_exp).
 * Returns a signed char: the digit with sign applied if SIGN != 0.
 * MANT_BITS must be defined by the including file. */
inline char ozaki_slice_digit(uint_repr_t aligned, int sign, int s)
{
  const int high = MANT_BITS - (7 * s);
  const int low  = MAX(0, high - 6);
  const int width = high - low + 1;
  char digit = 0;
  if (width > 0 && high >= 0) {
    digit = (char)((aligned >> low) & ((1U << width) - 1U));
  }
  if (sign) digit = -digit;
  return digit;
}

#endif /*OZAKI_COMMON_CL*/
