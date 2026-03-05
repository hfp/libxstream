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

/* Decompose an IEEE-754 value into sign, biased exponent, and implicit-1 mantissa.
 * Zero and subnormal inputs yield exp=0, mant=0. */
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

#endif /*OZAKI_COMMON_CL*/
