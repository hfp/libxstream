/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

#if !defined(LIBXSTREAM_C_VERSION)
#  define LIBXSTREAM_C_VERSION __OPENCL_C_VERSION__
#endif
#if !defined(LIBXSTREAM_VERSION)
#  define LIBXSTREAM_VERSION __OPENCL_VERSION__
#endif

#if (200 /*CL_VERSION_2_0*/ <= LIBXSTREAM_C_VERSION) || defined(__NV_CL_C_VERSION)
#  define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#  define UNROLL_AUTO __attribute__((opencl_unroll_hint))
#else
#  define UNROLL_FORCE(N)
#  define UNROLL_AUTO
#endif

#if !defined(LU) || (-1 == LU)
#  define UNROLL_OUTER(N)
#  define UNROLL(N)
#else /* (-2) full, (-1) no hints, (0) inner, (1) outer-dehint, (2) block-m */
#  if (1 <= LU) /* outer-dehint */
#    define UNROLL_OUTER(N) UNROLL_FORCE(1)
#  elif (-1 > LU) /* full */
#    define UNROLL_OUTER(N) UNROLL_FORCE(N)
#  else /* inner */
#    define UNROLL_OUTER(N)
#  endif
#  define UNROLL(N) UNROLL_FORCE(N)
#endif

#define BCST_NO(V, I) (V)
#if defined(WG) && (0 < WG) && defined(GPU) && (200 <= LIBXSTREAM_VERSION)
#  define BCST_WG(V, I) work_group_broadcast(V, I)
#endif
#if defined(SG) && (0 < SG) && defined(GPU) && (200 <= LIBXSTREAM_VERSION)
#  define BCST_SG(V, I) sub_group_broadcast(V, I)
#endif

#if !defined(MIN)
#  define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#if !defined(MAX)
#  define MAX(A, B) ((A) < (B) ? (B) : (A))
#endif
#if !defined(MAD)
#  define MAD fma
#endif

#define DIVUP(A, B) (((A) + (B) - 1) / (B))
#define NUP(N, UP) (DIVUP(N, UP) * (UP))
#define BLR(N, BN) (NUP(N, BN) - (N))

#define IDX(I, J, M, N) ((int)(I) * (N) + (J))
#define IDT(I, J, M, N) IDX(J, I, N, M)

/* Floating-point type and IEEE bit-manipulation utilities.
 * Controlled by USE_DOUBLE (define to 1 for fp64, 0 or undef for fp32). */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
# pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
  typedef ulong  uint_repr_t;
# define EXP_MASK 2047U
# define AS_UINT(x) as_ulong(x)
#else
  typedef float  real_t;
  typedef uint   uint_repr_t;
# define EXP_MASK 255U
# define AS_UINT(x) as_uint(x)
#endif

/* BF16 conversion helpers.
 * Controlled by USE_BF16_EXT (define to 1 for cl_intel_bfloat16_conversions hw path). */
#if defined(USE_BF16_EXT) && (0 < USE_BF16_EXT)
/* Hardware round-to-nearest-even via cl_intel_bfloat16_conversions.
 * Extension pragmas trigger warnings on some drivers; availability
 * is checked at init time. */
/*# pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable*/
# define ROUND_TO_BF16(x) intel_convert_bfloat16_as_ushort(x)
# define BF16_TO_F32(x)   intel_convert_as_bfloat16_float(x)
#elif !defined(ROUND_TO_BF16)
/** Round a float to BF16 (round-to-nearest-even).
 *  Portable uint32 bit-manipulation (no __bf16 intrinsic required). */
inline ushort round_to_bf16(float f)
{
  uint bits = as_uint(f);
  bits = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
  return (ushort)(bits >> 16);
}

/** Expand a BF16 encoding to float32 (exact). */
inline float bf16_to_f32(ushort v)
{
  return as_float((uint)v << 16);
}
# define ROUND_TO_BF16(x) round_to_bf16(x)
# define BF16_TO_F32(x)   bf16_to_f32(x)
#endif

#endif /*OPENCL_COMMON_H*/
