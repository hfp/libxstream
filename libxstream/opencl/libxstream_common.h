/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_OPENCL_COMMON_H
#define LIBXSTREAM_OPENCL_COMMON_H

/**
 * Token concatenation. CAT expands its arguments before pasting, which the
 * bare ## operator does not: it suppresses expansion of its operands, so
 * CAT_(PREFIX_, MACRO) pastes the macro's *name* instead of its value.
 * Needed to branch on a build-time -D token such as -DCONSTANT=global.
 */
#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

/* address space of read-only kernel arguments; hosts pass it explicitly */
#if !defined(CONSTANT)
# define CONSTANT global
#endif

#if !defined(LIBXSTREAM_OCLVER_C)
# define LIBXSTREAM_OCLVER_C __OPENCL_C_VERSION__
#endif
#if !defined(LIBXSTREAM_OCLVER)
# define LIBXSTREAM_OCLVER __OPENCL_VERSION__
#endif

/**
 * Either version admits it: a 3.0 device may report OpenCL C 1.2 and still take
 * the attribute, so requiring the C version alone silently drops every hint.
 * A host build of the same source supplies its own spelling beforehand.
 */
#if !defined(UNROLL_FORCE)
# if (200 /*CL_VERSION_2_0*/ <= LIBXSTREAM_OCLVER_C) || (200 <= LIBXSTREAM_OCLVER) || \
   defined(__NV_CL_C_VERSION)
#   define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#   define UNROLL_AUTO __attribute__((opencl_unroll_hint))
# else
#   define UNROLL_FORCE(N)
#   define UNROLL_AUTO
# endif
#endif

#if !defined(LU) || (-1 == LU)
# define UNROLL_OUTER(N)
# define UNROLL(N)
#else /* (-2) full, (-1) no hints, (0) inner, (1) outer-dehint, (2) block-m */
# if (1 <= LU) /* outer-dehint */
#   define UNROLL_OUTER(N) UNROLL_FORCE(1)
# elif (-1 > LU) /* full */
#   define UNROLL_OUTER(N) UNROLL_FORCE(N)
# else /* inner */
#   define UNROLL_OUTER(N)
# endif
# define UNROLL(N) UNROLL_FORCE(N)
#endif

#define BCST_NO(V, I) (V)
/* optional in 3.0: 300 clears "200 <=" yet may provide neither */
#if defined(WG) && (0 < WG) && defined(GPU) && \
  (defined(__opencl_c_work_group_collective_functions) || \
    (200 <= LIBXSTREAM_OCLVER_C && 300 > LIBXSTREAM_OCLVER_C))
# define BCST_WG(V, I) work_group_broadcast(V, I)
#endif
#if defined(SG) && (0 < SG) && defined(GPU) && \
  (defined(__opencl_c_subgroups) || defined(cl_khr_subgroups) || \
    (200 <= LIBXSTREAM_OCLVER_C && 300 > LIBXSTREAM_OCLVER_C))
# define BCST_SG(V, I) sub_group_broadcast(V, I)
#endif

/**
 * Sub-group lane and group ID: use sub-group builtins when available,
 * fall back to local IDs for vendors without cl_khr_subgroups (e.g. NVIDIA).
 * Requires work-group layout (SG, num_sub_groups, 1).
 */
#if defined(INTEL) && (0 < INTEL)
# define SGLID() get_sub_group_local_id()
# define SGID()  get_sub_group_id()
#else
# define SGLID() get_local_id(0)
# define SGID()  get_local_id(1)
#endif

#if !defined(MIN)
# define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#if !defined(MAX)
# define MAX(A, B) ((A) < (B) ? (B) : (A))
#endif
#if !defined(MAD)
# define MAD fma
#endif

#define DIVUP(A, B) (((A) + (B) - 1) / (B))
#define NUP(N, UP) (DIVUP(N, UP) * (UP))
#define BLR(N, BN) (NUP(N, BN) - (N))

#define IDX(I, J, M, N) ((int)(I) * (N) + (J))
#define IDT(I, J, M, N) IDX(J, I, N, M)

/**
 * Floating-point type and IEEE bit-manipulation utilities.
 * Controlled by USE_DOUBLE (define to 1 for fp64, 0 or undef for fp32).
 */
#if (defined(USE_DOUBLE) && (1 == USE_DOUBLE)) || (defined(TAN) && (2 == TAN))
# pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
typedef ulong uint_repr_t;
# define EXP_MASK 2047U
# define AS_UINT(x) as_ulong(x)
# if !defined(ZERO)
#   define ZERO 0.0
# endif
#elif !defined(TAN) || (1 == TAN)
typedef float real_t;
typedef uint uint_repr_t;
# define EXP_MASK 255U
# define AS_UINT(x) as_uint(x)
# if !defined(ZERO)
#   define ZERO 0.f
# endif
#endif

/**
 * Integer power of two via bit manipulation: 2^N exactly.
 * Avoids FP transcendental - one integer add, one shift, one bitcast.
 */
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
# define EXP2I(N) as_double((long)((N) + 1023) << 52)
#else
# define EXP2I(N) as_float(((N) + 127) << 23)
#endif

/* BF16 conversion helpers. Controlled by USE_BF16_EXT/USE_BF16. */
#if defined(USE_BF16_EXT) && (0 < USE_BF16_EXT)
/**
 * Hardware round-to-nearest-even via cl_intel_bfloat16_conversions.
 * Extension pragmas trigger warnings on some drivers; availability
 * is checked at init time.
 */
/*# pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable*/
# define ROUND_TO_BF16(x) intel_convert_bfloat16_as_ushort(x)
# define BF16_TO_F32(x) intel_convert_as_bfloat16_float(x)
#elif !defined(ROUND_TO_BF16) && defined(USE_BF16) && (0 < USE_BF16)
/**
 * Round a float to BF16 (round-to-nearest-even).
 * Portable uint32 bit-manipulation (no __bf16 intrinsic required).
 */
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
# define BF16_TO_F32(x) bf16_to_f32(x)
#endif

/**
 * IEEE FP16 storage conversions via core OpenCL half built-ins.
 * vload_half/vstore_half_rte convert to/from float without requiring
 * cl_khr_fp16 or half arithmetic; the raw storage is uint16_t.
 */
#if !defined(ROUND_TO_F16) && defined(USE_F16) && (0 < USE_F16)
/** Round a float to FP16 (round-to-nearest-even) and return raw bits. */
inline ushort round_to_f16(float f)
{
  ushort v;
  vstore_half_rte(f, 0, (half*)&v);
  return v;
}
/** Expand an FP16 encoding to float32 (exact). */
inline float f16_to_f32(ushort v)
{
  return vload_half(0, (const half*)&v);
}
# define ROUND_TO_F16(x) round_to_f16(x)
# define F16_TO_F32(x) f16_to_f32(x)
#endif

#endif /*LIBXSTREAM_OPENCL_COMMON_H*/
