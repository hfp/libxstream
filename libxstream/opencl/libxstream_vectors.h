/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_OPENCL_VECTORS_H
#define LIBXSTREAM_OPENCL_VECTORS_H

/**
 * OpenCL C vector types where the compiler has none, so that a kernel written
 * against them also translates for the host (libxstream_cpu_begin.h). The
 * device keeps the language's own types and operators; the host gets a struct
 * per type and reaches the operations on a whole vector through VEC_ZERO,
 * VEC_ADD and VEC_SET4.
 *
 * Only a whole-vector operation needs the detour. Component access does not:
 * a union of the vector and an array of its scalar means the same thing on both
 * sides, and that is what the kernels already spell.
 *
 * The host type is a struct rather than a compiler vector extension, which
 * would have let a kernel keep "(int8)(0)" and "a + b" verbatim: clang splats a
 * scalar cast onto a vector and GCC does not, so the two would disagree on the
 * operation that appears most, and the host mapping would stop being swappable.
 * The component loops are short and constant-trip, hence the vectorizer still
 * sees what it needs.
 *
 * VEC_ADD takes the type because C has no way to select on the type of an
 * argument: the caller knows it, and passing it keeps one spelling for both
 * sides. A caller that has the type in a macro (OZAKI_ACC_T) passes that.
 *
 * The host mapping is selected by LIBXSTREAM_CPU, which libxstream_cpu_begin.h
 * defines, and not by the absence of __OPENCL_VERSION__: a kernel is also run
 * through the host preprocessor on its way to the device (libxstream_opencl_dump
 * instantiates it with cpp), and that pass has no OpenCL predefine either. It
 * would take the host branch and bake the structs into what the device compiles.
 * Whoever states the host build is therefore the one who says so.
 */

#if !defined(LIBXSTREAM_CPU)

#define VEC_ZERO(TYPE) ((TYPE)(0))
#define VEC_ADD(TYPE, A, B) ((A) + (B))
#define VEC_SET4(TYPE, A, B, C, D) ((TYPE)((A), (B), (C), (D)))

#else

/**
 * A kernel stores a whole vector through a pointer to a narrower element, which
 * is how a 16-byte block leaves a byte buffer in one instruction. That promise
 * is the one strict aliasing does not make, and the attribute belongs to the
 * type rather than to the cast, hence it is attached where the type is made:
 * to the struct itself, as GCC ignores it after the name of a typedef whose
 * struct is defined in place. The host branch runs only under
 * libxstream_cpu_begin.h, which has included LIBXS and includes this header
 * before it neutralizes __attribute__.
 *
 * The host spelling of one vector type: TYPE is the OpenCL name, SCALAR the
 * component type, and N the width. The scalar array is named "s" so that a
 * union with an array of SCALAR sees the components either way.
 */
#define LIBXSTREAM_VEC_TYPE(TYPE, SCALAR, N) \
  typedef struct LIBXS_MAY_ALIAS { SCALAR s[N]; } TYPE; \
  static TYPE libxstream_vec_zero_##TYPE(void) { \
    TYPE zero_; \
    int i_; \
    for (i_ = 0; i_ < (N); ++i_) zero_.s[i_] = 0; \
    return zero_; \
  } \
  static TYPE libxstream_vec_add_##TYPE(TYPE a_, TYPE b_) { \
    TYPE sum_; \
    int i_; \
    for (i_ = 0; i_ < (N); ++i_) sum_.s[i_] = (SCALAR)(a_.s[i_] + b_.s[i_]); \
    return sum_; \
  }

/**
 * A vector built from its components, which OpenCL spells as a cast that C
 * reads as a comma expression instead. One width per macro, because the width
 * is part of what is being said, and only the widths a kernel writes exist.
 */
#define LIBXSTREAM_VEC_SET4(TYPE, SCALAR) \
  static TYPE libxstream_vec_set4_##TYPE(SCALAR a_, SCALAR b_, SCALAR c_, SCALAR d_) { \
    TYPE set_; \
    set_.s[0] = a_; \
    set_.s[1] = b_; \
    set_.s[2] = c_; \
    set_.s[3] = d_; \
    return set_; \
  }

/**
 * Two levels, the way CAT is spelled in libxstream_common.h: the argument of a
 * macro that pastes is not expanded first, so VEC_ZERO(OZAKI_ACC_T) has to
 * become VEC_ZERO_(int8) before the paste.
 */
#define VEC_ZERO(TYPE) VEC_ZERO_(TYPE)
#define VEC_ZERO_(TYPE) libxstream_vec_zero_##TYPE()
#define VEC_ADD(TYPE, A, B) VEC_ADD_(TYPE, A, B)
#define VEC_ADD_(TYPE, A, B) libxstream_vec_add_##TYPE(A, B)
#define VEC_SET4(TYPE, A, B, C, D) VEC_SET4_(TYPE, A, B, C, D)
#define VEC_SET4_(TYPE, A, B, C, D) libxstream_vec_set4_##TYPE(A, B, C, D)

/* A type the kernel at hand does not use brings its helpers along. */
LIBXS_PRAGMA_DIAG_PUSH()
LIBXS_PRAGMA_DIAG_OFF("-Wunused-function")

/**
 * The widths the kernels ask for. The scalars are spelled in full: the header
 * precedes the uchar/ushort/uint names libxstream_cpu_begin.h establishes, and
 * the types stay after the bracket, where a host launcher builds the same data.
 */
LIBXSTREAM_VEC_TYPE(char8, signed char, 8)
LIBXSTREAM_VEC_TYPE(uchar4, unsigned char, 4)
LIBXSTREAM_VEC_TYPE(ushort8, unsigned short, 8)
LIBXSTREAM_VEC_TYPE(int2, int, 2)
LIBXSTREAM_VEC_TYPE(int4, int, 4)
LIBXSTREAM_VEC_TYPE(int8, int, 8)
LIBXSTREAM_VEC_TYPE(uint4, unsigned int, 4)
LIBXSTREAM_VEC_TYPE(uint8, unsigned int, 8)
LIBXSTREAM_VEC_TYPE(float4, float, 4)
LIBXSTREAM_VEC_TYPE(float8, float, 8)
LIBXSTREAM_VEC_SET4(uchar4, unsigned char)
LIBXSTREAM_VEC_SET4(int4, int)
LIBXSTREAM_VEC_SET4(uint4, unsigned int)
LIBXSTREAM_VEC_SET4(float4, float)

LIBXS_PRAGMA_DIAG_POP()

#endif

#endif /*LIBXSTREAM_OPENCL_VECTORS_H*/
