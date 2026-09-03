/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <stddef.h>
#if defined(__LIBXS) || defined(LIBXS_SOURCE)
# include <libxs/libxs_macros.h>
# include <libxs/libxs_math.h>
#endif

/**
 * OpenCL C to host C shim: lets an ordinary C compiler translate a kernel that
 * was written for a device. Unlike its neighbours here, this header is included
 * by host sources rather than by OpenCL sources.
 *
 * Include it after every other header, then the kernel source, then
 * libxstream_cpu_end.h. The bracket is not cosmetic: the OpenCL keywords are
 * empty macros, and gcc expands macros in a pragma line, so an in-scope
 * "private" silently deletes an OpenMP private clause rather than failing.
 *
 * The work-item state is file-static, so the launcher belongs to the same
 * translation unit as the kernel it launches.
 *
 * Work-group model, selected by LIBXSTREAM_CPU_TEAM:
 * 0 (default) A work-item runs to completion, hence barrier() is a no-op and
 *   local memory is an automatic array. The kernel must then carry no
 *   cross-lane dependency, which a kernel arranges either by turning its lanes
 *   into loops or by being launched with a 1x1 work-group. Work-groups stay
 *   independent, so the launcher may run them in parallel.
 * 1 Lanes form an OpenMP team of one thread each and barrier() is an orphaned
 *   OpenMP barrier that binds to it. Local memory becomes static, hence one
 *   work-group at a time. Reproduces the on-device local memory tiling and is a
 *   debugging aid rather than a fast path.
 *
 * Host long is 32-bit on LLP64 targets whereas OpenCL long is 64-bit, so a host
 * build indexes smaller buffers there than the device does.
 */

#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM) && !defined(_OPENMP)
# error LIBXSTREAM_CPU_TEAM needs OpenMP to implement barrier()
#endif

/* Address spaces: a host build has only the one. */
#define global
#define private
#define constant const
#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM)
# define local static
#else
# define local
#endif

/* Kernels become ordinary functions that the launcher calls directly. */
#define kernel static

/**
 * A device-side helper becomes file-local, otherwise an external inline
 * definition refers to the static conversion helpers below. LIBXS already spells
 * "inline" for C89, hence the undef before the redefinition.
 */
#undef inline
#define inline static
/**
 * A helper the kernel at hand does not call is expected here, and the shim
 * neutralizes __attribute__ (the device spellings mean nothing to a host
 * compiler), so the attribute route to silence it is not available.
 */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

/* Kernel attributes carry no meaning on the host. */
#if !defined(__attribute__)
# define __attribute__(A)
#endif

/* C89 has no restrict, yet GNU compilers still honor the alias promise. */
#if !defined(__STDC_VERSION__) || (199901L > __STDC_VERSION__)
# if defined(__GNUC__) || defined(__clang__)
#   define restrict __restrict__
# else
#   define restrict
# endif
#endif

/* Scalar type names that OpenCL C provides as built-ins. */
#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int

/* Fence flags: barrier() has nothing to order on a host build. */
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2

/**
 * The kernel's unroll hints carry over: libxstream_common.h keeps whatever is
 * defined here rather than dropping the hints off-device. Unrolling the short
 * constant-trip loops matters more on the host, because it is what leaves a
 * lane loop innermost with a straight-line body for the vectorizer.
 */
#if defined(LIBXS_PRAGMA_UNROLL_N)
# define UNROLL_FORCE(N) LIBXS_PRAGMA_UNROLL_N(N)
# define UNROLL_AUTO LIBXS_PRAGMA_UNROLL
#else
# define UNROLL_FORCE(N)
# define UNROLL_AUTO
#endif

/* Vectorize a lane loop nest of N levels. */
#if defined(LIBXS_PRAGMA_SIMD_COLLAPSE)
# define SIMD_COLLAPSE(N) LIBXS_PRAGMA_SIMD_COLLAPSE(N)
#else
# define SIMD_COLLAPSE(N)
#endif

/**
 * Storage conversions: OpenCL C reaches FP16 through the half built-ins and a
 * device may convert BF16 in hardware, neither of which a host build has. LIBXS
 * carries the host implementations, and a consumer that does not have LIBXS
 * defines the four names before this header rather than growing the shim.
 */
#if defined(__LIBXS) || defined(LIBXS_SOURCE)
# if !defined(ROUND_TO_BF16)
#   define ROUND_TO_BF16(X) libxs_round_bf16_f32(X)
# endif
# if !defined(BF16_TO_F32)
#   define BF16_TO_F32(X) libxs_bf16_to_f32(X)
# endif
# if !defined(ROUND_TO_F16)
#   define ROUND_TO_F16(X) libxs_round_f16_f32(X)
# endif
# if !defined(F16_TO_F32)
#   define F16_TO_F32(X) libxs_f16_to_f32(X)
# endif
#endif

#define get_group_id(D) ((size_t)libxstream_cpu_gid[D])
#define get_local_id(D) ((size_t)libxstream_cpu_lid[D])
#define get_local_size(D) ((size_t)libxstream_cpu_lsz[D])
#define get_global_id(D) \
  ((size_t)(libxstream_cpu_gid[D] * libxstream_cpu_lsz[D] + libxstream_cpu_lid[D]))

/* Publish one work-item; the third dimension is degenerate. Survives _end.h. */
#define LIBXSTREAM_CPU_WORKITEM(G0, G1, G2, L0, L1, S0, S1) do { \
  libxstream_cpu_gid[0] = (G0); \
  libxstream_cpu_gid[1] = (G1); \
  libxstream_cpu_gid[2] = (G2); \
  libxstream_cpu_lid[0] = (L0); \
  libxstream_cpu_lid[1] = (L1); \
  libxstream_cpu_lid[2] = 0; \
  libxstream_cpu_lsz[0] = (S0); \
  libxstream_cpu_lsz[1] = (S1); \
  libxstream_cpu_lsz[2] = 1; \
} while (0)

#define barrier(FLAGS) libxstream_cpu_barrier()

/**
 * Defined once even where a translation unit brackets several kernels, whereas
 * the spellings above are re-established by every bracket because
 * libxstream_cpu_end.h retires them.
 *
 * Work-item coordinates the launcher publishes before each kernel call.
 * Threadprivate because one thread runs one whole work-group (default) or one
 * work-item of a work-group (LIBXSTREAM_CPU_TEAM) at a time.
 */
#if !defined(LIBXSTREAM_CPU_STATE)
#define LIBXSTREAM_CPU_STATE
static int libxstream_cpu_gid[3];
static int libxstream_cpu_lid[3];
static int libxstream_cpu_lsz[3];
#if defined(_OPENMP)
# pragma omp threadprivate(libxstream_cpu_gid, libxstream_cpu_lid, libxstream_cpu_lsz)
#endif


/**
 * Work-group barrier. An orphaned OpenMP barrier binds to the team the launcher
 * opened for the work-group, which spells it without the _Pragma operator that
 * C89 lacks.
 */
static void libxstream_cpu_barrier(void)
{
#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM)
# pragma omp barrier
#endif
}

#endif /*LIBXSTREAM_CPU_STATE*/
