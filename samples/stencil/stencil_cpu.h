/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_CPU_H
#define STENCIL_CPU_H

#include <libxs/libxs_macros.h>
#include <stddef.h>

/**
 * OpenCL C to host C shim: lets an ordinary C compiler translate a kernel from
 * kernels/ for execution on the host CPU. Include this header from
 * stencil_cpu.c only, and only after every other header: it redefines OpenCL
 * keywords (kernel, global, local, barrier) as macros and drops __attribute__,
 * which would corrupt any header included afterwards.
 *
 * Everything is a macro rather than a function so the translated kernel keeps
 * the same shape it has on the device, with the work-item coordinates folded
 * into the surrounding expressions.
 *
 * Three work-group models exist; stencil_cpu.c selects one. This header covers
 * the two that need barrier() and local memory to differ:
 * STENCIL_CPU_TEAM=0 (default) A work-item runs to completion, so barrier() is
 *   a no-op and local memory is an ordinary automatic array. The kernel must
 *   then carry no cross-lane dependency, which STENCIL_CPU_LANES arranges by
 *   turning the lanes into loops inside the kernel, and which a 1x1
 *   work-group arranges by not having any other lane. Work-groups stay
 *   independent, so the launcher runs them in parallel.
 * STENCIL_CPU_TEAM=1 Lanes form an OpenMP team of one thread each and
 *   barrier() is an orphaned OpenMP barrier that binds to it. Local memory
 *   becomes static, hence one work-group at a time. This reproduces the
 *   on-device local memory tiling and is a debugging aid, not a fast path.
 *
 * Host long is 32-bit on LLP64 targets whereas OpenCL long is 64-bit, so a
 * host build indexes smaller grids there than the device does.
 */

#if defined(STENCIL_CPU_TEAM) && (0 != STENCIL_CPU_TEAM) && !defined(_OPENMP)
# error STENCIL_CPU_TEAM needs OpenMP to implement barrier()
#endif

/* Address spaces: a host build has only the one. */
#define global
#define private
#define constant const
#if defined(STENCIL_CPU_TEAM) && (0 != STENCIL_CPU_TEAM)
# define local static
#else
# define local
#endif

/* Kernels become ordinary functions that the launcher calls directly. */
#define kernel static

/**
 * Kernel attributes carry no meaning on the host. The translation unit holds
 * nothing but this shim, one kernel source and the launcher, so dropping every
 * attribute costs less than enumerating the OpenCL ones.
 */
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
 * The kernel's unroll hints carry over to the host: LIBXSTREAM_OPENCL_COMMON_H
 * keeps whatever is defined here rather than dropping the hints off-device.
 * Unrolling the short constant-trip loops matters more on the host than on the
 * device, because it is what leaves the lane loop as the innermost loop with a
 * straight-line body for the vectorizer to work on.
 */
#define UNROLL_FORCE(N) LIBXS_PRAGMA_UNROLL_N(N)
#define UNROLL_AUTO LIBXS_PRAGMA_UNROLL

/**
 * Vectorize a lane loop nest of N levels. The promise the directive makes holds:
 * a lane writes its own grid point and its own column of the slow-axis window,
 * and it reads the previous time step at the very point it overwrites, so no
 * dependence crosses lanes.
 */
#define SIMD_COLLAPSE(N) LIBXS_PRAGMA_SIMD_COLLAPSE(N)

/**
 * Work-item coordinates that the launcher publishes before each kernel call.
 * Threadprivate because one thread runs one whole work-group (default) or one
 * work-item of a work-group (STENCIL_CPU_TEAM) at a time.
 */
static int stencil_cpu_gid[3];
static int stencil_cpu_lid[3];
static int stencil_cpu_lsz[3];
#if defined(_OPENMP)
# pragma omp threadprivate(stencil_cpu_gid, stencil_cpu_lid, stencil_cpu_lsz)
#endif

#define get_group_id(D) ((size_t)stencil_cpu_gid[D])
#define get_local_id(D) ((size_t)stencil_cpu_lid[D])
#define get_local_size(D) ((size_t)stencil_cpu_lsz[D])
#define get_global_id(D) \
  ((size_t)(stencil_cpu_gid[D] * stencil_cpu_lsz[D] + stencil_cpu_lid[D]))

/**
 * Array geometry the JIT supplies as -D per launch and a host build cannot:
 * {sx, sy} strides and {lx, ly, lz} halo of the wavefield. Uniform for the whole
 * launch, hence not threadprivate. Bound below so that the layout macros in
 * stencil_common.cl read them instead of compile-time constants.
 */
static long stencil_cpu_stride[4];
static int stencil_cpu_halo[3];

#define STENCIL_P_SX stencil_cpu_stride[0]
#define STENCIL_P_SY stencil_cpu_stride[1]
#define STENCIL_P_LX stencil_cpu_halo[0]
#define STENCIL_P_LY stencil_cpu_halo[1]
#define STENCIL_P_LZ stencil_cpu_halo[2]
/* The velocity carries no halo, hence a stride of its own. */
#define STENCIL_V_SX stencil_cpu_stride[2]
#define STENCIL_V_SY stencil_cpu_stride[3]
#define STENCIL_V_LX 0
#define STENCIL_V_LY 0
#define STENCIL_V_LZ 0

/**
 * Work-group barrier. An orphaned OpenMP barrier binds to the team the
 * launcher opened for the work-group, which spells the barrier without the
 * _Pragma operator that C89 lacks.
 */
static void stencil_cpu_barrier(void)
{
#if defined(STENCIL_CPU_TEAM) && (0 != STENCIL_CPU_TEAM)
# pragma omp barrier
#endif
}

#define barrier(FLAGS) stencil_cpu_barrier()

#endif /*STENCIL_CPU_H*/
