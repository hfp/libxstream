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
 * Closes libxstream_cpu_begin.h: retires the OpenCL spellings so that the rest
 * of the translation unit is ordinary C. Keeps the work-item state and
 * LIBXSTREAM_CPU_WORKITEM and LIBXSTREAM_CPU_GRID, which the launcher needs.
 *
 * What is ours is kept as well: the vector types and VEC_ZERO and VEC_ADD come
 * from a guarded header, so retiring them would leave the next bracket in the
 * translation unit without them and no way to get them back.
 *
 * No include guard: a translation unit may bracket more than one kernel.
 */

#undef global
#undef private
#undef constant
#undef local
#undef kernel
#undef inline
/* Restore the spelling LIBXS establishes where the compiler lacks inline. */
#if defined(LIBXS_INLINE_FIXUP)
# define inline LIBXS_INLINE_KEYWORD
#endif
LIBXS_PRAGMA_DIAG_POP()
#undef barrier
#undef restrict
#undef uchar
#undef ushort
#undef uint
#undef ulong
#undef as_uint
#undef as_int
#undef as_float
#undef as_ulong
#undef as_double
#undef clz
#undef mul_hi
#undef fma
#undef atomic_max
#undef atomic_or
#undef atomic_cmpxchg
#undef atom_cmpxchg
#undef min
#undef CLK_LOCAL_MEM_FENCE
#undef CLK_GLOBAL_MEM_FENCE
#undef get_group_id
#undef get_local_id
#undef get_local_size
#undef get_global_id
#undef get_num_groups
#undef get_global_size
#undef UNROLL_FORCE
#undef UNROLL_AUTO
#undef SIMD_COLLAPSE
#undef __attribute__
