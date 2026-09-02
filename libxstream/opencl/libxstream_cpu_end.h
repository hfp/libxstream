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
 * LIBXSTREAM_CPU_WORKITEM, which the launcher needs.
 *
 * No include guard: a translation unit may bracket more than one kernel.
 */

#undef global
#undef private
#undef constant
#undef local
#undef kernel
#undef barrier
#undef restrict
#undef uchar
#undef ushort
#undef uint
#undef CLK_LOCAL_MEM_FENCE
#undef CLK_GLOBAL_MEM_FENCE
#undef get_group_id
#undef get_local_id
#undef get_local_size
#undef get_global_id
#undef UNROLL_FORCE
#undef UNROLL_AUTO
#undef SIMD_COLLAPSE
#undef __attribute__
