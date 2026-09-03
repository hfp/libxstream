/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * One instance of the FP32 kernel, selected by STENCIL_CPU_IDX and included
 * once per instance, hence no include guard. The JIT specializes the device
 * kernel per launch; a host build compiles the combinations it may be asked
 * for and picks between them at run time.
 *
 * The index encodes three axes: the boundary treatment (which the grid decides,
 * see stencil_configure), the wavefield storage format and the operator radius.
 *
 *   bit 0    0 = clamp the gather, 1 = read into the halo
 *   bits 1-2 0 = FP32, 1 = BF16 one limb, 2 = BF16 two limbs, 3 = FP16
 *   bits 3+  0 = full radius (direct), 1-3 = that radius (compact)
 *
 * With STENCIL_CPU_TABLE defined the include emits a function pointer instead
 * of the instance, which is how the dispatch table stays in step with the set.
 */

#if !defined(STENCIL_CPU_IDX)
# error STENCIL_CPU_IDX selects the instance
#endif

#define STENCIL_CPU_KERNEL STENCIL_CPU_CAT(stencil_cpu_kernel_, STENCIL_CPU_IDX)

#if defined(STENCIL_CPU_TABLE)
STENCIL_CPU_KERNEL,
#else

#if (0 != (STENCIL_CPU_IDX % 2))
# define STENCIL_PADDED 1
#endif
#if (1 == ((STENCIL_CPU_IDX / 2) % 4))
# define STENCIL_BF16S 1
#elif (2 == ((STENCIL_CPU_IDX / 2) % 4))
# define STENCIL_BF16S 2
#elif (3 == ((STENCIL_CPU_IDX / 2) % 4))
# define STENCIL_FP16S 1
#endif
#if (1 == (STENCIL_CPU_IDX / 8))
# define RADIUS 1
#elif (2 == (STENCIL_CPU_IDX / 8))
# define RADIUS 2
#elif (3 == (STENCIL_CPU_IDX / 8))
# define RADIUS 3
#else
# define RADIUS STENCIL_RADIUS
#endif

/**
 * Everything the kernel sources define textually is redefined identically by
 * the next instance, which is what keeps the repeated include legal. Retired
 * here are the macros whose text itself depends on the axes above, plus the
 * guard of the file that derives them.
 */
#undef STENCIL_COMMON_CL
#undef STENCIL_CLAMP_COORD
#undef STENCIL_P_ELEM
#undef STENCIL_LOAD_P
#undef STENCIL_LOAD_P_BITS
#undef STENCIL_STORE_P
#undef STENCIL_BF16S_NDIGITS
#undef STENCIL_TTI_X_NDIGITS
#undef STENCIL_GATHER_STORE
#undef STENCIL_GATHER_STORE_ZERO
#undef STENCIL_GATHER_STORE_BF16S

#define stencil_apply_direct STENCIL_CPU_KERNEL
#include <libxstream/opencl/libxstream_cpu_begin.h>
#include "kernels/stencil_fp32.cl"
#include <libxstream/opencl/libxstream_cpu_end.h>
#undef stencil_apply_direct

#undef RADIUS
#undef STENCIL_PADDED
#undef STENCIL_BF16S
#undef STENCIL_FP16S
#endif /*STENCIL_CPU_TABLE*/

#undef STENCIL_CPU_KERNEL
#undef STENCIL_CPU_IDX
