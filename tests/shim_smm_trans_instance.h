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
 * One transpose.cl instance, hence no include guard: the kernel, a launcher of the
 * host's geometry (UPDIV(stack, BS) work-groups of WG lanes), and the descriptor
 * CAT(FN, _desc), after which the build string is retired. T is what the host moves
 * the element as, which for double is char8: the transpose never reads a value.
 */

#include "../samples/smm/kernels/transpose.cl"

static void CAT(FN, _launch)(const int* stack, void* matrix, int stack_size)
{
  const int ngroups = (stack_size + BS - 1) / BS;
  int g;
  LIBXSTREAM_CPU_GRID(ngroups, 1, 1);
  for (g = 0; g < ngroups; ++g) {
#pragma omp parallel num_threads(WG)
    {
      LIBXSTREAM_CPU_WORKITEM(g, 0, 0, omp_get_thread_num(), 0, WG, 1);
#if (1 < BS)
      FN(0 /*offset*/, stack, (T*)matrix, stack_size, BS);
#else
      FN(0 /*offset*/, stack, (T*)matrix);
#endif
    }
  }
}

static const shim_trans_desc_t CAT(FN, _desc) = { LIBXS_STRINGIFY(FN), SM, SN, BS, WG, INPLACE, CAT(FN, _launch) };

#undef SLM_PAD
#undef T
#undef FN
#undef WG
#undef SM
#undef SN
#undef BS
#undef INPLACE
