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
 * One multiply.cl instance, hence no include guard. The includer states the host's
 * build string and FN, which names the instance; this compiles the kernel and a
 * launcher of the host's geometry, UPDIV(stack, BS) work-groups of WG lanes, and
 * declares the descriptor CAT(FN, _desc). Then it retires what the build string and
 * the kernel defined, so that the next instance starts from nothing but the device
 * part and the atomics, which a unit shares. LU is the exception it cannot reset:
 * libxstream_common.h settles the unroll hints once, and on the host they are
 * pragmas that do not change a result.
 */

#include "../samples/smm/kernels/multiply.cl"

static void CAT(FN, _launch)(SHIM_SMM_ELEM* c, const SHIM_SMM_ELEM* a, const SHIM_SMM_ELEM* b, const int* stack,
  int stack_size)
{
  const int ngroups = (stack_size + BS - 1) / BS;
  int g;
  LIBXSTREAM_CPU_GRID(ngroups, 1, 1);
  for (g = 0; g < ngroups; ++g) {
#pragma omp parallel num_threads(WG)
    {
      LIBXSTREAM_CPU_WORKITEM(g, 0, 0, omp_get_thread_num(), 0, WG, 1);
#if (1 < BS)
      FN(c, a, b, stack, 0 /*param_format*/, stack_size, BS);
#else
      FN(c, a, b, stack, 0 /*param_format*/);
#endif
    }
  }
}

static const shim_smm_desc_t CAT(FN, _desc) = { LIBXS_STRINGIFY(FN), SM, SN, SK, BS, WG, CAT(FN, _launch) };

#undef ACTIVE
#undef ADX
#undef AMK
#undef BDX
#undef BNK
#undef CDX
#undef CNM
#undef NBM
#undef NBN
#undef REPEAT
#undef SINT
#undef UM
#undef VM
#undef WRK
#undef TILE_M
#undef T
#undef FN
#undef LU
#undef WG
#undef SM
#undef SN
#undef SK
#undef BS
#undef BSC
#undef VL
#undef BM
#undef BN
#undef BK
#undef TRACK_B
#undef TRACK_C
#undef ATOMIC_INC_NZ
#undef AL
#undef SLM_P
#undef SLM_A
#undef REG_A
#undef SLM_B
#undef REG_B
#undef SLM_C
