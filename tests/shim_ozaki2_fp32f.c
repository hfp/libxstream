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
 * Ozaki scheme 2 through the host shim (see shim_ozaki2.h): fp32, fractional
 * reconstruction per group fused into the GEMM (the default off a GPU). The
 * tables are the ones ozaki_crt_moduli_flags emitted for 9 moduli, captured
 * from a run rather than derived again here, so the kernels are held to what
 * the host actually builds.
 */
#define USE_DOUBLE 0
#define MANT_BITS 23
#define KGROUPS 0
#define OZAKI_U8 1
#define NMODULI 9
#define BIAS_PLUS_MANT 150
#define MANT_TRUNC 0
#define POW2_PIDX 3
#define OZAKI_HIER 1
#define HIER_GS 3
#define OZAKI_HIER_L2 0
#define HIER_GPROD {6844207u,14329088u,7994457u}
#define HIER_L2B {2695234681491ul,1287363443766ul,2307441777935ul}
#define HIER_L2INV {0u,13850575u,3083056u,0u,0u,456488u,0u,0u,0u}
#define HIER_L1W {1200169u,5434094u,209945u,2518785u,11360512u,449792u,7953876u,4547745u,3487294u}
#define OZAKI_FRACCRT 2
#define OZ2G_FRAC_L 5
#define OZ2G_FRAC_K {37,158,5,45,199,7,196,95,106}
#define OZ2G_FRAC_CLIMB {1,54,152,223,61,1,73,83,158,59,1,146,15,180,157,1,0,0,0,0,1,5,25,127,125, \
  1,37,226,39,8,1,76,171,136,114,1,136,110,95,10,1,13,178,10,136}
#define OZ2G_FRAC_GMH {6.84420700000000000000e+06,1.43290880000000000000e+07,7.99445700000000000000e+06}

#include "shim_ozaki2.h"


int main(void)
{
  return shim_ozaki2("fp32");
}
