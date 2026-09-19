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
 * Ozaki scheme 2 through the host shim (see shim_ozaki2.h): fp64, fractional
 * reconstruction per group fused into the GEMM (the default off a GPU). The
 * tables are the ones ozaki_crt_moduli_flags emitted for 16 moduli, captured
 * from a run rather than derived again here, so the kernels are held to what
 * the host actually builds.
 */
#define USE_DOUBLE 1
#define MANT_BITS 52
#define KGROUPS 0
#define OZAKI_U8 1
#define NMODULI 16
#define BIAS_PLUS_MANT 1075
#define MANT_TRUNC 0
#define POW2_PIDX 3
#define OZAKI_HIER 1
#define HIER_GS 4
#define OZAKI_HIER_L2 0
#define HIER_GPROD {1752116992u,1841455727u,1799186337u,1823610127u}
#define HIER_L2B {10528260474ul,10017478999ul,10252825788ul,10115508682ul}
#define HIER_L2INV {0u,828768696u,1255745875u,96929798u,0u,0u,1062200843u,1479311133u, \
  0u,0u,0u,1583419479u,0u,0u,0u,0u}
#define HIER_L1W {1486393088u,977311488u,1375895552u,1416750849u,917059625u,718415463u, \
  1308648740u,738787627u,1036568260u,1791260406u,46611045u,723932964u,847486864u,732629396u, \
  496480244u,1570623751u}
#define OZAKI_FRACCRT 2
#define OZ2G_FRAC_L 6
#define OZ2G_FRAC_K {179,111,128,207,125,87,140,67,140,226,5,68,112,92,52,149}
#define OZ2G_FRAC_CLIMB {1,54,152,223,61,224,1,73,83,158,59,45,1,146,15,180,157,14,1,0,0,0,0,0, \
  1,5,25,127,125,115,1,37,226,39,8,9,1,76,171,136,114,90,1,136,110,95,10,187,1,13,178,10,136,244, \
  1,32,180,112,198,124,1,83,144,148,143,64,1,131,201,119,171,43,1,15,239,1,15,239,1,30,46,243,179,251, \
  1,87,30,211,197,6,1,122,210,32,142,14}
#define OZ2G_FRAC_GMH {1.75211699200000000000e+09,1.84145572700000000000e+09, \
  1.79918633700000000000e+09,1.82361012700000000000e+09}

#include "shim_ozaki2.h"


int main(void)
{
  return shim_ozaki2("fp64");
}
