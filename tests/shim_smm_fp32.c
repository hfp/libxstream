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
 * The SMM kernels in fp32 through the host shim (see shim_smm.h),
 * one instance per build string smm_kernel.c or smm_trans.c
 * emitted for acc_bench, captured rather than composed: generated
 * by tests/shim_smm.py, whose header says how to regenerate. The
 * device part is what libxstream_opencl_flags_atomics emits for an
 * OpenCL 1.2 device without sub-groups.
 */

#define SHIM_SMM_ELEM float
#include "shim_smm.h"

/* a work-group is an OpenMP team, the barriers need its threads */
#if defined(_OPENMP)

#define LIBXSTREAM_CPU_TEAM 1
#define GPU 1
#define CONSTANT global
#define SG 0
#define INTEL 0
#define TAN 1
#define TA int
#define CMPXCHG atomic_cmpxchg
#define ATOMIC_ADD_GLOBAL(A, B) atomic_add_global_cmpxchg(A, B)
#define BARRIER(A) barrier(A)
#include <libxstream/opencl/libxstream_cpu_begin.h>

#define T float
#define WG 12
#define FN shim_ssmm_1
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 1
#include "shim_smm_instance.h"

#define T float
#define WG 14
#define FN shim_ssmm_2
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#include "shim_smm_instance.h"

#define T float
#define WG 14
#define FN shim_ssmm_3
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 2
#define TRACK_C
#define SLM_A 1
#include "shim_smm_instance.h"

#define T float
#define WG 160
#define FN shim_ssmm_4
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 160
#define FN shim_ssmm_5
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 17
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 160
#define FN shim_ssmm_6
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 176
#define FN shim_ssmm_7
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 3
#define BN 2
#define BK 6
#define TRACK_C
#define SLM_A 2
#define REG_B
#define SLM_C 2
#include "shim_smm_instance.h"

#define T float
#define WG 256
#define FN shim_ssmm_8
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 256
#define FN shim_ssmm_9
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 256
#define FN shim_ssmm_10
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 6
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_11
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 1
#define VL 8
#define BM 23
#define BN 1
#define BK 8
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_12
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 20
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_13
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_B
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_14
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_C
#define ATOMIC_INC_NZ
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_15
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_16
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_C
#define SLM_P
#define SLM_A 1
#define REG_B
#define SLM_C 1
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_17
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 23
#define BN 1
#define BK 17
#define TRACK_C
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_18
#define REPEAT 1
#define LU 0
#define SM 32
#define SN 32
#define SK 32
#define BS 1
#define VL 8
#define BM 32
#define BN 1
#define BK 8
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_19
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 30
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_20
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_B
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_21
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_C
#define ATOMIC_INC_NZ
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_22
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_23
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_C
#define SLM_P
#define SLM_A 2
#define REG_B
#define SLM_C 2
#include "shim_smm_instance.h"

#define T float
#define WG 32
#define FN shim_ssmm_24
#define REPEAT 1
#define LU -1
#define SM 32
#define SN 32
#define SK 32
#define BS 4
#define VL 8
#define BSC
#define BM 32
#define BN 1
#define BK 6
#define TRACK_C
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_25
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 16
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_26
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 1
#define VL 8
#define BM 4
#define BN 1
#define BK 4
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_27
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define REG_A
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_28
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 3
#define BN 2
#define BK 1
#define TRACK_C
#define REG_A
#define SLM_C 2
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_29
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_B
#define TRACK_C
#define REG_A
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_30
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define ATOMIC_INC_NZ
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_31
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_32
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define SLM_P
#define REG_A
#define SLM_C 2
#include "shim_smm_instance.h"

#define T float
#define WG 4
#define FN shim_ssmm_33
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 4
#define VL 8
#define BSC
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_34
#define REPEAT 1
#define LU 0
#define SM 5
#define SN 7
#define SK 9
#define BS 1
#define VL 8
#define BM 5
#define BN 1
#define BK 5
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_35
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 10
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_C
#define SLM_A 1
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_36
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_B
#define TRACK_C
#define SLM_A 1
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_37
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_C
#define ATOMIC_INC_NZ
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_38
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_C
#define SLM_A 1
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_39
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_C
#define SLM_P
#define SLM_A 1
#define SLM_C 1
#include "shim_smm_instance.h"

#define T float
#define WG 7
#define FN shim_ssmm_40
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 5
#define BN 1
#define BK 2
#define TRACK_C
#include "shim_smm_instance.h"

#define T float
#define WG 8
#define FN shim_ssmm_41
#define REPEAT 1
#define LU -1
#define SM 5
#define SN 7
#define SK 9
#define BS 4
#define VL 8
#define BSC
#define BM 3
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 1
#define SLM_C 1
#include "shim_smm_instance.h"

#define T float
#define WG 96
#define FN shim_ssmm_42
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 4
#define VL 8
#define BSC
#define BM 3
#define BN 2
#define BK 17
#define TRACK_C
#define SLM_A 1
#define REG_B
#define SLM_C 1
#include "shim_smm_instance.h"

#define INPLACE 0
#define FN shim_strans_1
#define SM 23
#define SN 23
#define WG 23
#define T float
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_2
#define SM 23
#define SN 23
#define WG 23
#define T float
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_3
#define SM 32
#define SN 32
#define WG 32
#define T float
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_4
#define SM 32
#define SN 32
#define WG 4
#define T float
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_5
#define SM 32
#define SN 32
#define WG 8
#define T float
#define BS 4
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_6
#define SM 4
#define SN 4
#define WG 4
#define T float
#define BS 4
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_7
#define SM 4
#define SN 4
#define WG 4
#define T float
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_8
#define SM 7
#define SN 9
#define WG 7
#define T float
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_9
#define SM 7
#define SN 9
#define WG 7
#define T float
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_10
#define SM 9
#define SN 7
#define WG 9
#define T float
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_strans_11
#define SM 9
#define SN 7
#define WG 9
#define T float
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#include <libxstream/opencl/libxstream_cpu_end.h>

static const shim_smm_desc_t* const shim_smm_set[] = {
  &shim_ssmm_1_desc,
  &shim_ssmm_2_desc,
  &shim_ssmm_3_desc,
  &shim_ssmm_4_desc,
  &shim_ssmm_5_desc,
  &shim_ssmm_6_desc,
  &shim_ssmm_7_desc,
  &shim_ssmm_8_desc,
  &shim_ssmm_9_desc,
  &shim_ssmm_10_desc,
  &shim_ssmm_11_desc,
  &shim_ssmm_12_desc,
  &shim_ssmm_13_desc,
  &shim_ssmm_14_desc,
  &shim_ssmm_15_desc,
  &shim_ssmm_16_desc,
  &shim_ssmm_17_desc,
  &shim_ssmm_18_desc,
  &shim_ssmm_19_desc,
  &shim_ssmm_20_desc,
  &shim_ssmm_21_desc,
  &shim_ssmm_22_desc,
  &shim_ssmm_23_desc,
  &shim_ssmm_24_desc,
  &shim_ssmm_25_desc,
  &shim_ssmm_26_desc,
  &shim_ssmm_27_desc,
  &shim_ssmm_28_desc,
  &shim_ssmm_29_desc,
  &shim_ssmm_30_desc,
  &shim_ssmm_31_desc,
  &shim_ssmm_32_desc,
  &shim_ssmm_33_desc,
  &shim_ssmm_34_desc,
  &shim_ssmm_35_desc,
  &shim_ssmm_36_desc,
  &shim_ssmm_37_desc,
  &shim_ssmm_38_desc,
  &shim_ssmm_39_desc,
  &shim_ssmm_40_desc,
  &shim_ssmm_41_desc,
  &shim_ssmm_42_desc
};
static const shim_trans_desc_t* const shim_trans_set[] = {
  &shim_strans_1_desc,
  &shim_strans_2_desc,
  &shim_strans_3_desc,
  &shim_strans_4_desc,
  &shim_strans_5_desc,
  &shim_strans_6_desc,
  &shim_strans_7_desc,
  &shim_strans_8_desc,
  &shim_strans_9_desc,
  &shim_strans_10_desc,
  &shim_strans_11_desc
};


int main(void)
{
  return shim_smm(shim_smm_set, (int)(sizeof(shim_smm_set) / sizeof(*shim_smm_set)), shim_trans_set,
    (int)(sizeof(shim_trans_set) / sizeof(*shim_trans_set)), "fp32");
}

#else

int main(void)
{
  /* stated rather than skipped, as a pass would be indistinguishable */
  printf("shim: smm fp32 NOT ATTEMPTED (work-group barriers need OpenMP)\n");
  return EXIT_SUCCESS;
}

#endif
