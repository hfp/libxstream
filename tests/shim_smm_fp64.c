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
 * The SMM kernels in fp64 through the host shim (see shim_smm.h),
 * one instance per build string smm_kernel.c or smm_trans.c
 * emitted for acc_bench, captured rather than composed: generated
 * by tests/shim_smm.py, whose header says how to regenerate. The
 * device part is what libxstream_opencl_flags_atomics emits for an
 * OpenCL 1.2 device without sub-groups.
 */

#define SHIM_SMM_ELEM double
#include "shim_smm.h"

/* a work-group is an OpenMP team, the barriers need its threads */
#if defined(_OPENMP)

#define LIBXSTREAM_CPU_TEAM 1
#define GPU 1
#define CONSTANT global
#define SG 0
#define INTEL 0
#define TAN 2
#define TA long
#define CMPXCHG atom_cmpxchg
#define ATOMIC_ADD_GLOBAL(A, B) atomic_add_global_cmpxchg(A, B)
#define BARRIER(A) barrier(A)
#include <libxstream/opencl/libxstream_cpu_begin.h>

#define T double
#define WG 12
#define FN shim_dsmm_1
#define REPEAT 1
#define LU 0
#define SM 5
#define SN 7
#define SK 9
#define BS 1
#define VL 8
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 12
#define FN shim_dsmm_2
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

#define T double
#define WG 14
#define FN shim_dsmm_3
#define REPEAT 1
#define LU 0
#define SM 5
#define SN 7
#define SK 9
#define BS 1
#define VL 8
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 14
#define FN shim_dsmm_4
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

#define T double
#define WG 14
#define FN shim_dsmm_5
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

#define T double
#define WG 160
#define FN shim_dsmm_6
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 1
#define VL 8
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 160
#define FN shim_dsmm_7
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 1
#define VL 8
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 160
#define FN shim_dsmm_8
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

#define T double
#define WG 160
#define FN shim_dsmm_9
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

#define T double
#define WG 160
#define FN shim_dsmm_10
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

#define T double
#define WG 176
#define FN shim_dsmm_11
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

#define T double
#define WG 23
#define FN shim_dsmm_12
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

#define T double
#define WG 256
#define FN shim_dsmm_13
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

#define T double
#define WG 256
#define FN shim_dsmm_14
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

#define T double
#define WG 256
#define FN shim_dsmm_15
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

#define T double
#define WG 32
#define FN shim_dsmm_16
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
#define TRACK_B
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_17
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
#define ATOMIC_INC_NZ
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_18
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
#define REG_A
#define REG_B
#define SLM_C 1
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_19
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

#define T double
#define WG 32
#define FN shim_dsmm_20
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
#define SLM_B 1
#define SLM_C 1
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_21
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
#define SLM_P
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_22
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
#include "shim_smm_instance.h"

#define T double
#define WG 32
#define FN shim_dsmm_23
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

#define T double
#define WG 32
#define FN shim_dsmm_24
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

#define T double
#define WG 32
#define FN shim_dsmm_25
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

#define T double
#define WG 32
#define FN shim_dsmm_26
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

#define T double
#define WG 32
#define FN shim_dsmm_27
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

#define T double
#define WG 32
#define FN shim_dsmm_28
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

#define T double
#define WG 32
#define FN shim_dsmm_29
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

#define T double
#define WG 32
#define FN shim_dsmm_30
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

#define T double
#define WG 32
#define FN shim_dsmm_31
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

#define T double
#define WG 32
#define FN shim_dsmm_32
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

#define T double
#define WG 32
#define FN shim_dsmm_33
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

#define T double
#define WG 32
#define FN shim_dsmm_34
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

#define T double
#define WG 32
#define FN shim_dsmm_35
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

#define T double
#define WG 4
#define FN shim_dsmm_36
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

#define T double
#define WG 4
#define FN shim_dsmm_37
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 1
#define VL 8
#define BM 2
#define BN 2
#define BK 2
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_38
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 1
#define VL 8
#define BM 3
#define BN 2
#define BK 4
#define TRACK_C
#define ATOMIC_INC_NZ
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_39
#define REPEAT 1
#define LU 0
#define SM 4
#define SN 4
#define SK 4
#define BS 1
#define VL 8
#define BM 4
#define BN 1
#define BK 1
#define TRACK_C
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_40
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
#define TRACK_B
#define TRACK_C
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_41
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
#define ATOMIC_INC_NZ
#define SLM_A 2
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_42
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
#define REG_A
#define REG_B
#define SLM_C 2
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_43
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

#define T double
#define WG 4
#define FN shim_dsmm_44
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
#define SLM_B 2
#define SLM_C 2
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_45
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
#define SLM_P
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_46
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
#include "shim_smm_instance.h"

#define T double
#define WG 4
#define FN shim_dsmm_47
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

#define T double
#define WG 4
#define FN shim_dsmm_48
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

#define T double
#define WG 4
#define FN shim_dsmm_49
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

#define T double
#define WG 4
#define FN shim_dsmm_50
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

#define T double
#define WG 4
#define FN shim_dsmm_51
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

#define T double
#define WG 4
#define FN shim_dsmm_52
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

#define T double
#define WG 4
#define FN shim_dsmm_53
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

#define T double
#define WG 7
#define FN shim_dsmm_54
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
#define TRACK_B
#define TRACK_C
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_55
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
#define ATOMIC_INC_NZ
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_56
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
#define REG_A
#define REG_B
#define SLM_C 1
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_57
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

#define T double
#define WG 7
#define FN shim_dsmm_58
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
#define SLM_B 1
#define SLM_C 1
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_59
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
#define SLM_P
#define REG_A
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_60
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
#include "shim_smm_instance.h"

#define T double
#define WG 7
#define FN shim_dsmm_61
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

#define T double
#define WG 7
#define FN shim_dsmm_62
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

#define T double
#define WG 7
#define FN shim_dsmm_63
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

#define T double
#define WG 7
#define FN shim_dsmm_64
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

#define T double
#define WG 7
#define FN shim_dsmm_65
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

#define T double
#define WG 7
#define FN shim_dsmm_66
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

#define T double
#define WG 8
#define FN shim_dsmm_67
#define REPEAT 1
#define LU 0
#define SM 5
#define SN 7
#define SK 9
#define BS 1
#define VL 8
#define BM 3
#define BN 2
#define BK 5
#define TRACK_C
#define ATOMIC_INC_NZ
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 8
#define FN shim_dsmm_68
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

#define T double
#define WG 96
#define FN shim_dsmm_69
#define REPEAT 1
#define LU 0
#define SM 23
#define SN 23
#define SK 23
#define BS 1
#define VL 8
#define BM 3
#define BN 2
#define BK 8
#define TRACK_C
#define ATOMIC_INC_NZ
#define SLM_A 1
#define REG_B
#include "shim_smm_instance.h"

#define T double
#define WG 96
#define FN shim_dsmm_70
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
#define FN shim_dtrans_1
#define SM 23
#define SN 23
#define WG 23
#define T char8
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_2
#define SM 23
#define SN 23
#define WG 23
#define T char8
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_3
#define SM 32
#define SN 32
#define WG 32
#define T char8
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_4
#define SM 32
#define SN 32
#define WG 4
#define T char8
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_5
#define SM 32
#define SN 32
#define WG 8
#define T char8
#define BS 4
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_6
#define SM 4
#define SN 4
#define WG 4
#define T char8
#define BS 4
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_7
#define SM 4
#define SN 4
#define WG 4
#define T char8
#define BS 8
#define SLM_PAD 1
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_8
#define SM 7
#define SN 9
#define WG 7
#define T char8
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_9
#define SM 7
#define SN 9
#define WG 7
#define T char8
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_10
#define SM 9
#define SN 7
#define WG 9
#define T char8
#define BS 4
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 0
#define FN shim_dtrans_11
#define SM 9
#define SN 7
#define WG 9
#define T char8
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 1
#define FN shim_dtrans_12
#define SM 23
#define SN 23
#define WG 23
#define T char8
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#define INPLACE 1
#define FN shim_dtrans_13
#define SM 4
#define SN 4
#define WG 4
#define T char8
#define BS 8
#define SLM_PAD 0
#include "shim_smm_trans_instance.h"

#include <libxstream/opencl/libxstream_cpu_end.h>

static const shim_smm_desc_t* const shim_smm_set[] = {
  &shim_dsmm_1_desc,
  &shim_dsmm_2_desc,
  &shim_dsmm_3_desc,
  &shim_dsmm_4_desc,
  &shim_dsmm_5_desc,
  &shim_dsmm_6_desc,
  &shim_dsmm_7_desc,
  &shim_dsmm_8_desc,
  &shim_dsmm_9_desc,
  &shim_dsmm_10_desc,
  &shim_dsmm_11_desc,
  &shim_dsmm_12_desc,
  &shim_dsmm_13_desc,
  &shim_dsmm_14_desc,
  &shim_dsmm_15_desc,
  &shim_dsmm_16_desc,
  &shim_dsmm_17_desc,
  &shim_dsmm_18_desc,
  &shim_dsmm_19_desc,
  &shim_dsmm_20_desc,
  &shim_dsmm_21_desc,
  &shim_dsmm_22_desc,
  &shim_dsmm_23_desc,
  &shim_dsmm_24_desc,
  &shim_dsmm_25_desc,
  &shim_dsmm_26_desc,
  &shim_dsmm_27_desc,
  &shim_dsmm_28_desc,
  &shim_dsmm_29_desc,
  &shim_dsmm_30_desc,
  &shim_dsmm_31_desc,
  &shim_dsmm_32_desc,
  &shim_dsmm_33_desc,
  &shim_dsmm_34_desc,
  &shim_dsmm_35_desc,
  &shim_dsmm_36_desc,
  &shim_dsmm_37_desc,
  &shim_dsmm_38_desc,
  &shim_dsmm_39_desc,
  &shim_dsmm_40_desc,
  &shim_dsmm_41_desc,
  &shim_dsmm_42_desc,
  &shim_dsmm_43_desc,
  &shim_dsmm_44_desc,
  &shim_dsmm_45_desc,
  &shim_dsmm_46_desc,
  &shim_dsmm_47_desc,
  &shim_dsmm_48_desc,
  &shim_dsmm_49_desc,
  &shim_dsmm_50_desc,
  &shim_dsmm_51_desc,
  &shim_dsmm_52_desc,
  &shim_dsmm_53_desc,
  &shim_dsmm_54_desc,
  &shim_dsmm_55_desc,
  &shim_dsmm_56_desc,
  &shim_dsmm_57_desc,
  &shim_dsmm_58_desc,
  &shim_dsmm_59_desc,
  &shim_dsmm_60_desc,
  &shim_dsmm_61_desc,
  &shim_dsmm_62_desc,
  &shim_dsmm_63_desc,
  &shim_dsmm_64_desc,
  &shim_dsmm_65_desc,
  &shim_dsmm_66_desc,
  &shim_dsmm_67_desc,
  &shim_dsmm_68_desc,
  &shim_dsmm_69_desc,
  &shim_dsmm_70_desc
};
static const shim_trans_desc_t* const shim_trans_set[] = {
  &shim_dtrans_1_desc,
  &shim_dtrans_2_desc,
  &shim_dtrans_3_desc,
  &shim_dtrans_4_desc,
  &shim_dtrans_5_desc,
  &shim_dtrans_6_desc,
  &shim_dtrans_7_desc,
  &shim_dtrans_8_desc,
  &shim_dtrans_9_desc,
  &shim_dtrans_10_desc,
  &shim_dtrans_11_desc,
  &shim_dtrans_12_desc,
  &shim_dtrans_13_desc
};


int main(void)
{
  return shim_smm(shim_smm_set, (int)(sizeof(shim_smm_set) / sizeof(*shim_smm_set)), shim_trans_set,
    (int)(sizeof(shim_trans_set) / sizeof(*shim_trans_set)), "fp64");
}

#else

int main(void)
{
  /* stated rather than skipped, as a pass would be indistinguishable */
  printf("shim: smm fp64 NOT ATTEMPTED (work-group barriers need OpenMP)\n");
  return EXIT_SUCCESS;
}

#endif
