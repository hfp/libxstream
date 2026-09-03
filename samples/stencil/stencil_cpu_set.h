/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * The set of instances, included twice by stencil_cpu.c: once to emit the
 * kernels and once (with STENCIL_CPU_TABLE) to emit the dispatch table, so the
 * two cannot drift. stencil_cpu_instance.h carries the body of one of them and
 * documents the encoding of the index.
 *
 * Written out rather than generated because the preprocessor cannot include a
 * file from within a macro expansion.
 */
#define STENCIL_CPU_IDX 0
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 1
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 2
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 3
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 4
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 5
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 6
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 7
#include "stencil_cpu_instance.h"

#if (8 < STENCIL_CPU_NINST) /* compact radii */
#define STENCIL_CPU_IDX 8
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 9
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 10
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 11
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 12
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 13
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 14
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 15
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 16
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 17
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 18
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 19
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 20
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 21
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 22
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 23
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 24
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 25
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 26
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 27
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 28
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 29
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 30
#include "stencil_cpu_instance.h"
#define STENCIL_CPU_IDX 31
#include "stencil_cpu_instance.h"
#endif
