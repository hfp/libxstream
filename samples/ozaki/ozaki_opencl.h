/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#ifndef OZAKI_OPENCL_H
#define OZAKI_OPENCL_H

#if !defined(CL_TARGET_OPENCL_VERSION)
# define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <stddef.h>
#include <stdint.h>

/* Block dimensions matching the OpenCL kernel defaults */
#if !defined(OZAKI_BM)
# define OZAKI_BM 16
#endif
#if !defined(OZAKI_BN)
# define OZAKI_BN 16
#endif
#if !defined(OZAKI_BK)
# define OZAKI_BK 16
#endif

/* Number of mantissa slices (default for double precision) */
#if !defined(OZAKI_NSLICES)
# define OZAKI_NSLICES 8
#endif

/* Batch K sub-panels to reduce kernel launch overhead */
#if !defined(OZAKI_BATCH_K)
# define OZAKI_BATCH_K 4
#endif

/* Ozaki flags */
#define OZAKI_TRIANGULAR 1
#define OZAKI_SYMMETRIZE 2

/* State for an Ozaki OpenCL session */
typedef struct ozaki_context_t {
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kern_preprocess_a;
  cl_kernel kern_preprocess_b;
  cl_kernel kern_dotprod;
  cl_device_id device;
  int use_double;  /* 1: fp64, 0: fp32 */
  int nslices;
  int ozflags;     /* bitmask: OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE */
  int oztrim;
  int gpu;         /* 1: GPU device, 0: other */
  int verbosity;   /* 0: quiet, 1: info, 2+: debug */
  size_t wgsize;   /* max work-group size */
  size_t sgsize;   /* sub-group size (0 if not applicable) */
} ozaki_context_t;

#endif /* OZAKI_OPENCL_H */
