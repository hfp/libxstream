/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef OZAKI_OPENCL_H
#define OZAKI_OPENCL_H

#include "libxstream_opencl.h"

/* Ozaki flags */
#define OZAKI_TRIANGULAR 1
#define OZAKI_SYMMETRIZE 2


/* State for an Ozaki OpenCL session.
 * All tuning parameters are set by ozaki_init (0 = auto). */
typedef struct ozaki_context_t {
  cl_kernel kern_preprocess_a;
  cl_kernel kern_preprocess_b;
  cl_kernel kern_dotprod;
  int bm, bn, bk;  /* block dimensions (JIT-compiled into kernels) */
  int batch_k;     /* K sub-panels per kernel launch */
  int use_double;  /* 1: fp64, 0: fp32 */
  int use_bf16;    /* 1: bf16 Dekker slices, 0: int8 mantissa slices */
  int use_xmx;     /* 1: hardware matrix multiply (DPAS/XMX) */
  int sg;          /* sub-group size used for compilation */
  int nslices;
  int ozflags;     /* bitmask: OZAKI_TRIANGULAR | OZAKI_SYMMETRIZE */
  int oztrim;
  int verbosity;   /* 0: quiet, 1: info, 2+: debug */
} ozaki_context_t;


/* Function prototypes (public API).
 * Pass 0 for bm/bn/bk/nslices/batch_k to use auto defaults.
 * Pass -1 for ozflags to use the default (TRIANGULAR | SYMMETRIZE);
 * 0 disables both flags.  Auto defaults choose XMX-friendly sizes
 * when hardware support is detected.
 * use_bf16: 1 selects Dekker bf16 slices, 0 selects int8 mantissa slices.
 * Can be overridden by OZAKI_BF16 environment variable. */
int ozaki_init(ozaki_context_t* ctx, int bm, int bn, int bk,
               int use_double, int use_bf16, int nslices, int batch_k,
               int ozflags, int oztrim);
void ozaki_destroy(ozaki_context_t* ctx);
int ozaki_gemm(ozaki_context_t* ctx, libxstream_stream_t* stream,
               char transa, char transb,
               int M, int N, int K,
               double alpha, const void* a, int lda,
                             const void* b, int ldb,
               double beta,        void* c, int ldc);

#endif /* OZAKI_OPENCL_H */
