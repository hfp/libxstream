/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_BLAS_H
#define LIBXSTREAM_BLAS_H

/**
 * Device BLAS: a vendor's library on the device of a stream, provided by a plugin (libxstream_mkl:
 * oneMKL's SYCL interface on Intel GPUs) which is loaded at the first call. Matrices are column-
 * major and in device memory, the datatype is LIBXS_DATATYPE_F64 or LIBXS_DATATYPE_F32, and the
 * work is enqueued on the stream in order with the stream's other work (asynchronous). Operands must
 * stay valid until the stream is synchronized. Every function returns EXIT_SUCCESS, or EXIT_FAILURE
 * if there is no plugin for the stream's device (or it failed its self-test), and then the caller
 * computes otherwise. LIBXSTREAM_BLAS=0 disables the plugin, and a path selects it.
 */
#include "libxstream.h"
#include <libxs/libxs.h>


/**
 * Name of the plugin for the stream's device (e.g., "mkl"), or NULL if none is usable. xmem (if not
 * NULL) receives non-zero if the plugin takes memory of libxstream_mem_allocate. The oneMKL plugin
 * takes it where it is Intel USM or SVM (see LIBXSTREAM_USM), and is unavailable otherwise.
 */
LIBXSTREAM_API const char* libxstream_blas_query(libxstream_stream_t* stream, int* xmem);

/** C := alpha*op(A)*op(B) + beta*C, with op given by transa and transb ('N' or 'T'). */
LIBXSTREAM_API int libxstream_blas_gemm(libxstream_stream_t* stream, libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc);

/**
 * Triangle uplo ('U' or 'L') of C := alpha*op(A)*op(A)^T + beta*C (b is NULL), with trans 'N' (A is
 * n x k) or 'T' (A is k x n). A plugin may decline SYR2K (b is not NULL) by returning EXIT_FAILURE.
 */
LIBXSTREAM_API int libxstream_blas_syrk(libxstream_stream_t* stream, libxs_data_t datatype, char uplo, char trans,
  int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc);

/** C := A*diag(x) (side 'R', x has n elements) or C := diag(x)*A (side 'L', m elements), A and C m x n. */
LIBXSTREAM_API int libxstream_blas_dgmm(libxstream_stream_t* stream, libxs_data_t datatype, char side, int m, int n,
  const void* a, int lda, const void* x, int incx, void* c, int ldc);

#endif /*LIBXSTREAM_BLAS_H*/
