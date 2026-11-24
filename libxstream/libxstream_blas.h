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
 * oneMKL's SYCL interface on Intel devices) which is loaded at the first call. Matrices are column-
 * major and in device memory, the datatype is LIBXS_DATATYPE_F64 or LIBXS_DATATYPE_F32, and the
 * work is enqueued on the stream in order with the stream's other work (asynchronous). Operands must
 * stay valid until the stream is synchronized. Every function returns EXIT_SUCCESS, or EXIT_FAILURE
 * if no plugin serves the stream's device (see libxstream_blas_query) or the precision, and then the
 * caller computes otherwise. LIBXSTREAM_BLAS=0 disables the plugin, and a path selects it.
 */
#include "libxstream.h"
#include <libxs/libxs.h>


/**
 * Name of the plugin which serves the stream's device with memory of libxstream_mem_allocate (e.g.,
 * "mkl"), or NULL. Asked once per context: the plugin's own self-test, and a GEMM on such memory at an
 * offset against the host. The oneMKL plugin serves an Intel device with Intel USM, with buffers where
 * memory is unified (LIBXSTREAM_USM=0), and with SVM on a CPU.
 */
LIBXSTREAM_API const char* libxstream_blas_query(libxstream_stream_t* stream);

/** C := alpha*op(A)*op(B) + beta*C, with op given by transa and transb ('N' or 'T'). */
LIBXSTREAM_API int libxstream_blas_gemm(libxstream_stream_t* stream, libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc);

/**
 * A batch of size GEMMs as above, with A, B, and C given by host arrays of device pointers (copied
 * before the call returns). The entries may run concurrently, hence no two of one call may update
 * the same C (successive calls are ordered by the stream).
 */
LIBXSTREAM_API int libxstream_blas_gemm_batch(libxstream_stream_t* stream, libxs_data_t datatype, char transa,
  char transb, int m, int n, int k, double alpha, const void* const* a, int lda, const void* const* b, int ldb,
  double beta, void* const* c, int ldc, int size);

/**
 * Triangle uplo ('U' or 'L') of C := alpha*op(A)*op(A)^T + beta*C (b is NULL), with trans 'N' (A is
 * n x k) or 'T' (A is k x n). A plugin may decline SYR2K (b is not NULL) by returning EXIT_FAILURE.
 */
LIBXSTREAM_API int libxstream_blas_syrk(libxstream_stream_t* stream, libxs_data_t datatype, char uplo, char trans,
  int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc);

/** C := A*diag(x) (side 'R', x has n elements) or C := diag(x)*A (side 'L', m elements), A and C m x n. */
LIBXSTREAM_API int libxstream_blas_dgmm(libxstream_stream_t* stream, libxs_data_t datatype, char side, int m, int n,
  const void* a, int lda, const void* x, int incx, void* c, int ldc);

/* header-only: include implementation after the declarations */
#if defined(LIBXSTREAM_SOURCE) && !defined(LIBXSTREAM_SOURCE_H)
# include "libxstream_source.h"
#endif

#endif /*LIBXSTREAM_BLAS_H*/
