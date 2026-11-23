/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_BLAS_PLUGIN_H
#define LIBXSTREAM_BLAS_PLUGIN_H

/* version of the table: a plugin refuses any other */
#define LIBXSTREAM_BLAS_PLUGIN_VERSION 1
/* the one function a plugin exports */
#define LIBXSTREAM_BLAS_PLUGIN_ENTRY "libxstream_blas_plugin"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * What a plugin of the device BLAS provides (see libxstream_blas.h): context and queue are the
 * OpenCL handles of a stream, f64 selects double (else single) precision, and every function returns
 * zero on success. Functions may be called concurrently, and they enqueue without waiting. A plugin
 * is never unloaded: its runtime may be gone by the time the process exits.
 */
typedef struct libxstream_blas_plugin_t {
  int version;
  /** Name, and whether memory of libxstream_mem_allocate (Intel USM) is taken. */
  const char* name;
  int xmem;
  int (*gemm)(void* context, void* queue, int f64, char transa, char transb, int m, int n, int k, double alpha,
    const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc);
  int (*syrk)(void* context, void* queue, int f64, char uplo, char trans, int n, int k, double alpha, const void* a,
    int lda, const void* b, int ldb, double beta, void* c, int ldc);
  int (*dgmm)(void* context, void* queue, int f64, char side, int m, int n, const void* a, int lda, const void* x,
    int incx, void* c, int ldc);
} libxstream_blas_plugin_t;

/** Fills the table (version set by the caller) and returns zero, or non-zero to refuse. */
typedef int (*libxstream_blas_plugin_fn)(libxstream_blas_plugin_t* table);

#if defined(__cplusplus)
}
#endif

#endif /*LIBXSTREAM_BLAS_PLUGIN_H*/
