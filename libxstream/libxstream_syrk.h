/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_SYRK_H
#define LIBXSTREAM_SYRK_H

/**
 * Device of LIBXS' SYRK and SYR2K (libxs_syrk_device_t) on the active device: the panels of a plan
 * run there, local or distributed, while LIBXS keeps the algorithm (and MPI). A panel is computed by
 * the device BLAS where it serves the device (libxstream_blas.h, LIBXSTREAM_SYRK_BLAS=0 disables
 * it), else by an own kernel; a panel the device cannot take is declined, and LIBXS computes it on
 * the host. Operands are copied at the first panel of every update, or once if they are resident.
 */
#include "libxstream.h"
#include <libxs/libxs_syrk.h>


/** Context of the device: its stream, buffers, and kernels. */
typedef struct libxstream_syrk_t libxstream_syrk_t;

/** Creates the context on the active device (libxstream_init and an active device are required). */
LIBXSTREAM_API int libxstream_syrk_create(libxstream_syrk_t** ctx);

/** Releases the context, which must outlive every plan that uses it. NULL is accepted. */
LIBXSTREAM_API void libxstream_syrk_destroy(libxstream_syrk_t* ctx);

/** Populates the device of a shape (libxs_syrk_shape_t, before libxs_syrk_dispatch), whose context is ctx. */
LIBXSTREAM_API void libxstream_syrk_device(libxstream_syrk_t* ctx, libxs_syrk_device_t* device);

/**
 * Weights d (k elements of the plan's datatype) for C := alpha*op(A)*diag(d)*op(A)^T + beta*C, or
 * NULL. They are read at the first panel of every update, e.g., to vary them between calls while A
 * stays resident.
 */
LIBXSTREAM_API void libxstream_syrk_weights(libxstream_syrk_t* ctx, const void* d);

/**
 * Non-zero: the operands A and B stay on the device after they were copied, and the caller promises
 * that neither changes. Zero: they are copied for every update (default). Changing the setting lets
 * the next update copy them again.
 */
LIBXSTREAM_API void libxstream_syrk_resident(libxstream_syrk_t* ctx, int keep);

/**
 * What computed the panels so far: the device BLAS by its plugin's name (e.g., "mkl"), "kernel"
 * (SYR2K, a negative weight, or no device BLAS for the device or precision), "host" for panels the
 * device declined, a combination such as "mkl+host", or "none". Valid until the next call with ctx.
 */
LIBXSTREAM_API const char* libxstream_syrk_backend(libxstream_syrk_t* ctx);

/* header-only: include implementation after the declarations */
#if defined(LIBXSTREAM_SOURCE) && !defined(LIBXSTREAM_SOURCE_H)
# include "libxstream_source.h"
#endif

#endif /*LIBXSTREAM_SYRK_H*/
