/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SYRK_OPENCL_H
#define SYRK_OPENCL_H

#include <libxstream/libxstream_opencl.h>
#include <libxs/libxs_syrk.h>


/**
 * Device of LIBXS' SYRK and SYR2K (libxs_syrk_device_t) on the active OpenCL device: the panels of
 * a plan run there, local or distributed, while LIBXS keeps the algorithm (and MPI). Operands are
 * copied at the first panel of every update, or once if they are resident (syrk_opencl_resident).
 */
typedef struct syrk_opencl_t syrk_opencl_t;


/** Creates the context on the active device (libxstream_init and an active device are required). */
int syrk_opencl_create(syrk_opencl_t** ctx);

/** Releases the context, which must outlive every plan that uses it. NULL is accepted. */
void syrk_opencl_destroy(syrk_opencl_t* ctx);

/** Populates the device of a shape (libxs_syrk_shape_t), whose context is ctx. */
void syrk_opencl_device(syrk_opencl_t* ctx, libxs_syrk_device_t* device);

/**
 * Weights d (k elements of the plan's datatype) for C := alpha*op(A)*diag(d)*op(A)^T + beta*C, or
 * NULL. They are read at the first panel of every update, e.g., to vary them between calls while A
 * stays resident.
 */
void syrk_opencl_weights(syrk_opencl_t* ctx, const void* d);

/**
 * Non-zero: the operands A and B stay on the device after they were copied, and the caller promises
 * that neither changes. Zero: they are copied for every update (default). Changing the setting lets
 * the next update copy them again.
 */
void syrk_opencl_resident(syrk_opencl_t* ctx, int keep);

#endif /*SYRK_OPENCL_H*/
