/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_PML_H
#define STENCIL_PML_H

#include "stencil_opencl.h"

/**
 * Absorbing boundary setup, shared by the device and the host backend.
 *
 * A caller that models its own absorbing boundary passes eta and phi in and then
 * owns them; the profile is only built when it does not, and ctx->pml_owned
 * records which of the two happened.
 */
int stencil_pml_setup(stencil_context_t* ctx, int nx, int ny, int nz,
                      libxstream_opencl_mem_hint_t mem_hint);

#endif /*STENCIL_PML_H*/
