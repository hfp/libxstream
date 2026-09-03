/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_WEIGHTS_H
#define STENCIL_WEIGHTS_H

#include "stencil_opencl.h"

/**
 * Operator weights, shared by the device and the host backend: the method
 * decides the radius the kernel is built for, and the weights that go with it.
 */

/* Sub-steps and per-step radius of a method; STENCIL_RADIUS_FIT tunes the fit. */
int stencil_method_params(stencil_method_t method, int* k_steps, int* r_per_step);

/* Plain finite-difference weight at distance dist. */
double stencil_fd_weight(const double* fd_weights, int radius, int dist);

/* Weight of a compact operator whose cascade reproduces the wide one. */
double stencil_compact_weight(int radius, int dist, double inv_h2);

/* fit_method: 0 = L2, 1 = Ricker, 2 = minimax over points-per-wavelength ppw. */
void stencil_fit_coeffs(int radius, double ppw, int fit_method, double* coeffs);
double stencil_fit_weight(int radius, int dist, double inv_h2,
                          double ppw, int fit_method);

#endif /*STENCIL_WEIGHTS_H*/
