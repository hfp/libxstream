/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SMM_SYCL_H
#define SMM_SYCL_H

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * C += A * op(B) for every entry of a homogeneous stack (three one-based element
 * offsets per entry, DBCSR's accelerator format) by oneMKL's SYCL interface on a
 * SYCL queue made of the given OpenCL context and command queue. The matrices are
 * Intel USM in that context, op(B) is B transposed if transb is non-zero, and the
 * call returns once the work is done. Returns 0 on success, and -1 if the work was
 * not done, which leaves C untouched.
 */
int opencl_libsmm_sycl_dgemm(void* cl_context, void* cl_command_queue, const int* stack, int stack_size, int m, int n, int k,
  int transb, const double* a, const double* b, double* c);

/** Releases the SYCL context made of the OpenCL context (repeated calls are allowed). */
void opencl_libsmm_sycl_finalize(void);

#if defined(__cplusplus)
}
#endif

#endif /*SMM_SYCL_H*/
