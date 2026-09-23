/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef DBM_OPENCL_H
#define DBM_OPENCL_H

/** Integers per task in the native format: m, n, k, offset_a, offset_b, offset_c. */
#define DBM_OPENCL_TASK_SIZE 6


/**
 * Enqueues C += alpha * A * B for a batch of small matrix products on the given
 * stream (libxstream_opencl_stream_t). A zero param_format selects the native
 * format with DBM_OPENCL_TASK_SIZE integers per task and zero-based offsets.
 * A non-zero param_format packs one shape for the whole batch (8 bits each for
 * m, n, and k) and params then holds three one-based offsets per task. The
 * shape is derived from params_host, whereas params is read by the device.
 */
int dbm_multiply_opencl_launch_kernel(void* stream, double alpha, int ntasks, int param_format,
  const int* params_host, const int* params, const double* pack_a_data,
  const double* pack_b_data, double* shard_c_data);

#endif /*DBM_OPENCL_H*/
