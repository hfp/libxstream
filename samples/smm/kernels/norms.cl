/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/* squared Frobenius norm per block, as DBCSR computes it on the host (DDOT) */
__attribute__((reqd_work_group_size(WG, 1, 1))) kernel void FN(global const double* restrict mat,
  global const int* restrict offsets, global const int* restrict nelems, global float* restrict norms)
{
  local double buf[WG]; /* WG is a power of two */
  const int gid = get_group_id(0), idx = get_local_id(0);
  const int offset = offsets[gid], nelem = nelems[gid];
  double sum = 0;
  int i;
  for (i = idx; i < nelem; i += WG) {
    const double d = mat[offset + i];
    sum += d * d;
  }
  buf[idx] = sum;
  for (i = WG / 2; 0 < i; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < i) buf[idx] += buf[idx + i];
  }
  if (0 == idx) norms[gid] = (float)buf[0];
}
