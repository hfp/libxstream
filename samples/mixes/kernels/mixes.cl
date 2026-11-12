/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Named mixes: each kernel is a known mixture of instructions, which another
 * kernel's counts are expressed in. T, TAN and ATOMIC_ADD_GLOBAL come from the
 * host (libxstream_opencl_flags_atomics), as for the SMM and DBM kernels. */
#include <opencl/libxstream_atomics.h>

#if !defined(TILE)
# define TILE 16
#endif


/* STREAM triad: two loads, one store, one FMA per element */
kernel void mixes_stream(global T* restrict a, global const T* restrict b, global const T* restrict c, T s, int n)
{
  const int i = (int)get_global_id(0);
  if (i < n) a[i] = b[i] + s * c[i];
}


/* GEMM, row-major n x n (n a multiple of TILE): tiles in local memory, one FMA per pair */
__attribute__((reqd_work_group_size(TILE, TILE, 1))) kernel void mixes_gemm(
  global T* restrict c, global const T* restrict a, global const T* restrict b, int n)
{
  local T la[TILE][TILE], lb[TILE][TILE];
  const int col = (int)get_global_id(0), row = (int)get_global_id(1);
  const int lx = (int)get_local_id(0), ly = (int)get_local_id(1);
  T sum = 0;
  int t = 0;
  for (; t < n; t += TILE) {
    int k = 0;
    la[ly][lx] = a[row * n + t + lx];
    lb[ly][lx] = b[(t + ly) * n + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (; k < TILE; ++k) sum = fma(la[ly][k], lb[k][lx], sum);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  c[row * n + col] = sum;
}


/* FP atomics: element i accumulates into slot i % m, hence n / m updates contend per slot */
kernel void mixes_atomics(global T* restrict c, global const T* restrict a, int n, int m)
{
  const int i = (int)get_global_id(0);
  if (i < n) ACCUMULATE(c + i % m, a[i]);
}
