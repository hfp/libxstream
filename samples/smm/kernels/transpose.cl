/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#if !defined(SLM_PAD)
#  define SLM_PAD 0
#endif

__attribute__((reqd_work_group_size(WG, 1, 1))) kernel void FN(
#if (1 < BS)
  int trs_offset, CONSTANT const int* restrict trs_stack, global T* restrict matrix,
  int stack_size, int bs)
{
#else
  int trs_offset, CONSTANT const int* restrict trs_stack, global T* restrict matrix)
{
#endif
  const int gid = get_group_id(0), idx = get_local_id(0);
  local T buf[SM][SN + SLM_PAD]; /* local memory buffer */
#if (1 < BS)
  const int batchsize = min(bs, stack_size - bs * gid);
  int i;
  for (i = 0; i < batchsize; ++i) {
    const int offset = trs_stack[trs_offset + bs * gid + i];
#else
  {
    const int offset = trs_stack[trs_offset + gid];
#endif
    /* matrix according to the index (transpose-stack) */
    global T* const restrict mat = matrix + offset;
#if (WG == SM)
    const int m = idx;
    int n;
    /* copy matrix elements into local buffer */
    for (n = 0; n < SN; ++n) buf[m][n] = mat[SM * n + m];
    barrier(CLK_LOCAL_MEM_FENCE);
    /* overwrite matrix elements (gather) */
    for (n = 0; n < SN; ++n) mat[SN * m + n] = buf[m][n];
#else
    T prv[SN]; /* private buffer */
    int m, n;
    /* copy matrix elements into local buffer */
    for (m = idx; m < SM; m += WG) {
      for (n = 0; n < SN; ++n) buf[m][n] = mat[SM * n + m];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (m = idx; m < SM; m += WG) {
      for (n = 0; n < SN; ++n) prv[n] = buf[m][n];
      /* overwrite matrix elements (gather) */
      for (n = 0; n < SN; ++n) mat[SN * m + n] = prv[n];
    }
#endif
#if (1 < BS)
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
  }
}
