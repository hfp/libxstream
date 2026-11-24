/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#define OPENCL_KERNELS_SOURCE_LIBXSTREAM_SYRK opencl_kernels_source_libxstream_syrk
#define OPENCL_KERNELS_STRING_LIBXSTREAM_SYRK \
  "#if !defined(T)\n" \
  "#define T double\n" \
  "#endif\n" \
  "#if !defined(BS)\n" \
  "#define BS 16\n" \
  "#endif\n" \
  "#if !defined(RB)\n" \
  "#define RB 4\n" \
  "#endif\n" \
  "#if !defined(BK)\n" \
  "#define BK 16\n" \
  "#endif\n" \
  "#define BT (BS * RB)\n" \
  "#define SYRK_OP(X,LD,TRANS,I,P) ((0 == (TRANS)) ? (X)[(I) + (size_t)(P) * (LD)] : (X)[(P) + (size_t)(I) * (LD)])\n" \
  "__attribute__((reqd_work_group_size(BS, BS, 1))) kernel void syrk_panel(global T* restrict out,\n" \
  "  global const T* restrict a, global const T* restrict b, global const T* restrict d, int m, int w, int r0, int c0,\n" \
  "  int k, int lda, int ldb, int trans, int twok, int hasd, T alpha, int upper)\n" \
  "{\n" \
  "  local T lx[BK][BT + 1], ly[BK][BT + 1];\n" \
  "  const int tx = (int)get_local_id(0), ty = (int)get_local_id(1);\n" \
  "  const int i0 = (int)get_group_id(0) * BT, j0 = (int)get_group_id(1) * BT;\n" \
  "  const int tid = ty * BS + tx;\n" \
  "  T acc[RB][RB];\n" \
  "  int term = 0, u, v;\n" \
  "  if (0 != upper ? (r0 + i0 > c0 + j0 + BT - 1) : (r0 + i0 + BT - 1 < c0 + j0)) return;\n" \
  "  for (u = 0; u < RB; ++u) {\n" \
  "    for (v = 0; v < RB; ++v) acc[u][v] = 0;\n" \
  "  }\n" \
  "  for (; term <= twok; ++term) {\n" \
  "    global const T* const x = (0 == term ? a : b);\n" \
  "    global const T* const y = (0 == term ? b : a);\n" \
  "    const int ldx = (0 == term ? lda : ldb), ldy = (0 == term ? ldb : lda);\n" \
  "    int p0 = 0;\n" \
  "    for (; p0 < k; p0 += BK) {\n" \
  "      int e = tid;\n" \
  "      for (; e < BK * BT; e += BS * BS) {\n" \
  "        const int pp = e / BT, q = e % BT, p = p0 + pp;\n" \
  "        const int gi = i0 + q, gj = j0 + q;\n" \
  "        T xv = 0, yv = 0;\n" \
  "        if (p < k) {\n" \
  "          if (gi < m) {\n" \
  "            xv = SYRK_OP(x, ldx, trans, r0 + gi, p);\n" \
  "            if (0 != hasd) xv *= d[p];\n" \
  "          }\n" \
  "          if (gj < w) yv = SYRK_OP(y, ldy, trans, c0 + gj, p);\n" \
  "        }\n" \
  "        lx[pp][q] = xv;\n" \
  "        ly[pp][q] = yv;\n" \
  "      }\n" \
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" \
  "      { int pp = 0;\n" \
  "        for (; pp < BK; ++pp) {\n" \
  "          T xr[RB], yr[RB];\n" \
  "          for (u = 0; u < RB; ++u) xr[u] = lx[pp][tx + u * BS];\n" \
  "          for (v = 0; v < RB; ++v) yr[v] = ly[pp][ty + v * BS];\n" \
  "          for (u = 0; u < RB; ++u) {\n" \
  "            for (v = 0; v < RB; ++v) acc[u][v] = fma(xr[u], yr[v], acc[u][v]);\n" \
  "          }\n" \
  "        }\n" \
  "      }\n" \
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" \
  "    }\n" \
  "  }\n" \
  "  for (v = 0; v < RB; ++v) {\n" \
  "    const int j = j0 + ty + v * BS;\n" \
  "    if (j < w) {\n" \
  "      for (u = 0; u < RB; ++u) {\n" \
  "        const int i = i0 + tx + u * BS;\n" \
  "        if (i < m) out[i + (size_t)j * m] = alpha * acc[u][v];\n" \
  "      }\n" \
  "    }\n" \
  "  }\n" \
  "}\n" \
  ""
static const char opencl_kernels_source_libxstream_syrk[] = OPENCL_KERNELS_STRING_LIBXSTREAM_SYRK;
