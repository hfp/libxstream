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
  "#if !defined(BSX)\n" \
  "#define BSX 16\n" \
  "#endif\n" \
  "#if !defined(BSY)\n" \
  "#define BSY 16\n" \
  "#endif\n" \
  "#if !defined(RX)\n" \
  "#define RX 4\n" \
  "#endif\n" \
  "#if !defined(RY)\n" \
  "#define RY 4\n" \
  "#endif\n" \
  "#if !defined(BK)\n" \
  "#define BK 16\n" \
  "#endif\n" \
  "#define BM (BSX * RX)\n" \
  "#define BN (BSY * RY)\n" \
  "#define SYRK_OP(X,LD,TRANS,I,P) ((0 == (TRANS)) ? (X)[(I) + (size_t)(P) * (LD)] : (X)[(P) + (size_t)(I) * (LD)])\n" \
  "__attribute__((reqd_work_group_size(BSX, BSY, 1))) kernel void syrk_panel(global T* restrict out,\n" \
  "  global const T* restrict a, global const T* restrict b, global const T* restrict d, int m, int w, int r0, int c0,\n" \
  "  int k, int lda, int ldb, int trans, int twok, int hasd, T alpha, int upper, global T* restrict part, int kslice)\n" \
  "{\n" \
  "  local T lx[BK][BM + 1], ly[BK][BN + 1];\n" \
  "  const int tx = (int)get_local_id(0), ty = (int)get_local_id(1);\n" \
  "  const int i0 = (int)get_group_id(0) * BM, j0 = (int)get_group_id(1) * BN;\n" \
  "  const int tid = ty * BSX + tx, slice = (int)get_group_id(2);\n" \
  "  const int kb = slice * kslice, ke = min(k, kb + kslice);\n" \
  "  global T* const dst = (0 == slice ? out : (part + (size_t)(slice - 1) * m * w));\n" \
  "  T acc[RX][RY];\n" \
  "  int term = 0, u, v;\n" \
  "  if (0 != upper ? (r0 + i0 > c0 + j0 + BN - 1) : (r0 + i0 + BM - 1 < c0 + j0)) return;\n" \
  "  for (u = 0; u < RX; ++u) {\n" \
  "    for (v = 0; v < RY; ++v) acc[u][v] = 0;\n" \
  "  }\n" \
  "  for (; term <= twok; ++term) {\n" \
  "    global const T* const x = (0 == term ? a : b);\n" \
  "    global const T* const y = (0 == term ? b : a);\n" \
  "    const int ldx = (0 == term ? lda : ldb), ldy = (0 == term ? ldb : lda);\n" \
  "    int p0 = kb;\n" \
  "    for (; p0 < ke; p0 += BK) {\n" \
  "      int e = tid;\n" \
  "      for (; e < BK * BM; e += BSX * BSY) {\n" \
  "        const int pp = e / BM, q = e % BM, p = p0 + pp, gi = i0 + q;\n" \
  "        T xv = 0;\n" \
  "        if (p < ke && gi < m) {\n" \
  "          xv = SYRK_OP(x, ldx, trans, r0 + gi, p);\n" \
  "          if (0 != hasd) xv *= d[p];\n" \
  "        }\n" \
  "        lx[pp][q] = xv;\n" \
  "      }\n" \
  "      for (e = tid; e < BK * BN; e += BSX * BSY) {\n" \
  "        const int pp = e / BN, q = e % BN, p = p0 + pp, gj = j0 + q;\n" \
  "        ly[pp][q] = ((p < ke && gj < w) ? SYRK_OP(y, ldy, trans, c0 + gj, p) : 0);\n" \
  "      }\n" \
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" \
  "      { int pp = 0;\n" \
  "        for (; pp < BK; ++pp) {\n" \
  "          T xr[RX], yr[RY];\n" \
  "          for (u = 0; u < RX; ++u) xr[u] = lx[pp][tx + u * BSX];\n" \
  "          for (v = 0; v < RY; ++v) yr[v] = ly[pp][ty + v * BSY];\n" \
  "          for (u = 0; u < RX; ++u) {\n" \
  "            for (v = 0; v < RY; ++v) acc[u][v] = fma(xr[u], yr[v], acc[u][v]);\n" \
  "          }\n" \
  "        }\n" \
  "      }\n" \
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" \
  "    }\n" \
  "  }\n" \
  "  for (v = 0; v < RY; ++v) {\n" \
  "    const int j = j0 + ty + v * BSY;\n" \
  "    if (j < w) {\n" \
  "      for (u = 0; u < RX; ++u) {\n" \
  "        const int i = i0 + tx + u * BSX;\n" \
  "        if (i < m) dst[i + (size_t)j * m] = alpha * acc[u][v];\n" \
  "      }\n" \
  "    }\n" \
  "  }\n" \
  "}\n" \
  "kernel void syrk_sum(global T* restrict out, global const T* restrict part, int n, int nparts)\n" \
  "{\n" \
  "  const size_t e = get_global_id(0);\n" \
  "  if (e < (size_t)n) {\n" \
  "    T sum = out[e];\n" \
  "    int s = 0;\n" \
  "    for (; s < nparts; ++s) sum += part[e + (size_t)s * n];\n" \
  "    out[e] = sum;\n" \
  "  }\n" \
  "}\n" \
  ""
static const char opencl_kernels_source_libxstream_syrk[] = OPENCL_KERNELS_STRING_LIBXSTREAM_SYRK;
