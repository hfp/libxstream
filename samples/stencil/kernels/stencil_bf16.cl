/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_common.cl"

/**
 * stencil_apply: Fused 3-dim gather + Dekker-split + DPAS + leapfrog.
 *
 * WG = (SG, WG_M_TILES, 1). Each WG handles one block and STRIPS_PER_WG
 * adjacent N-strips. For each strip and dimension: cooperatively
 * gathers K_PAD x XMX_N floats, Dekker-splits into SLM, DPAS
 * accumulates. A-side D loads are shared across strips within a WG.
 *
 * SLM budget: NDIGITS_X * K_PAD * XMX_N * sizeof(ushort)
 *           = 3 * 48 * 16 * 2 = 4608 bytes (reused per strip/dim).
 *
 * Dispatch:
 *   global = (nblocks * SG, M_TILES, N_STRIP_GROUPS)
 *   local  = (SG, WG_M_TILES, 1)
 */
#if defined(INTEL) && (2 <= INTEL)
__attribute__((reqd_work_group_size(SG, WG_M_TILES, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void stencil_apply(
  global const ushort* restrict dk_x,
  global const ushort* restrict dk_y,
  global const ushort* restrict dk_z,
  global const float* restrict p_grid,
  global float* restrict p_old,
  global float* restrict p_new,
  global const float* restrict vel,
  int nterms, float dt2,
  int nx, int ny, int nz,
  int nbx, int nby)
{
  const int blk_idx = (int)get_group_id(0);
  const int strip_grp = (int)get_group_id(2);
  const int sg_id = (int)get_sub_group_id();
  const int sg_lid = (int)get_sub_group_local_id();
  const int mi = sg_id * XMX_M;

  const int bz = blk_idx / (nbx * nby);
  const int by = (blk_idx / nbx) % nby;
  const int bx = blk_idx % nbx;
  const int ox = bx * BLK;
  const int oy = by * BLK;
  const int oz = bz * BLK;

  local ushort x_slm[NDIGITS_X * K_PAD * XMX_N];
  local int exp_range[2];

  const int d_wb = K_PAD * 2;
  const int fill_id = sg_id * SG + sg_lid;
  const int fill_total = WG_M_TILES * SG;
  float8 acc[STRIPS_PER_WG];
  int strip_local, dim;

  for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
    acc[strip_local] = (float8)(0.0f);
  }

  for (dim = 0; dim < NTERMS; ++dim) {
    global const ushort* dk = (0 == dim) ? dk_x : ((1 == dim) ? dk_y : dk_z);

    for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
      const int nj = (strip_grp * STRIPS_PER_WG + strip_local) * XMX_N;
      int idx, ndigits_eff;
      int local_max_exp = 0, local_min_exp = 255;

      if (0 == fill_id) { exp_range[0] = 0; exp_range[1] = 255; }
      barrier(CLK_LOCAL_MEM_FENCE);

      for (idx = fill_id; idx < K_PAD * XMX_N; idx += fill_total) {
        const int k = idx / XMX_N;
        const int col_local = idx % XMX_N;
        const int nc = nj + col_local;
        const int ci = nc % BLK;
        const int cj = nc / BLK;
        int gx, gy, gz;
        float val, residual;
        unsigned int bits;
        int e, s;

        STENCIL_GATHER_COORD(dim, ox, oy, oz, k, ci, cj, gx, gy, gz);
        STENCIL_CLAMP_COORD(gx, gy, gz, nx, ny, nz);

        if (k < K_BASE) {
          val = p_grid[STENCIL_GRID_IDX(gz, gy, gx, ny, nx)];

          bits = as_uint(val);
          e = (int)((bits >> 23) & 0xFFu);
          if (0 != e) {
            if (e > local_max_exp) local_max_exp = e;
            if (e < local_min_exp) local_min_exp = e;
          }

          residual = val;
          for (s = 0; s < NDIGITS_X; ++s) {
            const ushort bf = ROUND_TO_BF16(residual);
            x_slm[s * K_PAD * XMX_N + k * XMX_N + col_local] = bf;
            residual -= BF16_TO_F32(bf);
          }
        }
        else {
          for (s = 0; s < NDIGITS_X; ++s) {
            x_slm[s * K_PAD * XMX_N + k * XMX_N + col_local] = (ushort)0;
          }
        }
      }

      atomic_max(&exp_range[0], local_max_exp);
      atomic_min(&exp_range[1], local_min_exp);
      barrier(CLK_LOCAL_MEM_FENCE);

      { const int span = exp_range[0] - exp_range[1];
        ndigits_eff = (span <= 7) ? 1 : ((span <= 14) ? 2 : NDIGITS_X);
        if (0 == exp_range[0]) ndigits_eff = 1;
      }

      STENCIL_DPAS_ACC(dk, ndigits_eff, x_slm, d_wb, mi, acc[strip_local]);

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
    const int nj = (strip_grp * STRIPS_PER_WG + strip_local) * XMX_N;
    union { float8 v; float a[8]; } u;
    int m;
    u.v = acc[strip_local];
    UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
      const int row = mi + m;
      const int col = nj + sg_lid;
      if (0 <= row && row < BLK && col < N_TOTAL) {
        const int gx = ox + row;
        const int gy = oy + (col % BLK);
        const int gz = oz + (col / BLK);
        if (gx < nx && gy < ny && gz < nz) {
          const long i = STENCIL_GRID_IDX(gz, gy, gx, ny, nx);
          p_new[i] = 2.0f * p_grid[i] - p_old[i] + dt2 * vel[i] * u.a[m];
        }
      }
    }
  }
}


/**
 * stencil_apply_tti: TTI cross-derivative term via fused
 * gather + two-phase DPAS with SLM intermediate.
 *
 * Computes: Y += D_i * (c_ij . (D_j * P))
 *
 * Per (block, N-strip) WG:
 *   1. Cooperative gather along dim j into x_slm (same as stencil_apply)
 *   2. DPAS: D_j * X -> T (float8 per sub-group)
 *   3. Scale: T *= c_ij (point-wise)
 *   4. Dekker re-split T into t_slm as BF16
 *   5. DPAS: D_i * T -> accumulate into Y
 *
 * SLM: x_slm (4.5 KB) for phase 1, t_slm (3 KB) for phase 2.
 * Both fit within 128 KB. Reused across (sa, sb) pairs.
 *
 * Dispatch:
 *   global = (nblocks * SG, M_TILES, N_STRIPS)
 *   local  = (SG, M_TILES, 1)
 */
#if defined(INTEL) && (2 <= INTEL)
__attribute__((reqd_work_group_size(SG, M_TILES, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void stencil_apply_tti(
  global const ushort* restrict dk_i,
  global const ushort* restrict dk_j,
  global const float* restrict p_grid,
  global float* restrict p_new,
  global const float* restrict c_ij,
  int y_stride,
  int dim_j, int nx, int ny, int nz,
  int nbx, int nby)
{
  const int blk_idx = (int)get_group_id(0);
  const int nstrip = (int)get_group_id(2);
  const int sg_id = (int)get_sub_group_id();
  const int sg_lid = (int)get_sub_group_local_id();
  const int mi = sg_id * XMX_M;
  const int nj = nstrip * XMX_N;

  const int bz = blk_idx / (nbx * nby);
  const int by = (blk_idx / nbx) % nby;
  const int bx = blk_idx % nbx;
  const int ox = bx * BLK;
  const int oy = by * BLK;
  const int oz = bz * BLK;

  local ushort x_slm[NDIGITS_X * K_PAD * XMX_N];
  local ushort t_slm[NDIGITS_A * K_PAD * XMX_N];

  const int d_wb = K_PAD * 2;
  const int fill_id = sg_id * SG + sg_lid;
  const int fill_total = M_TILES * SG;
  float8 y_acc = (float8)(0.0f);
  int sa, sb, kstep, m, idx;

  for (idx = fill_id; idx < K_PAD * XMX_N; idx += fill_total) {
    const int k = idx / XMX_N;
    const int col_local = idx % XMX_N;
    const int nc = nj + col_local;
    const int ci = nc % BLK;
    const int cj = nc / BLK;
    int gx, gy, gz;
    float val, residual;
    int s;

    STENCIL_GATHER_COORD(dim_j, ox, oy, oz, k, ci, cj, gx, gy, gz);
    STENCIL_CLAMP_COORD(gx, gy, gz, nx, ny, nz);

    if (k < K_BASE) {
      val = p_grid[STENCIL_GRID_IDX(gz, gy, gx, ny, nx)];

      residual = val;
      for (s = 0; s < NDIGITS_X; ++s) {
        const ushort bf = ROUND_TO_BF16(residual);
        x_slm[s * K_PAD * XMX_N + k * XMX_N + col_local] = bf;
        residual -= BF16_TO_F32(bf);
      }
    }
    else {
      for (s = 0; s < NDIGITS_X; ++s) {
        x_slm[s * K_PAD * XMX_N + k * XMX_N + col_local] = (ushort)0;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (sb = 0; sb < NDIGITS_X; ++sb) {
    local const ushort* x_digit = x_slm + sb * K_PAD * XMX_N;

    for (sa = 0; sa < NDIGITS_A; ++sa) {
      global const ushort* dj_digit = dk_j + (long)sa * BLK * K_PAD;
      float8 t_acc = (float8)(0.0f);

      for (kstep = 0; kstep < K_PAD; kstep += 16) {
        ushort8 a_bf;
        uint8 b_bf;
        BF16_LOAD_A(dj_digit, d_wb, BLK, mi, kstep, &a_bf);
        b_bf = *(local const uint8*)(x_digit + kstep * XMX_N);
        BF16_DPAS_ONE(a_bf, b_bf, t_acc);
      }

      {
        union { float8 v; float a[8]; } t_u;
        const long cij_base = (long)blk_idx * BLK * BLK * BLK;
        t_u.v = t_acc;

        for (idx = fill_id; idx < NDIGITS_A * K_PAD * XMX_N; idx += fill_total) {
          t_slm[idx] = (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
          const int row = mi + m;
          const int col = nj + sg_lid;
          if (row < BLK && col < N_TOTAL) {
            t_u.a[m] *= c_ij[cij_base + (long)row * N_TOTAL + col];
          }
        }

        UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
          const int row = mi + m;
          STENCIL_SPLIT_F32_TO_SLM(t_slm, NDIGITS_A, row, sg_lid, t_u.a[m]);
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      { int sa2;
        UNROLL_FORCE(NDIGITS_A) for (sa2 = 0; sa2 < NDIGITS_A; ++sa2) {
          global const ushort* di_digit = dk_i + (long)sa2 * BLK * K_PAD;
          local const ushort* t_digit = t_slm + sa2 * K_PAD * XMX_N;

          for (kstep = 0; kstep < K_PAD; kstep += 16) {
            ushort8 a_bf;
            uint8 b_bf;
            BF16_LOAD_A(di_digit, d_wb, BLK, mi, kstep, &a_bf);
            b_bf = *(local const uint8*)(t_digit + kstep * XMX_N);
            BF16_DPAS_ONE(a_bf, b_bf, y_acc);
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  {
    union { float8 v; float a[8]; } u;
    u.v = y_acc;
    UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
      const int row = mi + m;
      const int col = nj + sg_lid;
      if (row < BLK && col < N_TOTAL) {
        const int gx = ox + row;
        const int gy = oy + (col % BLK);
        const int gz = oz + (col / BLK);
        if (gx < nx && gy < ny && gz < nz) {
          const long i = STENCIL_GRID_IDX(gz, gy, gx, ny, nx);
          p_new[i] += c_ij[i] * u.a[m];
        }
      }
    }
  }
}


