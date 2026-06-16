/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_common.cl"

/* Stencil via Dekker-split BF16 DPAS.
 *
 * Dimension-split formulation:
 *   Y = D_x * P + D_y * P + D_z * P   (isotropic, 3 terms)
 *   Y = sum_{i,j} D_i * (c_ij . D_j * P)  (TTI, 9 terms)
 *
 * Each term is a GEMM: D is BLK x BLK (banded FD operator),
 * P is BLK x BLK^2 (wavefield block reshaped along one axis).
 * With Dekker splitting: D has NDIGITS_A bf16 digits,
 * P (or intermediate) has NDIGITS_X bf16 digits.
 * Total products per term: NDIGITS_A * NDIGITS_X.
 *
 * Memory layout:
 *   D surfaces: precomputed BF16 digit arrays, one per operator per digit.
 *     Shape [NDIGITS_A][BLK][K_PAD] ushort, row-major.
 *     K_PAD >= BLK ensures 2D block I/O surface width >= 64 bytes.
 *     D is banded (only STENCIL_WIDTH diagonals non-zero) but stored
 *     dense -- zero entries cost nothing in DPAS (compute-bound on X).
 *
 *   X surfaces: Dekker-split wavefield along target dimension.
 *     Shape [NDIGITS_X][K_PAD][N_PAD] ushort, K-major for VNNI load.
 *     K_PAD rows, N_PAD columns (N_PAD >= 32 for surface constraint).
 *     Preprocess kernel gathers along the stencil dimension and splits.
 *
 *   Y output: float [BLK][BLK*BLK], accumulated in registers.
 *
 * Kernel: stencil_apply
 *   One work-group computes one output block (BLK^3 elements).
 *   Work-group: (SG, M_TILES * N_STRIPS, 1).
 *   Each sub-group owns one 8x16 output tile, accumulates all
 *   operator terms, then writes to global Y.
 *
 * Kernel: preprocess_x
 *   Gathers wavefield data along one dimension from the super-block
 *   (including halo), Dekker-splits into BF16 digits, and writes
 *   K-major surfaces suitable for 2D block B-side loads.
 */

#if defined(INTEL) && (2 <= INTEL)


/**
 * preprocess_x: Gather along one dimension and Dekker-split into BF16.
 *
 * For dimension d, the super-block P_super has shape:
 *   d=x: (BLK+2R) x BLK x BLK  -- x-fastest, K-axis contiguous
 *   d=y: BLK x (BLK+2R) x BLK  -- stride-BLK along K-axis
 *   d=z: BLK x BLK x (BLK+2R)  -- stride-BLK^2 along K-axis
 *
 * Output: xk[digit][k][n] in K-major layout, K=BLK (padded to K_PAD),
 *         N = BLK*BLK (padded to N_PAD).
 *
 * For d=x the gather is trivial (memcpy + split).
 * For d=y,z we gather with stride into contiguous K-rows.
 *
 * Work-group: (BLK, BLK, 1).  One WG per (z-plane, digit) for d=x,
 * or per (appropriate plane, digit) for other dims.
 * Dispatch: global = (BLK, BLK, BLK) covers all N = BLK^2 columns
 * and BLK rows of K.  One launch per digit.
 */
__attribute__((reqd_work_group_size(BLK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void preprocess_x(
  global const float* restrict p_super,
  int dim, int stride_k,
  int super_m, int super_n, int super_p,
  global ushort* restrict xk,
  int digit_offset, int ndigits)
{
  const int ki = (int)get_global_id(0);
  const int ni = (int)get_global_id(1);
  const int nj = (int)get_global_id(2);
  const int n_col = ni * BLK + nj;
  int result = EXIT_SUCCESS;

  if (ki < BLK && n_col < N_TOTAL) {
    int src_idx;
    float val;
    int s;

    if (0 == dim) {
      const int iz = n_col / BLK;
      const int iy = n_col % BLK;
      src_idx = (iz * super_n + iy) * super_m + (ki + RADIUS);
    }
    else if (1 == dim) {
      const int iz = n_col / BLK;
      const int ix = n_col % BLK;
      src_idx = (iz * super_n + (ki + RADIUS)) * super_m + ix;
    }
    else {
      const int iy = n_col / BLK;
      const int ix = n_col % BLK;
      src_idx = ((ki + RADIUS) * super_n + iy) * super_m + ix;
    }

    val = p_super[src_idx];

    {
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
      double residual = (double)val;
#else
      float residual = val;
#endif
      for (s = 0; s < ndigits; ++s) {
        const ushort bf = ROUND_TO_BF16((float)residual);
        const long dst_idx = ((long)(digit_offset + s) * K_PAD + ki)
                           * (long)N_PAD + n_col;
#if defined(USE_DOUBLE) && (1 == USE_DOUBLE)
        residual -= (double)BF16_TO_F32(bf);
#else
        residual -= BF16_TO_F32(bf);
#endif
        xk[dst_idx] = bf;
      }
    }
  }
}


/**
 * stencil_apply: One dimension-split operator term via BF16 DPAS.
 *
 * Computes: Y += D * X  (or Y = c^2 * D * X if first==1)
 *
 * One sub-group owns one 8x16 output tile of Y.
 * Loops over NDIGITS_A * NDIGITS_X digit pairs, each pair uses
 * K_PAD/16 DPAS K-steps.
 *
 * D surface (A-side): BLK x K_PAD bf16, read via 2D block load.
 *   BLK=32, K_PAD=32: 2 K-steps of 16.  The banded zeros (only
 *   STENCIL_WIDTH diagonals non-zero) are structurally zero in
 *   the surface -- DPAS multiplies them by zero at no extra cost
 *   since the kernel is memory-bound on X reads.
 *
 * X surface (B-side): K_PAD x N_PAD bf16, K-major for VNNI.
 *   Read via intel_sub_group_2d_block_read_transform_16b_16r16x1c.
 *   Preprocess kernel ensures K-contiguity for all dimensions.
 *
 * Parameters:
 *   dk       - D digit surface [NDIGITS_A][BLK][K_PAD] ushort
 *   xk       - X digit surface [NDIGITS_X][K_PAD][N_PAD] ushort
 *   y        - output block [BLK * N_TOTAL] float
 *   vel      - velocity field [BLK^3] float (c^2, applied if first)
 *   first    - if 1, write Y = c^2 * result; else Y += result
 *   y_stride - leading dimension of Y (>= N_TOTAL)
 *
 * Dispatch:
 *   local  = (SG, M_TILES * N_STRIPS, 1)
 *   global = (nblocks * SG, M_TILES * N_STRIPS, 1)
 */
__attribute__((reqd_work_group_size(SG, M_TILES * N_STRIPS, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void stencil_apply(
  global const ushort* restrict dk,
  global const ushort* restrict xk,
  global float* restrict y,
  global const float* restrict vel,
  int first, int y_stride)
{
  const int blk_idx = (int)get_group_id(0);
  const int sg_id = (int)get_sub_group_id();
  const int sg_lid = (int)get_sub_group_local_id();
  const int mi = (sg_id / N_STRIPS) * XMX_M;
  const int nj = (sg_id % N_STRIPS) * XMX_N;

  float8 acc = (float8)(0.0f);

  const int d_wb = K_PAD * 2;
  const int x_wb = N_PAD * 2;
  int sa, sb, kstep;

  for (sa = 0; sa < NDIGITS_A; ++sa) {
    global const ushort* d_digit = dk + (long)sa * BLK * K_PAD;

    for (sb = 0; sb < NDIGITS_X; ++sb) {
      global const ushort* x_digit = xk + (long)sb * K_PAD * N_PAD;

      for (kstep = 0; kstep < K_PAD; kstep += 16) {
        BF16_DPAS(d_digit, x_digit,
                        d_wb, BLK, x_wb, K_PAD,
                        mi, nj, kstep, kstep, acc);
      }
    }
  }

  {
    const long y_base = (long)blk_idx * BLK * (long)y_stride;
    union { float8 v; float a[8]; } u;
    int m;
    u.v = acc;
    if (first) {
      const long vel_base = (long)blk_idx * BLK * BLK * BLK;
      UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
        const int row = mi + m;
        const int col = nj + sg_lid;
        if (row < BLK && col < N_TOTAL) {
          const float v2 = vel[vel_base + (long)row * N_TOTAL + col];
          y[y_base + (long)row * y_stride + col] = u.a[m] * v2;
        }
      }
    }
    else {
      UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
        const int row = mi + m;
        const int col = nj + sg_lid;
        if (row < BLK && col < N_TOTAL) {
          y[y_base + (long)row * y_stride + col] += u.a[m];
        }
      }
    }
  }
}


/**
 * stencil_apply_tti: TTI cross-derivative term via two GEMMs with
 * inter-GEMM point-wise scaling through SLM.
 *
 * Computes: Y += D_i * (c_ij . (D_j * P))
 *
 * Two-phase approach per N-strip:
 *   Phase 1: All M_TILES sub-groups cooperate to produce a full
 *            BLK-row column of T = D_j * P, scale by c_ij,
 *            Dekker-split, and store to SLM as BF16 surface.
 *   Phase 2: Each sub-group reads the SLM surface (now K=BLK rows
 *            of BF16) as B-side, applies D_i via DPAS, accumulates Y.
 *
 * SLM budget: NDIGITS_A * K_PAD * XMX_N * sizeof(ushort)
 *   = 2 * 32 * 16 * 2 = 2048 bytes per N-strip.
 *   With barrier between strips, only one strip active at a time.
 *
 * Work-group: (SG, M_TILES, 1).  Each WG handles one N-strip at a
 * time in a loop, synchronizing via barriers.
 *
 * Parameters:
 *   dk_i     - D digit surface for dimension i [NDIGITS_A][BLK][K_PAD]
 *   dk_j     - D digit surface for dimension j [NDIGITS_A][BLK][K_PAD]
 *   xk       - X digit surface along dim j [NDIGITS_X][K_PAD][N_PAD]
 *   y        - output block [BLK][N_TOTAL] float (read-modify-write)
 *   c_ij     - anisotropy coefficient [BLK^3] float
 *   y_stride - leading dimension of Y
 */
__attribute__((reqd_work_group_size(SG, M_TILES, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void stencil_apply_tti(
  global const ushort* restrict dk_i,
  global const ushort* restrict dk_j,
  global const ushort* restrict xk,
  global float* restrict y,
  global const float* restrict c_ij,
  int y_stride)
{
  const int blk_idx = (int)get_group_id(0);
  const int sg_id = (int)get_sub_group_id();
  const int sg_lid = (int)get_sub_group_local_id();
  const int mi = sg_id * XMX_M;
  int result = EXIT_SUCCESS;

  const int d_wb = K_PAD * 2;
  const int x_wb = N_PAD * 2;

  local ushort t_slm[NDIGITS_A * K_PAD * XMX_N];

  int nstrip, sa, sb, kstep, m;

  for (nstrip = 0; nstrip < N_STRIPS; ++nstrip) {
    const int nj = nstrip * XMX_N;
    float8 y_acc = (float8)(0.0f);

    for (sb = 0; sb < NDIGITS_X; ++sb) {
      global const ushort* x_digit = xk + (long)sb * K_PAD * N_PAD;

      for (sa = 0; sa < NDIGITS_A; ++sa) {
        global const ushort* dj_digit = dk_j + (long)sa * BLK * K_PAD;
        float8 t_acc = (float8)(0.0f);

        for (kstep = 0; kstep < K_PAD; kstep += 16) {
          BF16_DPAS(dj_digit, x_digit,
                          d_wb, BLK, x_wb, K_PAD,
                          mi, nj, kstep, kstep, t_acc);
        }

        {
          union { float8 v; float a[8]; } t_u;
          const long cij_base = (long)blk_idx * BLK * BLK * BLK;
          t_u.v = t_acc;
          UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
            const int row = mi + m;
            const int col = nj + sg_lid;
            if (row < BLK && col < N_TOTAL) {
              t_u.a[m] *= c_ij[cij_base + row * N_TOTAL + col];
            }
          }
          t_acc = t_u.v;
        }

        {
          union { float8 v; float a[8]; } t_u;
          int sa2;
          t_u.v = t_acc;
          UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
            float residual = t_u.a[m];
            const int row = mi + m;
            UNROLL_FORCE(NDIGITS_A) for (sa2 = 0; sa2 < NDIGITS_A; ++sa2) {
              const ushort bf = ROUND_TO_BF16(residual);
              t_slm[sa2 * K_PAD * XMX_N + row * XMX_N + sg_lid] = bf;
              residual -= BF16_TO_F32(bf);
            }
          }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        {
          const int t_wb = XMX_N * 2;
          UNROLL_FORCE(NDIGITS_A) for (int sa2 = 0; sa2 < NDIGITS_A; ++sa2) {
            global const ushort* di_digit = dk_i + (long)sa2 * BLK * K_PAD;
            local const ushort* t_digit = t_slm + sa2 * K_PAD * XMX_N;

            for (kstep = 0; kstep < K_PAD; kstep += 16) {
              ushort8 a_bf;
              uint8 b_bf;
              intel_sub_group_2d_block_read_16b_8r16x1c(
                (global void*)di_digit, d_wb, BLK, d_wb,
                (int2)(kstep * 2, mi), (private ushort*)&a_bf);
              b_bf = *(local const uint8*)(t_digit + kstep * XMX_N);
              BF16_DPAS_ONE(a_bf, b_bf, y_acc);
            }
          }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    {
      const long y_base = (long)blk_idx * BLK * y_stride;
      union { float8 v; float a[8]; } u;
      u.v = y_acc;
      UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
        const int row = mi + m;
        const int col = nj + sg_lid;
        if (row < BLK && col < N_TOTAL) {
          y[y_base + row * y_stride + col] += u.a[m];
        }
      }
    }
  }
}


#endif /* INTEL >= 2 */
