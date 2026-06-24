/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_common.cl"

#if !defined(NSLICES_A)
# define NSLICES_A 1
#endif
#if !defined(NSLICES_X)
# define NSLICES_X 3
#endif
#if !defined(MANT_BITS)
# define MANT_BITS 23
#endif
#define BIAS 127

#define I8_K4_BASE ((K_BASE + 3) / 4)
#define I8_K4_PAD (K_PAD_I8 / 4)
#define I8_SLM_INTS (NSLICES_X * I8_K4_PAD * XMX_N)
#define I8_FILL_COUNT (I8_K4_PAD * XMX_N)
#if !defined(I8_EXP_MARGIN)
# define I8_EXP_MARGIN 1
#endif

inline char i8_slice_digit(uint aligned, int sign, int s)
{
  const int high = MANT_BITS - (7 * s);
  const int low = (high - 6 > 0) ? (high - 6) : 0;
  const int width = high - low + 1;
  char digit = 0;
  if (width > 0 && high >= 0) {
    digit = (char)((aligned >> low) & ((1U << width) - 1U));
  }
  if (0 != sign) digit = (char)(-digit);
  return digit;
}

/**
 * stencil_apply_int8: single-pass INT8 Ozaki-1 with carried-forward exponent.
 *
 * exp_buf is seeded by the host. Each step, the kernel updates exp_buf to
 * max(own_observed, left_neighbor, right_neighbor) + margin, propagating
 * growth at the wavefront speed (one block per step per dimension).
 */
#if defined(INTEL) && (2 <= INTEL)
__attribute__((reqd_work_group_size(SG, WG_M_TILES, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
#endif
kernel void stencil_apply_int8(
  global const char* restrict dk_x,
  global const char* restrict dk_y,
  global const char* restrict dk_z,
  global const float* restrict dk_scale,
  global const float* restrict p_grid,
  global float* restrict p_old,
  global float* restrict p_new,
  global const float* restrict vel,
  global const int* restrict exp_buf,
  global int* restrict exp_buf_out,
  float dt2,
  int nx, int ny, int nz,
  int nbx, int nby)
{
  const int bx = (int)get_group_id(0);
  const int by = (int)get_group_id(1);
  const int strip_grp = (int)get_group_id(2) & (N_STRIP_GROUPS - 1);
  const int bz = (int)get_group_id(2) >> STENCIL_NSTRIP_SHIFT;
  const int sg_id = (int)get_sub_group_id();
  const int sg_lid = (int)get_sub_group_local_id();
  const int mi = sg_id * XMX_M;
  const int ox = bx * BLK;
  const int oy = by * BLK;
  const int oz = bz * BLK;

  local int x_slm[I8_SLM_INTS];
  local int exp_sg[WG_M_TILES];

  const int fill_id = sg_id * SG + sg_lid;
  const int fill_total = WG_M_TILES * SG;
  const int blk_linear = bz * nby * nbx + by * nbx + bx;
  float8 acc[STRIPS_PER_WG];
  int strip_local, dim;

  UNROLL_FORCE(STRIPS_PER_WG) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
    acc[strip_local] = (float8)(0.0f);
  }

  UNROLL_OUTER(1) for (dim = 0; dim < NTERMS; ++dim) {
    global const char* dk = (0 == dim) ? dk_x : ((1 == dim) ? dk_y : dk_z);

    UNROLL_OUTER(1) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
      const int nj = (strip_grp * STRIPS_PER_WG + strip_local) * XMX_N;
      const int strip_abs = strip_grp * STRIPS_PER_WG + strip_local;
      const int exp_idx = (blk_linear * NTERMS + dim) * N_STRIPS + strip_abs;
      const int assumed_exp = exp_buf[exp_idx];
      const int nslices_eff = (0 == assumed_exp) ? 1
        : ((assumed_exp <= 7) ? 1 : ((assumed_exp <= 14) ? 2 : NSLICES_X));
      int idx;

      for (idx = fill_id; idx < I8_FILL_COUNT; idx += fill_total) {
        const int k4 = idx / XMX_N;
        const int col_local = idx % XMX_N;
        const int k_base = k4 * 4;
        const int nc = nj + col_local;
        const int ci = nc % BLK;
        const int cj = nc / BLK;
        int s, ki;
        uint pack[NSLICES_X];
        UNROLL_FORCE(NSLICES_X) for (s = 0; s < NSLICES_X; ++s) pack[s] = 0;

        UNROLL_FORCE(4) for (ki = 0; ki < 4; ++ki) {
          const int k = k_base + ki;
          uint mantissa = 0;
          int sign_bit = 0;
          if (k < K_BASE) {
            int gx, gy, gz;
            uint bits;
            int e, shift;

            STENCIL_GATHER_COORD(dim, ox, oy, oz, k, ci, cj, gx, gy, gz);
            STENCIL_CLAMP_COORD(gx, gy, gz, nx, ny, nz);

            bits = as_uint(p_grid[STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby)]);
            e = (int)((bits >> 23) & 0xFFu);
            sign_bit = (int)(bits >> 31);
            mantissa = (0 != e) ? ((bits & 0x7FFFFFu) | 0x800000u) : 0;

            shift = assumed_exp - e;
            if (shift < 0 || shift >= 24 || 0 == e) mantissa = 0;
            else if (shift > 0) mantissa >>= shift;
          }
          UNROLL_FORCE(NSLICES_X) for (s = 0; s < NSLICES_X; ++s) {
            pack[s] |= ((uint)(uchar)i8_slice_digit(mantissa, sign_bit, s)) << (ki * 8);
          }
        }

        UNROLL_FORCE(NSLICES_X) for (s = 0; s < NSLICES_X; ++s) {
          x_slm[s * I8_K4_PAD * XMX_N + k4 * XMX_N + col_local] = (int)pack[s];
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      { int sa, sb, ks;
        UNROLL_FORCE(NSLICES_A) for (sa = 0; sa < NSLICES_A; ++sa) {
          global const char* d_digit = dk + (long)sa * BLK * K_PAD_I8;
          UNROLL_AUTO for (sb = 0; sb < nslices_eff; ++sb) {
            int8 pair_acc = (int8)(0);
            const float pair_scale = dk_scale[sa * BLK + mi]
              * EXP2I(assumed_exp - BIAS - MANT_BITS + 7 * sb);
#if defined(TRIM) && (0 < TRIM)
            if (sa + sb >= NSLICES_A + nslices_eff - 1 - (TRIM - 1)) continue;
#endif
            for (ks = 0; ks < K_PAD_I8 / 4; ks += 8) {
              ushort8 a_i8;
              int8 b_i8;
              intel_sub_group_2d_block_read_8b_8r32x1c(
                (global void*)d_digit, K_PAD_I8, BLK, K_PAD_I8,
                (int2)(ks * 4, mi), (private ushort*)&a_i8);
              b_i8 = as_int8(intel_sub_group_block_read8(
                (local const uint*)(x_slm + sb * I8_K4_PAD * XMX_N + ks * XMX_N)));
              pair_acc = intel_sub_group_i8_i8_matrix_mad_k32(
                as_short8(a_i8), b_i8, pair_acc);
            }
            { union { int8 v; int a[8]; } ui;
              int m;
              ui.v = pair_acc;
              UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
                ((float*)&acc[strip_local])[m] += (float)ui.a[m] * pair_scale;
              }
            }
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  UNROLL_FORCE(STRIPS_PER_WG) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
    const int nj = (strip_grp * STRIPS_PER_WG + strip_local) * XMX_N;
    const int strip_abs = strip_grp * STRIPS_PER_WG + strip_local;
    union { float8 v; float a[8]; } u;
    int out_max_exp = 0;
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
          const long i = STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby);
          const float val = 2.0f * p_grid[i] - p_old[i] + dt2 * vel[i] * u.a[m];
          const int oe = (int)((as_uint(val) >> 23) & 0xFFu);
          p_new[i] = val;
          if (oe > out_max_exp) out_max_exp = oe;
        }
      }
    }
    { const int sg_out = sub_group_reduce_max(out_max_exp);
      if (0 == sg_lid) exp_sg[sg_id] = sg_out;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (0 == fill_id) {
      int ti, wg_out = 0;
      UNROLL_FORCE(WG_M_TILES) for (ti = 0; ti < WG_M_TILES; ++ti) {
        if (exp_sg[ti] > wg_out) wg_out = exp_sg[ti];
      }
      { const int exp_idx = (blk_linear * NTERMS + 0) * N_STRIPS + strip_abs;
        exp_buf_out[exp_idx] = wg_out + I8_EXP_MARGIN;
        exp_buf_out[exp_idx + N_STRIPS] = wg_out + I8_EXP_MARGIN;
        exp_buf_out[exp_idx + 2 * N_STRIPS] = wg_out + I8_EXP_MARGIN;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
