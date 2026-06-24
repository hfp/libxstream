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
# define NSLICES_A 2
#endif
#if !defined(NSLICES_X)
# define NSLICES_X 3
#endif
#if !defined(MANT_BITS)
# define MANT_BITS 23
#endif
#define BIAS 127

#define I8_SLM_COUNT (K_PAD_I8 * XMX_N)

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
 * stencil_apply_int8: Ozaki-1 INT8 slicing for the D*X stencil.
 *
 * D stored as char[NSLICES_A][BLK][K_PAD] with per-row FP scale dk_scale[BLK].
 * X sliced on-the-fly from FP32 p_grid with per-WG shared exponent.
 * Each (sa, sb) digit pair is accumulated in a separate int8 register,
 * converted to float and rescaled before summation.
 *
 * SLM: one X slice at a time (K_PAD * XMX_N bytes = 768 B).
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

  local char x_slm[I8_SLM_COUNT];
  local int exp_sg[WG_M_TILES];

  const int fill_id = sg_id * SG + sg_lid;
  const int fill_total = WG_M_TILES * SG;
  float8 acc[STRIPS_PER_WG];
  int strip_local, dim;

  UNROLL_FORCE(STRIPS_PER_WG) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
    acc[strip_local] = (float8)(0.0f);
  }

  UNROLL_OUTER(1) for (dim = 0; dim < NTERMS; ++dim) {
    global const char* dk = (0 == dim) ? dk_x : ((1 == dim) ? dk_y : dk_z);

    UNROLL_OUTER(1) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
      const int nj = (strip_grp * STRIPS_PER_WG + strip_local) * XMX_N;
      int local_max_exp = 0;
      int wg_max_exp, nslices_eff, idx, sb;

      { uint max_bits = 0;
        for (idx = fill_id; idx < K_PAD * XMX_N; idx += fill_total) {
          const int k = idx / XMX_N;
          const int col_local = idx % XMX_N;
          const int nc = nj + col_local;
          const int ci = nc % BLK;
          const int cj = nc / BLK;
          int gx, gy, gz;

          STENCIL_GATHER_COORD(dim, ox, oy, oz, k, ci, cj, gx, gy, gz);
          STENCIL_CLAMP_COORD(gx, gy, gz, nx, ny, nz);

          if (k < K_BASE) {
            uint bits = as_uint(p_grid[STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby)]);
            int e = (int)((bits >> 23) & 0xFFu);
            if (e > local_max_exp) local_max_exp = e;
          }
        }
      }

      { const int sg_max = sub_group_reduce_max(local_max_exp);
        if (0 == sg_lid) {
          exp_sg[sg_id] = sg_max;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      { int ti;
        wg_max_exp = 0;
        UNROLL_FORCE(WG_M_TILES) for (ti = 0; ti < WG_M_TILES; ++ti) {
          if (exp_sg[ti] > wg_max_exp) wg_max_exp = exp_sg[ti];
        }
        nslices_eff = (0 == wg_max_exp) ? 1
          : ((wg_max_exp - local_max_exp <= 7) ? 1
            : ((wg_max_exp - local_max_exp <= 14) ? 2 : NSLICES_X));
      }

      UNROLL_AUTO for (sb = 0; sb < nslices_eff; ++sb) {
        for (idx = fill_id; idx < K_PAD_I8 * XMX_N; idx += fill_total) {
          const int k = idx / XMX_N;
          const int col_local = idx % XMX_N;
          if (k < K_BASE) {
            const int nc = nj + col_local;
            const int ci = nc % BLK;
            const int cj = nc / BLK;
            int gx, gy, gz;
            float val;
            uint bits;
            int e, sign_bit, shift;
            uint mantissa;

            STENCIL_GATHER_COORD(dim, ox, oy, oz, k, ci, cj, gx, gy, gz);
            STENCIL_CLAMP_COORD(gx, gy, gz, nx, ny, nz);

            val = p_grid[STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby)];
            bits = as_uint(val);
            e = (int)((bits >> 23) & 0xFFu);
            sign_bit = (int)(bits >> 31);
            mantissa = (0 != e) ? ((bits & 0x7FFFFFu) | 0x800000u) : 0;
            shift = wg_max_exp - e;
            if (shift > 0) mantissa >>= shift;
            x_slm[k * XMX_N + col_local] = i8_slice_digit(mantissa, sign_bit, sb);
          }
          else {
            x_slm[k * XMX_N + col_local] = 0;
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        { int sa, ks;
          const float x_scale = EXP2I(wg_max_exp - BIAS - MANT_BITS + 7 * sb);
          UNROLL_FORCE(NSLICES_A) for (sa = 0; sa < NSLICES_A; ++sa) {
            global const char* d_digit = dk + (long)sa * BLK * K_PAD_I8;
            int8 pair_acc = (int8)(0);
            const float pair_scale = dk_scale[sa * BLK + mi] * x_scale;
#if defined(TRIM) && (0 < TRIM)
            if (sa + sb >= NSLICES_A + nslices_eff - 1 - (TRIM - 1)) continue;
#endif
            for (ks = 0; ks < K_PAD_I8; ks += 32) {
              ushort8 a_i8;
              int8 b_i8;
              int bi;
              intel_sub_group_2d_block_read_8b_8r32x1c(
                (global void*)d_digit, K_PAD_I8, BLK, K_PAD_I8,
                (int2)(ks, mi), (private ushort*)&a_i8);
              UNROLL_FORCE(8) for (bi = 0; bi < 8; ++bi) {
                const int k0 = ks + bi * 4;
                ((int*)&b_i8)[bi] =
                  ((int)(uchar)x_slm[(k0 + 0) * XMX_N + sg_lid]) |
                  ((int)(uchar)x_slm[(k0 + 1) * XMX_N + sg_lid] << 8) |
                  ((int)(uchar)x_slm[(k0 + 2) * XMX_N + sg_lid] << 16) |
                  ((int)(uchar)x_slm[(k0 + 3) * XMX_N + sg_lid] << 24);
              }
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

        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }

  UNROLL_FORCE(STRIPS_PER_WG) for (strip_local = 0; strip_local < STRIPS_PER_WG; ++strip_local) {
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
          const long i = STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby);
          p_new[i] = 2.0f * p_grid[i] - p_old[i] + dt2 * vel[i] * u.a[m];
        }
      }
    }
  }
}
