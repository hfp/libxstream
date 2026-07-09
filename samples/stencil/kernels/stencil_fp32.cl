/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "stencil_common.cl"

#if !defined(WG_X)
# define WG_X 32
#endif
#if !defined(WG_Y)
# define WG_Y 8
#endif
#define WG_SIZE (WG_X * WG_Y)
#define SLM_F (WG_X + 2 * RADIUS)
#define SLM_M (WG_Y + 2 * RADIUS)
#define SLM_TOTAL (SLM_M * SLM_F)
#define S_WINDOW (2 * RADIUS + 1)

#if defined(STENCIL_PADDED) && (0 < STENCIL_PADDED) && defined(INTEL) && (2 <= INTEL) \
    && (!defined(STENCIL_LAYOUT) || STENCIL_LAYOUT_XYZ == STENCIL_LAYOUT) \
  && (!defined(STENCIL_BF16S) || 0 >= STENCIL_BF16S) \
    && (!defined(STENCIL_PML) || 0 >= STENCIL_PML)
# define FP32_USE_BLOCK_IO 1
#endif

/* Logical-to-physical axis mapping:
 * XYZ: fast=X, medium=Y, slow=Z (Z-sliding window).
 * ZYX: fast=Z, medium=Y, slow=X (X-sliding window). */
#if (STENCIL_LAYOUT_ZYX == STENCIL_LAYOUT)
# define FP32_NFAST nz
# define FP32_NSLOW nx
# define FP32_COEFF_FAST (coeff + 2 * STENCIL_WIDTH)
# define FP32_COEFF_MED  (coeff + STENCIL_WIDTH)
# define FP32_COEFF_SLOW (coeff)
# define FP32_P_FMS(F, M, S) STENCIL_ZYX_P_IDX(F, M, S)
# define FP32_V_FMS(F, M, S) STENCIL_ZYX_V_IDX(F, M, S)
# define FP32_E_FMS(F, M, S) STENCIL_ZYX_E_IDX(F, M, S)
# define FP32_HD_FAST hdz_2
# define FP32_HD_MED  hdy_2
# define FP32_HD_SLOW hdx_2
#else
# define FP32_NFAST STENCIL_NX
# define FP32_NSLOW STENCIL_NZ
# define FP32_COEFF_FAST (coeff)
# define FP32_COEFF_MED  (coeff + STENCIL_WIDTH)
# define FP32_COEFF_SLOW (coeff + 2 * STENCIL_WIDTH)
# define FP32_P_FMS(F, M, S) ((long)(S) * (STENCIL_NY) * (STENCIL_NX) + (long)(M) * (STENCIL_NX) + (F))
# define FP32_V_FMS(F, M, S) FP32_P_FMS(F, M, S)
# define FP32_E_FMS(F, M, S) FP32_P_FMS(F, M, S)
# define FP32_HD_FAST hdx_2
# define FP32_HD_MED  hdy_2
# define FP32_HD_SLOW hdz_2
#endif

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
#if defined(FP32_USE_BLOCK_IO)
__attribute__((intel_reqd_sub_group_size(16)))
#endif
kernel void stencil_apply_direct(
  global const float* restrict p_grid,
  global const float* restrict p_old,
  global float* restrict p_new,
  global const float* restrict vel,
  CONSTANT float* restrict coeff,
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  global const float* restrict eta,
  global float* restrict phi,
  float hdx_2, float hdy_2, float hdz_2,
#endif
  float dt2,
  int nx, int ny, int nz)
{
  local float fm_slm[SLM_TOTAL];
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  local float eta_slm[SLM_TOTAL];
  float eta_s[3];
#endif

  const int i_f = (int)get_global_id(0);
  const int i_m = (int)get_global_id(1);
  const int is_base = (int)get_group_id(2) * BLK;
  const int lf = (int)get_local_id(0);
  const int lm = (int)get_local_id(1);
  const int lid = lm * WG_X + lf;
  const int valid_fm = (i_f < FP32_NFAST && i_m < ny);

  const int gf0 = (int)get_group_id(0) * WG_X - RADIUS;
  const int gm0 = (int)get_group_id(1) * WG_Y - RADIUS;
  float s_win[S_WINDOW];
  int i_s, r, idx, w;

#if !defined(NTERMS) || (2 < NTERMS)
  if (valid_fm) {
    UNROLL_FORCE(S_WINDOW) for (w = 0; w < S_WINDOW - 1; ++w) {
      int cs = is_base - RADIUS + w;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
      if (cs < 0) cs = 0; else if (cs >= FP32_NSLOW) cs = FP32_NSLOW - 1;
#endif
      s_win[w] = STENCIL_LOAD_P(p_grid, FP32_P_FMS(i_f, i_m, cs));
    }
  }
#endif

#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  { const int pml_w = 20;
    const int blk_f0 = (int)get_group_id(0) * WG_X;
    const int blk_m0 = (int)get_group_id(1) * WG_Y;
    const int blk_interior =
      (blk_f0 >= pml_w && blk_f0 + WG_X <= FP32_NFAST - pml_w &&
       blk_m0 >= pml_w && blk_m0 + WG_Y <= ny - pml_w &&
       is_base >= pml_w && is_base + BLK <= FP32_NSLOW - pml_w) ? 1 : 0;

    if (valid_fm && 0 == blk_interior) {
      int cs0 = (is_base > 0) ? is_base - 1 : 0;
      int cs1 = is_base;
      eta_s[0] = eta[FP32_E_FMS(i_f, i_m, cs0)];
      eta_s[1] = eta[FP32_E_FMS(i_f, i_m, cs1)];
    }
#endif

    UNROLL_OUTER(1) for (i_s = is_base; i_s < is_base + BLK && i_s < FP32_NSLOW; ++i_s) {
      float lap, p_center;

#if defined(FP32_USE_BLOCK_IO)
      { const int sgid = get_sub_group_id();
        const int sglid = get_sub_group_local_id();
        global const void* plane = (global const void*)(
          p_grid + (long)i_s * (STENCIL_NY) * (STENCIL_NX));
        const int wb = STENCIL_NX * 4;
        float8 blk_data;
        int col_base, row_base, c;
        col_base = (sgid % 3) * 16;
        row_base = (sgid / 3) * 8;
        if (sgid < 6) {
          intel_sub_group_2d_block_read_32b_8r16x1c(
            plane, wb, STENCIL_NY, wb,
            (int2)((gf0 + col_base) * 4, gm0 + row_base),
            (private uint*)&blk_data);
          for (c = 0; c < 8; ++c) {
            const int sf = col_base + sglid;
            const int sm = row_base + c;
            if (sf < SLM_F && sm < SLM_M)
              fm_slm[sm * SLM_F + sf] = ((float*)&blk_data)[c];
          }
        }
      }
#else
      for (idx = lid; idx < SLM_TOTAL; idx += WG_SIZE) {
        const int sm = idx / SLM_F;
        const int sf = idx % SLM_F;
        int gf = gf0 + sf;
        int gm = gm0 + sm;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
        if (gf < 0) gf = 0; else if (gf >= FP32_NFAST) gf = FP32_NFAST - 1;
        if (gm < 0) gm = 0; else if (gm >= ny) gm = ny - 1;
#endif
        { fm_slm[idx] = STENCIL_LOAD_P(p_grid, FP32_P_FMS(gf, gm, i_s));
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
          if (0 == blk_interior) eta_slm[idx] = eta[FP32_E_FMS(gf, gm, i_s)];
#endif
        }
      }
#endif
      barrier(CLK_LOCAL_MEM_FENCE);

      if (valid_fm) {
#if !defined(NTERMS) || (2 < NTERMS)
        { int cs = i_s + RADIUS;
#if !defined(STENCIL_PADDED) || (0 >= STENCIL_PADDED)
          if (cs >= FP32_NSLOW) cs = FP32_NSLOW - 1;
#endif
          s_win[S_WINDOW - 1] = STENCIL_LOAD_P(p_grid, FP32_P_FMS(i_f, i_m, cs));
        }
#endif
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
        if (0 == blk_interior) {
          int cs = i_s + 1;
          if (cs >= FP32_NSLOW) cs = FP32_NSLOW - 1;
          eta_s[2] = eta[FP32_E_FMS(i_f, i_m, cs)];
        }
#endif
        { const int c = (lm + RADIUS) * SLM_F + lf + RADIUS;
          CONSTANT const float* cf = FP32_COEFF_FAST;
          CONSTANT const float* cm = FP32_COEFF_MED;
          p_center = fm_slm[c];
          lap = cf[RADIUS] * p_center;
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cf[RADIUS + r] * (fm_slm[c + r] + fm_slm[c - r]);
          }
#if !defined(NTERMS) || (1 < NTERMS)
          lap += cm[RADIUS] * p_center;
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cm[RADIUS + r] * (fm_slm[c + r * SLM_F] + fm_slm[c - r * SLM_F]);
          }
#endif
        }
#if !defined(NTERMS) || (2 < NTERMS)
        { CONSTANT const float* cs = FP32_COEFF_SLOW;
          lap += cs[RADIUS] * s_win[RADIUS];
          UNROLL_FORCE(RADIUS) for (r = 1; r <= RADIUS; ++r) {
            lap += cs[RADIUS + r] * (s_win[RADIUS + r] + s_win[RADIUS - r]);
          }
        }
#endif
        { const long ip = FP32_P_FMS(i_f, i_m, i_s);
          const long iv = FP32_V_FMS(i_f, i_m, i_s);
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
          if (0 != blk_interior) {
            STENCIL_STORE_P(p_new, ip,
              2.0f * p_center - STENCIL_LOAD_P(p_old, ip) + dt2 * vel[iv] * lap);
          }
          else {
            const int c = (lm + RADIUS) * SLM_F + lf + RADIUS;
            const float eta1 = eta_slm[c];
            const float phi_val = phi[iv];
            const float p_old_val = STENCIL_LOAD_P(p_old, ip);
            const float numerator =
              (2.0f - eta1 * eta1 + 2.0f * eta1) * p_center - p_old_val
              + dt2 * vel[iv] * (lap + phi_val);
            STENCIL_STORE_P(p_new, ip, numerator / (1.0f + 2.0f * eta1));
            { const float uf_p = fm_slm[c + 1];
              const float uf_m = fm_slm[c - 1];
              const float um_p = fm_slm[c + SLM_F];
              const float um_m = fm_slm[c - SLM_F];
              const float us_p = s_win[RADIUS + 1];
              const float us_m = s_win[RADIUS - 1];
              const float eta_fp = eta_slm[c + 1];
              const float eta_fm = eta_slm[c - 1];
              const float eta_mp = eta_slm[c + SLM_F];
              const float eta_mm = eta_slm[c - SLM_F];
              const float tmp =
                (eta_fp - eta_fm) * (uf_p - uf_m) * FP32_HD_FAST
                + (eta_mp - eta_mm) * (um_p - um_m) * FP32_HD_MED
                + (eta_s[2] - eta_s[0]) * (us_p - us_m) * FP32_HD_SLOW;
              phi[iv] = (phi_val - tmp) / (1.0f + eta1);
            }
          }
#else
          STENCIL_STORE_P(p_new, ip,
            2.0f * p_center - STENCIL_LOAD_P(p_old, ip) + dt2 * vel[iv] * lap);
#endif
        }

#if !defined(NTERMS) || (2 < NTERMS)
        UNROLL_FORCE(S_WINDOW - 1) for (w = 0; w < S_WINDOW - 1; ++w) {
          s_win[w] = s_win[w + 1];
        }
#endif

#if defined(STENCIL_PML) && (0 < STENCIL_PML)
        if (0 == blk_interior) {
          eta_s[0] = eta_s[1];
          eta_s[1] = eta_s[2];
        }
#endif
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  }
#endif
}
