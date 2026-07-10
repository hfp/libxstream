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

#if defined(R_GATHER) && (R_GATHER >= 1) && (R_GATHER < RADIUS)
# define I8_RG R_GATHER
#else
# define I8_RG RADIUS
#endif
#define I8_K_BASE_G (BLK + 2 * I8_RG)

#undef KSTEP_I8_HI
#define KSTEP_I8_HI(MI) ((((MI) + XMX_M - 1 + 2 * I8_RG) >> 2) & ~7)

#define I8_GATHER_COORD(DIM, OX, OY, OZ, K, CI, CJ, GX, GY, GZ) \
  do { \
    if (0 == (DIM))      { (GX) = (OX) + (K) - I8_RG; (GY) = (OY) + (CI); (GZ) = (OZ) + (CJ); } \
    else if (1 == (DIM)) { (GX) = (OX) + (CI); (GY) = (OY) + (K) - I8_RG; (GZ) = (OZ) + (CJ); } \
    else                 { (GX) = (OX) + (CI); (GY) = (OY) + (CJ); (GZ) = (OZ) + (K) - I8_RG; } \
  } while (0)

#define I8_K4_BASE ((K_BASE + 3) / 4)
#define I8_K4_PAD (K_PAD_I8 / 4)
#define I8_SLM_INTS (NSLICES_X * I8_K4_PAD * XMX_N)
#define I8_FILL_COUNT (I8_K4_PAD * XMX_N)
#if !defined(I8_EXP_MARGIN)
# define I8_EXP_MARGIN 1
#endif

#define I8_TOTAL_ITERS (NTERMS * STRIPS_PER_WG)

#if !defined(INTEL) || (INTEL < 2)
# define I8_RED_SLM_SIZE (WG_M_TILES * SG)
#endif

#if defined(INTEL) && (2 <= INTEL)
#define STENCIL_I8_ACC(CUR_DK, BUF_CUR, CUR_NSLICES_EFF, CUR_ASSUMED_EXP, MI, ACC_SLOT) \
  do { int sa_, sb_, ks_; \
    const int ks_lo_ = KSTEP_I8_LO(MI); \
    const int ks_hi_ = KSTEP_I8_HI(MI); \
    UNROLL_FORCE(NSLICES_A) for (sa_ = 0; sa_ < NSLICES_A; ++sa_) { \
      global const char* d_digit_ = (CUR_DK) + (long)sa_ * BLK * K_PAD_I8; \
      UNROLL_AUTO for (sb_ = 0; sb_ < (CUR_NSLICES_EFF); ++sb_) { \
        int8 pair_acc_ = (int8)(0); \
        const float pair_scale_ = dk_scale[sa_ * BLK + (MI)] \
          * EXP2I((CUR_ASSUMED_EXP) - BIAS - MANT_BITS + 7 * sb_); \
        STENCIL_I8_TRIM_CHECK(sa_, sb_, CUR_NSLICES_EFF); \
        UNROLL_FORCE(KSTEP_I8_MAX_COUNT) \
        for (ks_ = ks_lo_; ks_ <= ks_hi_; ks_ += 8) { \
          ushort8 a_i8_; \
          int8 b_i8_; \
          intel_sub_group_2d_block_read_8b_8r32x1c( \
            (global void*)d_digit_, K_PAD_I8, BLK, K_PAD_I8, \
            (int2)(ks_ * 4, (MI)), (private ushort*)&a_i8_); \
          b_i8_ = as_int8(intel_sub_group_block_read8( \
            (local const uint*)(x_slm + (BUF_CUR) + sb_ * I8_K4_PAD * XMX_N + ks_ * XMX_N))); \
          pair_acc_ = intel_sub_group_i8_i8_matrix_mad_k32( \
            as_short8(a_i8_), b_i8_, pair_acc_); \
        } \
        { union { int8 v; int a[8]; } ui_; \
          int m_; \
          ui_.v = pair_acc_; \
          UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) { \
            ((float*)&(ACC_SLOT))[m_] += (float)ui_.a[m_] * pair_scale_; \
          } \
        } \
      } \
    } \
  } while (0)
#elif defined(NV) && (2 <= NV) && defined(STENCIL_INT8) && (1 == STENCIL_INT8)
#define STENCIL_I8_DP4A(D, A, B, C) \
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(D) : "r"(A), "r"(B), "r"(C))
#define STENCIL_I8_ACC(CUR_DK, BUF_CUR, CUR_NSLICES_EFF, CUR_ASSUMED_EXP, MI, ACC_SLOT) \
  do { int sa_, sb_; \
    const int sg_lid_i8_ = (int)SGLID(); \
    if (sg_lid_i8_ < XMX_N) { \
      UNROLL_FORCE(NSLICES_A) for (sa_ = 0; sa_ < NSLICES_A; ++sa_) { \
        global const char* d_digit_ = (CUR_DK) + (long)sa_ * BLK * K_PAD_I8; \
        UNROLL_AUTO for (sb_ = 0; sb_ < (CUR_NSLICES_EFF); ++sb_) { \
          const float pair_scale_ = dk_scale[sa_ * BLK + (MI)] \
            * EXP2I((CUR_ASSUMED_EXP) - BIAS - MANT_BITS + 7 * sb_); \
          int m_; \
          STENCIL_I8_TRIM_CHECK(sa_, sb_, CUR_NSLICES_EFF); \
          UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) { \
            const int row_ = (MI) + m_; \
            const int k4_lo_ = row_ >> 2; \
            const int k4_hi_ = (row_ + 2 * RADIUS) >> 2; \
            global const int* d_row_ = (global const int*)(d_digit_ + (long)row_ * K_PAD_I8); \
            local const int* x_col_ = x_slm + (BUF_CUR) \
              + sb_ * I8_K4_PAD * XMX_N + sg_lid_i8_; \
            int dot_ = 0, k4_; \
            for (k4_ = k4_lo_; k4_ <= k4_hi_; ++k4_) { \
              STENCIL_I8_DP4A(dot_, d_row_[k4_], x_col_[k4_ * XMX_N], dot_); \
            } \
            ((float*)&(ACC_SLOT))[m_] += (float)dot_ * pair_scale_; \
          } \
        } \
      } \
    } \
  } while (0)
#else
#define STENCIL_I8_ACC(CUR_DK, BUF_CUR, CUR_NSLICES_EFF, CUR_ASSUMED_EXP, MI, ACC_SLOT) \
  do { int sa_, sb_; \
    const int sg_lid_i8_ = (int)SGLID(); \
    if (sg_lid_i8_ < XMX_N) { \
      UNROLL_FORCE(NSLICES_A) for (sa_ = 0; sa_ < NSLICES_A; ++sa_) { \
        global const char* d_digit_ = (CUR_DK) + (long)sa_ * BLK * K_PAD_I8; \
        UNROLL_AUTO for (sb_ = 0; sb_ < (CUR_NSLICES_EFF); ++sb_) { \
          const float pair_scale_ = dk_scale[sa_ * BLK + (MI)] \
            * EXP2I((CUR_ASSUMED_EXP) - BIAS - MANT_BITS + 7 * sb_); \
          int m_; \
          STENCIL_I8_TRIM_CHECK(sa_, sb_, CUR_NSLICES_EFF); \
          UNROLL_FORCE(XMX_M) for (m_ = 0; m_ < XMX_M; ++m_) { \
            const int row_ = (MI) + m_; \
            const int k4_lo_ = row_ >> 2; \
            const int k4_hi_ = (row_ + 2 * RADIUS) >> 2; \
            global const char* d_row_ = d_digit_ + (long)row_ * K_PAD_I8; \
            local const int* x_col_ = x_slm + (BUF_CUR) \
              + sb_ * I8_K4_PAD * XMX_N + sg_lid_i8_; \
            int dot_ = 0, k4_; \
            for (k4_ = k4_lo_; k4_ <= k4_hi_; ++k4_) { \
              const int xpacked_ = x_col_[k4_ * XMX_N]; \
              dot_ += (int)d_row_[k4_ * 4 + 0] * (int)(char)(xpacked_ & 0xFF); \
              dot_ += (int)d_row_[k4_ * 4 + 1] * (int)(char)((xpacked_ >> 8) & 0xFF); \
              dot_ += (int)d_row_[k4_ * 4 + 2] * (int)(char)((xpacked_ >> 16) & 0xFF); \
              dot_ += (int)d_row_[k4_ * 4 + 3] * (int)(char)((xpacked_ >> 24) & 0xFF); \
            } \
            ((float*)&(ACC_SLOT))[m_] += (float)dot_ * pair_scale_; \
          } \
        } \
      } \
    } \
  } while (0)
#endif

#if defined(TRIM) && (0 < TRIM)
# define STENCIL_I8_TRIM_CHECK(SA, SB, ND) \
    if ((SA) + (SB) >= NSLICES_A + (ND) - 1 - (TRIM - 1)) continue
#else
# define STENCIL_I8_TRIM_CHECK(SA, SB, ND) ((void)0)
#endif

#define I8_SLICE_SHIFT_0 (MANT_BITS - 6)
#define I8_SLICE_SHIFT_1 (MANT_BITS - 13)
#define I8_SLICE_SHIFT_2 (MANT_BITS - 20)

#define I8_EXTRACT_MANTISSA(BITS, ASSUMED_EXP, MANT, SIGN) \
  do { \
    const int e_ = (int)(((BITS) >> 23) & 0xFFu); \
    const int sh_ = (ASSUMED_EXP) - e_; \
    (SIGN) = (int)((BITS) >> 31); \
    (MANT) = (0 != e_) ? (((BITS) & 0x7FFFFFu) | 0x800000u) : 0; \
    if (sh_ < 0 || sh_ >= 24 || 0 == e_) (MANT) = 0; \
    else if (sh_ > 0) (MANT) >>= sh_; \
  } while (0)

#define I8_PACK_BYTE(MANT, SIGN, SHIFT) \
  ((uchar)((0 != (SIGN)) \
    ? (char)(-((char)(((MANT) >> (SHIFT)) & 0x7Fu))) \
    : (char)(((MANT) >> (SHIFT)) & 0x7Fu)))

#define I8_GATHER_PACK4(PACK, MANT0, S0, MANT1, S1, MANT2, S2, MANT3, S3) \
  do { \
    (PACK)[0] = (uint)I8_PACK_BYTE(MANT0, S0, I8_SLICE_SHIFT_0) \
              | ((uint)I8_PACK_BYTE(MANT1, S1, I8_SLICE_SHIFT_0) << 8) \
              | ((uint)I8_PACK_BYTE(MANT2, S2, I8_SLICE_SHIFT_0) << 16) \
              | ((uint)I8_PACK_BYTE(MANT3, S3, I8_SLICE_SHIFT_0) << 24); \
    (PACK)[1] = (uint)I8_PACK_BYTE(MANT0, S0, I8_SLICE_SHIFT_1) \
              | ((uint)I8_PACK_BYTE(MANT1, S1, I8_SLICE_SHIFT_1) << 8) \
              | ((uint)I8_PACK_BYTE(MANT2, S2, I8_SLICE_SHIFT_1) << 16) \
              | ((uint)I8_PACK_BYTE(MANT3, S3, I8_SLICE_SHIFT_1) << 24); \
    (PACK)[2] = (uint)I8_PACK_BYTE(MANT0, S0, I8_SLICE_SHIFT_2) \
              | ((uint)I8_PACK_BYTE(MANT1, S1, I8_SLICE_SHIFT_2) << 8) \
              | ((uint)I8_PACK_BYTE(MANT2, S2, I8_SLICE_SHIFT_2) << 16) \
              | ((uint)I8_PACK_BYTE(MANT3, S3, I8_SLICE_SHIFT_2) << 24); \
  } while (0)

#if !defined(STENCIL_BLOCKED) || (0 >= STENCIL_BLOCKED)
#if (STENCIL_LAYOUT_XYZ == STENCIL_LAYOUT)
#define I8_GATHER_LOAD4(DIM, OX, OY, OZ, K_BASE4, CI, CJ, NX, NY, NZ, P_GRID, BITS4) \
  do { \
    if (0 == (DIM) && (K_BASE4) + 3 < I8_K_BASE_G) { \
      const int gx4_ = (OX) + (K_BASE4) - I8_RG; \
      const int gy4_ = (OY) + (CI); \
      const int gz4_ = (OZ) + (CJ); \
      if (gx4_ >= 0 && gx4_ + 3 < (NX) \
          && gy4_ >= 0 && gy4_ < (NY) && gz4_ >= 0 && gz4_ < (NZ)) { \
        const uint4 v4_ = as_uint4(vload4(0, \
          (P_GRID) + (long)(gz4_) * (NY) * (NX) + (long)(gy4_) * (NX) + (gx4_))); \
        (BITS4)[0] = v4_.s0; (BITS4)[1] = v4_.s1; \
        (BITS4)[2] = v4_.s2; (BITS4)[3] = v4_.s3; \
      } \
      else { \
        int ki_; \
        UNROLL_FORCE(4) for (ki_ = 0; ki_ < 4; ++ki_) { \
          const int k_ = (K_BASE4) + ki_; \
          if (k_ < I8_K_BASE_G) { \
            int gx_, gy_, gz_; \
            I8_GATHER_COORD(DIM, OX, OY, OZ, k_, CI, CJ, gx_, gy_, gz_); \
            STENCIL_CLAMP_COORD(gx_, gy_, gz_, NX, NY, NZ); \
            (BITS4)[ki_] = STENCIL_LOAD_P_BITS(P_GRID, STENCIL_GRID_IDX(gz_, gy_, gx_, NY, NX)); \
          } \
          else (BITS4)[ki_] = 0; \
        } \
      } \
    } \
    else { \
      int ki_; \
      UNROLL_FORCE(4) for (ki_ = 0; ki_ < 4; ++ki_) { \
        const int k_ = (K_BASE4) + ki_; \
        if (k_ < I8_K_BASE_G) { \
          int gx_, gy_, gz_; \
          I8_GATHER_COORD(DIM, OX, OY, OZ, k_, CI, CJ, gx_, gy_, gz_); \
          STENCIL_CLAMP_COORD(gx_, gy_, gz_, NX, NY, NZ); \
          (BITS4)[ki_] = STENCIL_LOAD_P_BITS(P_GRID, STENCIL_P_IDX(gz_, gy_, gx_, NY, NX, 0, 0)); \
        } \
        else (BITS4)[ki_] = 0; \
      } \
    } \
  } while (0)
#else /* non-XYZ (e.g. ZYX): no X-contiguous vload4 fast-path */
#define I8_GATHER_LOAD4(DIM, OX, OY, OZ, K_BASE4, CI, CJ, NX, NY, NZ, P_GRID, BITS4) \
  do { int ki_; \
    UNROLL_FORCE(4) for (ki_ = 0; ki_ < 4; ++ki_) { \
      const int k_ = (K_BASE4) + ki_; \
      if (k_ < I8_K_BASE_G) { \
        int gx_, gy_, gz_; \
        I8_GATHER_COORD(DIM, OX, OY, OZ, k_, CI, CJ, gx_, gy_, gz_); \
        STENCIL_CLAMP_COORD(gx_, gy_, gz_, NX, NY, NZ); \
        (BITS4)[ki_] = STENCIL_LOAD_P_BITS(P_GRID, STENCIL_P_IDX(gz_, gy_, gx_, NY, NX, 0, 0)); \
      } \
      else (BITS4)[ki_] = 0; \
    } \
  } while (0)
#endif
#else
#define I8_GATHER_LOAD4(DIM, OX, OY, OZ, K_BASE4, CI, CJ, NX, NY, NZ, P_GRID, BITS4) \
  do { int ki_; \
    UNROLL_FORCE(4) for (ki_ = 0; ki_ < 4; ++ki_) { \
      const int k_ = (K_BASE4) + ki_; \
      if (k_ < I8_K_BASE_G) { \
        int gx_, gy_, gz_; \
        I8_GATHER_COORD(DIM, OX, OY, OZ, k_, CI, CJ, gx_, gy_, gz_); \
        STENCIL_CLAMP_COORD(gx_, gy_, gz_, NX, NY, NZ); \
        (BITS4)[ki_] = STENCIL_LOAD_P_BITS(P_GRID, STENCIL_P_IDX(gz_, gy_, gx_, NY, NX, nbx, nby)); \
      } \
      else (BITS4)[ki_] = 0; \
    } \
  } while (0)
#endif


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
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  global const float* restrict eta,
  global float* restrict phi,
  float hdx_2, float hdy_2, float hdz_2,
#endif
  float dt2,
  int nx, int ny, int nz,
  int nbx, int nby)
{
  const int bx = (int)get_group_id(0);
  const int by = (int)get_group_id(1);
  const int strip_grp = (int)get_group_id(2) & (N_STRIP_GROUPS - 1);
  const int bz = (int)get_group_id(2) >> STENCIL_NSTRIP_SHIFT;
  const int sg_id = (int)SGID();
  const int sg_lid = (int)SGLID();
  const int mi = sg_id * XMX_M;
  const int ox = bx * BLK;
  const int oy = by * BLK;
  const int oz = bz * BLK;

  local int x_slm[2 * I8_SLM_INTS];
  local int exp_sg[WG_M_TILES];
#if !defined(INTEL) || (INTEL < 2)
  local int red_slm[I8_RED_SLM_SIZE];
#endif

  const int fill_id = sg_id * SG + sg_lid;
  const int fill_total = WG_M_TILES * SG;
  const int blk_linear = bz * nby * nbx + by * nbx + bx;
  float8 acc[STRIPS_PER_WG];
  int iter, buf_cur, buf_next;

  UNROLL_FORCE(STRIPS_PER_WG) for (iter = 0; iter < STRIPS_PER_WG; ++iter) {
    acc[iter] = (float8)(0.0f);
  }

  /* Prolog: gather first (dim=0, strip=0) into buf[0]. */
  {
    const int nj = (strip_grp * STRIPS_PER_WG + 0) * XMX_N;
    const int strip_abs = strip_grp * STRIPS_PER_WG + 0;
    const int exp_idx = (blk_linear * NTERMS + 0) * N_STRIPS + strip_abs;
    const int assumed_exp = exp_buf[exp_idx];
    int idx;

    for (idx = fill_id; idx < I8_FILL_COUNT; idx += fill_total) {
      const int k4 = idx / XMX_N;
      const int col_local = idx % XMX_N;
      const int k_base = k4 * 4;
      const int nc = nj + col_local;
      const int ci = nc % BLK;
      const int cj = nc / BLK;
      uint bits4[4], mant4[4];
      int sign4[4], ki;
      uint pack[NSLICES_X];

      I8_GATHER_LOAD4(STENCIL_DIM(0), ox, oy, oz, k_base, ci, cj, nx, ny, nz, p_grid, bits4);

      UNROLL_FORCE(4) for (ki = 0; ki < 4; ++ki) {
        I8_EXTRACT_MANTISSA(bits4[ki], assumed_exp, mant4[ki], sign4[ki]);
      }

      I8_GATHER_PACK4(pack, mant4[0], sign4[0], mant4[1], sign4[1],
        mant4[2], sign4[2], mant4[3], sign4[3]);

      UNROLL_FORCE(NSLICES_X) for (ki = 0; ki < NSLICES_X; ++ki) {
        x_slm[ki * I8_K4_PAD * XMX_N + k4 * XMX_N + col_local] = (int)pack[ki];
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Steady state: overlap gather[iter+1]->buf[next] with DPAS[iter]<-buf[cur].
   * Global loads issued by gather fly while DPAS uses the systolic array. */
  buf_cur = 0;
  buf_next = I8_SLM_INTS;

  UNROLL_OUTER(1) for (iter = 1; iter < I8_TOTAL_ITERS; ++iter) {
    const int cur_dim = (iter - 1) / STRIPS_PER_WG;
    const int cur_strip = (iter - 1) % STRIPS_PER_WG;
    const int cur_strip_abs = strip_grp * STRIPS_PER_WG + cur_strip;
    const int cur_exp_idx = (blk_linear * NTERMS + cur_dim) * N_STRIPS + cur_strip_abs;
    const int cur_assumed_exp = exp_buf[cur_exp_idx];
    const int cur_nslices_eff = (0 == cur_assumed_exp) ? 1
      : ((cur_assumed_exp <= 7) ? 1 : ((cur_assumed_exp <= 14) ? 2 : NSLICES_X));

    const int next_dim = iter / STRIPS_PER_WG;
    const int next_strip = iter % STRIPS_PER_WG;
    const int next_nj = (strip_grp * STRIPS_PER_WG + next_strip) * XMX_N;
    const int next_strip_abs = strip_grp * STRIPS_PER_WG + next_strip;
    const int next_exp_idx = (blk_linear * NTERMS + next_dim) * N_STRIPS + next_strip_abs;
    const int next_assumed_exp = exp_buf[next_exp_idx];

    { const int cur_dim_l = STENCIL_DIM(cur_dim);
      const int next_dim_l = STENCIL_DIM(next_dim);
      global const char* cur_dk = (0 == cur_dim_l) ? dk_x : ((1 == cur_dim_l) ? dk_y : dk_z);

    STENCIL_I8_ACC(cur_dk, buf_cur, cur_nslices_eff, cur_assumed_exp, mi, acc[cur_strip]);

    { int idx;
      for (idx = fill_id; idx < I8_FILL_COUNT; idx += fill_total) {
        const int k4 = idx / XMX_N;
        const int col_local = idx % XMX_N;
        const int k_base = k4 * 4;
        const int nc = next_nj + col_local;
        const int ci = nc % BLK;
        const int cj = nc / BLK;
        uint bits4[4], mant4[4];
        int sign4[4], ki;
        uint pack[NSLICES_X];

        I8_GATHER_LOAD4(next_dim_l, ox, oy, oz, k_base, ci, cj, nx, ny, nz, p_grid, bits4);

        UNROLL_FORCE(4) for (ki = 0; ki < 4; ++ki) {
          I8_EXTRACT_MANTISSA(bits4[ki], next_assumed_exp, mant4[ki], sign4[ki]);
        }

        I8_GATHER_PACK4(pack, mant4[0], sign4[0], mant4[1], sign4[1],
          mant4[2], sign4[2], mant4[3], sign4[3]);

        UNROLL_FORCE(NSLICES_X) for (ki = 0; ki < NSLICES_X; ++ki) {
          x_slm[buf_next + ki * I8_K4_PAD * XMX_N + k4 * XMX_N + col_local] = (int)pack[ki];
        }
      }
    }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Swap buffers. */
    { const int tmp = buf_cur;
      buf_cur = buf_next;
      buf_next = tmp;
    }
  }

  /* Epilog: DPAS on the last iteration's data (already in buf[cur]). */
  {
    const int cur_dim = (I8_TOTAL_ITERS - 1) / STRIPS_PER_WG;
    const int cur_strip = (I8_TOTAL_ITERS - 1) % STRIPS_PER_WG;
    const int cur_strip_abs = strip_grp * STRIPS_PER_WG + cur_strip;
    const int cur_exp_idx = (blk_linear * NTERMS + cur_dim) * N_STRIPS + cur_strip_abs;
    const int cur_assumed_exp = exp_buf[cur_exp_idx];
    const int cur_nslices_eff = (0 == cur_assumed_exp) ? 1
      : ((cur_assumed_exp <= 7) ? 1 : ((cur_assumed_exp <= 14) ? 2 : NSLICES_X));

    { const int cur_dim_l = STENCIL_DIM(cur_dim);
      global const char* cur_dk = (0 == cur_dim_l) ? dk_x : ((1 == cur_dim_l) ? dk_y : dk_z);
      STENCIL_I8_ACC(cur_dk, buf_cur, cur_nslices_eff, cur_assumed_exp, mi, acc[cur_strip]);
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  /* Output: leapfrog update + exponent scan for exp_buf_out. */
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  { const int pml_w = 20;
    const int blk_interior =
      (ox >= pml_w && ox + BLK <= nx - pml_w &&
       oy >= pml_w && oy + BLK <= ny - pml_w &&
       oz >= pml_w && oz + BLK <= nz - pml_w) ? 1 : 0;
#endif

    UNROLL_FORCE(STRIPS_PER_WG) for (iter = 0; iter < STRIPS_PER_WG; ++iter) {
      const int nj = (strip_grp * STRIPS_PER_WG + iter) * XMX_N;
      const int strip_abs = strip_grp * STRIPS_PER_WG + iter;
      union { float8 v; float a[8]; } u;
      int out_max_exp = 0;
      int m;
      u.v = acc[iter];
      UNROLL_FORCE(XMX_M) for (m = 0; m < XMX_M; ++m) {
        const int row = mi + m;
        const int col = nj + sg_lid;
        if (0 <= row && row < BLK && sg_lid < XMX_N && col < N_TOTAL) {
          const int gx = ox + row;
          const int gy = oy + (col % BLK);
          const int gz = oz + (col / BLK);
          if (gx < nx && gy < ny && gz < nz) {
            const long i = STENCIL_P_IDX(gz, gy, gx, ny, nx, nbx, nby);
            const long iv = STENCIL_V_IDX(gz, gy, gx, ny, nx);
            float val;
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
            if (0 != blk_interior) {
              val = 2.0f * STENCIL_LOAD_P(p_grid, i) - STENCIL_LOAD_P(p_old, i)
                  + dt2 * vel[iv] * u.a[m];
            }
            else {
              const long ie = STENCIL_E_IDX(gz, gy, gx, ny, nx);
              const float eta1 = eta[ie];
              const float phi_val = phi[iv];
              const float p_cur = STENCIL_LOAD_P(p_grid, i);
              const float p_old_val = STENCIL_LOAD_P(p_old, i);
              const float numerator =
                (2.0f - eta1 * eta1 + 2.0f * eta1) * p_cur - p_old_val
                + dt2 * vel[iv] * (u.a[m] + phi_val);
              float tmp = 0.0f;
              val = numerator / (1.0f + 2.0f * eta1);
              if (gx > 0 && gx < nx - 1) {
                const long ip = STENCIL_P_IDX(gz, gy, gx + 1, ny, nx, nbx, nby);
                const long im = STENCIL_P_IDX(gz, gy, gx - 1, ny, nx, nbx, nby);
                const long ep = STENCIL_E_IDX(gz, gy, gx + 1, ny, nx);
                const long em = STENCIL_E_IDX(gz, gy, gx - 1, ny, nx);
                tmp += (eta[ep] - eta[em])
                     * (STENCIL_LOAD_P(p_grid, ip) - STENCIL_LOAD_P(p_grid, im)) * hdx_2;
              }
              if (gy > 0 && gy < ny - 1) {
                const long ip = STENCIL_P_IDX(gz, gy + 1, gx, ny, nx, nbx, nby);
                const long im = STENCIL_P_IDX(gz, gy - 1, gx, ny, nx, nbx, nby);
                const long ep = STENCIL_E_IDX(gz, gy + 1, gx, ny, nx);
                const long em = STENCIL_E_IDX(gz, gy - 1, gx, ny, nx);
                tmp += (eta[ep] - eta[em])
                     * (STENCIL_LOAD_P(p_grid, ip) - STENCIL_LOAD_P(p_grid, im)) * hdy_2;
              }
              if (gz > 0 && gz < nz - 1) {
                const long ip = STENCIL_P_IDX(gz + 1, gy, gx, ny, nx, nbx, nby);
                const long im = STENCIL_P_IDX(gz - 1, gy, gx, ny, nx, nbx, nby);
                const long ep = STENCIL_E_IDX(gz + 1, gy, gx, ny, nx);
                const long em = STENCIL_E_IDX(gz - 1, gy, gx, ny, nx);
                tmp += (eta[ep] - eta[em])
                     * (STENCIL_LOAD_P(p_grid, ip) - STENCIL_LOAD_P(p_grid, im)) * hdz_2;
              }
              phi[iv] = (phi_val - tmp) / (1.0f + eta1);
            }
#else
            val = 2.0f * STENCIL_LOAD_P(p_grid, i) - STENCIL_LOAD_P(p_old, i)
                + dt2 * vel[iv] * u.a[m];
#endif
            { const int oe = (int)((as_uint(val) >> 23) & 0xFFu);
              STENCIL_STORE_P(p_new, i, val);
              if (oe > out_max_exp) out_max_exp = oe;
            }
          }
        }
      }
#if defined(INTEL) && (0 < INTEL)
      { const int sg_out = sub_group_reduce_max(out_max_exp);
        if (0 == sg_lid) exp_sg[sg_id] = sg_out;
      }
#else
      { red_slm[sg_id * SG + sg_lid] = out_max_exp;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (0 == sg_lid) {
          int lane, sg_out = 0;
          for (lane = 0; lane < SG; ++lane) {
            const int v = red_slm[sg_id * SG + lane];
            if (v > sg_out) sg_out = v;
          }
          exp_sg[sg_id] = sg_out;
        }
      }
#endif
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
#if defined(STENCIL_PML) && (0 < STENCIL_PML)
  }
#endif
}
