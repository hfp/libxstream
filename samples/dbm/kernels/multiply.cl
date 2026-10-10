/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <opencl/libxstream_atomics.h>

#define SINT short

#if defined(PRECISION) && (1 == PRECISION)
#  define CVT(A) convert_float(A)
#  define TYPE float
#else
#  define TYPE double
#  define CVT(A) A
#endif

#if !defined(CLINEAR)
#  define XM(T) T[0]
#  define XN(T) T[1]
#  define XI IDT
#else
#  define XM(T) T[1]
#  define XN(T) T[0]
#  define XI IDX
#endif

#define XK(T) T[2]
#define XA(T, IBASE) (XM(T) - IBASE)
#define XB(T, IBASE) (XN(T) - IBASE)
#define XC(T, IBASE) (XK(T) - IBASE)

/* When exact shape is known (homogeneous batches), use K for unrolling
 * but cap at 8 to avoid instruction-cache pressure for larger K values.
 * Otherwise fall back to the BK threshold from the host. */
#if defined(DBM_M) && defined(DBM_N) && defined(DBM_K)
#  undef BK
#  if (DBM_K <= 8)
#    define BK DBM_K
#  else
#    define BK 8
#  endif
#elif !defined(BK) || (0 >= BK)
#  define BK 1
#endif

/* K-block for ROWLANE: cap at 8 to limit register pressure
 * (a_reg[BKP] + c_acc[SG] must fit in available GRF). */
#if (BK <= 8)
#  define BKP BK
#else
#  define BKP 8
#endif

/* Broadcast tile size: use exact M when known and fits in one tile,
 * avoiding wasted broadcast iterations when M < BN. */
#if defined(DBM_M) && (0 < DBM_M) && (DBM_M <= BN)
#  define BM DBM_M
#else
#  define BM BN
#endif

/* Skip zero contributions (NZ): avoids atomic overhead for zero values. */
#if defined(NZ) && (0 != NZ)
#  define DBM_ACCUMULATE(PTR, VAL) \
    do { \
      const TYPE dbm_accumulate_v_ = (VAL); \
      if ((TYPE)0 != dbm_accumulate_v_) { \
        ACCUMULATE(PTR, dbm_accumulate_v_); \
      } \
    } while (0)
#else
#  define DBM_ACCUMULATE(PTR, VAL) ACCUMULATE(PTR, VAL)
#endif

/* Override SG for optimal tile geometry.
 * For BLKRD_A (homogeneous): SG = DBM_M so each sub-group = one task.
 * For ROWLANE (row per lane): SG = 16 for efficient sub-group ops (Intel). */
#if defined(BLKRD_A) && defined(DBM_M) && defined(SG) && (DBM_M != SG)
#  undef SG
#  define SG DBM_M
#elif defined(ROWLANE) && defined(SG) && (16 < SG) && defined(INTEL) && (0 != INTEL)
#  undef SG
#  define SG 16
#endif

/* Sub-group block read for A: each lane loads one row-element from a
 * contiguous column of A. Requires DBM_M == SG (flat dispatch). */
#if defined(BLKRD_A) && defined(SG) && (0 < SG) && defined(INTEL) && \
  (0 != INTEL)
#  if defined(PRECISION) && (1 == PRECISION)
#    define A_BLOCK_READ(PTR) as_float(intel_sub_group_block_read((const global uint*)(PTR)))
#  else
#    define A_BLOCK_READ(PTR) \
      as_double(intel_sub_group_block_read_ul((const global ulong*)(PTR)))
#  endif
#endif

#if defined(A_BLOCK_READ)
#  define LOAD_A(A, IBASE, SHIFT, SHAPE, M, K) \
    CVT(A_BLOCK_READ((A) + XA(SHIFT, IBASE) + (K) * XM(SHAPE)))
#else
#  define LOAD_A(A, IBASE, SHIFT, SHAPE, M, K) \
    CVT((A)[XA(SHIFT, IBASE) + IDT(M, K, XM(SHAPE), XK(SHAPE))])
#endif

/* Row-oriented store: flush CVEC[0..N1-1] to C[M, N0..N0+N1-1] */
#define DBM_MUL_STORE(ALPHA, IBASE, SHIFT, SHAPE, C, CVEC, M, N0, N1) \
  do { \
    UNROLL_AUTO for (SINT n = 0; n < (N1); ++n) \
    { \
      DBM_ACCUMULATE( \
        (C) + XC(SHIFT, IBASE) + XI(M, n + (N0), XM(SHAPE), XN(SHAPE)), (ALPHA) * (CVEC)[n]); \
    } \
  } while (0)

/* Broadcast B values when all lanes in a sub-group handle the same task.
 * Requires WG > 0 (guarantees intel_reqd_sub_group_size) and stride >= SG
 * so sub-group boundaries align with task boundaries. Only valid for
 * homogeneous batches (DBM_M) where all lanes are unconditionally active. */
#if defined(WG) && (0 < WG) && defined(BCST_SG) && defined(DBM_M) && (DBM_M >= SG)
#  define LOAD_B(V) BCST_SG(V, 0)
#else
#  define LOAD_B(V) (V)
#endif

/* Row-oriented kernel: one row M, columns N0..N0+BN-1 */
#define DBM_MUL_KERNEL(IBASE, SHIFT, SHAPE, A, B, CVEC, M, N0, BN) \
  do { \
    UNROLL_AUTO for (SINT k = 0; k < XK(SHAPE); ++k) \
    { \
      const int dbm_mul_kernel_ik_ = IDX(k, N0, XK(SHAPE), XN(SHAPE)); \
      const TYPE dbm_mul_kernel_ak_ = LOAD_A(A, IBASE, SHIFT, SHAPE, M, k); \
      UNROLL_AUTO for (SINT n = 0; n < (BN); ++n) \
      { \
        (CVEC)[n] = MAD(dbm_mul_kernel_ak_, LOAD_B(CVT((B)[dbm_mul_kernel_ik_ + n])), (CVEC)[n]); \
      } \
    } \
  } while (0)

/* Row-oriented multiply: tiles N by BN, compile-time BK */
#define DBM_MUL(ALPHA, IBASE, SHIFT, SHAPE, A, B, C, CVEC, M, BN) \
  do { \
    SINT dbm_mul_n0_ = 0, dbm_mul_n1_ = XN(SHAPE) - (BN); \
    UNROLL_FORCE(BN) for (SINT n = 0; n < (BN); ++n) \
    { \
      (CVEC)[n] = ZERO; \
    } \
    UNROLL_AUTO for (; dbm_mul_n0_ <= dbm_mul_n1_; dbm_mul_n0_ += (BN)) \
    { \
      DBM_MUL_KERNEL(IBASE, SHIFT, SHAPE, A, B, CVEC, M, dbm_mul_n0_, BN); \
      DBM_MUL_STORE(ALPHA, IBASE, SHIFT, SHAPE, C, CVEC, M, dbm_mul_n0_, BN); \
      UNROLL_FORCE(BN) for (SINT n = 0; n < (BN); ++n) \
      { \
        (CVEC)[n] = ZERO; \
      } \
    } \
    dbm_mul_n1_ = XN(SHAPE) - dbm_mul_n0_; \
    DBM_MUL_KERNEL(IBASE, SHIFT, SHAPE, A, B, CVEC, M, dbm_mul_n0_, dbm_mul_n1_); \
    DBM_MUL_STORE(ALPHA, IBASE, SHIFT, SHAPE, C, CVEC, M, dbm_mul_n0_, dbm_mul_n1_); \
  } while (0)

#if defined(SGBCST) && defined(BCST_SG)
/* Column-oriented store: flush CVEC[0..ME-1] to C[MB..MB+ME-1, N] */
#  define DBM_BCST_STORE(ALPHA, IBASE, SHIFT, SHAPE, C, CVEC, N, MB, ME) \
    do { \
      UNROLL_AUTO for (SINT m = 0; m < (ME); ++m) \
      { \
        DBM_ACCUMULATE((C) + XC(SHIFT, IBASE) + XI(m + (MB), N, XM(SHAPE), XN(SHAPE)), \
          (ALPHA) * (CVEC)[m]); \
      } \
    } while (0)

/* Column-oriented kernel: one column N, rows MB..MB+BN-1 via broadcast */
#  define DBM_BCST_KERNEL(IBASE, SHIFT, SHAPE, A, B, CVEC, SID, N, MB, BN) \
    do { \
      UNROLL_AUTO for (SINT k = 0; k < XK(SHAPE); ++k) \
      { \
        const TYPE dbm_bcst_kernel_ak_ = CVT( \
          (A)[XA(SHIFT, IBASE) + \
              IDT(MIN((SINT)(SID) + (MB), (SINT)(XM(SHAPE) - 1)), k, XM(SHAPE), XK(SHAPE))]); \
        const TYPE dbm_bcst_kernel_bv_ = CVT((B)[IDX(k, N, XK(SHAPE), XN(SHAPE))]); \
        UNROLL_FORCE(BN) for (SINT m = 0; m < (BN); ++m) \
        { \
          (CVEC)[m] = MAD(BCST_SG(dbm_bcst_kernel_ak_, (uint)m), dbm_bcst_kernel_bv_, (CVEC)[m]); \
        } \
      } \
    } while (0)

/* Column-oriented multiply: tiles M by BN, compile-time BK.
 * ACTIVE=0 for inactive lanes (participate in broadcasts, skip C writes). */
#  define DBM_BCST(ALPHA, IBASE, SHIFT, SHAPE, A, B, C, CVEC, SID, N, ACTIVE, BN) \
    do { \
      const SINT dbm_bcst_xm_ = XM(SHAPE); \
      UNROLL_AUTO for (SINT mb = 0; mb < dbm_bcst_xm_; mb += (BN)) \
      { \
        UNROLL_FORCE(BN) for (SINT m = 0; m < (BN); ++m) \
        { \
          (CVEC)[m] = ZERO; \
        } \
        DBM_BCST_KERNEL(IBASE, SHIFT, SHAPE, A, B, CVEC, SID, N, mb, BN); \
        if (ACTIVE) { \
          DBM_BCST_STORE( \
            ALPHA, IBASE, SHIFT, SHAPE, C, CVEC, N, mb, MIN(dbm_bcst_xm_ - mb, (SINT)(BN))); \
        } \
      } \
    } while (0)
#endif

/* the host launches a work-group per FUSE tasks for these, which the flat path misreads */
#if (defined(SGBCST) || defined(ROWLANE)) && !defined(BCST_SG)
#  error "SGBCST and ROWLANE need sub-group broadcast (SG_EXACT)"
#endif
#if defined(ROWLANE) && defined(CLINEAR)
#  error "ROWLANE stores C column-major (no CLINEAR)"
#endif

/* Bits per shape field of a packed param_format (host: DBM_OPENCL_LIBSMM_PFORMAT) */
#if !defined(PFORMAT) || (0 >= PFORMAT)
#  undef PFORMAT
#  define PFORMAT 8
#endif
#define PSHAPE(PFMT, I) (((1 << PFORMAT) - 1) & ((PFMT) >> (PFORMAT * (I))))

/* Decode task parameters: shape, ibase, params offset */
#define DBM_TASK_DECODE(ITASK, TID, PFMT, PARAMS, SHAPE, IBASE) \
  do { \
    if (0 == (PFMT)) { \
      const int dbm_task_decode_i_ = ((ITASK) + (TID)) * 6; \
      (SHAPE)[0] = (PARAMS)[dbm_task_decode_i_ + 0]; \
      (SHAPE)[1] = (PARAMS)[dbm_task_decode_i_ + 1]; \
      (SHAPE)[2] = (PARAMS)[dbm_task_decode_i_ + 2]; \
      (PARAMS) += dbm_task_decode_i_ + 3; \
    } \
    else { \
      (SHAPE)[0] = (SINT)PSHAPE(PFMT, 0); \
      (SHAPE)[1] = (SINT)PSHAPE(PFMT, 1); \
      (SHAPE)[2] = (SINT)PSHAPE(PFMT, 2); \
      (PARAMS) += ((ITASK) + (TID)) * 3; \
      (IBASE) = 1; \
    } \
  } while (0)

/* Override shape with compile-time constants (homogeneous batches).
 * Applied after DBM_TASK_DECODE to enable constant folding. The order is the
 * decoded one: XM and XN already swap the roles of M and N for CLINEAR. */
#if defined(DBM_M) && defined(DBM_N) && defined(DBM_K)
#  define DBM_SHAPE_OVERRIDE(SHAPE) \
    do { \
      (SHAPE)[0] = DBM_M; \
      (SHAPE)[1] = DBM_N; \
      (SHAPE)[2] = DBM_K; \
    } while (0)
#else
#  define DBM_SHAPE_OVERRIDE(SHAPE) \
    do { \
      (void)(SHAPE); \
    } while (0)
#endif

/* Tasks per work-group of the per-task dispatch: a run of consecutive tasks sharing A
 * (hence M and K) is one product over the run's concatenated columns */
#if !defined(FUSE) || (1 > FUSE)
#  undef FUSE
#  define FUSE 1
#endif

/* Run of tasks sharing A from T0 up to T (before T1), with task T0's SHIFT/SHAPE/IBASE */
#define DBM_RUN(ITASK, T0, T1, PFMT, PARAMS, SHIFT, SHAPE, IBASE, T, NCOL) \
  do { \
    (SHIFT) = (PARAMS); \
    DBM_TASK_DECODE(ITASK, T0, PFMT, SHIFT, SHAPE, IBASE); \
    DBM_SHAPE_OVERRIDE(SHAPE); \
    (NCOL) = XN(SHAPE); \
    for ((T) = (T0) + 1; (T) < (T1); ++(T)) { \
      CONSTANT const int* restrict dbm_run_shift_ = (PARAMS); \
      SINT dbm_run_shape_[3], dbm_run_ibase_ = 0; \
      DBM_TASK_DECODE(ITASK, T, PFMT, dbm_run_shift_, dbm_run_shape_, dbm_run_ibase_); \
      DBM_SHAPE_OVERRIDE(dbm_run_shape_); \
      if (XA(dbm_run_shift_, dbm_run_ibase_) != XA(SHIFT, IBASE) || \
          XM(dbm_run_shape_) != XM(SHAPE) || XK(dbm_run_shape_) != XK(SHAPE)) \
      { \
        break; \
      } \
      (NCOL) += XN(dbm_run_shape_); \
    } \
  } while (0)

/* Advances a lane's task J (run ends before T, J's columns start at JCOL0) to cover COL */
#define DBM_RUN_SEEK(ITASK, T, PFMT, PARAMS, J, JSHIFT, JSHAPE, JBASE, JCOL0, COL) \
  do { \
    while ((JCOL0) + XN(JSHAPE) <= (COL) && (J) + 1 < (T)) { \
      (JCOL0) += XN(JSHAPE); \
      ++(J); \
      (JSHIFT) = (PARAMS); \
      (JBASE) = 0; \
      DBM_TASK_DECODE(ITASK, J, PFMT, JSHIFT, JSHAPE, JBASE); \
      DBM_SHAPE_OVERRIDE(JSHAPE); \
    } \
  } while (0)

/* Row M of A (XM x XK at AL, the sub-group's rows start at MB) times column BCOL of B
 * (BIDX columns at offset BOFF): accumulates SG columns into CACC by SG-broadcast */
#define DBM_ROW_MAD(AL, XM, XK, MB, M, B, BOFF, BCOL, BIDX, VALID, CACC) \
  do { \
    SINT dbm_row_k_ = 0; \
    UNROLL_AUTO for (; dbm_row_k_ + BKP <= (XK); dbm_row_k_ += BKP) { \
      TYPE dbm_row_a_[BKP]; \
      if ((MB) + SG <= (XM)) { /* no row of the sub-group is out of range */ \
        UNROLL_FORCE(BKP) for (SINT kb = 0; kb < BKP; ++kb) { \
          dbm_row_a_[kb] = CVT((AL)[IDT(M, dbm_row_k_ + kb, XM, XK)]); \
        } \
      } \
      else { \
        UNROLL_FORCE(BKP) for (SINT kb = 0; kb < BKP; ++kb) { \
          dbm_row_a_[kb] = ((M) < (XM)) ? CVT((AL)[IDT(M, dbm_row_k_ + kb, XM, XK)]) : ZERO; \
        } \
      } \
      UNROLL_FORCE(BKP) for (SINT kb = 0; kb < BKP; ++kb) { \
        const TYPE dbm_row_b_ = \
          (VALID) ? CVT((B)[(BOFF) + IDX(dbm_row_k_ + kb, BCOL, XK, BIDX)]) : ZERO; \
        UNROLL_FORCE(SG) for (SINT n = 0; n < SG; ++n) { \
          (CACC)[n] = MAD(dbm_row_a_[kb], BCST_SG(dbm_row_b_, (uint)n), (CACC)[n]); \
        } \
      } \
    } \
    UNROLL_AUTO for (; dbm_row_k_ < (XK); ++dbm_row_k_) { /* K remainder */ \
      const TYPE dbm_row_a_ = \
        ((MB) + SG <= (XM) || (M) < (XM)) ? CVT((AL)[IDT(M, dbm_row_k_, XM, XK)]) : ZERO; \
      const TYPE dbm_row_b_ = (VALID) ? CVT((B)[(BOFF) + IDX(dbm_row_k_, BCOL, XK, BIDX)]) : ZERO; \
      UNROLL_FORCE(SG) for (SINT n = 0; n < SG; ++n) { \
        (CACC)[n] = MAD(dbm_row_a_, BCST_SG(dbm_row_b_, (uint)n), (CACC)[n]); \
      } \
    } \
  } while (0)

#if defined(WG) && (0 < WG)
__attribute__((reqd_work_group_size(WG, 1, 1))) REQD_SG
#endif
kernel void
dbm_multiply(double alpha, int itask, int ntasks, int size, int param_format,
  CONSTANT const int* restrict params,
/* CLINEAR swaps a/b in the signature so that the host's
 * fixed arg order (adata=arg6, bdata=arg7) transposes
 * the access pattern for coalesced memory reads. */
#if !defined(CLINEAR)
  CONSTANT const double* restrict a, CONSTANT const double* restrict b,
#else
  CONSTANT const double* restrict b, CONSTANT const double* restrict a,
#endif
  global double* restrict c)
{
#if defined(SM) && (0 < SM) && defined(WG) && (0 < WG) /* tls needs a fixed work-group size */
  local TYPE tls[WG][BN + SM - 1];
  local TYPE* restrict const cvec = &tls[get_local_id(0)][0];
#else
  TYPE cvec[BN];
#endif
#if defined(ROWLANE) && defined(BCST_SG) && defined(FUSEC)
  /* per-task dispatch, row per lane, tasks grouped by C: a run of tasks sharing C
   * (hence M and N) accumulates in registers, and C is written once per run */
  const int t1 = MIN((int)get_group_id(0) * FUSE + FUSE, ntasks);
  const SINT sid = (SINT)SGLID();
  int t0 = (int)get_group_id(0) * FUSE;
  while (t0 < t1) {
    CONSTANT const int* restrict shift = params;
    SINT shape[3], ibase = 0;
    int t = t0 + 1;
    DBM_TASK_DECODE(itask, t0, param_format, shift, shape, ibase);
    DBM_SHAPE_OVERRIDE(shape);
    for (; t < t1; ++t) { /* extend the run */
      CONSTANT const int* restrict tshift = params;
      SINT tshape[3], tbase = 0;
      DBM_TASK_DECODE(itask, t, param_format, tshift, tshape, tbase);
      if (XC(tshift, tbase) != XC(shift, ibase)) break;
    }
    {
      const SINT xm = XM(shape), xn = XN(shape);
      const int coff = XC(shift, ibase);
      TYPE c_acc[SG];
      UNROLL_OUTER(1)
      for (SINT mb = (SINT)(get_local_id(0) / SG * SG); mb < xm; mb += (SINT)get_local_size(0)) {
        const SINT m = mb + sid;
        UNROLL_AUTO for (SINT n0 = 0; n0 < xn; n0 += SG) {
          const SINT col = n0 + sid;
          UNROLL_FORCE(SG) for (SINT i = 0; i < SG; ++i) c_acc[i] = ZERO;
          for (int j = t0; j < t; ++j) {
            CONSTANT const int* restrict jshift = params;
            SINT jshape[3], jbase = 0;
            DBM_TASK_DECODE(itask, j, param_format, jshift, jshape, jbase);
            DBM_SHAPE_OVERRIDE(jshape);
            DBM_ROW_MAD(a + XA(jshift, jbase), xm, XK(jshape), mb, m, b, XB(jshift, jbase), col, xn,
              col < xn, c_acc);
          }
          if (m < xm) {
            const SINT ncols = MIN((SINT)SG, xn - n0);
            UNROLL_AUTO for (SINT n = 0; n < ncols; ++n) {
              DBM_ACCUMULATE(c + coff + XI(m, n0 + n, xm, xn), alpha * c_acc[n]);
            }
          }
        }
      }
    }
    t0 = t;
  }
#elif defined(ROWLANE) && defined(BCST_SG)
  /* per-task dispatch, row per lane: N is tiled by SG columns, which
   * sub_group_broadcast fans out per K-step (block-reading A gave wrong
   * results at some offsets and was slower than plain loads) */
  const int t1 = MIN((int)get_group_id(0) * FUSE + FUSE, ntasks);
  const SINT sid = (SINT)SGLID();
  int t0 = (int)get_group_id(0) * FUSE;
  while (t0 < t1) {
    CONSTANT const int* restrict shift;
    SINT shape[3], ibase = 0;
    int t, ncol;
    DBM_RUN(itask, t0, t1, param_format, params, shift, shape, ibase, t, ncol);
    {
      CONSTANT const double* restrict al = a + XA(shift, ibase);
      const SINT xm = XM(shape), xk = XK(shape);
      TYPE c_acc[SG];
      /* M-tiling: each sub-group handles SG consecutive rows */
      UNROLL_OUTER(1)
      for (SINT mb = (SINT)(get_local_id(0) / SG * SG); mb < xm; mb += (SINT)get_local_size(0)) {
        const SINT m = mb + sid;
        CONSTANT const int* restrict jshift = shift;
        SINT jshape[3] = {shape[0], shape[1], shape[2]}, jbase = ibase;
        int j = t0, jcol0 = 0;
        UNROLL_AUTO for (int n0 = 0; n0 < ncol; n0 += SG) {
          const int col = n0 + sid;
          SINT jcol;
          int boff, coff;
          DBM_RUN_SEEK(itask, t, param_format, params, j, jshift, jshape, jbase, jcol0, col);
          jcol = (SINT)(col < ncol ? (col - jcol0) : 0);
          boff = XB(jshift, jbase);
          coff = XC(jshift, jbase);
          UNROLL_FORCE(SG) for (SINT i = 0; i < SG; ++i) c_acc[i] = ZERO;
          DBM_ROW_MAD(al, xm, xk, mb, m, b, boff, jcol, XN(jshape), col < ncol, c_acc);
          { /* store: column n belongs to the task of lane n (broadcast by all lanes) */
            const int ncols = MIN(SG, ncol - n0);
            UNROLL_AUTO for (int n = 0; n < ncols; ++n) {
              const int cn = BCST_SG(coff, (uint)n), jn = BCST_SG((int)jcol, (uint)n);
              if (m < xm) {
                DBM_ACCUMULATE(c + cn + XI(m, jn, xm, 0), alpha * c_acc[n]);
              }
            }
          }
        }
      }
    }
    t0 = t;
  }
#elif defined(SGBCST) && defined(BCST_SG) && !defined(BLKRD_A)
  /* per-task dispatch, column per lane: broadcast shares A */
  const int t1 = MIN((int)get_group_id(0) * FUSE + FUSE, ntasks);
  const SINT sid = (SINT)get_local_id(0);
  /* rows are broadcast within a sub-group, which is narrower than WG if WG > SG */
  const SINT lane = (SINT)SGLID();
  int t0 = (int)get_group_id(0) * FUSE;
  while (t0 < t1) {
    CONSTANT const int* restrict shift;
    SINT shape[3], ibase = 0;
    int t, ncol;
    DBM_RUN(itask, t0, t1, param_format, params, shift, shape, ibase, t, ncol);
    {
      CONSTANT const int* restrict jshift = shift;
      SINT jshape[3] = {shape[0], shape[1], shape[2]}, jbase = ibase;
      int j = t0, jcol0 = 0;
      UNROLL_AUTO for (int nb0 = 0; nb0 < ncol; nb0 += WG) { /* all lanes */
        const int col = nb0 + sid, active = (col < ncol);
        DBM_RUN_SEEK(itask, t, param_format, params, j, jshift, jshape, jbase, jcol0, col);
        DBM_BCST(alpha, jbase, jshift, jshape, a, b + XB(jshift, jbase), c, cvec, lane,
          active ? (col - jcol0) : 0, active, BM);
      }
    }
    t0 = t;
  }
#else
  /* flat dispatch: global work-item maps to (task, row) */
  const int i = (int)get_global_id(0);
#  if defined(WG) && (0 < WG)
  if (i < size)
#  endif
  {
    /* rows per task as the host counted them: N instead of M for CLINEAR */
#  if defined(DBM_M) && defined(DBM_N) && !defined(CLINEAR)
    const SINT rows = DBM_M;
#  elif defined(DBM_M) && defined(DBM_N)
    const SINT rows = DBM_N;
#  elif defined(MAX_M)
    const SINT rows = MAX_M;
#  elif !defined(CLINEAR)
    const SINT rows = (0 != param_format ? (SINT)PSHAPE(param_format, 0) : (SINT)(size / ntasks));
#  else
    const SINT rows = (0 != param_format ? (SINT)PSHAPE(param_format, 1) : (SINT)(size / ntasks));
#  endif
    SINT shape[3], ibase = 0, m;
    const int tid = i / rows;
    m = i - tid * rows;
    DBM_TASK_DECODE(itask, tid, param_format, params, shape, ibase);
    DBM_SHAPE_OVERRIDE(shape);
    if (m < XM(shape)) {
      b += XB(params, ibase);
      DBM_MUL(alpha, ibase, params, shape, a, b, c, cvec, m, BN);
    }
  }
#endif
}
