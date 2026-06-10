/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
#  include "smm_acc_opencl.h"
#  include "smm_kernels.h"
#  include "acc_bench.h"
#  include <libxs/libxs_predict.h>
#  include <libxs/libxs_timer.h>
#  include <libxs/libxs_hash.h>

#  if !defined(LIBXSTREAM_CHECK)
#    define LIBXSTREAM_CHECK(RESULT, EXPR, MSG) \
      do { \
        const int libxstream_check_r_ = (EXPR); \
        if (EXIT_SUCCESS != libxstream_check_r_) { \
          if (EXIT_SUCCESS == (RESULT)) (RESULT) = libxstream_check_r_; \
          fprintf(stderr, "ERROR: %s (%i)\n", (MSG), libxstream_check_r_); \
        } \
      } while (0)
#  endif
#  if !defined(OPENCL_KERNELS_SOURCE_MULTIPLY)
#    error "OpenCL SMM-kernel code not found!"
#  endif
#  if !defined(OPENCL_LIBSMM_KERNELNAME_SMM)
#    define OPENCL_LIBSMM_KERNELNAME_SMM "smm"
#  endif
#  if !defined(OPENCL_LIBSMM_NLOCKS_SMM)
#    define OPENCL_LIBSMM_NLOCKS_SMM 16
#  endif
#  if !defined(OPENCL_LIBSMM_DEFAULT_BM)
#    define OPENCL_LIBSMM_DEFAULT_BM INT_MAX
#  endif
#  if !defined(OPENCL_LIBSMM_DEFAULT_BN)
#    define OPENCL_LIBSMM_DEFAULT_BN 1
#  endif
#  if !defined(OPENCL_LIBSMM_DEFAULT_BK)
#    define OPENCL_LIBSMM_DEFAULT_BK INT_MAX
#  endif
#  if !defined(OPENCL_LIBSMM_DEFAULT_BS)
#    define OPENCL_LIBSMM_DEFAULT_BS 8
#  endif
#  if !defined(OPENCL_LIBSMM_BS_MIN) && 1
#    define OPENCL_LIBSMM_BS_MIN 32
#  endif
#  if !defined(OPENCL_LIBSMM_SMM_S)
#    define OPENCL_LIBSMM_SMM_S 64
#  endif
#  if !defined(OPENCL_LIBSMM_VMIN)
#    define OPENCL_LIBSMM_VMIN 8
#  endif
#  define OPENCL_LIBSMM_TYPESIZE(TYPEID) \
    (dbcsr_type_real_8 == (TYPEID) ? ((int)sizeof(double)) : (dbcsr_type_real_4 == (TYPEID) ? ((int)sizeof(float)) : 0 /*unknown*/))
#  define OPENCL_LIBSMM_SMMENV(KEY) opencl_libsmm_getenv("OPENCL_LIBSMM_SMM", KEY)

#  if defined(__cplusplus)
extern "C" {
#  endif

#  if defined(OPENCL_KERNELS_PREDICT_MODELS)
extern libxs_predict_t* opencl_libsmm_predict_model;
#  endif
extern unsigned int opencl_libsmm_devuid;

opencl_libsmm_acc_dbm_launch_fn_t opencl_libsmm_acc_dbm_launch_fn;

#  if defined(OPENCL_LIBSMM_PFORMAT) && (0 < OPENCL_LIBSMM_PFORMAT)
void opencl_libsmm_acc_set_dbm_launch_fn(opencl_libsmm_acc_dbm_launch_fn_t launch_fn) {
  opencl_libsmm_acc_dbm_launch_fn = launch_fn;
}
#  else
int opencl_libsmm_acc_process(const int* host_param_stack, const int* dev_param_stack, int stack_size, libsmm_acc_data_t datatype,
  const void* dev_a_data, const void* dev_b_data, void* dev_c_data, int m_max, int n_max, int k_max, int max_kernel_dim,
  libxstream_bool_t def_mnk, void* stream, void* c_stream, int param_format, cl_event* event);
#  endif
int opencl_libsmm_acc_process(const int* host_param_stack, const int* dev_param_stack, int stack_size, libsmm_acc_data_t datatype,
  const void* dev_a_data, const void* dev_b_data, void* dev_c_data, int m_max, int n_max, int k_max, int max_kernel_dim,
  libxstream_bool_t def_mnk, void* stream, void* c_stream, int param_format, cl_event* event) {
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(host_param_stack); /* TODO */
  LIBXS_UNUSED(c_stream); /* TODO */
  LIBXS_ASSERT(0 == stack_size || (NULL != dev_a_data && NULL != dev_b_data && NULL != dev_c_data && NULL != dev_param_stack));
  LIBXS_ASSERT(0 < max_kernel_dim && NULL != stream && 0 <= stack_size && 0 <= m_max && 0 <= n_max && 0 <= k_max);
  if (0 != libsmm_acc_process_suitable(def_mnk, datatype, stack_size, m_max, n_max, k_max, max_kernel_dim)) {
    static libxs_lock_t locks[OPENCL_LIBSMM_NLOCKS_SMM];
    const libxs_timer_tick_t start = libxs_timer_tick();
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    const libxstream_opencl_stream_t* const str = (const libxstream_opencl_stream_t*)stream;
    const char *const env_s = OPENCL_LIBSMM_SMMENV("S"), *const env_bs = OPENCL_LIBSMM_SMMENV("BS");
    const int s = ((NULL == env_s || '\0' == *env_s) ? OPENCL_LIBSMM_SMM_S : atoi(env_s));
    int kernel_idx = 0, bs = ((NULL == env_bs || '\0' == *env_bs) ? 0 : atoi(env_bs));
    opencl_libsmm_smm_t* config;
    libxs_lock_t* lock = locks;
    opencl_libsmm_smmkey_t key;
    LIBXS_MEMZERO(&key); /* potentially heterogeneous key-data */
    key.devuid = (0 != opencl_libsmm_devuid) ? opencl_libsmm_devuid
      : ((1 != libxstream_opencl_config.devmatch && ((unsigned int)-1) != libxstream_opencl_config.devmatch)
          ? libxstream_opencl_config.devmatch
          : devinfo->uid);
    key.type = datatype;
    key.m = m_max;
    key.n = n_max;
    key.k = k_max;
#  if (1 < OPENCL_LIBSMM_NLOCKS_SMM)
    LIBXS_ASSERT(!(OPENCL_LIBSMM_NLOCKS_SMM & (OPENCL_LIBSMM_NLOCKS_SMM - 1))); /* POT */
    lock += LIBXS_MOD2(libxs_hash(&key, sizeof(key), 25071975 /*seed*/), OPENCL_LIBSMM_NLOCKS_SMM);
#  endif
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock); /* calling clSetKernelArg/clEnqueueNDRangeKernel must be consistent */
    config = (opencl_libsmm_smm_t*)libxs_registry_get(
      opencl_libsmm_registry, &key, sizeof(key), libxs_registry_lock(opencl_libsmm_registry));
#  if defined(OPENCL_KERNELS_PREDICT_MODELS)
    if (NULL == config && NULL != opencl_libsmm_predict_model) {
      libxs_predict_info_t pinfo = { 0 };
      double inputs[3], outputs[16];
      const double thr = 0.9;
      inputs[0] = (double)key.m;
      inputs[1] = (double)key.n;
      inputs[2] = (double)key.k;
      libxs_predict_eval(NULL, opencl_libsmm_predict_model, inputs, outputs, &pinfo, 0);
      if (pinfo.distance <= 2.0 && NULL != pinfo.confidence) {
        opencl_libsmm_smm_t predicted;
        LIBXS_MEMZERO(&predicted);
        if (pinfo.confidence[0] >= thr) predicted.bs = LIBXS_MAX(LIBXS_ROUNDX(int, outputs[0]), 1);
        if (pinfo.confidence[1] >= thr) predicted.bm = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[1]), 1, key.m);
        if (pinfo.confidence[2] >= thr) predicted.bn = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[2]), 1, key.n);
        if (pinfo.confidence[3] >= thr) predicted.bk = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[3]), 1, key.m);
        if (pinfo.confidence[4] >= thr) predicted.ws = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[4]), 1, key.m * key.n);
        if (pinfo.confidence[5] >= thr) predicted.wg = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[5]), -2, 1);
        if (pinfo.confidence[6] >= thr) predicted.lu = LIBXS_MAX(LIBXS_ROUNDX(int, outputs[6]), -2);
        if (pinfo.confidence[7] >= thr) predicted.nz = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[7]), 0, 1);
        if (pinfo.confidence[8] >= thr) predicted.al = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[8]), 0, 1);
        if (pinfo.confidence[9] >= thr) predicted.tb = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[9]), 0, 1);
        if (pinfo.confidence[10] >= thr) predicted.tc = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[10]), 0, 1);
        if (pinfo.confidence[11] >= thr) predicted.ap = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[11]), 0, 1);
        if (pinfo.confidence[12] >= thr) predicted.aa = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[12]), 0, 2);
        if (pinfo.confidence[13] >= thr) predicted.ab = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[13]), 0, 2);
        if (pinfo.confidence[14] >= thr) predicted.ac = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[14]), 0, 1);
        if (pinfo.confidence[15] >= thr) predicted.flags = LIBXS_CLMP(LIBXS_ROUNDX(int, outputs[15]), 0, 1);
        config = (opencl_libsmm_smm_t*)libxs_registry_set(
          opencl_libsmm_registry, &key, sizeof(key), &predicted, sizeof(predicted), libxs_registry_lock(opencl_libsmm_registry));
      }
    }
#  endif
    if (0 >= bs) bs = ((NULL != config && 0 < config->bs) ? config->bs : OPENCL_LIBSMM_DEFAULT_BS);
    /* determine kernel-kind (mini-batch vs. mini-kernel) */
    if (1 == bs || 0 > s || (bs * s) > stack_size) kernel_idx = bs = 1;
    if (NULL == config || NULL == config->kernel[kernel_idx]) {
      char buffer[LIBXSTREAM_BUFFERSIZE], build_params[LIBXSTREAM_BUFFERSIZE], fname[LIBXSTREAM_MAXSTRLEN];
      int nchar = LIBXS_SNPRINTF(fname, sizeof(fname),
        /* kernel name are meant to be unambiguous (BLAS-typeprefix and kernelsize) */
        "x" OPENCL_LIBSMM_KERNELNAME_SMM "%ix%ix%i", m_max, n_max, k_max);
#  if defined(__DBCSR_ACC)
      int routine_handle;
      c_dbcsr_timeset(LIBSMM_ACC_PROCESS_ROUTINE_NAME_STRPTR, LIBSMM_ACC_PROCESS_ROUTINE_NAME_LENPTR, &routine_handle);
#  endif
      result = ((0 < nchar && (int)sizeof(fname) > nchar) ? EXIT_SUCCESS : EXIT_FAILURE);
      if (EXIT_SUCCESS == result) {
        libxstream_opencl_atomic_fp_t tkind = libxstream_opencl_atomic_fp_no;
        const char* tname = NULL;
        switch (datatype) {
          case dbcsr_type_real_8: {
            tkind = libxstream_opencl_atomic_fp_64;
            tname = "double";
            fname[0] = 'd';
          } break;
          case dbcsr_type_real_4: {
            tkind = libxstream_opencl_atomic_fp_32;
            tname = "float";
            fname[0] = 's';
          } break;
          default: LIBXS_ASSERT(NULL == tname);
        }
        if (NULL != tname && dbcsr_type_real_8 == datatype) {
          const char* const fp64ext[] = {"cl_khr_fp64"};
          if (EXIT_SUCCESS !=
              libxstream_opencl_device_ext(libxstream_opencl_config.devices[libxstream_opencl_config.device_id], fp64ext, 1))
          {
            fprintf(stderr, "ERROR ACC/LIBSMM: device does not support FP64 (cl_khr_fp64) -- build with ELEM_TYPE=float\n");
            result = EXIT_FAILURE;
            tname = NULL;
          }
        }
        if (NULL != tname) {
          const char *extensions[] = {NULL, NULL};
          const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
          const unsigned int devuid = (0 != opencl_libsmm_devuid) ? opencl_libsmm_devuid : devinfo->uid;
          size_t nextensions = sizeof(extensions) / sizeof(*extensions), sgs = 0, wgsize_prf = 1;
          const char *const env_bm = OPENCL_LIBSMM_SMMENV("BM"), *const env_bn = OPENCL_LIBSMM_SMMENV("BN");
          const char *const env_bk = OPENCL_LIBSMM_SMMENV("BK"), *const env_ws = OPENCL_LIBSMM_SMMENV("WS");
          const char *const env_wg = OPENCL_LIBSMM_SMMENV("WG"), *const env_lu = OPENCL_LIBSMM_SMMENV("LU");
          const char *const env_nz = OPENCL_LIBSMM_SMMENV("NZ"), *const env_al = OPENCL_LIBSMM_SMMENV("AL");
          const char *const env_tb = OPENCL_LIBSMM_SMMENV("TB"), *const env_tc = OPENCL_LIBSMM_SMMENV("TC");
          const char *const env_ap = OPENCL_LIBSMM_SMMENV("AP"), *const env_aa = OPENCL_LIBSMM_SMMENV("AA");
          const char *const env_ab = OPENCL_LIBSMM_SMMENV("AB"), *const env_ac = OPENCL_LIBSMM_SMMENV("AC");
          const char *const env_xf = OPENCL_LIBSMM_SMMENV("XF"), *const env_cl = OPENCL_LIBSMM_SMMENV("BUILDOPTS");
          const char* const intel_xf = "-cl-intel-256-GRF-per-thread";
          const int blockn = ((NULL == env_bn || '\0' == *env_bn) ? 0 : atoi(env_bn));
          const int blockk = ((NULL == env_bk || '\0' == *env_bk) ? 0 : atoi(env_bk));
          const int wgmin = ((NULL == env_ws || '\0' == *env_ws) ? 0 : atoi(env_ws));
          const int default_aa = (((0x0bd0 > devuid || 0x0bdb < devuid)) ? ((k_max % OPENCL_LIBSMM_VMIN) ? 1 : 2) : 0);
          const int default_ab = (((0x0bd0 > devuid || 0x0bdb < devuid) && 0x020a != devuid) ? 3 : 0), default_ac = 0;
          const int default_bk = (((0x0bd0 > devuid || 0x0bdb < devuid || n_max < k_max) && 0x020a != devuid)
                                    ? (0 == kernel_idx ? LIBXS_MIN(OPENCL_LIBSMM_DEFAULT_BK, m_max)
                                                       : LIBXS_MIN(OPENCL_LIBSMM_VMIN, m_max))
                                    : 1);
          const int default_wg = (((0x0bd0 > devuid || 0x0bdb < devuid)) ? (0 == kernel_idx ? 0 : -2) : -1);
          const int default_lu = (0 != devinfo->intel ? -1 : 0);
          int defaults, blockm, nbm, nbn;
          opencl_libsmm_smm_t new_config;
          if (NULL == config) {
            LIBXS_MEMZERO(&new_config);
          }
          else { /* preserve kernels, performance counters, etc. */
            memcpy(&new_config, config, sizeof(opencl_libsmm_smm_t));
          }
          if (NULL == env_xf || '\0' == *env_xf) {
            if (0 != devinfo->intel && CL_DEVICE_TYPE_GPU == devinfo->type && NULL != env_cl && NULL != strstr(env_cl, intel_xf)) {
              new_config.flags = 1;
            }
          }
          else {
            new_config.flags = atoi(env_xf);
          }
          defaults = ((NULL == config || 0 != kernel_idx || (NULL != config && new_config.flags != config->flags)) ? 1 : 0);
          new_config.lu = LIBXS_MAX(-2, (NULL == env_lu || '\0' == *env_lu) ? (0 != defaults ? default_lu : config->lu)
                                                                            : atoi(env_lu)); /* populate only lower bound */
          blockm = ((NULL == env_bm || '\0' == *env_bm || 1 < new_config.lu) /* 1<LU ignores BM */
                      ? (1 >= new_config.lu ? 0 : LIBXS_UP(m_max / new_config.lu, OPENCL_LIBSMM_VMIN))
                      : atoi(env_bm));
          /* two defaults for new_config parameters: 1st - regular, 2nd - BS=1 kernel */
          new_config.bm = (0 >= blockm
                             ? (0 == kernel_idx ? ((0 != defaults || 0 >= config->bm) ? LIBXS_MIN(OPENCL_LIBSMM_DEFAULT_BM, m_max)
                                                                                      : LIBXS_CLMP(config->bm, 1, m_max))
                                                : LIBXS_MIN(OPENCL_LIBSMM_DEFAULT_BM, m_max))
                             : LIBXS_MIN(blockm, m_max));
          new_config.bn = (0 >= blockn
                             ? (0 == kernel_idx ? ((0 != defaults || 0 >= config->bn) ? LIBXS_MIN(OPENCL_LIBSMM_DEFAULT_BN, n_max)
                                                                                      : LIBXS_CLMP(config->bn, 1, n_max))
                                                : LIBXS_MIN(OPENCL_LIBSMM_DEFAULT_BN, n_max))
                             : LIBXS_MIN(blockn, n_max));
          new_config.bk = (0 >= blockk ? ((0 != defaults || 0 >= config->bk) ? default_bk : LIBXS_CLMP(config->bk, 1, m_max))
                                       : LIBXS_MIN(blockk, m_max));
          new_config.ws = (0 >= wgmin
                             ? (0 == kernel_idx ? ((0 != defaults || 0 >= config->ws) ? LIBXS_MAX(m_max, n_max)
                                                                                      : LIBXS_CLMP(config->ws, 1, n_max * m_max))
                                                : LIBXS_MAX(m_max, n_max))
                             : LIBXS_MIN(wgmin, n_max * m_max));
          new_config.wg = LIBXS_CLMP(
            (NULL == env_wg || '\0' == *env_wg) ? (0 != defaults ? default_wg : config->wg) : atoi(env_wg), -2, 2);
          new_config.nz = LIBXS_CLMP(
            (NULL == env_nz || '\0' == *env_nz) ? (0 != defaults ? /*default*/ 0 : config->nz) : atoi(env_nz), 0, 1);
#  if defined(OPENCL_LIBSMM_TODO)
          new_config.al = LIBXS_CLMP(/* bug with AL=1 and XF=1? */
            (NULL == env_al || '\0' == *env_al) ? (0 != defaults ? /*default*/ 0 : config->al) : atoi(env_al), 0, 1);
#  else
          LIBXS_UNUSED(env_al);
          new_config.al = 0;
#  endif
          new_config.tb = LIBXS_CLMP(
            (NULL == env_tb || '\0' == *env_tb) ? (0 != defaults ? /*default*/ 0 : config->tb) : atoi(env_tb), 0, 1);
          new_config.tc = LIBXS_CLMP(
            (NULL == env_tc || '\0' == *env_tc) ? (0 != defaults ? /*default*/ 1 : config->tc) : atoi(env_tc), 0, 1);
          new_config.ap = LIBXS_CLMP(
            (NULL == env_ap || '\0' == *env_ap) ? (0 != defaults ? /*default*/ 0 : config->ap) : atoi(env_ap), 0, 1);
          new_config.aa = LIBXS_CLMP(/* bug with AA=2 and XF=1? */
            (NULL == env_aa || '\0' == *env_aa) ? (0 != defaults ? default_aa : config->aa) : atoi(env_aa), 0, 2);
          new_config.ab = LIBXS_CLMP(
            (NULL == env_ab || '\0' == *env_ab) ? (0 != defaults ? default_ab : config->ab) : atoi(env_ab), 0, 2);
          new_config.ac = LIBXS_CLMP(
            (NULL == env_ac || '\0' == *env_ac) ? (0 != defaults ? default_ac : config->ac) : atoi(env_ac), 0, 1);
          if (0 >= new_config.s) new_config.s = stack_size;
          if (0 == kernel_idx || 1 >= new_config.bs) new_config.bs = bs;
          nbm = LIBXS_UPDIV(m_max, new_config.bm);
          nbn = LIBXS_UPDIV(n_max, new_config.bn);
          new_config.wgsize[kernel_idx] = LIBXS_MAX(nbm * nbn, new_config.ws);
          if (0 != new_config.wg) {
            if (1 < devinfo->wgsize[2]) { /* subgroups supported */
              if (new_config.wgsize[kernel_idx] <= devinfo->wgsize[2]) {
                sgs = devinfo->wgsize[2];
              }
              else if (new_config.wgsize[kernel_idx] <= devinfo->wgsize[1]) {
                sgs = devinfo->wgsize[1];
              }
            }
            wgsize_prf = LIBXS_UP(new_config.wgsize[kernel_idx], 0 != sgs ? sgs : devinfo->wgsize[1]);
          }
          else { /* cover exactly */
            wgsize_prf = new_config.wgsize[kernel_idx];
          }
          if (2 <= new_config.wg) wgsize_prf = LIBXS_UP2POT(wgsize_prf);
          if (wgsize_prf < (2 * new_config.wgsize[kernel_idx])) new_config.wgsize[kernel_idx] = wgsize_prf; /* limit */
          LIBXS_ASSERT(1 <= bs && 0 < new_config.wgsize[kernel_idx] && 0 < wgsize_prf);
          /* ensure minimum requested WG-size */
          while ((nbm * nbn) < new_config.ws && (nbm < m_max || nbn < n_max)) {
            if (nbn < n_max) {
              ++nbn;
            }
            else if (nbm < m_max) {
              ++nbm;
            }
          }
          if ((nbm * nbn) < new_config.ws) {
            new_config.bn = LIBXS_UPDIV(n_max, nbn);
            new_config.bm = LIBXS_UPDIV(m_max, nbm);
            new_config.wgsize[kernel_idx] = (2 > new_config.wg ? (nbm * nbn) : (LIBXS_CAST_INT(LIBXS_UP2POT(nbm * nbn))));
          }
          else { /* reset */
            nbm = LIBXS_UPDIV(m_max, new_config.bm);
            nbn = LIBXS_UPDIV(n_max, new_config.bn);
          }
          while (((0 != new_config.flags) ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0]) < new_config.wgsize[kernel_idx] &&
                 (new_config.bm < m_max || new_config.bn < n_max))
          {
            if (new_config.bn < n_max) {
              ++new_config.bn;
              nbn = LIBXS_UPDIV(n_max, new_config.bn);
            }
            else if (new_config.bm < m_max) {
              ++new_config.bm;
              nbm = LIBXS_UPDIV(m_max, new_config.bm);
            }
            new_config.wgsize[kernel_idx] = (2 > new_config.wg ? (nbm * nbn) : (LIBXS_CAST_INT(LIBXS_UP2POT(nbm * nbn))));
          }
          if (new_config.wgsize[kernel_idx] <= ((0 != new_config.flags) ? devinfo->wgsize[0] / 2 : devinfo->wgsize[0])) {
            /* SMM can be handled by device */
            const char* const cmem = (EXIT_SUCCESS != libxstream_opencl_use_cmem(devinfo) ? "global" : "constant");
            const char* const env_nrepeat = getenv("NREPEAT_SMM");
            const int typesize = OPENCL_LIBSMM_TYPESIZE(datatype);
            const int slm_a = (1 != new_config.aa ? 0 : (LIBXS_ISPOT(k_max * typesize) + 1));
            const int slm_b = (1 != new_config.ab ? 0 : (LIBXS_ISPOT(k_max * typesize) + 1));
            const int slm_c = (1 != new_config.ac ? 0 : (LIBXS_ISPOT(m_max * typesize) + 1));
            /* compose build parameters and flags */
            nchar = LIBXS_SNPRINTF(build_params, sizeof(build_params),
              "-DT=%s -DGPU=%u -DCONSTANT=%s -DWG=%i -DSG=%i -DINTEL=%i -DFN=%s -DREPEAT=%i -DLU=%i "
              "-DSM=%i -DSN=%i -DSK=%i -DBS=%i -DVL=%i %s -DBM=%i -DBN=%i -DBK=%i "
              "%s %s %s %s %s %s %s %s ", /* space! */
              tname, CL_DEVICE_TYPE_GPU == devinfo->type, cmem, LIBXS_CAST_INT(new_config.wgsize[kernel_idx]), LIBXS_CAST_INT(sgs),
              (int)(0 != devinfo->intel), fname, NULL == env_nrepeat ? 1 : atoi(env_nrepeat), new_config.lu, m_max, n_max, k_max,
              bs, OPENCL_LIBSMM_VMIN, bs == new_config.bs ? "-DBSC" : "", new_config.bm, new_config.bn, new_config.bk,
              0 == new_config.tb ? "" : "-DTRACK_B", 0 != new_config.tc ? "-DTRACK_C" : "",
              0 == new_config.nz ? "" : "-DATOMIC_INC_NZ", 0 == new_config.al ? "" : "-DAL", 0 == new_config.ap ? "" : "-DSLM_P",
              0 == new_config.aa ? "" : (1 == slm_a ? "-DSLM_A=1" : (0 != slm_a ? "-DSLM_A=2" : "-DREG_A")),
              0 == new_config.ab ? "" : (1 == slm_b ? "-DSLM_B=1" : (0 != slm_b ? "-DSLM_B=2" : "-DREG_B")),
              0 == new_config.ac ? "" : (1 == slm_c ? "-DSLM_C=1" : "-DSLM_C=2"));
            /* apply support for FP-atomics */
            if (0 < nchar && (int)sizeof(build_params) > nchar) {
              nchar = libxstream_opencl_flags_atomics(&libxstream_opencl_config.device, tkind, extensions, &nextensions,
                build_params + nchar, sizeof(build_params) - nchar);
            }
            else {
              result = EXIT_FAILURE;
            }
            if (0 < nchar && (int)sizeof(build_params) > nchar) {
              nchar = LIBXS_SNPRINTF(buffer, sizeof(buffer), "%s %s%s",
                (0 == new_config.flags || 0 == devinfo->intel || CL_DEVICE_TYPE_GPU != devinfo->type) ? "" : intel_xf,
                0 == libxstream_opencl_config.debug ? "-cl-fast-relaxed-math -cl-denorms-are-zero " : "",
                NULL == env_cl ? "" : env_cl);
              if (0 >= nchar || (int)sizeof(buffer) <= nchar) result = EXIT_FAILURE;
            }
            else {
              result = EXIT_FAILURE;
            }
          }
          else { /* matrix-size causes too large WG-size */
            result = EXIT_FAILURE;
          }
          if (EXIT_SUCCESS == result) {
            const char* const env_kernel = OPENCL_LIBSMM_SMMENV("KERNEL");
            result = libxstream_opencl_kernel(NULL == env_kernel ? 0 : 1,
              NULL == env_kernel ? OPENCL_KERNELS_SOURCE_MULTIPLY : env_kernel, fname, build_params, buffer, NULL /*cl_try*/,
              NULL /*cl_try_ok*/, extensions, nextensions, new_config.kernel + kernel_idx);
            if (EXIT_SUCCESS == result) {
              size_t wgsize_max_kernel = devinfo->wgsize[0];
              result = clGetKernelWorkGroupInfo(
                new_config.kernel[kernel_idx], device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgsize_max_kernel, NULL);
              if (EXIT_SUCCESS == result) {
                LIBXS_ASSERT(0 < new_config.wgsize[kernel_idx] && 0 < wgsize_max_kernel);
                LIBXS_ASSERT(wgsize_max_kernel <= devinfo->wgsize[0]);
                if (new_config.wgsize[kernel_idx] <= wgsize_max_kernel) { /* check planned WG-size vs kernel-specific WG-size */
                  if (NULL == config || NULL == config->kernel[kernel_idx]) {
                    config = (opencl_libsmm_smm_t*)libxs_registry_set(opencl_libsmm_registry, &key, sizeof(key), &new_config,
                      sizeof(new_config), libxs_registry_lock(opencl_libsmm_registry));
                  }
                  if (NULL != config) {
                    if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
                      const double duration = libxs_timer_duration(start, libxs_timer_tick());
                      LIBXS_STDIO_ACQUIRE();
                      fprintf(stderr, "INFO ACC/LIBSMM: SMM-kernel ");
                      opencl_libsmm_write_smm_params(
                        stderr, 0 /*only_key*/, &key, NULL /*config*/, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
                      fprintf(stderr, "=");
                      opencl_libsmm_write_smm_params(
                        stderr, 0 /*only_key*/, &key, &new_config, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
                      fprintf(stderr, " gen=%.1f ms\n", 1E3 * duration);
                      LIBXS_STDIO_RELEASE();
                    }
                  }
                  /* failed to register config */
                  else {
                    result = EXIT_FAILURE;
                  }
                }
                else {
                  if (0 != libxstream_opencl_config.verbosity) {
                    fprintf(stderr, "ERROR LIBSMM: tile-size causes too large WG-size (min(%u,%u) < %u)!\n",
                      LIBXS_CAST_UINT(wgsize_max_kernel), LIBXS_CAST_UINT(devinfo->wgsize[0]),
                      LIBXS_CAST_UINT(new_config.wgsize[kernel_idx]));
                  }
                  result = EXIT_FAILURE; /* tile-size causes too large WG-size */
                }
              }
            }
            else if (0 != libxstream_opencl_config.verbosity) {
              LIBXS_STDIO_ACQUIRE();
              fprintf(stderr, "ERROR ACC/LIBSMM: SMM-kernel ");
              opencl_libsmm_write_smm_params(
                stderr, 0 /*only_key*/, &key, NULL /*config*/, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
              fprintf(stderr, "=");
              opencl_libsmm_write_smm_params(
                stderr, 0 /*only_key*/, &key, &new_config, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
              fprintf(stderr, " failed to compile!\n");
              LIBXS_STDIO_RELEASE();
            }
          }
        }
        /* insufficient device capabilities */
        else {
          result = EXIT_FAILURE;
        }
      }
      /* remove configuration from registry to avoid infinitely retrying code generation */
      if (EXIT_SUCCESS != result && NULL != config) {
        libxs_registry_remove(opencl_libsmm_registry, &key, sizeof(key), libxs_registry_lock(opencl_libsmm_registry));
      }
#  if defined(__DBCSR_ACC)
      c_dbcsr_timestop(&routine_handle);
#  endif
    }
    LIBXS_ASSERT(EXIT_SUCCESS != result || (NULL != config && NULL != config->kernel[kernel_idx]));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->bm && config->bm <= m_max));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->bn && config->bn <= n_max));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->bk && config->bk <= m_max));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->ws && config->ws <= (m_max * n_max)));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (-2 <= config->wg && 2 >= config->wg));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (-2 <= config->lu /*&& 2 >= config->lu*/));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->nz && 1 >= config->nz));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->al && 1 >= config->al));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->tb && 1 >= config->tb));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->tc && 1 >= config->tc));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->ap && 1 >= config->ap));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->aa && 2 >= config->aa));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->ab && 2 >= config->ab));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (0 <= config->ac && 1 >= config->ac));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->wgsize[kernel_idx]));
    LIBXS_ASSERT(EXIT_SUCCESS != result || (1 <= config->s && 1 <= config->bs));
    if (EXIT_SUCCESS == result) {
      size_t work_size;
      /* scale intra-kernel batchsize according to stacksize */
      if (0 == kernel_idx && 1 < config->bs && stack_size < config->s) {
#  if defined(OPENCL_LIBSMM_BS_MIN)
        const int config_bs = LIBXS_MAX(config->bs, OPENCL_LIBSMM_BS_MIN);
#  else
        const int config_bs = config->bs;
#  endif
        bs = LIBXS_UPDIV(stack_size * config_bs, config->s - 1);
        if (config->bs < bs) bs = config->bs;
      }
      /* adjust launchsize according to intra-kernel batchsize */
      work_size = LIBXS_UPDIV(stack_size, bs) * config->wgsize[kernel_idx];
      /* calling clSetKernelArg/clEnqueueNDRangeKernel must be consistent */
      LIBXSTREAM_CHECK(
        result, libxstream_opencl_set_kernel_ptr(config->kernel[kernel_idx], 0, dev_c_data), "set C-matrix argument of SMM-kernel");
      LIBXSTREAM_CHECK(
        result, libxstream_opencl_set_kernel_ptr(config->kernel[kernel_idx], 1, dev_a_data), "set A-matrix argument of SMM-kernel");
      LIBXSTREAM_CHECK(
        result, libxstream_opencl_set_kernel_ptr(config->kernel[kernel_idx], 2, dev_b_data), "set B-matrix argument of SMM-kernel");
      LIBXSTREAM_CHECK(result, libxstream_opencl_set_kernel_ptr(config->kernel[kernel_idx], 3, dev_param_stack),
        "set batch-list argument of SMM-kernel");
      LIBXSTREAM_CHECK(result, clSetKernelArg(config->kernel[kernel_idx], 4, sizeof(int), &param_format),
        "set batch-format argument of SMM-kernel");
      if (0 == kernel_idx) {
        LIBXS_ASSERT(bs <= config->bs);
        LIBXSTREAM_CHECK(
          result, clSetKernelArg(config->kernel[kernel_idx], 5, sizeof(int), &stack_size), "set stacksize argument of SMM-kernel");
        LIBXSTREAM_CHECK(
          result, clSetKernelArg(config->kernel[kernel_idx], 6, sizeof(int), &bs), "set minibatch argument of SMM-kernel");
      }
      LIBXSTREAM_CHECK(result,
        clEnqueueNDRangeKernel(str->queue, config->kernel[kernel_idx], 1 /*work_dim*/, NULL /*offset*/, &work_size,
          config->wgsize + kernel_idx, 0, NULL, event),
        "launch SMM-kernel");
      /* eventually update performance counters inside of locked region */
      if ((3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) && 0 == param_format &&
          EXIT_SUCCESS == result)
      {
        LIBXS_STDIO_ACQUIRE();
        fprintf(stderr, "INFO ACC/LIBSMM: SMM-kernel ");
        opencl_libsmm_write_smm_params(
          stderr, 1 /*only_key*/, &key, NULL /*config*/, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
        fprintf(stderr, "=");
        opencl_libsmm_write_smm_params(stderr, 1 /*only_key*/, &key, config, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
        fprintf(stderr, " ss=%i\n", stack_size);
        LIBXS_STDIO_RELEASE();
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  else if (0 < stack_size) { /* inhomogeneous, large kernel, or unsupported datatype */
    return -1; /* TODO: document result code to trigger host-fallback */
  }
  return result;
}

#  if defined(__cplusplus)
}
#  endif

#endif /*defined(__OPENCL)*/
