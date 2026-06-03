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
#  include <libxs_timer.h>
#  include <libxs_hash.h>
#  include <libxs_str.h>

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
#  if !defined(OPENCL_KERNELS_SOURCE_TRANSPOSE)
#    error "OpenCL transpose-kernel code not found!"
#  endif
#  if !defined(OPENCL_LIBSMM_KERNELNAME_TRANS)
#    define OPENCL_LIBSMM_KERNELNAME_TRANS "trans"
#  endif
#  if !defined(OPENCL_LIBSMM_NLOCKS_TRANS)
#    define OPENCL_LIBSMM_NLOCKS_TRANS 16
#  endif
#  if !defined(OPENCL_LIBSMM_DEFAULT_BS)
#    define OPENCL_LIBSMM_DEFAULT_BS 8
#  endif
#  define OPENCL_LIBSMM_TYPESIZE(TYPEID) \
    (dbcsr_type_real_8 == (TYPEID) ? ((int)sizeof(double)) : (dbcsr_type_real_4 == (TYPEID) ? ((int)sizeof(float)) : 0 /*unknown*/))
#  define OPENCL_LIBSMM_TRANSENV(KEY) opencl_libsmm_getenv("OPENCL_LIBSMM_TRANS", KEY)

#  if defined(__cplusplus)
extern "C" {
#  endif


int libsmm_acc_transpose(const int* dev_trs_stack, int offset, int stack_size, void* dev_data, libsmm_acc_data_t datatype, int m,
  int n, int max_kernel_dim, void* stream) {
  int result = EXIT_SUCCESS;
  const int mn = m * n;
  LIBXS_ASSERT((NULL != dev_trs_stack && NULL != stream && NULL != dev_data && 0 <= offset && 0 < stack_size) || 0 == stack_size);
  LIBXS_ASSERT(0 < m && 0 < n);
  if (0 != stack_size && 1 != mn &&
      (
#  if defined(OPENCL_LIBSMM_F64)
        dbcsr_type_real_8 == datatype
#  else
        0
#  endif
        ||
#  if defined(OPENCL_LIBSMM_F32)
        dbcsr_type_real_4 == datatype
#  else
        0
#  endif
        ) &&
      mn <= (max_kernel_dim * max_kernel_dim))
  {
    static libxs_lock_t locks[OPENCL_LIBSMM_NLOCKS_TRANS];
    const libxs_timer_tick_t start = libxs_timer_tick();
    const libxstream_opencl_stream_t* const str = (const libxstream_opencl_stream_t*)stream;
    opencl_libsmm_trans_t* config;
    libxs_lock_t* lock = locks;
    opencl_libsmm_transkey_t key;
#  if (1 < OPENCL_LIBSMM_NLOCKS_TRANS)
    unsigned int hash;
#  endif
    LIBXS_MEMZERO(&key); /* potentially heterogeneous key-data (alignment gaps) */
    key.type = datatype;
    key.m = m;
    key.n = n; /* initialize key */
#  if (1 < OPENCL_LIBSMM_NLOCKS_TRANS)
    LIBXS_ASSERT(!(OPENCL_LIBSMM_NLOCKS_TRANS & (OPENCL_LIBSMM_NLOCKS_TRANS - 1))); /* POT */
    hash = libxs_hash(&key, sizeof(key), 25071975 /*seed*/);
    lock += LIBXS_MOD2(hash, OPENCL_LIBSMM_NLOCKS_TRANS);
#  endif
    /* calling clSetKernelArg/clEnqueueNDRangeKernel must be consistent */
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    config = (opencl_libsmm_trans_t*)libxs_registry_get(
      opencl_libsmm_registry, &key, sizeof(key), libxs_registry_lock(opencl_libsmm_registry));
    if (NULL == config) {
      char buffer[LIBXSTREAM_BUFFERSIZE], build_params[LIBXSTREAM_BUFFERSIZE];
      char fname[LIBXSTREAM_MAXSTRLEN];
      int nchar = LIBXS_SNPRINTF(fname, sizeof(fname),
        /* kernel name are meant to be unambiguous (BLAS-typeprefix and kernelsize) */
        "x" OPENCL_LIBSMM_KERNELNAME_TRANS "%ix%i", m, n);
#  if defined(__DBCSR_ACC)
      int routine_handle;
      c_dbcsr_timeset(LIBSMM_ACC_TRANSPOSE_ROUTINE_NAME_STRPTR, LIBSMM_ACC_TRANSPOSE_ROUTINE_NAME_LENPTR, &routine_handle);
#  endif
      if (0 < nchar && (int)sizeof(fname) > nchar) {
        const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
        const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
        const char *const env_cl = OPENCL_LIBSMM_TRANSENV("BUILDOPTS"), *const env_bm = OPENCL_LIBSMM_TRANSENV("BM");
        const char* const env_bs = OPENCL_LIBSMM_TRANSENV("BS");
        const char* const cmem = (EXIT_SUCCESS != libxstream_opencl_use_cmem(devinfo) ? "global" : "constant");
        const char* const build_format = "-DCONSTANT=%s -DINPLACE=%i -DFN=%s -DSM=%i -DSN=%i -DWG=%i -DT=%s -DBS=%i -DSLM_PAD=%i";
        const char *const env_inplace = OPENCL_LIBSMM_TRANSENV("INPLACE"), *tname = "";
#  if defined(OPENCL_LIBSMM_TRANS_INPLACE)
        const int inplace = ((m == n) && (NULL == env_inplace ? 1 : ('0' != *env_inplace)));
#  else
        const int inplace = ((m == n) && (NULL == env_inplace ? 0 : ('0' != *env_inplace)));
#  endif
        const int blockm = ((NULL == env_bm || '\0' == *env_bm) ? 0 : atoi(env_bm));
        const int bm = (0 >= blockm ? m : LIBXS_MIN(blockm, m));
        const int typesize = OPENCL_LIBSMM_TYPESIZE(datatype);
        const int slm_pad = (0 != inplace ? 0 : (LIBXS_ISPOT(n * typesize) ? 1 : 0));
        int tbs = ((NULL == env_bs || '\0' == *env_bs) ? OPENCL_LIBSMM_DEFAULT_BS : atoi(env_bs));
        opencl_libsmm_trans_t new_config;
        LIBXS_MEMZERO(&new_config);
        if (1 >= tbs) tbs = 1;
        switch (datatype) {
          case dbcsr_type_real_8: {
            tname = "char8"; /* double */
            fname[0] = 'd';
          } break;
          case dbcsr_type_real_4: {
            tname = "float";
            fname[0] = 's';
          } break;
          default: LIBXS_ASSERT('\0' == *tname);
        }
        new_config.wgsize = LIBXS_MIN((size_t)((m == bm || 0 == (m % bm)) ? bm : m), devinfo->wgsize[0]);
        new_config.bs = tbs;
        nchar = LIBXS_SNPRINTF(buffer, sizeof(buffer), "%s", NULL == env_cl ? "" : env_cl);
        if (0 <= /*<*/ nchar && (int)sizeof(buffer) > nchar) {
          nchar = LIBXS_SNPRINTF(build_params, sizeof(build_params), build_format, cmem, inplace, fname, m, n,
            LIBXS_CAST_INT(new_config.wgsize), tname, tbs, slm_pad);
        }
        if ('\0' != *tname && 0 < nchar && (int)sizeof(build_params) > nchar) {
          result = libxstream_opencl_kernel(0 /*source_kind*/, OPENCL_KERNELS_SOURCE_TRANSPOSE, fname, build_params, buffer,
            NULL /*try*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &new_config.kernel);
          if (EXIT_SUCCESS == result) {
            size_t wgsize_max;
            LIBXS_ASSERT(NULL != new_config.kernel);
            result = clGetKernelWorkGroupInfo(
              new_config.kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgsize_max, NULL);
            if (EXIT_SUCCESS == result) {
              LIBXS_ASSERT(0 < wgsize_max);
              if (wgsize_max < new_config.wgsize) {
                new_config.wgsize = wgsize_max;
                nchar = LIBXS_SNPRINTF(build_params, sizeof(build_params), build_format, cmem, inplace, fname, m, n,
                  LIBXS_CAST_INT(new_config.wgsize), tname, tbs, slm_pad);
                if (0 < nchar && (int)sizeof(build_params) > nchar) {
                  result = libxstream_opencl_kernel(0 /*source_kind*/, OPENCL_KERNELS_SOURCE_TRANSPOSE, fname, build_params, buffer,
                    NULL /*try*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &new_config.kernel);
                }
                else {
                  result = EXIT_FAILURE;
                }
              }
              if (EXIT_SUCCESS == result) {
                config = (opencl_libsmm_trans_t*)libxs_registry_set(opencl_libsmm_registry, &key, sizeof(key), &new_config,
                  sizeof(new_config), libxs_registry_lock(opencl_libsmm_registry));
                if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
                  const double duration = libxs_timer_duration(start, libxs_timer_tick());
                  LIBXS_STDIO_ACQUIRE();
                  fprintf(stderr, "INFO ACC/LIBSMM: TRANS-kernel ");
                  opencl_libsmm_write_trans_params(
                    stderr, 0 /*only_key*/, &key, NULL /*config*/, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
                  fprintf(stderr, "=");
                  opencl_libsmm_write_trans_params(
                    stderr, 0 /*only_key*/, &key, config, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
                  fprintf(stderr, " gen=%.1f ms\n", 1E3 * duration);
                  LIBXS_STDIO_RELEASE();
                }
              }
            }
          }
        }
        else if (EXIT_SUCCESS == result) {
          result = EXIT_FAILURE;
        }
      }
      else {
        result = EXIT_FAILURE;
      }
#  if defined(__DBCSR_ACC)
      c_dbcsr_timestop(&routine_handle);
#  endif
    }
    LIBXS_ASSERT((NULL != config && NULL != config->kernel && 0 < config->wgsize && 1 <= config->bs) || EXIT_SUCCESS != result);
    if (EXIT_SUCCESS == result) {
      const size_t work_size = config->wgsize * LIBXS_UPDIV(stack_size, config->bs);
      LIBXSTREAM_CHECK(result, clSetKernelArg(config->kernel, 0, sizeof(int), &offset), "set offset argument of transpose kernel");
      LIBXSTREAM_CHECK(
        result, libxstream_opencl_set_kernel_ptr(config->kernel, 1, dev_trs_stack), "set batch-list argument of transpose kernel");
      LIBXSTREAM_CHECK(
        result, libxstream_opencl_set_kernel_ptr(config->kernel, 2, dev_data), "set matrix-data argument of transpose kernel");
      if (1 < config->bs) {
        LIBXSTREAM_CHECK(
          result, clSetKernelArg(config->kernel, 3, sizeof(int), &stack_size), "set stacksize argument of transpose kernel");
        LIBXSTREAM_CHECK(
          result, clSetKernelArg(config->kernel, 4, sizeof(int), &config->bs), "set minibatch argument of transpose kernel");
      }
      LIBXSTREAM_CHECK(result,
        clEnqueueNDRangeKernel(
          str->queue, config->kernel, 1 /*work_dim*/, NULL /*offset*/, &work_size, &config->wgsize, 0, NULL, NULL),
        "launch transpose kernel");
      /* eventually update performance counters inside of locked region */
      if ((3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) && EXIT_SUCCESS == result) {
        LIBXS_STDIO_ACQUIRE();
        fprintf(stderr, "INFO ACC/LIBSMM: TRANS-kernel ");
        opencl_libsmm_write_trans_params(
          stderr, 1 /*only_key*/, &key, NULL /*config*/, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
        fprintf(stderr, "=");
        opencl_libsmm_write_trans_params(stderr, 1 /*only_key*/, &key, config, NULL /*delim*/, NULL /*begin*/, NULL /*close*/);
        fprintf(stderr, " ss=%i\n", stack_size);
        LIBXS_STDIO_RELEASE();
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  return result;
}

#  if defined(__cplusplus)
}
#  endif

#endif /*defined(__OPENCL)*/
