/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_SRC_DBCSR_H
#define LIBXSTREAM_SRC_DBCSR_H

#include <libxstream_opencl.h>

/* Use DBCSR's profile for detailed timings (function name prefix-offset) */
#if !defined(LIBXSTREAM_PROFILE_DBCSR) && (defined(__OFFLOAD_PROFILING) || 1)
#  if defined(__DBCSR_ACC)
#    define LIBXSTREAM_PROFILE_DBCSR 8
#  endif
#endif

#if defined(LIBXSTREAM_PROFILE_DBCSR)
# define LIBXSTREAM_PROFILE_BEGIN \
  int routine_handle_; \
  if (0 != libxstream_opencl_config.profile) { \
    static const char* routine_name_ptr_ = LIBXS_FUNCNAME + LIBXSTREAM_PROFILE_DBCSR; \
    static const int routine_name_len_ = (int)sizeof(LIBXS_FUNCNAME) - (LIBXSTREAM_PROFILE_DBCSR + 1); \
    libxstream_timeset((const char**)&routine_name_ptr_, &routine_name_len_, &routine_handle_); \
  }
# define LIBXSTREAM_PROFILE_END \
  if (0 != libxstream_opencl_config.profile) libxstream_timestop(&routine_handle_)
#else
# define LIBXSTREAM_PROFILE_BEGIN
# define LIBXSTREAM_PROFILE_END
#endif

#endif /* LIBXSTREAM_SRC_DBCSR_H */
