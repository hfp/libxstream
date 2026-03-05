/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxstream.h>
#include <libxstream_dbcsr.h>
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
    c_dbcsr_timeset((const char**)&routine_name_ptr_, &routine_name_len_, &routine_handle_); \
  }
# define LIBXSTREAM_PROFILE_END \
  if (0 != libxstream_opencl_config.profile) c_dbcsr_timestop(&routine_handle_)
#else
# define LIBXSTREAM_PROFILE_BEGIN
# define LIBXSTREAM_PROFILE_END
#endif


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_init(void) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_init();
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_finalize(void) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_finalize();
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_get_ndevices(int* ndevices) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_get_ndevices(ndevices);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_set_active_device(int device_id) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_set_active_device(device_id);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_device_synchronize(void) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_device_synchronize();
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_stream_priority_range(int* least, int* greatest) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_stream_priority_range(least, greatest);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_stream_create((libxstream_stream_t**)stream_p, name, priority);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_stream_destroy(void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_stream_destroy((libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_stream_sync(void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_stream_sync((libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_stream_wait_event(void* stream, void* event) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_stream_wait_event((libxstream_stream_t*)stream, (libxstream_event_t*)event);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_event_create(void** event_p) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_event_create((libxstream_event_t**)event_p);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_event_destroy(void* event) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_event_destroy((libxstream_event_t*)event);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_event_record(void* event, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_event_record((libxstream_event_t*)event, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_event_query((libxstream_event_t*)event, has_occurred);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_event_synchronize(void* event) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_event_synchronize((libxstream_event_t*)event);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memdev_allocate(dev_mem, nbytes);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memdev_deallocate(dev_mem);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memdev_set_ptr(dev_mem, other, lb);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memhst_allocate(host_mem, nbytes, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memhst_deallocate(host_mem, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memcpy_h2d(host_mem, dev_mem, nbytes, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memcpy_d2h(dev_mem, host_mem, nbytes, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memcpy_d2d(devmem_src, devmem_dst, nbytes, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memset_zero(dev_mem, offset, nbytes, (libxstream_stream_t*)stream);
  LIBXSTREAM_PROFILE_END;
  return result;
}

int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total) {
  int result;
  LIBXSTREAM_PROFILE_BEGIN;
  result = libxstream_memdev_info(mem_free, mem_total);
  LIBXSTREAM_PROFILE_END;
  return result;
}

#if defined(__cplusplus)
}
#endif
