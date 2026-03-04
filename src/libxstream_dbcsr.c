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


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_init(void) { return libxstream_init(); }
int c_dbcsr_acc_finalize(void) { return libxstream_finalize(); }
void c_dbcsr_acc_clear_errors(void) { libxstream_clear_errors(); }

int c_dbcsr_acc_get_ndevices(int* ndevices) { return libxstream_get_ndevices(ndevices); }
int c_dbcsr_acc_set_active_device(int device_id) { return libxstream_set_active_device(device_id); }
int c_dbcsr_acc_device_synchronize(void) { return libxstream_device_synchronize(); }

int c_dbcsr_acc_stream_priority_range(int* least, int* greatest) {
  return libxstream_stream_priority_range(least, greatest);
}
int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  return libxstream_stream_create(stream_p, name, priority);
}
int c_dbcsr_acc_stream_destroy(void* stream) { return libxstream_stream_destroy(stream); }
int c_dbcsr_acc_stream_sync(void* stream) { return libxstream_stream_sync(stream); }
int c_dbcsr_acc_stream_wait_event(void* stream, void* event) {
  return libxstream_stream_wait_event(stream, event);
}

int c_dbcsr_acc_event_create(void** event_p) { return libxstream_event_create(event_p); }
int c_dbcsr_acc_event_destroy(void* event) { return libxstream_event_destroy(event); }
int c_dbcsr_acc_event_record(void* event, void* stream) { return libxstream_event_record(event, stream); }
int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred) {
  return libxstream_event_query(event, has_occurred);
}
int c_dbcsr_acc_event_synchronize(void* event) { return libxstream_event_synchronize(event); }

int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes) {
  return libxstream_dev_mem_allocate(dev_mem, nbytes);
}
int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem) { return libxstream_dev_mem_deallocate(dev_mem); }
int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb) {
  return libxstream_dev_mem_set_ptr(dev_mem, other, lb);
}
int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream) {
  return libxstream_host_mem_allocate(host_mem, nbytes, stream);
}
int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream) {
  return libxstream_host_mem_deallocate(host_mem, stream);
}
int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream) {
  return libxstream_memcpy_h2d(host_mem, dev_mem, nbytes, stream);
}
int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream) {
  return libxstream_memcpy_d2h(dev_mem, host_mem, nbytes, stream);
}
int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream) {
  return libxstream_memcpy_d2d(devmem_src, devmem_dst, nbytes, stream);
}
int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream) {
  return libxstream_memset_zero(dev_mem, offset, nbytes, stream);
}
int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total) {
  return libxstream_dev_mem_info(mem_free, mem_total);
}

#if defined(__cplusplus)
}
#endif
