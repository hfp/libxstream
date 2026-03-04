/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_H
#define LIBXSTREAM_H

#include "libxstream_version.h"
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** types */
typedef int libxstream_bool_t;

/** initialization and finalization */
int libxstream_acc_init(void);
int libxstream_acc_finalize(void);
void libxstream_acc_clear_errors(void);

/** devices */
int libxstream_acc_get_ndevices(int* ndevices);
int libxstream_acc_set_active_device(int device_id);
int libxstream_acc_device_synchronize(void);

/** streams */
int libxstream_acc_stream_priority_range(int* least, int* greatest);
int libxstream_acc_stream_create(void** stream_p, const char* name,
  /** lower number is higher priority */
  int priority);
int libxstream_acc_stream_destroy(void* stream);
int libxstream_acc_stream_sync(void* stream);
int libxstream_acc_stream_wait_event(void* stream, void* event);

/** events */
int libxstream_acc_event_create(void** event_p);
int libxstream_acc_event_destroy(void* event);
int libxstream_acc_event_record(void* event, void* stream);
int libxstream_acc_event_query(void* event, libxstream_bool_t* has_occurred);
int libxstream_acc_event_synchronize(void* event);

/** memory */
int libxstream_acc_dev_mem_allocate(void** dev_mem, size_t nbytes);
int libxstream_acc_dev_mem_deallocate(void* dev_mem);
int libxstream_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb);
int libxstream_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream);
int libxstream_acc_host_mem_deallocate(void* host_mem, void* stream);
int libxstream_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream);
int libxstream_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream);
int libxstream_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream);
int libxstream_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream);
int libxstream_acc_dev_mem_info(size_t* mem_free, size_t* mem_total);

#if defined(__cplusplus)
}
#endif

#endif /*LIBXSTREAM_H*/
