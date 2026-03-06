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
typedef struct libxstream_stream_t libxstream_stream_t;
typedef struct libxstream_event_t libxstream_event_t;

/** initialization and finalization */
int libxstream_init(void);
int libxstream_finalize(void);

/** devices */
int libxstream_get_ndevices(int* ndevices);
int libxstream_set_active_device(int device_id);
int libxstream_device_synchronize(void);

/** streams */
int libxstream_stream_priority_range(int* least, int* greatest);
int libxstream_stream_create(libxstream_stream_t** stream_p, const char* name,
  /** lower number is higher priority */
  int priority);
int libxstream_stream_destroy(libxstream_stream_t* stream);
int libxstream_stream_sync(libxstream_stream_t* stream);
int libxstream_stream_wait_event(libxstream_stream_t* stream, libxstream_event_t* event);

/** events */
int libxstream_event_create(libxstream_event_t** event_p);
int libxstream_event_destroy(libxstream_event_t* event);
int libxstream_event_record(libxstream_event_t* event, libxstream_stream_t* stream);
int libxstream_event_query(libxstream_event_t* event, libxstream_bool_t* has_occurred);
int libxstream_event_synchronize(libxstream_event_t* event);

/** memory */
void* libxstream_memdev_allocate(size_t nbytes);
void libxstream_memdev_deallocate(void* dev_mem);
int libxstream_memdev_set_ptr(void** dev_mem, void* other, size_t lb);
int libxstream_memdev_info(size_t* mem_free, size_t* mem_total);
void* libxstream_memhst_allocate(size_t nbytes, libxstream_stream_t* stream);
int libxstream_memhst_deallocate(void* host_mem, libxstream_stream_t* stream);
int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memset_zero(void* dev_mem, size_t offset, size_t nbytes, libxstream_stream_t* stream);

#if defined(__cplusplus)
}
#endif

#endif /*LIBXSTREAM_H*/
