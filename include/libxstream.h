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

#include "libxstream_macros.h"
#include <stddef.h>

/** types */
typedef int libxstream_bool_t;
typedef struct libxstream_stream_t libxstream_stream_t;
typedef struct libxstream_event_t libxstream_event_t;

/** initialization and finalization */
LIBXSTREAM_API int libxstream_init(void);
LIBXSTREAM_API int libxstream_finalize(void);

/** devices */
LIBXSTREAM_API int libxstream_device_count(int* ndevices);
LIBXSTREAM_API int libxstream_device_set_active(int device_id);
LIBXSTREAM_API int libxstream_device_sync(void);

/** streams */
typedef enum libxstream_stream_flags_t {
  LIBXSTREAM_STREAM_DEFAULT = 0,
  LIBXSTREAM_STREAM_LOW = 1,
  LIBXSTREAM_STREAM_HIGH = 2,
  LIBXSTREAM_STREAM_PROFILING = 4
} libxstream_stream_flags_t;
LIBXSTREAM_API int libxstream_stream_create(libxstream_stream_t** stream_p, const char* name, int flags);
LIBXSTREAM_API int libxstream_stream_destroy(libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_stream_sync(libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_stream_wait_event(libxstream_stream_t* stream, libxstream_event_t* event);
/** Enable CL_QUEUE_PROFILING_ENABLE on stream (no-op if already set). */
LIBXSTREAM_API int libxstream_stream_set_profiling(libxstream_stream_t* stream);

/** events */
LIBXSTREAM_API int libxstream_event_create(libxstream_event_t** event_p);
LIBXSTREAM_API int libxstream_event_destroy(libxstream_event_t* event);
LIBXSTREAM_API int libxstream_event_record(libxstream_event_t* event, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_event_query(libxstream_event_t* event, libxstream_bool_t* has_occurred);
LIBXSTREAM_API int libxstream_event_sync(libxstream_event_t* event);

/** memory */
LIBXSTREAM_API int libxstream_mem_allocate(void** dev_mem, size_t nbytes);
LIBXSTREAM_API int libxstream_mem_deallocate(void* dev_mem);
LIBXSTREAM_API int libxstream_mem_offset(void** dev_mem, void* other, size_t lb);
LIBXSTREAM_API int libxstream_mem_info(size_t* mem_free, size_t* mem_total);
LIBXSTREAM_API int libxstream_mem_host_allocate(void** host_mem, size_t nbytes, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_mem_host_deallocate(void* host_mem, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_mem_copy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_mem_copy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_mem_copy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, libxstream_stream_t* stream);
LIBXSTREAM_API int libxstream_mem_zero(void* dev_mem, size_t offset, size_t nbytes, libxstream_stream_t* stream);

#endif /*LIBXSTREAM_H*/
