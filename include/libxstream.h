/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSTREAM_H
#define LIBXSTREAM_H

#include "libxstream_macros.h"
#include <stdint.h>

#ifdef __cplusplus
# include <stddef.h>
#endif


/** Boolean state. */
LIBXSTREAM_EXPORT_C typedef int libxstream_bool;
/** Stream type. */
LIBXSTREAM_EXPORT_C typedef struct libxstream_stream libxstream_stream;
/** Event type. */
LIBXSTREAM_EXPORT_C typedef struct libxstream_event libxstream_event;
/** Enumeration of elemental "scalar" types. */
LIBXSTREAM_EXPORT_C typedef enum libxstream_type {
  LIBXSTREAM_TYPE_UNKNOWN,
  LIBXSTREAM_TYPE_I8, LIBXSTREAM_TYPE_U8,
  LIBXSTREAM_TYPE_I16, LIBXSTREAM_TYPE_U16,
  LIBXSTREAM_TYPE_I32, LIBXSTREAM_TYPE_U32,
  LIBXSTREAM_TYPE_I64, LIBXSTREAM_TYPE_U64,
  LIBXSTREAM_TYPE_F32, LIBXSTREAM_TYPE_F64,
  LIBXSTREAM_TYPE_C32, LIBXSTREAM_TYPE_C64,
  LIBXSTREAM_TYPE_CHAR,
  LIBXSTREAM_TYPE_BYTE = LIBXSTREAM_TYPE_U8
} libxstream_type;
/** Function argument type. */
LIBXSTREAM_EXPORT_C typedef struct LIBXSTREAM_TARGET(mic) libxstream_argument libxstream_argument;
/** Function type of an offloadable function. */
LIBXSTREAM_EXPORT_C typedef LIBXSTREAM_TARGET(mic) void (/*LIBXSTREAM_CDECL*/*libxstream_function)(LIBXSTREAM_VARIADIC);

/** Query the number of available devices. */
LIBXSTREAM_EXPORT_C int libxstream_get_ndevices(size_t* ndevices);
/** Query the device set active for this thread. */
LIBXSTREAM_EXPORT_C int libxstream_get_active_device(int* device);
/** Set the active device for this thread. */
LIBXSTREAM_EXPORT_C int libxstream_set_active_device(int device);

/** Query the memory metrics of the given device (valid to pass one NULL pointer). */
LIBXSTREAM_EXPORT_C int libxstream_mem_info(int device, size_t* allocatable, size_t* physical);
/** Allocate aligned memory (0: automatic) on the given device. */
LIBXSTREAM_EXPORT_C int libxstream_mem_allocate(int device, void** memory, size_t size, size_t alignment);
/** Deallocate memory; shall match the device where the memory was allocated. */
LIBXSTREAM_EXPORT_C int libxstream_mem_deallocate(int device, const void* memory);
/** Fill memory with zeros; allocated memory can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memset_zero(void* memory, size_t size, libxstream_stream* stream);
/** Copy memory from the host to the device; addresses can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t size, libxstream_stream* stream);
/** Copy memory from the device to the host; addresses can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t size, libxstream_stream* stream);
/** Copy memory from device to device; cross-device copies are allowed as well. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_d2d(const void* src, void* dst, size_t size, libxstream_stream* stream);

/** Query the range of valid priorities (inclusive bounds). */
LIBXSTREAM_EXPORT_C int libxstream_stream_priority_range(int* least, int* greatest);
/** Create a stream on a device (demux<0: auto-locks, 0: manual, demux>0: sync.). */
LIBXSTREAM_EXPORT_C int libxstream_stream_create(libxstream_stream** stream, int device, int demux, int priority, const char* name);
/** Destroy a stream; pending work must be completed if results are needed. */
LIBXSTREAM_EXPORT_C int libxstream_stream_destroy(libxstream_stream* stream);
/** Wait for a stream to complete pending work; NULL to synchronize all streams. */
LIBXSTREAM_EXPORT_C int libxstream_stream_sync(libxstream_stream* stream);
/** Wait for an event recorded earlier; NULL increases the match accordingly. */
LIBXSTREAM_EXPORT_C int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event);
/** Lock a stream such that the caller thread can safely enqueue work. */
LIBXSTREAM_EXPORT_C int libxstream_stream_lock(libxstream_stream* stream);
/** Unlock a stream such that another thread can acquire the stream. */
LIBXSTREAM_EXPORT_C int libxstream_stream_unlock(libxstream_stream* stream);

/** Create an event; can be used multiple times to record an event. */
LIBXSTREAM_EXPORT_C int libxstream_event_create(libxstream_event** event);
/** Destroy an event; does not implicitly waits for the completion of the event. */
LIBXSTREAM_EXPORT_C int libxstream_event_destroy(libxstream_event* event);
/** Record an event; an event can be re-recorded multiple times. */
LIBXSTREAM_EXPORT_C int libxstream_event_record(libxstream_event* event, libxstream_stream* stream);
/** Check whether an event has occurred or not (non-blocking). */
LIBXSTREAM_EXPORT_C int libxstream_event_query(const libxstream_event* event, libxstream_bool* has_occured);
/** Wait for an event to complete i.e., work queued prior to recording the event. */
LIBXSTREAM_EXPORT_C int libxstream_event_synchronize(libxstream_event* event);

/** Create an input function argument; takes the data, dim., shape, and pitch. */
LIBXSTREAM_EXPORT_C const libxstream_argument* libxstream_fn_create_in(const void* in, libxstream_type type, size_t dims, const size_t shape[], const size_t pitch[]);
/** Create an output function argument; takes the data, dim., shape, and pitch. */
LIBXSTREAM_EXPORT_C const libxstream_argument* libxstream_fn_create_out(void* out, libxstream_type type, size_t dims, const size_t shape[], const size_t pitch[]);
/** Create an in-out function argument; takes the data, dim., shape, and pitch. */
LIBXSTREAM_EXPORT_C const libxstream_argument* libxstream_fn_create_inout(void* inout, libxstream_type type, size_t dims, const size_t shape[], const size_t pitch[]);
/** Destroy a function argument. */
LIBXSTREAM_EXPORT_C void libxstream_fn_destroy_arg(const libxstream_argument* arg);

#endif // LIBXSTREAM_H
