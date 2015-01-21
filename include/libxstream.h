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

#if defined(LIBXSTREAM_DEBUG)
# include <assert.h>
#endif
#if defined(LIBXSTREAM_DEBUG) || defined(LIBXSTREAM_CHECK)
# if defined(LIBXSTREAM_OFFLOAD)
#   pragma offload_attribute(push,target(mic))
# endif
# include <stdio.h>
# if defined(LIBXSTREAM_OFFLOAD)
#   pragma offload_attribute(pop)
# endif
#endif

#if defined(_OPENMP)
# include <omp.h>
#endif

#ifdef __cplusplus
# include <stddef.h>
extern "C" {
#endif

/** Data type representing a signal. */
typedef uintptr_t libxstream_signal;
/** Forward declaration of the stream type (C++ API includes the definition). */
typedef struct libxstream_stream libxstream_stream;
/** Forward declaration of the event type (C++ API includes the definition). */
typedef struct libxstream_event libxstream_event;

/** Query the number of available devices. */
int libxstream_get_ndevices(size_t* ndevices);
/** Query the device set active for this thread. */
int libxstream_get_active_device(int* device);
/** Sets the active device for this thread. */
int libxstream_set_active_device(int device);

/** Query the range of valid priorities (inclusive). */
int libxstream_stream_priority_range(int* least, int* greatest);
/** Create a stream on a given device. The given priority shall be within the queried bounds. */
int libxstream_stream_create(libxstream_stream** stream, int device, int priority, const char* name);
/** Destroy a stream. Any pending work with results needed must be completed explicitly (prior). */
int libxstream_stream_destroy(libxstream_stream* stream);
/** Wait for a stream to complete pending work. A NULL-stream synchronizes all streams. */
int libxstream_stream_sync(libxstream_stream* stream);
/** Wait for an event recorded earlier. Passing NULL increases the match accordingly. */
int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event);

/** Create an event; can be re-used multiple times by re-recording the event. */
int libxstream_event_create(libxstream_event** event);
/** Destroy an event; does not implicitly waits for the completion of the event. */
int libxstream_event_destroy(libxstream_event* event);
/** Record an event; an event can be re-recorded multiple times. */
int libxstream_event_record(libxstream_event* event, libxstream_stream* stream);
/** Check whether an event has occured or not (non-blocking). */
int libxstream_event_query(const libxstream_event* event, int* has_occured);
/** Wait for an event to complete i.e., any work enqueued prior to recording the event. */
int libxstream_event_synchronize(libxstream_event* event);

/** Query the memory metrics of the given device (it is valid to pass one NULL pointer). */
int libxstream_mem_info(int device, size_t* allocatable, size_t* physical);
/** Allocates memory with automatic alignment (0) on the given device; mandatory for device-side memory. */
int libxstream_mem_allocate(int device, void** memory, size_t size, size_t alignment);
/** Deallocates the memory. The given device shall match the device where the memory was allocated. */
int libxstream_mem_deallocate(int device, const void* memory);
/** Fills the memory with zeros; allocated memory can carry an offset and a smaller size. */
int libxstream_memset_zero(void* memory, size_t size, libxstream_stream* stream);
/** Copies memory from the host to the device; allocated memory can carry an offset and a smaller size. */
int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t size, libxstream_stream* stream);
/** Copies memory from the device to the host; allocated memory can carry an offset and a smaller size. */
int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t size, libxstream_stream* stream);
/** Copies memory from the device to the device with cross-device copies being allowed as well. */
int libxstream_memcpy_d2d(const void* src, void* dst, size_t size, libxstream_stream* stream);

#ifdef __cplusplus
}
#endif

#endif // LIBXSTREAM_H
