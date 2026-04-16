/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_CP2K_H
#define LIBXSTREAM_CP2K_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "libxstream_macros.h"

typedef void* offloadStream_t;
typedef void* offloadEvent_t;
typedef int offloadError_t;

#define offloadSuccess EXIT_SUCCESS

#if !defined(OFFLOAD_CHECK)
# define OFFLOAD_CHECK(CMD) \
    do { \
      const offloadError_t offload_check_error_ = (CMD); \
      if (offloadSuccess != offload_check_error_) { \
        const char* const offload_check_name_ = offloadGetErrorName(offload_check_error_); \
        if (NULL != offload_check_name_ && '\0' != *offload_check_name_) { \
          fprintf(stderr, "ERROR: \"%s\" at %s:%i\n", offload_check_name_, __FILE__, __LINE__); \
        } \
        else { \
          fprintf(stderr, "ERROR %i: %s:%i\n", (int)offload_check_error_, __FILE__, __LINE__); \
        } \
        abort(); \
      } \
    } while (0)
#endif

LIBXSTREAM_API const char* offloadGetErrorName(offloadError_t error);
LIBXSTREAM_API offloadError_t offloadGetLastError(void);

LIBXSTREAM_API void offloadMemsetAsync(void* ptr, int val, size_t size, offloadStream_t stream);
LIBXSTREAM_API void offloadMemset(void* ptr, int val, size_t size);
LIBXSTREAM_API void offloadMemcpyAsyncHtoD(void* ptr_dev, const void* ptr_hst, size_t size, offloadStream_t stream);
LIBXSTREAM_API void offloadMemcpyAsyncDtoH(void* ptr_hst, const void* ptr_dev, size_t size, offloadStream_t stream);
LIBXSTREAM_API void offloadMemcpyAsyncDtoD(void* dst, const void* src, size_t size, offloadStream_t stream);
LIBXSTREAM_API void offloadMemcpyHtoD(void* ptr_dev, const void* ptr_hst, size_t size);
LIBXSTREAM_API void offloadMemcpyDtoH(void* ptr_hst, const void* ptr_dev, size_t size);
LIBXSTREAM_API void offloadMemcpyToSymbol(const void* symbol, const void* src, size_t count);

LIBXSTREAM_API void offloadEventCreate(offloadEvent_t* event);
LIBXSTREAM_API void offloadEventDestroy(offloadEvent_t event);
LIBXSTREAM_API void offloadEventSynchronize(offloadEvent_t event);
LIBXSTREAM_API void offloadEventRecord(offloadEvent_t event, offloadStream_t stream);
LIBXSTREAM_API bool offloadEventQuery(offloadEvent_t event);

LIBXSTREAM_API void offloadStreamCreate(offloadStream_t* stream);
LIBXSTREAM_API void offloadStreamDestroy(offloadStream_t stream);
LIBXSTREAM_API void offloadStreamSynchronize(offloadStream_t stream);
LIBXSTREAM_API void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event);

LIBXSTREAM_API void offloadMallocHost(void** ptr, size_t size);
LIBXSTREAM_API void offloadMalloc(void** ptr, size_t size);
LIBXSTREAM_API void offloadFree(void* ptr);
LIBXSTREAM_API void offloadFreeHost(void* ptr);

LIBXSTREAM_API void offloadDeviceSynchronize(void);
LIBXSTREAM_API void offloadEnsureMallocHeapSize(size_t required_size);

#endif /*LIBXSTREAM_CP2K_H*/
