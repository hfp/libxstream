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
#include <stdio.h>
#include <stdlib.h>

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

#if defined(__cplusplus)
extern "C" {
#endif

const char* offloadGetErrorName(offloadError_t error);
offloadError_t offloadGetLastError(void);

void offloadMemsetAsync(void* ptr, int val, size_t size, offloadStream_t stream);
void offloadMemset(void* ptr, int val, size_t size);
void offloadMemcpyAsyncHtoD(void* ptr_dev, const void* ptr_hst, size_t size, offloadStream_t stream);
void offloadMemcpyAsyncDtoH(void* ptr_hst, const void* ptr_dev, size_t size, offloadStream_t stream);
void offloadMemcpyAsyncDtoD(void* dst, const void* src, size_t size, offloadStream_t stream);
void offloadMemcpyHtoD(void* ptr_dev, const void* ptr_hst, size_t size);
void offloadMemcpyDtoH(void* ptr_hst, const void* ptr_dev, size_t size);
void offloadMemcpyToSymbol(const void* symbol, const void* src, size_t count);

void offloadEventCreate(offloadEvent_t* event);
void offloadEventDestroy(offloadEvent_t event);
void offloadEventSynchronize(offloadEvent_t event);
void offloadEventRecord(offloadEvent_t event, offloadStream_t stream);
bool offloadEventQuery(offloadEvent_t event);

void offloadStreamCreate(offloadStream_t* stream);
void offloadStreamDestroy(offloadStream_t stream);
void offloadStreamSynchronize(offloadStream_t stream);
void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event);

void offloadMallocHost(void** ptr, size_t size);
void offloadMalloc(void** ptr, size_t size);
void offloadFree(void* ptr);
void offloadFreeHost(void* ptr);

void offloadDeviceSynchronize(void);
void offloadEnsureMallocHeapSize(size_t required_size);

#if defined(__cplusplus)
}
#endif

#endif /*LIBXSTREAM_CP2K_H*/
