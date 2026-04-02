/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
# include <libxstream_cp2k.h>
# include <libxstream_opencl.h>

#define OFFLOAD_EXPECT(RESULT, NAME) \
  do { \
    if (EXIT_SUCCESS != (RESULT)) { \
      CL_ERROR_REPORT(NAME); \
      abort(); \
    } \
  } while (0)


# if defined(__cplusplus)
extern "C" {
# endif

const char* offloadGetErrorName(offloadError_t error) {
  return libxstream_opencl_strerror(error);
}


offloadError_t offloadGetLastError(void) {
  return libxstream_opencl_error_consume();
}


void offloadMemsetAsync(void* ptr, int val, size_t size, offloadStream_t stream) {
  const int result = libxstream_opencl_memset(ptr, val, 0 /*offset*/, size, (libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadMemsetAsync");
}


void offloadMemset(void* ptr, int val, size_t size) {
  offloadMemsetAsync(ptr, val, size, NULL);
}


void offloadMemcpyAsyncHtoD(void* ptr_dev, const void* ptr_hst, size_t size, offloadStream_t stream) {
  const int result = libxstream_mem_copy_h2d(ptr_hst, ptr_dev, size, (libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadMemcpyAsyncHtoD");
}


void offloadMemcpyAsyncDtoH(void* ptr_hst, const void* ptr_dev, size_t size, offloadStream_t stream) {
  const int result = libxstream_mem_copy_d2h(ptr_dev, ptr_hst, size, (libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadMemcpyAsyncDtoH");
}


void offloadMemcpyAsyncDtoD(void* dst, const void* src, size_t size, offloadStream_t stream) {
  const int result = libxstream_mem_copy_d2d(src, dst, size, (libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadMemcpyAsyncDtoD");
}


void offloadMemcpyHtoD(void* ptr_dev, const void* ptr_hst, size_t size) {
  offloadMemcpyAsyncHtoD(ptr_dev, ptr_hst, size, NULL);
}


void offloadMemcpyDtoH(void* ptr_hst, const void* ptr_dev, size_t size) {
  offloadMemcpyAsyncDtoH(ptr_hst, ptr_dev, size, NULL);
}


void offloadMemcpyToSymbol(const void* symbol, const void* src, size_t count) {
  LIBXS_UNUSED(symbol);
  LIBXS_UNUSED(src);
  LIBXS_UNUSED(count);
  LIBXS_ASSERT(NULL == symbol || NULL == src || 0 == count);
}


void offloadEventCreate(offloadEvent_t* event) {
  const int result = libxstream_event_create((libxstream_event_t**)event);
  OFFLOAD_EXPECT(result, "offloadEventCreate");
}


void offloadEventDestroy(offloadEvent_t event) {
  const int result = libxstream_event_destroy((libxstream_event_t*)event);
  OFFLOAD_EXPECT(result, "offloadEventDestroy");
}


void offloadEventSynchronize(offloadEvent_t event) {
  const int result = libxstream_event_sync((libxstream_event_t*)event);
  OFFLOAD_EXPECT(result, "offloadEventSynchronize");
}


void offloadEventRecord(offloadEvent_t event, offloadStream_t stream) {
  const int result = libxstream_event_record((libxstream_event_t*)event, (libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadEventRecord");
}


bool offloadEventQuery(offloadEvent_t event) {
  libxstream_bool_t has_occurred = 0;
  const int result = libxstream_event_query((libxstream_event_t*)event, &has_occurred);
  OFFLOAD_EXPECT(result, "offloadEventQuery");
  return 0 != has_occurred;
}


void offloadStreamCreate(offloadStream_t* stream) {
  const int result = libxstream_stream_create((libxstream_stream_t**)stream, "Offload Stream", LIBXSTREAM_STREAM_DEFAULT);
  OFFLOAD_EXPECT(result, "offloadStreamCreate");
}


void offloadStreamDestroy(offloadStream_t stream) {
  const int result = libxstream_stream_destroy((libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadStreamDestroy");
}


void offloadStreamSynchronize(offloadStream_t stream) {
  const int result = libxstream_stream_sync((libxstream_stream_t*)stream);
  OFFLOAD_EXPECT(result, "offloadStreamSynchronize");
}


void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event) {
  const int result = libxstream_stream_wait_event((libxstream_stream_t*)stream, (libxstream_event_t*)event);
  OFFLOAD_EXPECT(result, "offloadStreamWaitEvent");
}


void offloadMallocHost(void** ptr, size_t size) {
  const int result = libxstream_mem_host_allocate(ptr, size, NULL);
  OFFLOAD_EXPECT(result, "offloadMallocHost");
}


void offloadMalloc(void** ptr, size_t size) {
  const int result = libxstream_mem_allocate(ptr, size);
  OFFLOAD_EXPECT(result, "offloadMalloc");
}


void offloadFree(void* ptr) {
  const int result = libxstream_mem_deallocate(ptr);
  OFFLOAD_EXPECT(result, "offloadFree");
}


void offloadFreeHost(void* ptr) {
  const int result = libxstream_mem_host_deallocate(ptr, NULL);
  OFFLOAD_EXPECT(result, "offloadFreeHost");
}


void offloadDeviceSynchronize(void) {
  const int result = libxstream_device_sync();
  OFFLOAD_EXPECT(result, "offloadDeviceSynchronize");
}


void offloadEnsureMallocHeapSize(size_t required_size) {
  LIBXS_UNUSED(required_size);
  LIBXS_ASSERT(0 == required_size);
}

# if defined(__cplusplus)
}
# endif

#endif /*__OPENCL*/
