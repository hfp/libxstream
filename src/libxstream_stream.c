/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
#  include <libxstream_opencl.h>
#  include <string.h>


#  if defined(__cplusplus)
extern "C" {
#  endif

const libxstream_opencl_stream_t* libxstream_opencl_stream(libxs_lock_t* lock, int thread_id) {
  const libxstream_opencl_stream_t *result = NULL, *result_main = NULL;
  const size_t n = LIBXSTREAM_MAXNITEMS * libxstream_opencl_config.nthreads;
  size_t i;
  assert(NULL != libxstream_opencl_config.streams);
  assert(thread_id < libxstream_opencl_config.nthreads);
  if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  for (i = libxstream_opencl_config.nstreams; i < n; ++i) {
    const libxstream_opencl_stream_t* const str = libxstream_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (str->tid == thread_id || 0 > thread_id) { /* hit */
        result = str;
        break;
      }
      else if (NULL == result_main && 0 == str->tid) {
        result_main = str;
      }
    }
    else break; /* error */
  }
  if (NULL == result) { /* fallback */
    assert(NULL != libxstream_opencl_config.device.context);
    result = (NULL != result_main ? result_main : &libxstream_opencl_config.device.stream);
  }
  if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  return result;
}


const libxstream_opencl_stream_t* libxstream_opencl_stream_default(void) {
  const libxstream_opencl_stream_t* result = NULL;
  result = libxstream_opencl_stream(libxstream_opencl_config.lock_stream, libxs_tid());
  assert(NULL != result);
  return result;
}


int libxstream_stream_create(libxstream_stream_t** stream_p, const char* name, int flags) {
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  LIBXSTREAM_STREAM_PROPERTIES_TYPE properties[9] = {
    CL_QUEUE_PROPERTIES, 0 /*placeholder*/, 0 /* terminator */
  };
  int result, tid = 0, offset = 0;
  cl_command_queue queue = NULL;
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
  int priority = 0;
#  endif
  assert(NULL != stream_p);
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
  if (0 != (LIBXSTREAM_STREAM_LOW & flags)) {
    properties[3] = CL_QUEUE_PRIORITY_LOW_KHR;
  }
  else if (0 != (LIBXSTREAM_STREAM_HIGH & flags)) {
    properties[3] = CL_QUEUE_PRIORITY_HIGH_KHR;
  }
  else { /* default: auto-detect from config and name */
    int least = -1, greatest = -1;
    if (0 != (1 & libxstream_opencl_config.priority) && EXIT_SUCCESS == libxstream_stream_priority_range(&least, &greatest) &&
        least != greatest)
    {
      properties[3] = (0 != (2 & libxstream_opencl_config.priority) &&
                        (NULL != libxs_stristr(name, "calc") || (NULL != strstr(name, "priority"))))
                        ? CL_QUEUE_PRIORITY_HIGH_KHR
                        : CL_QUEUE_PRIORITY_MED_KHR;
    }
    else {
      properties[3] = least;
    }
  }
  if (CL_QUEUE_PRIORITY_HIGH_KHR <= properties[3] && CL_QUEUE_PRIORITY_LOW_KHR >= properties[3]) {
    priority = properties[3]; /* sanitize */
    properties[2] = CL_QUEUE_PRIORITY_KHR;
    properties[4] = 0; /* terminator */
  }
#  endif
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_stream);
#  if defined(_OPENMP)
  {
    static int libxstream_opencl_stream_counter_base = 0;
    static int libxstream_opencl_stream_counter = 0;
    if (1 < omp_get_num_threads()) {
      const int i = libxstream_opencl_stream_counter++;
      assert(0 < libxstream_opencl_config.nthreads);
      tid = (i < libxstream_opencl_config.nthreads ? i : (i % libxstream_opencl_config.nthreads));
    }
    else offset = libxstream_opencl_stream_counter_base++;
  }
#  endif
  if (NULL == devinfo->context)
#  if defined(LIBXSTREAM_ACTIVATE)
  {
    result = EXIT_FAILURE;
  }
  else
#  else
  {
    result = libxstream_opencl_set_active_device(NULL /*lock*/, libxstream_opencl_config.device_id);
  }
  if (NULL != devinfo->context)
#  endif
  {
    const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
    if (0 != (LIBXSTREAM_STREAM_PROFILING & flags) ||
        NULL != libxstream_opencl_config.hist_h2d || NULL != libxstream_opencl_config.hist_d2h ||
        NULL != libxstream_opencl_config.hist_d2d)
    {
      properties[1] |= CL_QUEUE_PROFILING_ENABLE;
    }
#  if defined(LIBXSTREAM_XHINTS)
    if ((2 & libxstream_opencl_config.xhints) && 0 != devinfo->intel) {
      properties[1] |= (((LIBXSTREAM_STREAM_PROPERTIES_TYPE)1) << 31); /* CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL */
    }
    if ((4 & libxstream_opencl_config.xhints) && 0 != devinfo->intel) {
      struct {
        cl_command_queue_properties properties;
        cl_bitfield capabilities;
        cl_uint count;
        char name[64 /*CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL*/];
      } intel_qfprops[16];
      const int j = (0 /*terminator*/ == properties[2] ? 2 : 4);
      size_t nbytes = 0, i;
      if (EXIT_SUCCESS == clGetDeviceInfo(device_id, 0x418B /*CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL*/, sizeof(intel_qfprops),
                            intel_qfprops, &nbytes))
      { /* enable queue families */
        for (i = 0; (i * sizeof(*intel_qfprops)) < nbytes; ++i) {
          if (0 /*CL_QUEUE_DEFAULT_CAPABILITIES_INTEL*/ == intel_qfprops[i].capabilities && 1 < intel_qfprops[i].count) {
            properties[j + 0] = 0x418C; /* CL_QUEUE_FAMILY_INTEL */
            properties[j + 1] = LIBXS_CAST_INT(i);
            properties[j + 2] = 0x418D; /* CL_QUEUE_INDEX_INTEL */
            properties[j + 3] = (i + offset) % intel_qfprops[i].count;
            properties[j + 4] = 0; /* terminator */
            break;
          }
        }
      }
    }
#  endif
    queue = LIBXSTREAM_CREATE_COMMAND_QUEUE(devinfo->context, device_id, properties, &result);
  }
  if (EXIT_SUCCESS == result) { /* register stream */
    assert(NULL != libxstream_opencl_config.streams && NULL != queue);
    *stream_p = libxs_pmalloc(
      (void**)libxstream_opencl_config.streams, &libxstream_opencl_config.nstreams);
    if (NULL != *stream_p) {
      libxstream_opencl_stream_t* const str = (libxstream_opencl_stream_t*)*stream_p;
#  if !defined(NDEBUG)
      LIBXS_MEMZERO(str);
#  endif
      str->queue = queue;
      str->tid = tid;
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
      str->priority = priority;
#  endif
    }
    else result = EXIT_FAILURE;
  }
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_stream);
  if (EXIT_SUCCESS != result && NULL != queue) {
    clReleaseCommandQueue(queue);
    *stream_p = NULL;
  }
  LIBXSTREAM_RETURN_CAUSE(result, name);
}


int libxstream_stream_destroy(libxstream_stream_t* stream) {
  int result = EXIT_SUCCESS;
  if (NULL != stream) {
    const libxstream_opencl_stream_t* const str = stream;
    const cl_command_queue queue = str->queue;
#  if !defined(NDEBUG)
    LIBXS_MEMZERO((libxstream_opencl_stream_t*)stream);
#  endif
    if (NULL != libxstream_opencl_config.streams) {
      libxs_pfree_lock(stream, (void**)libxstream_opencl_config.streams,
        &libxstream_opencl_config.nstreams, libxstream_opencl_config.lock_stream);
    }
    if (NULL != queue) {
      result = clReleaseCommandQueue(queue);
    }
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_stream_priority_range(int* least, int* greatest) {
  int result = ((NULL != least || NULL != greatest) ? EXIT_SUCCESS : EXIT_FAILURE);
  int priohi = -1, priolo = -1;
  assert(NULL == least || NULL == greatest || least != greatest); /* no alias */
#  if defined(LIBXSTREAM_STREAM_PRIORITIES)
  if (0 < libxstream_opencl_config.ndevices) {
    const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    char buffer[LIBXSTREAM_BUFFERSIZE];
    cl_platform_id platform = NULL;
    assert(NULL != devinfo->context);
    LIBXSTREAM_CHECK(result, clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL),
      "retrieve platform associated with active device");
    LIBXSTREAM_CHECK(result, clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, LIBXSTREAM_BUFFERSIZE, buffer, NULL),
      "retrieve platform extensions");
    if (EXIT_SUCCESS == result) {
      if (NULL != strstr(buffer, "cl_khr_priority_hints") ||
          EXIT_SUCCESS == libxstream_opencl_device_vendor(device_id, "nvidia", 0 /*use_platform_name*/))
      {
        priohi = CL_QUEUE_PRIORITY_HIGH_KHR;
        priolo = CL_QUEUE_PRIORITY_LOW_KHR;
      }
    }
  }
#  endif
  if (NULL != greatest) *greatest = priohi;
  if (NULL != least) *least = priolo;
  LIBXSTREAM_RETURN(result);
}


int libxstream_stream_sync(libxstream_stream_t* stream) {
  const libxstream_opencl_stream_t* str = NULL;
  int result = EXIT_SUCCESS;
  str = (NULL != stream ? stream : libxstream_opencl_stream_default());
  assert(NULL != str && NULL != str->queue);
  if (0 == (16 & libxstream_opencl_config.wa)) result = clFinish(str->queue);
  else {
    cl_event event = NULL;
#  if defined(CL_VERSION_1_2)
    result = clEnqueueMarkerWithWaitList(str->queue, 0, NULL, &event);
#  else
    result = clEnqueueMarker(str->queue, &event);
#  endif
    if (EXIT_SUCCESS == result) {
      assert(NULL != event);
      result = clWaitForEvents(1, &event);
    }
    if (NULL != event) LIBXS_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_opencl_device_synchronize(libxs_lock_t* lock, int thread_id) {
  int result = EXIT_SUCCESS;
  const size_t n = LIBXSTREAM_MAXNITEMS * libxstream_opencl_config.nthreads;
  size_t i;
  assert(thread_id < libxstream_opencl_config.nthreads);
  assert(NULL != libxstream_opencl_config.streams);
  if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  for (i = libxstream_opencl_config.nstreams; i < n; ++i) {
    const libxstream_opencl_stream_t* const str = libxstream_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (0 > thread_id || str->tid == thread_id) { /* hit */
        result = clFinish(str->queue);
        if (EXIT_SUCCESS != result) break;
      }
    }
    else { /* end of registered streams */
      break;
    }
  }
  if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  return result;
}


int libxstream_device_sync(void) {
  int result = EXIT_SUCCESS;
#  if defined(_OPENMP)
  if (1 == omp_get_num_threads()) {
    result = libxstream_opencl_device_synchronize(libxstream_opencl_config.lock_stream, -1 /*all*/);
  }
  else {
    result = libxstream_opencl_device_synchronize(NULL /*lock*/, omp_get_thread_num());
  }
#  else
  result = libxstream_opencl_device_synchronize(NULL /*lock*/, /*main*/ 0);
#  endif
  LIBXSTREAM_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
