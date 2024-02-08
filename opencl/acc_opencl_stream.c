/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#  include "acc_opencl.h"
#  include <string.h>

#  if defined(CL_VERSION_2_0)
#    define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_queue_properties
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#  else
#    define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_int
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
      clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#  endif

#  if defined(__cplusplus)
extern "C" {
#  endif

c_dbcsr_acc_opencl_lock_t c_dbcsr_acc_opencl_stream_lock;
int c_dbcsr_acc_opencl_stream_counter_base;
int c_dbcsr_acc_opencl_stream_counter;


const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream(c_dbcsr_acc_opencl_lock_t* lock, int thread_id) {
  const c_dbcsr_acc_opencl_stream_t* result = NULL;
  const size_t n = ACC_OPENCL_HANDLES_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
  size_t i;
  assert(NULL != c_dbcsr_acc_opencl_config.streams);
  assert(thread_id < c_dbcsr_acc_opencl_config.nthreads);
  if (NULL != lock) {
    LIBXSMM_ATOMIC_ACQUIRE(lock, LIBXSMM_SYNC_NPAUSE, ACC_OPENCL_ATOMIC_KIND);
  }
  for (i = c_dbcsr_acc_opencl_config.nstreams; i < n; ++i) {
    const c_dbcsr_acc_opencl_stream_t* const str = c_dbcsr_acc_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (str->tid == thread_id || 0 > thread_id) { /* hit */
        result = str;
        break;
      }
    }
    else break; /* error */
  }
  if (NULL != lock) {
    LIBXSMM_ATOMIC_RELEASE(lock, ACC_OPENCL_ATOMIC_KIND);
  }
  return result;
}


const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream_default(void) {
  const c_dbcsr_acc_opencl_stream_t* result = NULL;
  const int tid = ACC_OPENCL_OMP_TID();
  LIBXSMM_ATOMIC_ACQUIRE(&c_dbcsr_acc_opencl_stream_lock, LIBXSMM_SYNC_NPAUSE, ACC_OPENCL_ATOMIC_KIND);
  result = c_dbcsr_acc_opencl_stream(NULL /*lock*/, tid);
  if (0 != tid && NULL == result) {
    result = c_dbcsr_acc_opencl_stream(NULL /*lock*/, 0 /*main thread*/);
  }
  LIBXSMM_ATOMIC_RELEASE(&c_dbcsr_acc_opencl_stream_lock, ACC_OPENCL_ATOMIC_KIND);
  assert(NULL != result);
  return result;
}


int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  ACC_OPENCL_STREAM_PROPERTIES_TYPE properties[8] = {
    CL_QUEUE_PROPERTIES, 0 /*placeholder*/, 0 /* terminator */
  };
  int result, i, tid = 0, offset = 0;
  cl_command_queue queue = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != stream_p);
#  if !defined(ACC_OPENCL_STREAM_PRIORITIES)
  LIBXSMM_UNUSED(priority);
#  else
  if (CL_QUEUE_PRIORITY_HIGH_KHR <= priority && CL_QUEUE_PRIORITY_LOW_KHR >= priority) {
    properties[3] = priority;
  }
  else {
    int least = -1, greatest = -1;
    if (0 != (1 & c_dbcsr_acc_opencl_config.priority) && EXIT_SUCCESS == c_dbcsr_acc_stream_priority_range(&least, &greatest) &&
        least != greatest)
    {
      properties[3] = (0 != (2 & c_dbcsr_acc_opencl_config.priority) &&
                        (NULL != LIBXSMM_STRISTR(name, "calc") || (NULL != strstr(name, "priority"))))
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
  LIBXSMM_ATOMIC_ACQUIRE(&c_dbcsr_acc_opencl_stream_lock, LIBXSMM_SYNC_NPAUSE, ACC_OPENCL_ATOMIC_KIND);
#  if defined(_OPENMP)
  if (1 < omp_get_num_threads()) {
    assert(0 < c_dbcsr_acc_opencl_config.nthreads);
    i = c_dbcsr_acc_opencl_stream_counter++;
    tid = (i < c_dbcsr_acc_opencl_config.nthreads ? i : (i % c_dbcsr_acc_opencl_config.nthreads));
  }
  else offset = c_dbcsr_acc_opencl_stream_counter_base++;
#  endif
  if (NULL != c_dbcsr_acc_opencl_config.device.context) {
    cl_device_id device = NULL;
    result = clGetContextInfo(c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (EXIT_SUCCESS == result) {
      if (0 != c_dbcsr_acc_opencl_config.device.intel) {
        const int xhints = ((1 == c_dbcsr_acc_opencl_config.xhints || 0 > c_dbcsr_acc_opencl_config.xhints)
                              ? (0 != c_dbcsr_acc_opencl_config.device.intel ? 1 : 0)
                              : (c_dbcsr_acc_opencl_config.xhints >> 1));
        if (0 != (1 & xhints)) { /* attempt to enable command aggregation */
          const ACC_OPENCL_STREAM_PROPERTIES_TYPE props[4] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 /* terminator */
          };
          const cl_command_queue q = ACC_OPENCL_CREATE_COMMAND_QUEUE(
            c_dbcsr_acc_opencl_config.device.context, device, props, &result);
          if (EXIT_SUCCESS == result) {
            c_dbcsr_acc_opencl_config.timer = c_dbcsr_acc_opencl_timer_host; /* force host-timer */
            clReleaseCommandQueue(q);
          }
          else result = EXIT_SUCCESS;
        }
        if (0 != (2 & xhints)) { /* attempt to enable queue families */
          struct {
            cl_command_queue_properties properties;
            cl_bitfield capabilities;
            cl_uint count;
            char name[64 /*CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL*/];
          } intel_qfprops[16];
          size_t nbytes = 0, i;
          if (EXIT_SUCCESS == clGetDeviceInfo(device, 0x418B /*CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL*/, sizeof(intel_qfprops),
                                intel_qfprops, &nbytes))
          {
            for (i = 0; (i * sizeof(*intel_qfprops)) < nbytes; ++i) {
              if (0 /*CL_QUEUE_DEFAULT_CAPABILITIES_INTEL*/ == intel_qfprops[i].capabilities && 1 < intel_qfprops[i].count) {
                const int j = (0 /*terminator*/ == properties[2] ? 2 : 4);
                properties[j + 0] = 0x418C; /* CL_QUEUE_FAMILY_INTEL */
                properties[j + 1] = (int)i;
                properties[j + 2] = 0x418D; /* CL_QUEUE_INDEX_INTEL */
                properties[j + 3] = (i + offset) % intel_qfprops[i].count;
                properties[j + 4] = 0; /* terminator */
                break;
              }
            }
          }
        }
      }
      if ((c_dbcsr_acc_opencl_timer_device == c_dbcsr_acc_opencl_config.timer) &&
          (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity))
      {
        properties[1] = CL_QUEUE_PROFILING_ENABLE;
      }
      queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(c_dbcsr_acc_opencl_config.device.context, device, properties, &result);
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* register stream */
    assert(NULL != queue);
    *stream_p = (
#  if defined(ACC_OPENCL_PMALLOC)
      NULL != c_dbcsr_acc_opencl_config.streams
        ? libxsmm_pmalloc((void**)c_dbcsr_acc_opencl_config.streams, &c_dbcsr_acc_opencl_config.nstreams)
        :
#  endif
        malloc(sizeof(c_dbcsr_acc_opencl_stream_t)));
    if (NULL != *stream_p) {
      c_dbcsr_acc_opencl_stream_t* const str = (c_dbcsr_acc_opencl_stream_t*)*stream_p;
#  if !defined(NDEBUG)
      LIBXSMM_MEMZERO127(str);
#  endif
      str->queue = queue;
      str->priority = priority;
      str->tid = tid;
    }
    else result = EXIT_FAILURE;
  }
  LIBXSMM_ATOMIC_RELEASE(&c_dbcsr_acc_opencl_stream_lock, ACC_OPENCL_ATOMIC_KIND);
  if (EXIT_SUCCESS != result && NULL != queue) {
    clReleaseCommandQueue(queue);
    *stream_p = NULL;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN_CAUSE(result, name);
}


int c_dbcsr_acc_stream_destroy(void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != stream) {
    const c_dbcsr_acc_opencl_stream_t* const str = ACC_OPENCL_STREAM(stream);
    const cl_command_queue queue = str->queue;
#  if defined(ACC_OPENCL_PMALLOC)
    if (NULL != c_dbcsr_acc_opencl_config.streams) {
      libxsmm_pfree(stream, (void**)c_dbcsr_acc_opencl_config.streams, &c_dbcsr_acc_opencl_config.nstreams);
    }
    else
#  endif
    {
      free(stream);
    }
    if (NULL != queue) result = clReleaseCommandQueue(queue);
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_priority_range(int* least, int* greatest) {
  int result = ((NULL != least || NULL != greatest) ? EXIT_SUCCESS : EXIT_FAILURE);
  int priohi = -1, priolo = -1;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(least != greatest); /* no alias */
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  if (0 < c_dbcsr_acc_opencl_config.ndevices) {
    char buffer[ACC_OPENCL_BUFFERSIZE];
    cl_platform_id platform = NULL;
    cl_device_id active_id = NULL;
    if (EXIT_SUCCESS == result) {
      result = clGetContextInfo(
        c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &active_id, NULL);
    }
    ACC_OPENCL_CHECK(clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL),
      "retrieve platform associated with active device", result);
    ACC_OPENCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ACC_OPENCL_BUFFERSIZE, buffer, NULL),
      "retrieve platform extensions", result);
    if (EXIT_SUCCESS == result) {
      if (NULL != strstr(buffer, "cl_khr_priority_hints") ||
          EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/))
      {
        priohi = CL_QUEUE_PRIORITY_HIGH_KHR;
        priolo = CL_QUEUE_PRIORITY_LOW_KHR;
      }
    }
  }
#  endif
  if (NULL != greatest) *greatest = priohi;
  if (NULL != least) *least = priolo;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_sync(void* stream) {
  const c_dbcsr_acc_opencl_stream_t* str = NULL;
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(ACC_OPENCL_STREAM_NULL)
  str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream_default());
#  else
  str = ACC_OPENCL_STREAM(stream);
#  endif
  assert(NULL != str && NULL != str->queue);
  result = clFinish(str->queue);
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_synchronize(c_dbcsr_acc_opencl_lock_t* lock, int thread_id) {
  int result = EXIT_SUCCESS;
  const size_t n = ACC_OPENCL_HANDLES_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
  size_t i;
  assert(thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != c_dbcsr_acc_opencl_config.streams);
  if (NULL != lock) {
    LIBXSMM_ATOMIC_ACQUIRE(lock, LIBXSMM_SYNC_NPAUSE, ACC_OPENCL_ATOMIC_KIND);
  }
  for (i = c_dbcsr_acc_opencl_config.nstreams; i < n; ++i) {
    const c_dbcsr_acc_opencl_stream_t* const str = c_dbcsr_acc_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (str->tid == thread_id || 0 > thread_id) { /* hit */
        result = clFinish(str->queue);
        if (EXIT_SUCCESS != result) break;
      }
    }
    else { /* error */
      result = EXIT_FAILURE;
      break;
    }
  }
  if (NULL != lock) {
    LIBXSMM_ATOMIC_RELEASE(lock, ACC_OPENCL_ATOMIC_KIND);
  }
  return result;
}


int c_dbcsr_acc_device_synchronize(void) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(_OPENMP)
  if (1 == omp_get_num_threads()) {
    result = c_dbcsr_acc_opencl_device_synchronize(&c_dbcsr_acc_opencl_stream_lock, -1 /*all*/);
  }
  else {
    result = c_dbcsr_acc_opencl_device_synchronize(NULL /*lock*/, omp_get_thread_num());
  }
#  else
  result = c_dbcsr_acc_opencl_device_synchronize(NULL /*lock*/, /*main*/ 0);
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
