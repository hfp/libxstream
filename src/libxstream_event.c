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


#  if defined(__cplusplus)
extern "C" {
#  endif

int libxstream_event_create(libxstream_event_t** event_p) {
  int result = EXIT_SUCCESS;
  assert(NULL != libxstream_opencl_config.events && NULL != event_p);
  *event_p = libxs_pmalloc_lock(
    (void**)libxstream_opencl_config.events, &libxstream_opencl_config.nevents, libxstream_opencl_config.lock_event);
  if (NULL != *event_p) (*event_p)->cl_evt = NULL;
  else result = EXIT_FAILURE;
  CL_RETURN(result, "");
}


int libxstream_event_destroy(libxstream_event_t* event) {
  int result = EXIT_SUCCESS;
  if (NULL != event) {
    cl_event clevent;
    assert(NULL != libxstream_opencl_config.events);
    LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    clevent = event->cl_evt;
    event->cl_evt = NULL;
    LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
    libxs_pfree_lock(event, (void**)libxstream_opencl_config.events, &libxstream_opencl_config.nevents, libxstream_opencl_config.lock_event);
    if (NULL != clevent) {
      result = clReleaseEvent(clevent);
    }
  }
  CL_RETURN(result, "");
}


int libxstream_stream_wait_event(libxstream_stream_t* stream, libxstream_event_t* event) { /* wait for an event (device-side) */
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = NULL;
  cl_event clevent = NULL;
  str = (NULL != stream ? stream : libxstream_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
  LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  clevent = event->cl_evt;
  if (NULL != clevent) clRetainEvent(clevent);
  LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
  if (NULL != clevent) {
#  if defined(CL_VERSION_1_2)
    result = clEnqueueBarrierWithWaitList(str->queue, 1, &clevent, NULL);
#  else
    result = clEnqueueWaitForEvents(str->queue, 1, &clevent);
#  endif
    if (EXIT_SUCCESS != result) {
      LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      if (clevent == event->cl_evt) {
        event->cl_evt = NULL;
        LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(clevent));
      }
      else {
        LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
      }
    }
    clReleaseEvent(clevent);
  }
  else if (3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: libxstream_stream_wait_event discovered an empty event.\n");
  }
  CL_RETURN(result, "");
}


int libxstream_event_record(libxstream_event_t* event, libxstream_stream_t* stream) {
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = NULL;
  cl_event clevent = NULL, clevent_result = NULL;
  str = (NULL != stream ? stream : libxstream_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
#  if defined(CL_VERSION_1_2)
  result = clEnqueueMarkerWithWaitList(str->queue, 0, NULL, &clevent_result);
#  else
  result = clEnqueueMarker(str->queue, &clevent_result);
#  endif
  LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  clevent = event->cl_evt;
  if (EXIT_SUCCESS == result) {
    assert(NULL != clevent_result);
    event->cl_evt = clevent_result;
  }
  else {
    event->cl_evt = NULL;
  }
  LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
  if (NULL != clevent) {
    const int result_release = clReleaseEvent(clevent);
    if (EXIT_SUCCESS == result) result = result_release;
  }
  if (EXIT_SUCCESS != result && NULL != clevent_result) {
    LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(clevent_result));
  }
  CL_RETURN(result, "");
}


int libxstream_event_query(libxstream_event_t* event, int* has_occurred) {
  cl_int status = CL_COMPLETE;
  cl_event clevent;
  int result;
  assert(NULL != event && NULL != has_occurred);
  LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  clevent = event->cl_evt;
  if (NULL != clevent) clRetainEvent(clevent);
  LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
  if (NULL != clevent) {
    result = clGetEventInfo(clevent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
    clReleaseEvent(clevent);
  }
  else result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 0 <= status) *has_occurred = (CL_COMPLETE == status ? 1 : 0);
  else { /* error state */
    result = EXIT_SUCCESS; /* soft-error */
    *has_occurred = 1;
  }
  CL_RETURN(result, "");
}


int libxstream_event_sync(libxstream_event_t* event) { /* waits on the host-side */
  int result = EXIT_SUCCESS;
  cl_event clevent;
  assert(NULL != event);
  LIBXS_ATOMIC_ACQUIRE(libxstream_opencl_config.lock_event, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  clevent = event->cl_evt;
  if (NULL != clevent) clRetainEvent(clevent);
  LIBXS_ATOMIC_RELEASE(libxstream_opencl_config.lock_event, LIBXS_ATOMIC_LOCKORDER);
  if (NULL != clevent) {
    if (0 == (32 & libxstream_opencl_config.wa)) {
      cl_int status = CL_COMPLETE + 1;
      if (64 & libxstream_opencl_config.xhints) {
        result = clGetEventInfo(clevent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
        assert(EXIT_SUCCESS == result || CL_COMPLETE != status);
      }
      if (CL_COMPLETE != status) result = clWaitForEvents(1, &clevent);
    }
    else {
      cl_command_queue queue = NULL;
      result = clGetEventInfo(clevent, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &queue, NULL);
      CL_CHECK(result, clFinish(queue));
    }
    clReleaseEvent(clevent);
  }
  else if (3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: libxstream_event_sync discovered an empty event.\n");
  }
  CL_RETURN(result, "");
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
