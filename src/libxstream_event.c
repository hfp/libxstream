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

int libxstream_event_create(void** event_p) {
  int result = EXIT_SUCCESS;
  assert(NULL != libxstream_opencl_config.events && NULL != event_p);
  *event_p = libxs_pmalloc_lock(
    (void**)libxstream_opencl_config.events, &libxstream_opencl_config.nevents, libxstream_opencl_config.lock_event);
  if (NULL != *event_p) *(cl_event*)*event_p = NULL;
  else result = EXIT_FAILURE;
  LIBXSTREAM_RETURN(result);
}


int libxstream_event_destroy(void* event) {
  int result = EXIT_SUCCESS;
  if (NULL != event) {
    const cl_event clevent = *LIBXSTREAM_EVENT(event);
    assert(NULL != libxstream_opencl_config.events);
#  if !defined(NDEBUG)
    *(cl_event*)event = NULL;
#  endif
    libxs_pfree_lock(event, (void**)libxstream_opencl_config.events, &libxstream_opencl_config.nevents, libxstream_opencl_config.lock_event);
    if (NULL != clevent) {
      result = clReleaseEvent(clevent);
    }
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_stream_wait_event(void* stream, void* event) { /* wait for an event (device-side) */
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = NULL;
  cl_event clevent = NULL;
  str = (NULL != stream ? LIBXSTREAM_STREAM(stream) : libxstream_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
  clevent = *LIBXSTREAM_EVENT(event);
  if (NULL != clevent) {
#  if defined(CL_VERSION_1_2)
    result = clEnqueueBarrierWithWaitList(str->queue, 1, &clevent, NULL);
#  else
    result = clEnqueueWaitForEvents(str->queue, 1, &clevent);
#  endif
    if (EXIT_SUCCESS != result) {
      LIBXSTREAM_EXPECT(EXIT_SUCCESS == clReleaseEvent(clevent));
      *(cl_event*)event = NULL;
    }
  }
  else if (3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: libxstream_stream_wait_event discovered an empty event.\n");
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_event_record(void* event, void* stream) {
  int result = EXIT_SUCCESS;
  const libxstream_opencl_stream_t* str = NULL;
  cl_event clevent = NULL, clevent_result = NULL;
  str = (NULL != stream ? LIBXSTREAM_STREAM(stream) : libxstream_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
  clevent = *LIBXSTREAM_EVENT(event);
#  if defined(CL_VERSION_1_2)
  result = clEnqueueMarkerWithWaitList(str->queue, 0, NULL, &clevent_result);
#  else
  result = clEnqueueMarker(str->queue, &clevent_result);
#  endif
  if (NULL != clevent) {
    const int result_release = clReleaseEvent(clevent);
    if (EXIT_SUCCESS == result) result = result_release;
  }
  if (EXIT_SUCCESS == result) {
    assert(NULL != clevent_result);
    *(cl_event*)event = clevent_result;
  }
  else {
    if (NULL != clevent_result) LIBXSTREAM_EXPECT(EXIT_SUCCESS == clReleaseEvent(clevent_result));
    *(cl_event*)event = NULL;
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_event_query(void* event, int* has_occurred) {
  cl_int status = CL_COMPLETE;
  int result;
  assert(NULL != event && NULL != has_occurred);
  result = clGetEventInfo(*LIBXSTREAM_EVENT(event), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
  if (EXIT_SUCCESS == result && 0 <= status) *has_occurred = (CL_COMPLETE == status ? 1 : 0);
  else { /* error state */
    result = EXIT_SUCCESS; /* soft-error */
    *has_occurred = 1;
  }
  LIBXSTREAM_RETURN(result);
}


int libxstream_event_synchronize(void* event) { /* waits on the host-side */
  int result = EXIT_SUCCESS;
  cl_event clevent;
  assert(NULL != event);
  clevent = *LIBXSTREAM_EVENT(event);
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
      if (EXIT_SUCCESS == result) result = clFinish(queue);
    }
  }
  else if (3 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: libxstream_event_synchronize discovered an empty event.\n");
  }
  LIBXSTREAM_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
