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

#  if !defined(ACC_OPENCL_EVENT_BARRIER) && 1
#    define ACC_OPENCL_EVENT_BARRIER
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_event_create(void** event_p) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event_p);
  assert(NULL == c_dbcsr_acc_opencl_config.events || sizeof(void*) >= sizeof(cl_event));
  *event_p = (
#  if LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && defined(ACC_OPENCL_HANDLES_MAXCOUNT) && \
    (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    NULL != c_dbcsr_acc_opencl_config.events ? libxsmm_pmalloc(c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents)
                                             :
#  endif
                                             malloc(sizeof(cl_event)));
  if (NULL != *event_p) *(cl_event*)*event_p = NULL;
  else result = EXIT_FAILURE;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_destroy(void* event) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != event) {
    const cl_event clevent = *ACC_OPENCL_EVENT(event);
    if (NULL != clevent) result = clReleaseEvent(clevent);
#  if LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && defined(ACC_OPENCL_HANDLES_MAXCOUNT) && \
    (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    if (NULL != c_dbcsr_acc_opencl_config.events) {
      /**(cl_event*)event = NULL; assert(NULL == *ACC_OPENCL_EVENT(event));*/
      libxsmm_pfree(event, c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents);
    }
    else
#  endif
    {
      free(event);
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_wait_event(void* stream, void* event) { /* wait for an event (device-side) */
  int result = EXIT_SUCCESS;
  cl_command_queue queue = NULL;
  cl_event clevent = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(ACC_OPENCL_STREAM_NULL)
  queue = *ACC_OPENCL_STREAM(NULL != stream ? stream : c_dbcsr_acc_opencl_stream_default());
#  else
  queue = *ACC_OPENCL_STREAM(stream);
#  endif
  assert(NULL != queue && NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
  if (NULL != clevent) {
#  if defined(CL_VERSION_1_2)
#    if defined(ACC_OPENCL_EVENT_BARRIER)
    result = clEnqueueBarrierWithWaitList(queue, 1, &clevent, NULL);
#    else
    result = clEnqueueMarkerWithWaitList(queue, 1, &clevent, NULL);
#    endif
#  else
    result = clEnqueueWaitForEvents(queue, 1, &clevent);
#  endif
  }
  /*else result = EXIT_FAILURE;*/
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_record(void* event, void* stream) {
  int result;
  cl_command_queue queue = NULL;
  cl_event clevent = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(ACC_OPENCL_STREAM_NULL)
  queue = *ACC_OPENCL_STREAM(NULL != stream ? stream : c_dbcsr_acc_opencl_stream_default());
#  else
  queue = *ACC_OPENCL_STREAM(stream);
#  endif
  assert(NULL != queue && NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
  if (NULL != clevent) {
    ACC_OPENCL_EXPECT(CL_SUCCESS == clReleaseEvent(clevent));
#  if !defined(NDEBUG)
    clevent = NULL;
#  endif
  }
#  if defined(CL_VERSION_1_2)
#    if defined(ACC_OPENCL_EVENT_BARRIER)
  result = clEnqueueBarrierWithWaitList(queue, 0, NULL, &clevent);
#    else
  result = clEnqueueMarkerWithWaitList(queue, 0, NULL, &clevent);
#    endif
#  else
  result = clEnqueueMarker(queue, &clevent);
#  endif
  if (CL_SUCCESS == result) {
    assert(NULL != clevent);
    *(cl_event*)event = clevent;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred) {
  cl_int status = CL_COMPLETE;
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event && NULL != has_occurred);
  result = clGetEventInfo(*ACC_OPENCL_EVENT(event), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
  if (CL_SUCCESS == result && 0 <= status) *has_occurred = (CL_COMPLETE == status ? 1 : 0);
  else { /* error state */
    result = EXIT_SUCCESS; /* soft-error */
    *has_occurred = 1;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_synchronize(void* event) { /* waits on the host-side */
  int result = EXIT_SUCCESS;
  cl_event clevent;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
  if (NULL != clevent) {
    result = clWaitForEvents(1, &clevent);
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
