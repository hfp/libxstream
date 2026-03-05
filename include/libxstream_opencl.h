/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_OPENCL_H
#define LIBXSTREAM_OPENCL_H

/* Support for other libraries, e.g., CP2K's DBM/DBT */
#if defined(__OFFLOAD_OPENCL) && !defined(__OPENCL)
#  define __OPENCL
#endif

#if defined(__OPENCL)
#  if !defined(CL_TARGET_OPENCL_VERSION)
#    define CL_TARGET_OPENCL_VERSION 220
#  endif
#  if defined(__APPLE__)
#    include <OpenCL/cl.h>
#  else
#    include <CL/cl.h>
#  endif
#else
#  error Definition of __OPENCL preprocessor symbol is missing!
#endif

#if !defined(LIBXSTREAM_NOEXT)
#  if defined(__APPLE__)
#    include <OpenCL/cl_ext.h>
#  else
#    include <CL/cl_ext.h>
#  endif
#endif

#include "libxstream.h"
#if defined(__LIBXS)
#  include <libxs_malloc.h>
#  include <libxs_timer.h>
#  include <libxs_hist.h>
#  include <libxs_mem.h>
#else /* code depends on LIBXS */
#  include <libxs_source.h>
#  define __LIBXS
#endif

#if !defined(LIBXSTREAM_MAXALIGN)
#  define LIBXSTREAM_MAXALIGN (2 << 20 /*2MB*/)
#endif
#if !defined(LIBXSTREAM_BUFFERSIZE)
#  define LIBXSTREAM_BUFFERSIZE (8 << 10 /*8KB*/)
#endif
#if !defined(LIBXSTREAM_MAXSTRLEN)
#  define LIBXSTREAM_MAXSTRLEN 48
#endif
#if !defined(LIBXSTREAM_MAXNDEVS)
#  define LIBXSTREAM_MAXNDEVS 64
#endif
/* Counted on a per-thread basis! */
#if !defined(LIBXSTREAM_MAXNITEMS)
#  define LIBXSTREAM_MAXNITEMS 1024
#endif
/* First char is CSV-separator by default (w/o spaces) */
#if !defined(LIBXSTREAM_DELIMS)
#  define LIBXSTREAM_DELIMS ",;"
#endif
#if !defined(LIBXSTREAM_CMEM) && 1
#  define LIBXSTREAM_CMEM
#endif
#if !defined(LIBXSTREAM_ASYNC) && 1
#  define LIBXSTREAM_ASYNC getenv("LIBXSTREAM_ASYNC")
#endif
#if !defined(LIBXSTREAM_XHINTS) && 1
#  define LIBXSTREAM_XHINTS getenv("LIBXSTREAM_XHINTS")
#endif
#if !defined(LIBXSTREAM_STREAM_PRIORITIES) && 0
#  if defined(CL_QUEUE_PRIORITY_KHR)
#    define LIBXSTREAM_STREAM_PRIORITIES
#  endif
#endif
#if !defined(LIBXSTREAM_USM) && defined(CL_VERSION_2_0) && 1
#  if defined(__OFFLOAD_UNIFIED_MEMORY)
/* Do not rely on an Intel extension for pointer arithmetic */
#    define LIBXSTREAM_USM 2
#  else
/* Rely on OpenCL 2.0 (eventually mix-in an Intel ext.) */
#    define LIBXSTREAM_USM 1
#  endif
#else
#  define LIBXSTREAM_USM 0
#endif
/* Activate device by default */
#if !defined(LIBXSTREAM_ACTIVATE) && 0
#  define LIBXSTREAM_ACTIVATE 0
#endif

#if defined(CL_VERSION_2_0)
#  define LIBXSTREAM_STREAM_PROPERTIES_TYPE cl_queue_properties
#  define LIBXSTREAM_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#else
#  define LIBXSTREAM_STREAM_PROPERTIES_TYPE cl_int
#  define LIBXSTREAM_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
    clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#endif

#define LIBXSTREAM_ERROR() libxstream_opencl_config.device.error.code
#define LIBXSTREAM_ERROR_NAME(CODE) \
  ((EXIT_SUCCESS != libxstream_opencl_config.device.error.code && (CODE) == libxstream_opencl_config.device.error.code) \
      ? libxstream_opencl_config.device.error.name \
      : cl_strerror(CODE))

#define LIBXSTREAM_ERROR_REPORT(NAME) \
  do { \
    const char* const acc_opencl_error_report_name_ = (const char*)('\0' != *#NAME ? (uintptr_t)(NAME + 0) : 0); \
    if (0 != libxstream_opencl_config.verbosity) { \
      if (NULL != acc_opencl_error_report_name_ && '\0' != *acc_opencl_error_report_name_) { \
        fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", acc_opencl_error_report_name_); \
      } \
      else if (0 != libxstream_opencl_config.device.error.code) { \
        if (NULL != libxstream_opencl_config.device.error.name && '\0' != *libxstream_opencl_config.device.error.name) { \
          fprintf(stderr, "ERROR ACC/OpenCL: %s: %s (code=%i)\n", libxstream_opencl_config.device.error.name, \
            cl_strerror(libxstream_opencl_config.device.error.code), libxstream_opencl_config.device.error.code); \
        } \
        else if (-1001 == libxstream_opencl_config.device.error.code) { \
          fprintf(stderr, "ERROR ACC/OpenCL: incomplete OpenCL installation?\n"); \
        } \
        else { \
          fprintf(stderr, "ERROR ACC/OpenCL: %s (code=%i)\n", \
            cl_strerror(libxstream_opencl_config.device.error.code), libxstream_opencl_config.device.error.code); \
        } \
      } \
      memset(&libxstream_opencl_config.device.error, 0, sizeof(libxstream_opencl_config.device.error)); \
    } \
    assert(!"SUCCESS"); \
  } while (0)

#define LIBXSTREAM_CHECK(RESULT, CMD, MSG) \
  do { \
    if (EXIT_SUCCESS == (RESULT)) { \
      (RESULT) = (CMD); /* update result given code from cmd */ \
      libxstream_opencl_config.device.error.name = (MSG); \
      libxstream_opencl_config.device.error.code = (RESULT); \
      assert(EXIT_SUCCESS == (RESULT)); \
    } \
    else LIBXSTREAM_ERROR_REPORT(); \
  } while (0)

#define LIBXSTREAM_RETURN_CAUSE(RESULT, NAME) \
  do { \
    if (EXIT_SUCCESS == (RESULT)) { \
      assert(EXIT_SUCCESS == libxstream_opencl_config.device.error.code); \
      memset(&libxstream_opencl_config.device.error, 0, sizeof(libxstream_opencl_config.device.error)); \
    } \
    else LIBXSTREAM_ERROR_REPORT(NAME); \
    return (RESULT); \
  } while (0)

#define LIBXSTREAM_RETURN(RESULT) \
  do { \
    if (EXIT_SUCCESS == (RESULT)) { \
      assert(EXIT_SUCCESS == libxstream_opencl_config.device.error.code); \
      memset(&libxstream_opencl_config.device.error, 0, sizeof(libxstream_opencl_config.device.error)); \
    } \
    else LIBXSTREAM_ERROR_REPORT(); \
    return (RESULT); \
  } while (0)


#if defined(__cplusplus)
extern "C" {
#endif

/** Rich type denoting an error. */
typedef struct libxstream_opencl_error_t {
  const char* name;
  int code;
} libxstream_opencl_error_t;

/** Information about streams (libxstream_stream_create). */
typedef struct libxstream_stream_t {
  cl_command_queue queue;
  int tid;
#if defined(LIBXSTREAM_STREAM_PRIORITIES)
  int priority;
#endif
} libxstream_opencl_stream_t;

/** Information about events (libxstream_event_create). */
struct libxstream_event_t {
  cl_event cl_evt;
};

/** Settings updated during libxstream_set_active_device. */
typedef struct libxstream_opencl_device_t {
  /** Activated device context. */
  cl_context context;
  /**
   * Stream for internal purpose, e.g., stream-argument
   * (ACC-interface) can be NULL (synchronous)
   */
  libxstream_opencl_stream_t stream;
  /** Last error (not necessarily thread-safe/specific). */
  libxstream_opencl_error_t error;
  /** OpenCL compiler flag (language standard). */
  char std_flag[16];
  /** OpenCL support-level (major and minor). */
  cl_int std_level[2], std_clevel[2];
  /**
   * Maximum size of workgroup (WG), preferred multiple of WG-size (PM),
   * and size of subgoup (SG) only if larger-equal than PM. SG is signaled
   * smaller if an alternative SG-size exists (SG is zero if no support).
   */
  size_t wgsize[3];
  /** Maximum size of memory allocations and constant buffer. */
  cl_ulong size_maxalloc, size_maxcmem;
  /** Kind of device (GPU, CPU, or other). */
  cl_device_type type;
  /** Whether host memory is unified, and SVM/USM capabilities. */
  cl_int unified, usm;
  /** Device-UID. */
  cl_uint uid;
  /** Main vendor? */
  cl_int intel, amd, nv;
  /* USM support functions */
  cl_int (*clSetKernelArgMemPointerINTEL)(cl_kernel, cl_uint, const void*);
  cl_int (*clEnqueueMemFillINTEL)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*);
  cl_int (*clEnqueueMemcpyINTEL)(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*);
  void* (*clDeviceMemAllocINTEL)(cl_context, cl_device_id, const /*cl_mem_properties_intel*/ void*, size_t, cl_uint, cl_int*);
  void* (*clSharedMemAllocINTEL)(cl_context, cl_device_id, const /*cl_mem_properties_intel*/ void*, size_t, cl_uint, cl_int*);
  void* (*clHostMemAllocINTEL)(cl_context, const /*cl_mem_properties_intel*/ void*, size_t, cl_uint, cl_int*);
  cl_int (*clMemFreeINTEL)(cl_context, void*);
} libxstream_opencl_device_t;

typedef enum libxstream_event_kind_t {
  libxstream_event_kind_none,
  libxstream_event_kind_h2d,
  libxstream_event_kind_d2h,
  libxstream_event_kind_d2d
} libxstream_event_kind_t;

/** Information about host/device-memory pointer. */
typedef struct libxstream_opencl_info_memptr_t {
  cl_mem memory; /* first item! */
  void* memptr;
} libxstream_opencl_info_memptr_t;

/** Enumeration of FP-atomic kinds. */
typedef enum libxstream_opencl_atomic_fp_t {
  libxstream_opencl_atomic_fp_no = 0,
  libxstream_opencl_atomic_fp_32 = 1,
  libxstream_opencl_atomic_fp_64 = 2
} libxstream_opencl_atomic_fp_t;

/**
 * Settings discovered/setup during libxstream_init (independent of the device)
 * and settings updated during libxstream_set_active_device (devinfo).
 */
typedef struct libxstream_opencl_config_t {
  /** Table of ordered viable/discovered devices (matching criterion). */
  cl_device_id devices[LIBXSTREAM_MAXNDEVS];
  /** Active device (per process). */
  libxstream_opencl_device_t device;
  /** Locks used by domain. */
  libxs_lock_t *lock_main, *lock_stream, *lock_event, *lock_memory;
  /** All memptrs and related storage/counter. */
  libxstream_opencl_info_memptr_t **memptrs, *memptr_data;
  size_t nmemptrs; /* counter */
  /** Host memory pool (3-arg libxs_malloc). */
  libxs_malloc_pool_t* pool_hst;
  /** Handle-counter. */
  size_t nstreams, nevents;
  /** All streams and related storage. */
  libxstream_opencl_stream_t **streams, *stream_data;
  /** All events and related storage. */
  cl_event **events, *event_data;
  /** Device-ID to lookup devices-array. */
  cl_int device_id;
  /** Kernel-parameters are matched against device's UID */
  cl_uint devmatch;
  /** Split devices into sub-devices (if possible) */
  cl_int devsplit;
  /** Verbosity level (output on stderr). */
  cl_int verbosity;
  /** Guessed number of ranks per node (local), and rank-ID. */
  cl_int nranks, nrank;
  /** Non-zero if library is initialized (negative: no device). */
  cl_int ndevices;
  /** Maximum number of threads (omp_get_max_threads). */
  cl_int nthreads;
#if defined(LIBXSTREAM_STREAM_PRIORITIES)
  /** Runtime-adjust LIBXSTREAM_STREAM_PRIORITIES. */
  cl_int priority;
#endif
  /** Runtime-enable LIBXSTREAM_PROFILE_DBCSR. */
  cl_int profile;
  /** Detailed/optional insight. */
  libxs_hist_t *hist_h2d, *hist_d2h, *hist_d2d;
  /** Configuration and execution-hints. */
  cl_int xhints;
  /** Asynchronous memory operations. */
  cl_int async;
  /** Debug (output/symbols, etc.). */
  cl_int debug;
  /** Dump level. */
  cl_int dump;
  /** WA level */
  cl_int wa;
} libxstream_opencl_config_t;

/** Global configuration setup in libxstream_init. */
extern libxstream_opencl_config_t libxstream_opencl_config;

/** If buffers are hinted for non-concurrent writes aka "OpenCL constant". */
int libxstream_opencl_use_cmem(const libxstream_opencl_device_t* devinfo);
/** Determines host-pointer registration (for modification). Returns NULL if memory is SVM/USM. */
libxstream_opencl_info_memptr_t* libxstream_opencl_info_hostptr(const void* memory);
/**
 * Determines device-pointer registration (for modification; internal). The offset is measured in elsize.
 * Returns NULL if memory is SVM/USM (offset is zero in this case).
 */
libxstream_opencl_info_memptr_t* libxstream_opencl_info_devptr_modify(
  libxs_lock_t* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro (lock-control); offset is measured in elsize. */
int libxstream_opencl_info_devptr_lock(libxstream_opencl_info_memptr_t* info, libxs_lock_t* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro; offset is measured in elsize. */
int libxstream_opencl_info_devptr(
  libxstream_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Finds an existing stream for the given thread-ID (or NULL). */
const libxstream_opencl_stream_t* libxstream_opencl_stream(libxs_lock_t* lock, int thread_id);
/** Determines default-stream (see libxstream_opencl_device_t::stream). */
const libxstream_opencl_stream_t* libxstream_opencl_stream_default(void);
/** Like libxstream_memset_zero, but supporting an arbitrary value used as initialization pattern. */
int libxstream_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, libxstream_stream_t* stream);
/** Amount of device memory; local memory is only non-zero if separate from global. */
int libxstream_opencl_info_devmem(cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified);
/** Get device-ID for given device, and optionally global device-ID. */
int libxstream_opencl_device_id(cl_device_id device, int* device_id, int* global_id);
/** Confirm the vendor of the given device. */
int libxstream_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name);
/** Capture or calculate UID based on the device-name. */
int libxstream_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid);
/** Based on the device-ID, return the device's UID (capture or calculate), device name, and platform name. */
int libxstream_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup);
/** Return the OpenCL support-level for the given device. */
int libxstream_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[16], cl_device_type* type);
/** Check if given device supports the extensions. */
int libxstream_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts);
/** Create context for given device. */
int libxstream_opencl_create_context(cl_device_id device_id, cl_context* context);
/** Internal variant of libxstream_set_active_device. */
int libxstream_opencl_set_active_device(libxs_lock_t* lock, int device_id);
/** Assemble flags to support atomic operations. */
int libxstream_opencl_flags_atomics(const libxstream_opencl_device_t* devinfo, libxstream_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen);
/** Assemble given defines and internal definitions. */
int libxstream_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup);
/** Combines build-params, build-options, and extra flags. */
int libxstream_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size);
/**
 * Build kernel from source with given kernel_name, build_params and build_options.
 * The build_params are meant to instantiate the kernel (-D) whereas build_options
 * are are meant to be compiler-flags. The source_kind denotes source's content:
 *  0: OpenCL source code
 *  1: Filename (OpenCL or binary)
 * >1: Binary code (source_kind denotes size)
 */
int libxstream_opencl_kernel(size_t source_kind, const char source[], const char kernel_name[], const char build_params[],
  const char build_options[], const char try_build_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_kernel* kernel);
/** Per-thread variant of libxstream_device_synchronize. */
int libxstream_opencl_device_synchronize(libxs_lock_t* lock, int thread_id);
/** To support USM, call this function for pointer arguments instead of clSetKernelArg. */
int libxstream_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value);

/** Measure time in seconds for the given event. */
double libxstream_opencl_duration(cl_event event, int* result_code);

/** Map a raw cl_int error code to a human-readable string. */
const char* cl_strerror(cl_int err);

#if defined(__cplusplus)
}
#endif

#endif /*LIBXSTREAM_OPENCL_H*/
