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
# define __OPENCL
#endif

#if defined(__OPENCL)
# if !defined(CL_TARGET_OPENCL_VERSION)
#   define CL_TARGET_OPENCL_VERSION 220
# endif
# if defined(__APPLE__)
#   include <OpenCL/cl.h>
# else
#   include <CL/cl.h>
# endif
#else
# error Definition of __OPENCL preprocessor symbol is missing!
#endif

#if !defined(LIBXSTREAM_NOEXT)
# if defined(__APPLE__)
#   include <OpenCL/cl_ext.h>
# else
#   include <CL/cl_ext.h>
# endif
#endif

#if defined(__LIBXS)
# include <libxs_malloc.h>
# include <libxs_timer.h>
# include <libxs_hist.h>
# include <libxs_mem.h>
#else /* code depends on LIBXS */
# include <libxs_source.h>
# define __LIBXS
#endif
/* signal header-only mode before libxstream_macros.h selects build-kind */
#if !defined(LIBXSTREAM_SOURCE) && !defined(LIBXSTREAM_BUILD) && !defined(__LIBXSTREAM)
# define LIBXSTREAM_SOURCE
#endif
#include "libxstream.h"

#if !defined(LIBXSTREAM_MAXALIGN)
# define LIBXSTREAM_MAXALIGN (2 << 20 /*2MB*/)
#endif
#if !defined(LIBXSTREAM_BUFFERSIZE)
# define LIBXSTREAM_BUFFERSIZE (8 << 10 /*8KB*/)
#endif
#if !defined(LIBXSTREAM_MAXSTRLEN)
# define LIBXSTREAM_MAXSTRLEN 48
#endif
#if !defined(LIBXSTREAM_MAXNDEVS)
# define LIBXSTREAM_MAXNDEVS 64
#endif
/* Counted on a per-thread basis! */
#if !defined(LIBXSTREAM_MAXNITEMS)
# define LIBXSTREAM_MAXNITEMS 1024
#endif
/* First char is CSV-separator by default (w/o spaces) */
#if !defined(LIBXSTREAM_DELIMS)
# define LIBXSTREAM_DELIMS ",;"
#endif
#if !defined(LIBXSTREAM_CMEM) && 1
# define LIBXSTREAM_CMEM
#endif
#if !defined(LIBXSTREAM_ASYNC) && 1
# define LIBXSTREAM_ASYNC getenv("LIBXSTREAM_ASYNC")
#endif
#if !defined(LIBXSTREAM_XHINTS) && 1
# define LIBXSTREAM_XHINTS getenv("LIBXSTREAM_XHINTS")
#endif
#if !defined(LIBXSTREAM_STREAM_PRIORITIES) && 0
# if defined(CL_QUEUE_PRIORITY_KHR)
#   define LIBXSTREAM_STREAM_PRIORITIES
# endif
#endif
#if !defined(LIBXSTREAM_USM) && defined(CL_VERSION_2_0) && 1
# if defined(__OFFLOAD_UNIFIED_MEMORY)
/* Do not rely on an Intel extension for pointer arithmetic */
#   define LIBXSTREAM_USM 2
# else
/* Rely on OpenCL 2.0 (eventually mix-in an Intel ext.) */
#   define LIBXSTREAM_USM 1
# endif
#else
# define LIBXSTREAM_USM 0
#endif
/* Activate device by default */
#if !defined(LIBXSTREAM_ACTIVATE)
# define LIBXSTREAM_ACTIVATE -1
#endif

#if defined(CL_VERSION_2_0)
# define LIBXSTREAM_STREAM_PROPERTIES_TYPE cl_queue_properties
# define LIBXSTREAM_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#else
# define LIBXSTREAM_STREAM_PROPERTIES_TYPE cl_int
# define LIBXSTREAM_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
    clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#endif

/** Skip-on-error: if RESULT is success, execute CALL and record any error. */
#if !defined(CL_CHECK)
# define CL_CHECK(RESULT, CALL) \
    do { \
      if (EXIT_SUCCESS == (RESULT)) { \
        const cl_int cl_check_result_ = (CALL); \
        if (CL_SUCCESS != cl_check_result_) { \
          (RESULT) = (int)cl_check_result_; \
          libxstream_opencl_config.device.error.name = libxstream_opencl_strerror(cl_check_result_); \
          libxstream_opencl_config.device.error.code = (RESULT); \
        } \
      } \
    } while (0)
#endif

/** Report error details from the error slot to stderr (if verbose). */
#define CL_ERROR_REPORT(NAME) \
  do { \
    if (0 != libxstream_opencl_config.verbosity && 0 != libxstream_opencl_config.device.error.code) { \
      const char* const cl_error_report_name_ = (const char*)('\0' != *#NAME ? (uintptr_t)(&(NAME)[0]) : 0); \
      if (NULL != cl_error_report_name_ && '\0' != *cl_error_report_name_) { \
        fprintf(stderr, "ERROR ACC/OpenCL: %s: %s (code=%i)\n", cl_error_report_name_, \
          libxstream_opencl_strerror(libxstream_opencl_config.device.error.code), libxstream_opencl_config.device.error.code); \
      } \
      else if (NULL != libxstream_opencl_config.device.error.name) { \
        fprintf(stderr, "ERROR ACC/OpenCL: %s (code=%i)\n", libxstream_opencl_config.device.error.name, \
          libxstream_opencl_config.device.error.code); \
      } \
      else { \
        fprintf(stderr, "ERROR ACC/OpenCL: code=%i\n", libxstream_opencl_config.device.error.code); \
      } \
    } \
  } while (0)

/** Return RESULT from function, reporting error with optional cause NAME. */
#define CL_RETURN(RESULT, NAME) \
  do { \
    if (EXIT_SUCCESS != (RESULT)) CL_ERROR_REPORT(NAME); \
    return (RESULT); \
  } while (0)

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

/** Settings updated during libxstream_device_set_active. */
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
  char std_flag[32];
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
  /** Large GRF mode (opt-in via LIBXSTREAM_BIGGRF). */
  cl_int biggrf;
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
 * and settings updated during libxstream_device_set_active (devinfo).
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

LIBXSTREAM_API int libxstream_stream_priority_range(int* least, int* greatest);

/** Global configuration setup in libxstream_init. */
LIBXSTREAM_APIVAR_PUBLIC(libxstream_opencl_config_t libxstream_opencl_config);

/** If buffers are hinted for non-concurrent writes aka "OpenCL constant". */
LIBXSTREAM_API int libxstream_opencl_use_cmem(const libxstream_opencl_device_t* devinfo);
/** Determines host-pointer registration (for modification). Returns NULL if memory is SVM/USM. */
LIBXSTREAM_API libxstream_opencl_info_memptr_t* libxstream_opencl_info_hostptr(const void* memory);
/**
 * Determines device-pointer registration (for modification; internal). The offset is measured in elsize.
 * Returns NULL if memory is SVM/USM (offset is zero in this case).
 */
LIBXSTREAM_API libxstream_opencl_info_memptr_t* libxstream_opencl_info_devptr_modify(
  libxs_lock_t* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro (lock-control); offset is measured in elsize. */
LIBXSTREAM_API int libxstream_opencl_info_devptr_lock(libxstream_opencl_info_memptr_t* info, libxs_lock_t* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro; offset is measured in elsize. */
LIBXSTREAM_API int libxstream_opencl_info_devptr(
  libxstream_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Finds an existing stream for the given thread-ID (or NULL). */
LIBXSTREAM_API const libxstream_opencl_stream_t* libxstream_opencl_stream(libxs_lock_t* lock, int thread_id);
/** Determines default-stream (see libxstream_opencl_device_t::stream). */
LIBXSTREAM_API const libxstream_opencl_stream_t* libxstream_opencl_stream_default(void);
/** Like libxstream_mem_zero, but supporting an arbitrary value used as initialization pattern. */
LIBXSTREAM_API int libxstream_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, libxstream_stream_t* stream);
/** Amount of device memory; local memory is only non-zero if separate from global. */
LIBXSTREAM_API int libxstream_opencl_info_devmem(
  cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified);
/** Get device-ID for given device, and optionally global device-ID. */
LIBXSTREAM_API int libxstream_opencl_device_id(cl_device_id device, int* device_id, int* global_id);
/** Confirm the vendor of the given device. */
LIBXSTREAM_API int libxstream_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name);
/** Capture or calculate UID based on the device-name. */
LIBXSTREAM_API int libxstream_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid);
/** Based on the device-ID, return the device's UID (capture or calculate), device name, and platform name. */
LIBXSTREAM_API int libxstream_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup);
/** Return the OpenCL support-level for the given device. */
LIBXSTREAM_API int libxstream_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[32], cl_device_type* type);
/** Check if given device supports the extensions. */
LIBXSTREAM_API int libxstream_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts);
/** Create context for given device. */
LIBXSTREAM_API int libxstream_opencl_create_context(cl_device_id device_id, cl_context* context);
/** Internal variant of libxstream_device_set_active. */
LIBXSTREAM_API int libxstream_opencl_set_active_device(libxs_lock_t* lock, int device_id);
/** Assemble flags to support atomic operations. */
LIBXSTREAM_API int libxstream_opencl_flags_atomics(const libxstream_opencl_device_t* devinfo, libxstream_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen);
/** Assemble given defines and internal definitions. */
LIBXSTREAM_API int libxstream_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup);
/** Combines build-params, build-options, and extra flags. */
LIBXSTREAM_API int libxstream_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size);
/**
 * Build program from source with given build_params and build_options.
 * The build_params are meant to instantiate the kernel (-D) whereas build_options
 * are meant to be compiler-flags. The source_kind denotes source's content:
 *  0: OpenCL source code
 *  1: Filename (OpenCL or binary)
 * >1: Binary code (source_kind denotes size)
 * The name is used for dump-filenames and diagnostic messages.
 * try_build_options are extra flags appended speculatively: if the build fails
 * with them, the program is rebuilt without them. try_ok (if non-NULL) receives
 * EXIT_SUCCESS when try_build_options were accepted, EXIT_FAILURE otherwise.
 * extnames/num_exts prepend #pragma OPENCL EXTENSION ... enable directives
 * for the given extension names (only for OpenCL source, i.e. source_kind < 2).
 * Caller must release the program with clReleaseProgram.
 */
LIBXSTREAM_API int libxstream_opencl_program(size_t source_kind, const char source[], const char name[], const char build_params[],
  const char build_options[], const char try_build_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_program* program);
/** Extract a kernel from a built program. */
LIBXSTREAM_API int libxstream_opencl_kernel_query(cl_program program, const char kernel_name[], cl_kernel* kernel);
/** Convenience: build program, extract kernel, release program. */
LIBXSTREAM_API int libxstream_opencl_kernel(size_t source_kind, const char source[], const char kernel_name[],
  const char build_params[], const char build_options[], const char try_build_options[], int* try_ok, const char* const extnames[],
  size_t num_exts, cl_kernel* kernel);
/** Per-thread variant of libxstream_device_sync. */
LIBXSTREAM_API int libxstream_opencl_device_synchronize(libxs_lock_t* lock, int thread_id);
/** To support USM, call this function for pointer arguments instead of clSetKernelArg. */
LIBXSTREAM_API int libxstream_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value);

/** Measure time in seconds for the given event. */
LIBXSTREAM_API double libxstream_opencl_duration(cl_event event, int* result_code);

/** Map a raw cl_int error code to a human-readable string. */
LIBXSTREAM_API const char* libxstream_opencl_strerror(cl_int err);
/** Consume and clear the last error. */
LIBXSTREAM_API int libxstream_opencl_error_consume(void);

/* header-only: include implementation (deferred from libxstream_macros.h) */
#if defined(LIBXSTREAM_SOURCE) && !defined(LIBXSTREAM_SOURCE_H)
# include "libxstream_source.h"
#endif

#endif /*LIBXSTREAM_OPENCL_H*/
