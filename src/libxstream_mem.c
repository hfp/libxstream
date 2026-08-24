/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
# include <libxstream/libxstream_opencl.h>
# include <string.h>
# if defined(_WIN32)
#   include <Windows.h>
# else
#   if !defined(__linux__) && defined(__APPLE__) && defined(__MACH__)
#     include <sys/types.h>
#     include <sys/sysctl.h>
#   endif
#   include <unistd.h>
# endif

# if !defined(LIBXSTREAM_MEM_ALLOC)
#   if 1
#     define LIBXSTREAM_MEM_ALLOC(SIZE, ALIGNMENT) libxs_malloc(libxstream_opencl_config.pool_hst, SIZE, ALIGNMENT)
#     define LIBXSTREAM_MEM_FREE(PTR) libxs_free(PTR)
#   else
#     define LIBXSTREAM_MEM_ALLOC(SIZE, ALIGNMENT) aligned_alloc(ALIGNMENT, SIZE)
#     define LIBXSTREAM_MEM_FREE(PTR) free(PTR)
#   endif
# endif
# if !defined(LIBXSTREAM_MEM_ALIGNSCALE)
#   define LIBXSTREAM_MEM_ALIGNSCALE 8
# endif
/**
 * Staging buffer per thread, and the smallest transfer worth staging. The
 * buffer is a window rather than a mirror of the operand: a transfer larger
 * than it is copied and enqueued in chunks, which bounds the extra host memory
 * to this size per transferring thread however large the operands grow.
 */
# if !defined(LIBXSTREAM_MEM_STAGE_MIN)
#   define LIBXSTREAM_MEM_STAGE_MIN (1 << 20)
# endif
/**
 * Threads for the staging copy. Staging only pays if the copy outruns the
 * pageable transport it replaces: on an H100 a serial copy reaches 12.8 GB/s
 * against 10.9 GB/s direct, i.e. a loss, while 32 threads reach 32.8 GB/s
 * end-to-end against 55.1 for memory the runtime owns. More threads do not
 * help - 224 measured 29.6 - so the count is capped rather than taken from the
 * team size.
 */
/**
 * Staging needs both a parallel copy and a per-thread window: without OpenMP
 * the copy is slower than the transport it would replace, and without TLS the
 * window needs a lock in the transfer path, where a thread losing the race
 * would fall back to that same transport silently.
 */
# if defined(_OPENMP) && !defined(LIBXS_NO_TLS)
#   define LIBXSTREAM_MEM_STAGING
# endif
# if !defined(LIBXSTREAM_MEM_SVM_INTEL) && 0
#   define LIBXSTREAM_MEM_SVM_INTEL
# endif
# if !defined(LIBXSTREAM_MEM_HST_INTEL) && 0
#   define LIBXSTREAM_MEM_HST_INTEL
# endif
# if !defined(LIBXSTREAM_MEM_SVM_USM) && 0
#   define LIBXSTREAM_MEM_SVM_USM
# endif
# if !defined(LIBXSTREAM_MEM_DEBUG) && 0
#   if !defined(NDEBUG)
#     define LIBXSTREAM_MEM_DEBUG
#   endif
# endif


#if defined(LIBXSTREAM_MEM_STAGING)
/**
 * Staging buffer of the calling thread. Thread-local rather than shared so that
 * no transfer can lose a race for it and fall back to the transport this is
 * meant to avoid - a silent slow path is the failure mode that makes pageable
 * transfers hard to notice in the first place. Released with the context at
 * finalize, along with every other host allocation.
 */
static LIBXS_TLS void* libxstream_mem_stage_ptr = NULL;
static LIBXS_TLS size_t libxstream_mem_stage_nbytes = 0;
#endif


LIBXSTREAM_API_INTERN int libxstream_memptr_register(cl_mem /*memory*/, void** /*memptr_out*/);
LIBXSTREAM_API_INTERN int libxstream_memptr_register(cl_mem memory, void** memptr_out)
{
  static const char source[] =
    "kernel void memptr(global unsigned long* ptr) {\n"
    "  const union { global unsigned long* p; unsigned long u; } cast = { ptr };\n"
    "  const size_t i = get_global_id(0);\n"
    "  ptr[i] = cast.u + i;\n"
    "}\n";
  libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  libxstream_opencl_info_memptr_t* info = NULL;
  const size_t size = 1;
  int result = EXIT_SUCCESS;
  void* memptr = NULL;

  assert(NULL != memptr_out && NULL != memory);
  assert(NULL != devinfo->stream.queue);
  assert(sizeof(size_t) == sizeof(cl_ulong));

  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
  if (devinfo->context != devinfo->memptr_context) {
    if (NULL != devinfo->memptr_kernel) {
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseKernel(devinfo->memptr_kernel));
      devinfo->memptr_kernel = NULL;
    }
    devinfo->memptr_context = NULL;
  }
  if (NULL == devinfo->memptr_kernel) {
    result = libxstream_opencl_kernel(0, source, "memptr",
      NULL, NULL, NULL, NULL, NULL, 0, &devinfo->memptr_kernel);
    if (EXIT_SUCCESS == result) devinfo->memptr_context = devinfo->context;
  }
  CL_CHECK(result, clSetKernelArg(devinfo->memptr_kernel, 0, sizeof(cl_mem), &memory));
  if (EXIT_SUCCESS == result) {
    result = clEnqueueNDRangeKernel(devinfo->stream.queue, devinfo->memptr_kernel,
      1, NULL, &size, NULL, 0, NULL, NULL);
  }
  if (EXIT_SUCCESS == result) {
    result = clEnqueueReadBuffer(devinfo->stream.queue, memory, CL_TRUE,
      0, sizeof(void*), &memptr, 0, NULL, NULL);
  }
  assert(EXIT_SUCCESS != result || NULL != memptr);
  if (EXIT_SUCCESS == result) {
    info = (libxstream_opencl_info_memptr_t*)libxs_pmalloc(
      (void**)libxstream_opencl_config.memptrs, &libxstream_opencl_config.nmemptrs);
    if (NULL != info) {
      info->memory = memory;
      info->memptr = memptr;
    }
    else result = EXIT_FAILURE;
  }
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);

  *memptr_out = memptr;
  return result;
}


/**
 * Registers the pages of a host allocation with the CUDA runtime (no-op unless
 * the application linked it). Takes the base of the allocation and its padded
 * size rather than the aligned pointer handed to the caller: registration is
 * page-granular and cudaHostUnregister accepts only the very pointer that was
 * registered. Failure is not an error - memory the CUDA runtime rejects merely
 * stays pageable for its own transfers.
 */
LIBXSTREAM_API_INTERN void libxstream_mem_host_register(void* /*memptr*/, size_t /*nbytes*/);
LIBXSTREAM_API_INTERN void libxstream_mem_host_register(void* memptr, size_t nbytes)
{
  if (NULL != libxstream_opencl_config.cudaHostRegister && NULL != memptr) {
    /* cudaHostRegisterPortable: pinned in every context, i.e. also after cudaSetDevice */
    const int status = libxstream_opencl_config.cudaHostRegister(memptr, nbytes, 0x01);
    LIBXS_ATOMIC_SIZE(LIBXS_ATOMIC_ADD_FETCH)(&libxstream_opencl_config.nhostreg, 1, LIBXS_ATOMIC_RELAXED);
    if (EXIT_SUCCESS == status) {
      LIBXS_ATOMIC_SIZE(LIBXS_ATOMIC_ADD_FETCH)(&libxstream_opencl_config.nhostreg_ok, 1, LIBXS_ATOMIC_RELAXED);
    }
  }
}


/**
 * Counterpart of libxstream_mem_host_register, to be called before the memory is
 * given back: CUDA holding a mapping of pages the OpenCL runtime has released is
 * a use-after-free inside the CUDA driver.
 *
 * Whether this particular allocation was accepted is not tracked, so one that
 * was rejected is unregistered in vain. That is cheaper than a per-allocation
 * flag, and the case is confined to a process where registration failed at least
 * once (nhostreg_ok reports it).
 */
LIBXSTREAM_API_INTERN void libxstream_mem_host_unregister(void* /*memptr*/);
LIBXSTREAM_API_INTERN void libxstream_mem_host_unregister(void* memptr)
{
  if (NULL != libxstream_opencl_config.cudaHostUnregister && NULL != memptr &&
      0 != libxstream_opencl_config.nhostreg_ok)
  {
    LIBXS_ELIDE_RESULT(int, libxstream_opencl_config.cudaHostUnregister(memptr));
  }
}


LIBXSTREAM_API_INTERN void* libxstream_mem_hst_xmalloc(size_t size, const void* extra)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  void* result = NULL;
  int status = EXIT_SUCCESS;
  LIBXS_UNUSED(extra);
  if (libxstream_opencl_mem_hst_unknown == libxstream_opencl_config.mem_hst) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (libxstream_opencl_mem_hst_unknown == libxstream_opencl_config.mem_hst) {
      libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_malloc;
# if (1 >= LIBXSTREAM_USM)
      if (NULL != devinfo->clSharedMemAllocINTEL && NULL != devinfo->clMemFreeINTEL) {
        libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_shared_intel;
        libxstream_opencl_config.pool_hst_clSharedMemAllocINTEL = devinfo->clSharedMemAllocINTEL;
        libxstream_opencl_config.pool_hst_clMemFreeINTEL = devinfo->clMemFreeINTEL;
      }
      else if (NULL != devinfo->clHostMemAllocINTEL && NULL != devinfo->clMemFreeINTEL) {
        libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_host_intel;
        libxstream_opencl_config.pool_hst_clHostMemAllocINTEL = devinfo->clHostMemAllocINTEL;
        libxstream_opencl_config.pool_hst_clMemFreeINTEL = devinfo->clMemFreeINTEL;
      }
# endif
# if (0 != LIBXSTREAM_USM)
      if (libxstream_opencl_mem_hst_malloc == libxstream_opencl_config.mem_hst &&
          0 != devinfo->usm && 0 != devinfo->unified) {
        libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_svm;
      }
# endif
      if (libxstream_opencl_mem_hst_malloc != libxstream_opencl_config.mem_hst) {
        libxstream_opencl_config.pool_hst_context = devinfo->context;
        libxstream_opencl_config.pool_hst_device = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
# if (0 != LIBXSTREAM_USM)
        libxstream_opencl_config.pool_hst_usm = devinfo->usm;
# endif
        if (NULL != libxstream_opencl_config.pool_hst_context &&
            EXIT_SUCCESS != clRetainContext(libxstream_opencl_config.pool_hst_context)) {
          libxstream_opencl_config.pool_hst_context = NULL;
          libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_malloc;
        }
      }
# if (0 != LIBXSTREAM_USM)
      if (libxstream_opencl_mem_hst_svm == libxstream_opencl_config.mem_hst) {
        libxstream_opencl_config.pool_hst_queue = devinfo->stream.queue;
        if ((0 == ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) &
                   libxstream_opencl_config.pool_hst_usm) && NULL == libxstream_opencl_config.pool_hst_queue) ||
            (NULL != libxstream_opencl_config.pool_hst_queue &&
             EXIT_SUCCESS != clRetainCommandQueue(libxstream_opencl_config.pool_hst_queue))) {
          libxstream_opencl_config.pool_hst_queue = NULL;
          if (NULL != libxstream_opencl_config.pool_hst_context) {
            LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseContext(libxstream_opencl_config.pool_hst_context));
            libxstream_opencl_config.pool_hst_context = NULL;
          }
          libxstream_opencl_config.mem_hst = libxstream_opencl_mem_hst_malloc;
        }
      }
# endif
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
  }
  switch (libxstream_opencl_config.mem_hst) {
    case libxstream_opencl_mem_hst_shared_intel: {
# if (1 >= LIBXSTREAM_USM)
      result = libxstream_opencl_config.pool_hst_clSharedMemAllocINTEL(
        libxstream_opencl_config.pool_hst_context, libxstream_opencl_config.pool_hst_device, NULL, size, 0, &status);
# endif
    } break;
    case libxstream_opencl_mem_hst_host_intel: {
# if (1 >= LIBXSTREAM_USM)
      result = libxstream_opencl_config.pool_hst_clHostMemAllocINTEL(
        libxstream_opencl_config.pool_hst_context, NULL, size, 0, &status);
# endif
    } break;
    case libxstream_opencl_mem_hst_svm: {
# if (0 != LIBXSTREAM_USM)
      const int svmflags = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) &
                  libxstream_opencl_config.pool_hst_usm)
                            ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0);
      result = clSVMAlloc(libxstream_opencl_config.pool_hst_context,
        (cl_svm_mem_flags)(CL_MEM_READ_WRITE | svmflags), size, 0);
      if (NULL != result && 0 == svmflags) {
        status = clEnqueueSVMMap(libxstream_opencl_config.pool_hst_queue, CL_TRUE,
          (CL_MAP_READ | CL_MAP_WRITE), result, size, 0, NULL, NULL);
        if (EXIT_SUCCESS != status) {
          clSVMFree(libxstream_opencl_config.pool_hst_context, result);
          result = NULL;
        }
      }
# endif
    } break;
    default: {
      result = malloc(size);
    } break;
  }
  if (EXIT_SUCCESS != status) result = NULL;
  /**
   * Registered here rather than per libxs_malloc, because the pool hands out
   * pointers into these blocks: registration is page-granular, so two
   * sub-allocations sharing a page would collide and freeing one would unpin the
   * other. Only driver-provided memory qualifies - a malloc'ed block can share
   * a page with unrelated heap data, and that case has no OpenCL device anyway.
   */
  if (libxstream_opencl_mem_hst_malloc != libxstream_opencl_config.mem_hst) {
    libxstream_mem_host_register(result, size);
  }
  return result;
}


LIBXSTREAM_API_INTERN void libxstream_mem_hst_xfree(void* pointer, const void* extra)
{
  LIBXS_UNUSED(extra);
  if (libxstream_opencl_mem_hst_malloc != libxstream_opencl_config.mem_hst) {
    libxstream_mem_host_unregister(pointer);
  }
  switch (libxstream_opencl_config.mem_hst) {
    case libxstream_opencl_mem_hst_shared_intel:
    case libxstream_opencl_mem_hst_host_intel: {
# if (1 >= LIBXSTREAM_USM)
      libxstream_opencl_config.pool_hst_clMemFreeINTEL(libxstream_opencl_config.pool_hst_context, pointer);
# endif
    } break;
    case libxstream_opencl_mem_hst_svm: {
# if (0 != LIBXSTREAM_USM)
      if (0 == ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & libxstream_opencl_config.pool_hst_usm)) {
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clEnqueueSVMUnmap(libxstream_opencl_config.pool_hst_queue, pointer, 0, NULL, NULL));
      }
      clSVMFree(libxstream_opencl_config.pool_hst_context, pointer);
# endif
    } break;
    default: {
      free(pointer);
    } break;
  }
}


LIBXSTREAM_API_INTERN void* libxstream_mem_dev_xmalloc(size_t size, const void* extra)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  void* result = NULL;
  int status = EXIT_SUCCESS;
  LIBXS_UNUSED(extra);
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clDeviceMemAllocINTEL) {
    const cl_device_id did = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
#   if defined(LIBXSTREAM_XHINTS)
    if (0 != (8 & libxstream_opencl_config.xhints) && 0 != devinfo->intel && 0 == devinfo->unified) {
      const cl_ulong props[] = {0x4195 /*CL_MEM_ALLOC_FLAGS_INTEL*/, (1u << 22), 0};
      result = devinfo->clDeviceMemAllocINTEL(devinfo->context, did, props, size, 0, &status);
      if (CL_SUCCESS != status) { result = NULL; status = EXIT_SUCCESS; }
    }
    if (NULL == result)
#   endif
    {
      result = devinfo->clDeviceMemAllocINTEL(devinfo->context, did, NULL, size, 0, &status);
    }
  }
  else if (NULL != devinfo->clSharedMemAllocINTEL) {
    const cl_device_id did = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
#   if defined(LIBXSTREAM_XHINTS)
    if (0 != (8 & libxstream_opencl_config.xhints) && 0 != devinfo->intel && 0 == devinfo->unified) {
      const cl_ulong props[] = {0x4195 /*CL_MEM_ALLOC_FLAGS_INTEL*/, (1u << 22), 0};
      result = devinfo->clSharedMemAllocINTEL(devinfo->context, did, props, size, 0, &status);
      if (CL_SUCCESS != status) { result = NULL; status = EXIT_SUCCESS; }
    }
    if (NULL == result)
#   endif
    {
      result = devinfo->clSharedMemAllocINTEL(devinfo->context, did, NULL, size, 0, &status);
    }
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm) {
    cl_svm_mem_flags svmflags = CL_MEM_READ_WRITE;
    if (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)) {
      svmflags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }
#   if defined(LIBXSTREAM_XHINTS)
    if (0 != (8 & libxstream_opencl_config.xhints) && 0 != devinfo->intel && 0 == devinfo->unified &&
        0 != (CL_DEVICE_SVM_ATOMICS & devinfo->usm))
    {
      svmflags |= CL_MEM_SVM_ATOMICS;
    }
#   endif
    result = clSVMAlloc(devinfo->context, svmflags, size, 0);
  }
  else
# endif
  {
    LIBXS_UNUSED(devinfo);
  }
  return (EXIT_SUCCESS == status) ? result : NULL;
}


LIBXSTREAM_API_INTERN void libxstream_mem_dev_xfree(void* pointer, const void* extra)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  /**
   * The stream recorded at allocation time need not outlive the buffer: the
   * pool is drained at finalization, after the application destroyed its
   * streams. A destroyed stream has no queue and nothing left to wait for,
   * so syncing it is both unnecessary and invalid.
   */
  if (NULL != extra) {
    const uintptr_t addr = (uintptr_t)extra;
    const libxstream_opencl_stream_t* const str = (const libxstream_opencl_stream_t*)addr;
    if (NULL != str->queue) libxstream_stream_sync((libxstream_stream_t*)addr);
  }
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clMemFreeINTEL) {
    devinfo->clMemFreeINTEL(devinfo->context, pointer);
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm) {
    clSVMFree(devinfo->context, pointer);
  }
  else
# endif
  {
    LIBXS_UNUSED(devinfo);
    LIBXS_UNUSED(pointer);
  }
}


/**
 * Release the sub-buffers created for pointers into the given parent buffer.
 * Called when the parent is deallocated; the caller holds lock_memory.
 */
LIBXSTREAM_API_INTERN void libxstream_opencl_subbuffer_release(cl_mem parent);
LIBXSTREAM_API_INTERN void libxstream_opencl_subbuffer_release(cl_mem parent)
{
  size_t i = 0;
  assert(NULL != parent);
  for (; i < libxstream_opencl_config.nsubs; ++i) {
    if (parent == libxstream_opencl_config.subs[i].parent) {
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(libxstream_opencl_config.subs[i].memory));
      libxstream_opencl_config.subs[i].memory = NULL;
      libxstream_opencl_config.subs[i].parent = NULL;
    }
  }
}


LIBXSTREAM_API int libxstream_mem_dev_allocate_hint(void** dev_mem, size_t nbytes, libxstream_opencl_mem_hint_t hint)
{
  libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  void* memptr = NULL;
  assert(NULL != dev_mem && NULL != devinfo->context);
# if !defined(LIBXSTREAM_XHINTS)
  LIBXS_UNUSED(hint);
# endif
  if (0 != nbytes) {
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clDeviceMemAllocINTEL) {
      const cl_device_id did = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
      cl_int status = CL_SUCCESS;
#   if defined(LIBXSTREAM_XHINTS)
      if (libxstream_opencl_mem_hint_compress != hint && 0 != devinfo->intel && 0 == devinfo->unified) {
        const cl_ulong props[] = {0x4195 /*CL_MEM_ALLOC_FLAGS_INTEL*/, (1u << 22), 0};
        memptr = devinfo->clDeviceMemAllocINTEL(devinfo->context, did, props, nbytes, 0, &status);
        if (CL_SUCCESS != status) memptr = NULL;
      }
      if (NULL == memptr)
#   endif
      {
        memptr = devinfo->clDeviceMemAllocINTEL(devinfo->context, did, NULL, nbytes, 0, &status);
        if (CL_SUCCESS != status) { memptr = NULL; result = EXIT_FAILURE; }
      }
    }
    else if (NULL != devinfo->clSharedMemAllocINTEL) {
      const cl_device_id did = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
      cl_int status = CL_SUCCESS;
#   if defined(LIBXSTREAM_XHINTS)
      if (libxstream_opencl_mem_hint_compress != hint && 0 != devinfo->intel && 0 == devinfo->unified) {
        const cl_ulong props[] = {0x4195 /*CL_MEM_ALLOC_FLAGS_INTEL*/, (1u << 22), 0};
        memptr = devinfo->clSharedMemAllocINTEL(devinfo->context, did, props, nbytes, 0, &status);
        if (CL_SUCCESS != status) memptr = NULL;
      }
      if (NULL == memptr)
#   endif
      {
        memptr = devinfo->clSharedMemAllocINTEL(devinfo->context, did, NULL, nbytes, 0, &status);
        if (CL_SUCCESS != status) { memptr = NULL; result = EXIT_FAILURE; }
      }
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm) {
      cl_svm_mem_flags svmflags = CL_MEM_READ_WRITE;
      if (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)) {
        svmflags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
      }
#   if defined(LIBXSTREAM_XHINTS)
      if (libxstream_opencl_mem_hint_compress != hint && 0 != devinfo->intel && 0 == devinfo->unified &&
          0 != (CL_DEVICE_SVM_ATOMICS & devinfo->usm))
      {
        svmflags |= CL_MEM_SVM_ATOMICS;
      }
#   endif
      memptr = clSVMAlloc(devinfo->context, svmflags, nbytes, 0);
      if (NULL == memptr) result = EXIT_FAILURE;
    }
    else
# endif
    {
      cl_mem memory = NULL;
      const cl_mem_flags flags = (cl_mem_flags)(CL_MEM_READ_WRITE |
#   if defined(LIBXSTREAM_XHINTS)
        ((libxstream_opencl_mem_hint_compress != hint && 0 != devinfo->intel && 0 == devinfo->unified) ? (1u << 22) : 0)
#   else
        0
#   endif
      );
      memory = clCreateBuffer(devinfo->context, flags, nbytes, NULL, &result);
      if (EXIT_SUCCESS == result && NULL != memory) {
        result = libxstream_memptr_register(memory, &memptr);
        if (EXIT_SUCCESS != result) {
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(memory));
          memptr = NULL;
        }
      }
    }
  }
  *dev_mem = memptr;
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_dev_deallocate_hint(void* dev_mem)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  if (NULL != dev_mem) {
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clMemFreeINTEL) {
      result = devinfo->clMemFreeINTEL(devinfo->context, dev_mem);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm) {
      clSVMFree(devinfo->context, dev_mem);
    }
    else
# endif
    {
      libxstream_opencl_info_memptr_t* info = NULL;
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
      info = libxstream_opencl_info_devptr_modify(NULL, dev_mem, 1, NULL, NULL);
      if (NULL != info && info->memptr == dev_mem && NULL != info->memory) {
        libxstream_opencl_info_memptr_t* const pfree = libxstream_opencl_config.memptrs[libxstream_opencl_config.nmemptrs];
        /* sub-buffers of this buffer are owned here (see libxstream_opencl_subbuffer) */
        libxstream_opencl_subbuffer_release(info->memory);
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(info->memory));
        libxs_pfree(pfree, (void**)libxstream_opencl_config.memptrs, &libxstream_opencl_config.nmemptrs);
        *info = *pfree;
        LIBXS_MEMZERO(pfree);
      }
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API libxstream_opencl_info_memptr_t* libxstream_opencl_info_hostptr(const void* memory)
{
  libxstream_opencl_info_memptr_t* result = NULL;
  if (NULL == libxstream_opencl_config.device.clHostMemAllocINTEL &&
# if (0 != LIBXSTREAM_USM)
      0 == libxstream_opencl_config.device.usm &&
# endif
      NULL != memory)
  {
    assert(sizeof(libxstream_opencl_info_memptr_t) < (uintptr_t)memory);
    result = (libxstream_opencl_info_memptr_t*)((uintptr_t)memory - sizeof(libxstream_opencl_info_memptr_t));
  }
  return result;
}


LIBXSTREAM_API libxstream_opencl_info_memptr_t* libxstream_opencl_info_devptr_modify(
  libxs_lock_t* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset)
{
  libxstream_opencl_info_memptr_t* result = NULL;
# if !defined(LIBXSTREAM_MEM_DEBUG)
  LIBXS_UNUSED(amount);
# endif
  if (NULL != memory) {
    assert(NULL != libxstream_opencl_config.device.context);
    if (/* USM-pointer */
# if (0 != LIBXSTREAM_USM)
      0 != libxstream_opencl_config.device.usm ||
# endif
      NULL != libxstream_opencl_config.device.clDeviceMemAllocINTEL ||
      NULL != libxstream_opencl_config.device.clSharedMemAllocINTEL)
    { /* assume only first item of libxstream_opencl_info_memptr_t is accessed */
      assert(0 != libxstream_opencl_config.device.usm || NULL != libxstream_opencl_config.device.clDeviceMemAllocINTEL ||
        NULL != libxstream_opencl_config.device.clSharedMemAllocINTEL);
      result = NULL; /*(libxstream_opencl_info_memptr_t*)memory*/
      if (NULL != offset) *offset = 0;
    }
    else { /* info-augmented pointer */
      const uintptr_t pointer = (uintptr_t)memory;
      const size_t n = LIBXSTREAM_MAXNITEMS * libxstream_opencl_config.nthreads;
      size_t hit = (size_t)-1, i;
      const libxstream_opencl_info_memptr_t* miss = NULL;
      assert(0 == libxstream_opencl_config.device.usm && NULL == libxstream_opencl_config.device.clDeviceMemAllocINTEL &&
        NULL == libxstream_opencl_config.device.clSharedMemAllocINTEL);
      assert(NULL != libxstream_opencl_config.memptrs);
      if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
      for (i = libxstream_opencl_config.nmemptrs; i < n; ++i) {
        libxstream_opencl_info_memptr_t* const info = libxstream_opencl_config.memptrs[i];
        if (NULL != info) {
          const uintptr_t memptr = (uintptr_t)info->memptr;
          if (memptr == pointer) { /* fast-path */
            if (NULL != offset) *offset = 0;
            result = info;
            break;
          }
          else if (memptr < pointer && NULL != offset) {
            size_t d = pointer - memptr, s = d;
            assert(0 < elsize && 0 != d);
            if (d < hit) miss = info;
            if (d < hit && (1 == elsize || 0 == (d % elsize)) &&
# if defined(LIBXSTREAM_MEM_DEBUG) /* TODO: verify enclosed conditions */
                (EXIT_SUCCESS == clGetMemObjectInfo(info->memory, CL_MEM_SIZE, sizeof(size_t), &s, NULL)) &&
                (NULL == amount || (*amount * elsize + d) <= s) &&
# endif
                (1 == elsize || 0 == (s % elsize)) && d <= s)
            {
              *offset = (1 == elsize ? d : (d / elsize));
              result = info;
              hit = d;
            }
# if defined(LIBXSTREAM_MEM_DEBUG)
            else if (d < hit && 0 != libxstream_opencl_config.debug && 0 != libxstream_opencl_config.verbosity) {
              fprintf(stderr, "ERROR ACC/OpenCL: memory=%p pointer=%p size=%llu offset=%llu info failed\n",
                (const void*)info->memory, info->memptr, (unsigned long long)s,
                (unsigned long long)(1 == elsize ? d : (d / elsize)));
            }
# endif
          }
        }
        else break;
      }
      if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
      if (NULL == result && 0 != libxstream_opencl_config.debug && 0 != libxstream_opencl_config.verbosity) {
        fprintf(stderr, "ERROR ACC/OpenCL: pointer=%p base=%p size=%llu offset=%llu registry=%llu/%llu info failed\n",
          memory, NULL != miss ? miss->memptr : NULL,
          (unsigned long long)(NULL != amount ? (*amount * elsize) : 0),
          (unsigned long long)(NULL != miss ? (pointer - (uintptr_t)miss->memptr) : 0),
          (unsigned long long)(n - libxstream_opencl_config.nmemptrs),
          (unsigned long long)n);
      }
    }
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_info_devptr_lock(libxstream_opencl_info_memptr_t* info, libxs_lock_t* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset)
{
  const libxstream_opencl_info_memptr_t* meminfo = NULL;
  int result = EXIT_SUCCESS;
  void* non_const;
  LIBXS_UNION_ASSIGN(void*, non_const, const void*, memory);
  meminfo = libxstream_opencl_info_devptr_modify(lock, non_const, elsize, amount, offset);
  assert(NULL != info);
  if (NULL == meminfo) { /* USM-pointer */
    if (
# if (0 != LIBXSTREAM_USM)
      0 != libxstream_opencl_config.device.usm ||
# endif
      NULL != libxstream_opencl_config.device.clDeviceMemAllocINTEL ||
      NULL != libxstream_opencl_config.device.clSharedMemAllocINTEL)
    {
      LIBXS_MEMZERO(info);
      info->memory = (cl_mem)non_const;
    }
    else result = EXIT_FAILURE;
  }
  else { /* info-augmented pointer */
    assert(NULL != libxstream_opencl_config.device.context);
    LIBXS_ASSIGN(info, meminfo);
    info->memory = (cl_mem)meminfo->memptr;
  }
  return result;
}


LIBXSTREAM_API int libxstream_opencl_info_devptr(
  libxstream_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset)
{
  libxs_lock_t* const lock_memory = ((
# if (0 != LIBXSTREAM_USM)
                                       0 != libxstream_opencl_config.device.usm ||
# endif
                                       NULL != libxstream_opencl_config.device.clSetKernelArgMemPointerINTEL)
                                       ? NULL /* no lock required */
                                       : libxstream_opencl_config.lock_memory);
  return libxstream_opencl_info_devptr_lock(info, lock_memory, memory, elsize, amount, offset);
}


LIBXSTREAM_API_INTERN int libxstream_mem_host_deallocate_internal(void* /*host_ptr*/, cl_command_queue /*queue*/);
LIBXSTREAM_API_INTERN int libxstream_mem_host_deallocate_internal(void* host_ptr, cl_command_queue queue)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_FAILURE;
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clMemFreeINTEL) {
#   if defined(LIBXSTREAM_MEM_SVM_INTEL) || defined(LIBXSTREAM_MEM_HST_INTEL)
    result = devinfo->clMemFreeINTEL(devinfo->context, host_ptr);
#   else
    LIBXSTREAM_MEM_FREE(host_ptr);
    result = EXIT_SUCCESS;
#   endif
  }
  else
# endif
# if (0 != LIBXSTREAM_USM) && ((1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM))
    if (0 != devinfo->usm && 0 != devinfo->unified)
  {
    if (0 == ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)) {
      result = clEnqueueSVMUnmap(queue, host_ptr, 0, NULL, NULL); /* clSVMFree below synchronizes */
    }
    else result = EXIT_SUCCESS;
    clSVMFree(devinfo->context, host_ptr);
  }
  else
# endif
  {
    LIBXS_UNUSED(queue);
    LIBXSTREAM_MEM_FREE(host_ptr);
    result = EXIT_SUCCESS;
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_host_allocate(void** host_mem, size_t nbytes, libxstream_stream_t* stream)
{
  void* result_ptr = NULL;
  assert(NULL != host_mem);
  if (0 != nbytes) {
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    if (NULL != libxstream_opencl_config.pool_hst && (
# if (1 >= LIBXSTREAM_USM)
        NULL != devinfo->clMemFreeINTEL ||
# endif
# if (0 != LIBXSTREAM_USM)
        0 != devinfo->usm ||
# endif
        NULL == devinfo->context))
    {
      result_ptr = libxs_malloc(libxstream_opencl_config.pool_hst, nbytes, LIBXS_MALLOC_NATIVE);
    }
    else if (NULL != devinfo->context) {
      const libxstream_opencl_stream_t* str;
      int alignment = LIBXS_MAX(0x10000, sizeof(void*));
      int result = EXIT_SUCCESS;
      void* host_ptr = NULL;
      cl_mem memory = NULL;
      const size_t size_meminfo = sizeof(libxstream_opencl_info_memptr_t);
      int memflags = CL_MEM_ALLOC_HOST_PTR;
      str = (NULL != stream ? stream : libxstream_opencl_stream_default());
      assert(NULL != str);
      if ((LIBXSTREAM_MEM_ALIGNSCALE * LIBXS_CACHELINE) <= nbytes) {
        const int a = ((LIBXSTREAM_MEM_ALIGNSCALE * LIBXSTREAM_MAXALIGN) <= nbytes ? LIBXSTREAM_MAXALIGN : LIBXS_CACHELINE);
        if (alignment < a) alignment = a;
      }
      nbytes += alignment + size_meminfo - 1;
# if defined(LIBXSTREAM_XHINTS)
      if (0 != (4 & libxstream_opencl_config.xhints) && (0 != devinfo->nv || NULL != (LIBXSTREAM_XHINTS))) {
        host_ptr = LIBXSTREAM_MEM_ALLOC(nbytes, alignment);
        if (NULL != host_ptr) memflags = CL_MEM_USE_HOST_PTR;
      }
# endif
      memory = clCreateBuffer(devinfo->context, (cl_mem_flags)(CL_MEM_READ_WRITE | memflags), nbytes, host_ptr, &result);
      if (EXIT_SUCCESS == result) {
        void* mapped = host_ptr;
        if (NULL == host_ptr) {
          mapped = clEnqueueMapBuffer(str->queue, memory, CL_TRUE /*always block*/,
# if defined(LIBXSTREAM_XHINTS) && (defined(CL_VERSION_1_2) || defined(CL_MAP_WRITE_INVALIDATE_REGION))
            (16 & libxstream_opencl_config.xhints) ? CL_MAP_WRITE_INVALIDATE_REGION :
# endif
                                                   (CL_MAP_READ | CL_MAP_WRITE),
            0 /*offset*/, nbytes, 0, NULL, NULL, &result);
        }
        assert(EXIT_SUCCESS == result || NULL == mapped);
        if (EXIT_SUCCESS == result) {
          const uintptr_t address = (uintptr_t)mapped;
          const uintptr_t aligned = LIBXS_UP2(address + size_meminfo, alignment);
          libxstream_opencl_info_memptr_t* const meminfo = (libxstream_opencl_info_memptr_t*)(aligned - size_meminfo);
          assert(address + size_meminfo <= aligned && NULL != meminfo);
          meminfo->memory = memory;
          meminfo->memptr = mapped;
          result_ptr = (void*)aligned;
          assert(meminfo == libxstream_opencl_info_hostptr(result_ptr));
          libxstream_mem_host_register(mapped, nbytes);
        }
      }
      if (NULL == result_ptr) {
        if (NULL != memory) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(memory));
        if (NULL != host_ptr) LIBXSTREAM_MEM_FREE(host_ptr);
      }
    }
  }
  *host_mem = result_ptr;
  return (NULL != result_ptr || 0 == nbytes) ? EXIT_SUCCESS : EXIT_FAILURE;
}


LIBXSTREAM_API int libxstream_mem_host_deallocate(void* host_mem, libxstream_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  if (NULL != host_mem) {
    const libxstream_opencl_info_memptr_t* const meminfo = libxstream_opencl_info_hostptr(host_mem);
    if (NULL == meminfo || NULL == meminfo->memory) { /* USM/SVM pointer */
      libxs_free(host_mem);
    }
    else { /* info-augmented pointer (clCreateBuffer path) */
      const libxstream_opencl_stream_t* const str = (NULL != stream ? stream : libxstream_opencl_stream_default());
      const libxstream_opencl_info_memptr_t info = *meminfo;
      int result_release = EXIT_SUCCESS;
      void* host_ptr = NULL;
      assert(NULL != str);
      libxstream_mem_host_unregister(info.memptr);
      if (EXIT_SUCCESS == clGetMemObjectInfo(info.memory, CL_MEM_HOST_PTR, sizeof(void*), &host_ptr, NULL) && NULL != host_ptr) {
        LIBXSTREAM_MEM_FREE(host_ptr);
      }
      else {
        result = clEnqueueUnmapMemObject(str->queue, info.memory, info.memptr, 0, NULL, NULL);
      }
      result_release = clReleaseMemObject(info.memory);
      if (EXIT_SUCCESS == result) result = result_release;
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_host_pin(void* host_mem, size_t nbytes)
{
  int result = EXIT_SUCCESS;
  if (NULL != host_mem && 0 != nbytes) {
    libxstream_pin_resolve(); /* outside the lock: it may load a vendor runtime */
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (1 == libxstream_opencl_config.pin) {
      if (libxstream_opencl_config.npins < (LIBXSTREAM_MAXNPINS)) {
        libxstream_opencl_config.pinptr[libxstream_opencl_config.npins] = (const char*)host_mem;
        libxstream_opencl_config.pinsize[libxstream_opencl_config.npins] = nbytes;
        ++libxstream_opencl_config.npins;
      }
      else result = EXIT_FAILURE;
    }
    else if (2 <= libxstream_opencl_config.pin) {
      libxstream_mem_host_register(host_mem, nbytes);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
      fprintf(stderr, "INFO ACC/OpenCL: pin %p (%lu MB) mode=%i -> %s\n", host_mem,
        (unsigned long)(nbytes >> 20), libxstream_opencl_config.pin,
        EXIT_SUCCESS == result ? "ok" : "rejected");
    }
  }
  return result;
}


LIBXSTREAM_API int libxstream_mem_host_unpin(void* host_mem)
{
  int result = EXIT_SUCCESS;
  if (NULL != host_mem) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (1 == libxstream_opencl_config.pin) {
      size_t i;
      for (i = 0; i < libxstream_opencl_config.npins; ++i) {
        if (libxstream_opencl_config.pinptr[i] == (const char*)host_mem) {
          const size_t last = libxstream_opencl_config.npins - 1;
          libxstream_opencl_config.pinptr[i] = libxstream_opencl_config.pinptr[last];
          libxstream_opencl_config.pinsize[i] = libxstream_opencl_config.pinsize[last];
          libxstream_opencl_config.npins = last;
          break;
        }
      }
    }
    else if (2 <= libxstream_opencl_config.pin) {
      libxstream_mem_host_unregister(host_mem);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
  }
  return result;
}


LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_mem_copy_notify(cl_event /*event*/, cl_int /*event_status*/, void* /*data*/);
LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_mem_copy_notify(cl_event event, cl_int event_status, void* data)
{
#if defined(CL_VERSION_2_0)
  cl_command_type type = CL_COMMAND_SVM_MEMCPY;
#else
  cl_command_type type = 0;
#endif
  int result = EXIT_SUCCESS;
  /**
   * {size, size, duration}: vals[0] is the binning key, which
   * libxs_hist_query_percentile reconstructs from the bucket's position on the
   * axis rather than from the stored samples. A rate must therefore not use it
   * - the amount is carried a second time in vals[1], which is stored and
   * averaged like the duration, so amount and duration describe the same sample.
   */
  double vals[5];
  cl_ulong begin = 0, end = 0;
  result = libxstream_opencl_interval(event, &begin, &end);
  vals[2] = 1E-3 * LIBXS_DELTA(begin, end); /* Microseconds */
  LIBXS_UNUSED(event_status);
  assert(CL_COMPLETE == event_status && NULL != data && 8 == sizeof(data));
  if (EXIT_SUCCESS == result && EXIT_SUCCESS == clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL)) {
    const size_t size = LIBXSTREAM_EVENT_SIZE(data);
    const int kind = LIBXSTREAM_EVENT_KIND(data);
    libxs_hist_t* hist = NULL;
    const char* name = NULL;
    /**
     * The kind recorded at enqueue selects the histogram, but the command type
     * must corroborate it: a stale or foreign event carrying a plausible data
     * word would otherwise be attributed to a real transfer. The buffer command
     * types are checked explicitly; the SVM and Intel-USM paths raise types that
     * match no buffer command (SVM_MEMCPY, SVM_MEMFILL, or vendor-specific), and
     * keying on those alone is what silently dropped every sample on such
     * stacks - so an unrecognized type is accepted, a contradicting one is not.
     */
    int agrees = 1;
    switch (type) {
      case CL_COMMAND_WRITE_BUFFER: agrees = (libxstream_event_kind_h2d == kind); break;
      case CL_COMMAND_READ_BUFFER: agrees = (libxstream_event_kind_d2h == kind); break;
      case CL_COMMAND_COPY_BUFFER: agrees = (libxstream_event_kind_d2d == kind); break;
      case CL_COMMAND_FILL_BUFFER: agrees = (libxstream_event_kind_zero == kind); break;
      default: agrees = 1; /* not a buffer command: kind is the only source */
    }
    vals[0] = vals[1] = 1E-6 * size; /* Megabyte (key and stored amount) */
    if (0 != agrees) {
      switch (kind) {
        case libxstream_event_kind_h2d: {
          hist = libxstream_opencl_config.hist_h2d;
          name = "H2D";
        } break;
        case libxstream_event_kind_d2h: {
          hist = libxstream_opencl_config.hist_d2h;
          name = "D2H";
        } break;
        case libxstream_event_kind_d2d: {
          hist = libxstream_opencl_config.hist_d2d;
          name = "D2D";
        } break;
        case libxstream_event_kind_zero: {
          hist = libxstream_opencl_config.hist_zero;
          name = "ZERO";
        } break;
        default: assert(libxstream_event_kind_none == kind); /* should not happen */
      }
    }
    else assert(0 == "event kind contradicts command type");
    if (NULL != hist) {
      /**
       * Discard durations too close to the timer resolution to be meaningful:
       * the rate they imply is dominated by quantization, and a handful of such
       * samples would otherwise set the histogram range for the useful ones.
       */
      const double floor_us = 1E-3 * (double)(LIBXSTREAM_PROFILE_TICKS * libxstream_opencl_config.device.timer_ns);
      vals[3] = libxstream_opencl_reltime(begin);
      vals[4] = libxstream_opencl_reltime(end);
      if (vals[2] >= floor_us) {
        libxs_hist_push(libxstream_opencl_config.lock_memory, hist, vals);
        /**
         * The same interval device-wide, where a transfer can be seen to
         * overlap a kernel rather than merely coexist with one. Under
         * lock_event rather than lock_memory: the kernel callback pushes into
         * the same histogram, and one histogram cannot be guarded by two locks.
         * Not nested with the push above, so the pair cannot deadlock.
         */
        libxs_hist_push(libxstream_opencl_config.lock_event,
          libxstream_opencl_config.hist_device, vals + 3);
        LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nprofile, 1, LIBXS_ATOMIC_RELAXED);
        if (0 > libxstream_opencl_config.profile_mem) {
          /* relative to the epoch: an absolute timestamp does not survive a
             double, which is what the union depends on as well */
          fprintf(stderr, "PROF ACC/OpenCL: %s mb=%.1f us=%.0f ns=%.0f-%.0f\n",
            name, vals[1], vals[2], vals[3], vals[4]);
        }
      }
      else {
        LIBXS_ATOMIC_ADD_FETCH(&libxstream_opencl_config.nprofile_short, 1, LIBXS_ATOMIC_RELAXED);
        if (0 > libxstream_opencl_config.profile_mem) {
          fprintf(stderr, "PROF ACC/OpenCL: %s mb=%.1f us=%.0f ns=%.0f-%.0f (below %.0f us, discarded)\n",
            name, vals[1], vals[2], vals[3], vals[4], floor_us);
        }
      }
    }
  }
  if (NULL != event) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
}


LIBXSTREAM_API int libxstream_mem_allocate(void** dev_mem, size_t nbytes)
{
  /* assume no lock is needed to protect against context/device changes */
  libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  void* memptr = NULL;
  assert(NULL != dev_mem && NULL != devinfo->context);
  if (0 != nbytes) {
    if (NULL != libxstream_opencl_config.pool_dev && (
# if (1 >= LIBXSTREAM_USM)
        NULL != devinfo->clDeviceMemAllocINTEL ||
        NULL != devinfo->clSharedMemAllocINTEL ||
# endif
# if (0 != LIBXSTREAM_USM)
        0 != devinfo->usm ||
# endif
        0 /*sentinel*/))
    {
      memptr = libxs_malloc(libxstream_opencl_config.pool_dev, nbytes, LIBXS_MALLOC_NATIVE);
    }
    else {
      cl_mem memory = NULL;
# if defined(LIBXSTREAM_XHINTS)
      const int devuid = devinfo->uid, devuids = (0x4905 == devuid || 0x020a == devuid || (0x0bd0 <= devuid && 0x0bdb >= devuid));
      const int try_flag = ((0 != (8 & libxstream_opencl_config.xhints) && 0 != devinfo->intel && 0 == devinfo->unified &&
                              (devuids || NULL != (LIBXSTREAM_XHINTS)))
                              ? (1u << 22)
                              : 0);
      memory = clCreateBuffer(devinfo->context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag), nbytes, NULL /*host_ptr*/, &result);
      if (0 != try_flag && EXIT_SUCCESS != result) /* retry without try_flag */
# endif
      {
        memory = clCreateBuffer(devinfo->context, CL_MEM_READ_WRITE, nbytes, NULL /*host_ptr*/, &result);
      }
      if (EXIT_SUCCESS == result) {
        result = libxstream_memptr_register(memory, &memptr);
      }
      if (EXIT_SUCCESS != result) {
        if (0 != libxstream_opencl_config.verbosity) {
          fprintf(stderr, "ERROR ACC/OpenCL: memory=%p pointer=%p size=%llu failed to allocate (%s, code=%i)\n",
            (const void*)memory, memptr, (unsigned long long)nbytes,
            libxstream_opencl_strerror(result), result);
        }
        if (NULL != memory) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(memory));
        memptr = NULL;
      }
    }
  }
  *dev_mem = memptr;
  return (NULL != memptr || 0 == nbytes) ? EXIT_SUCCESS : EXIT_FAILURE;
}


LIBXSTREAM_API int libxstream_mem_deallocate(void* dev_mem)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  if (NULL != dev_mem) {
    assert(NULL != devinfo->context);
    if (NULL != libxstream_opencl_config.pool_dev && (
# if (1 >= LIBXSTREAM_USM)
        NULL != devinfo->clDeviceMemAllocINTEL ||
        NULL != devinfo->clSharedMemAllocINTEL ||
# endif
# if (0 != LIBXSTREAM_USM)
        0 != devinfo->usm ||
# endif
        0 /*sentinel*/))
    {
      libxs_free(dev_mem);
    }
    else {
      libxstream_opencl_info_memptr_t* info = NULL;
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
      info = libxstream_opencl_info_devptr_modify(NULL, dev_mem, 1 /*elsize*/, NULL /*amount*/, NULL /*offset*/);
      if (NULL != info && info->memptr == dev_mem && NULL != info->memory) {
        libxstream_opencl_info_memptr_t* const pfree = libxstream_opencl_config.memptrs[libxstream_opencl_config.nmemptrs];
        /* sub-buffers of this buffer are owned here (see libxstream_opencl_subbuffer) */
        libxstream_opencl_subbuffer_release(info->memory);
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseMemObject(info->memory));
        libxs_pfree(pfree, (void**)libxstream_opencl_config.memptrs, &libxstream_opencl_config.nmemptrs);
        *info = *pfree;
        LIBXS_MEMZERO(pfree);
      }
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_offset(void** dev_mem, void* other, size_t offset)
{
  int result = EXIT_SUCCESS;
  assert(NULL != dev_mem);
  if (NULL != other || 0 == offset) {
    *dev_mem = (char*)other + offset;
  }
  else {
    result = EXIT_FAILURE;
    *dev_mem = NULL;
  }
  CL_RETURN(result, "");
}


/**
 * True when the range lies inside one the caller declared with
 * libxstream_mem_host_pin. A miss is not an error: it only means the transfer
 * takes the ordinary route, which is correct for memory the runtime allocated.
 * The list is short by construction, so a scan costs less than the smallest
 * transfer that reaches it.
 */
#if defined(LIBXSTREAM_MEM_STAGING)
static int libxstream_mem_pinned(const void* host_mem, size_t nbytes)
{
  const char* const pointer = (const char*)host_mem;
  int result = 0;
  size_t i;
  for (i = 0; i < libxstream_opencl_config.npins && 0 == result; ++i) {
    const char* const lo = libxstream_opencl_config.pinptr[i];
    if (NULL != lo && lo <= pointer && (size_t)(pointer - lo) + nbytes <= libxstream_opencl_config.pinsize[i]) {
      result = 1;
    }
  }
  return result;
}
#endif


/**
 * Staging buffer of this thread, grown on demand up to the window size. Returns
 * NULL when staging is unavailable, which every caller treats as "transfer the
 * ordinary way" rather than as a failure.
 */
static void* libxstream_mem_stage(size_t* nbytes)
{
  void* result = NULL;
#if defined(LIBXSTREAM_MEM_STAGING)
  /* rounded up to an even size: the window is used as two halves */
  const size_t limit = libxstream_opencl_config.stage;
  const size_t want = LIBXS_UP2((*nbytes < limit ? *nbytes : limit), 2);
  if (libxstream_mem_stage_nbytes < want) {
    void* buffer = NULL;
    if (NULL != libxstream_mem_stage_ptr) {
      LIBXS_EXPECT(EXIT_SUCCESS == libxstream_mem_host_deallocate(libxstream_mem_stage_ptr, NULL));
      libxstream_mem_stage_ptr = NULL;
      libxstream_mem_stage_nbytes = 0;
    }
    if (EXIT_SUCCESS == libxstream_mem_host_allocate(&buffer, want, NULL) && NULL != buffer) {
      libxstream_mem_stage_ptr = buffer;
      libxstream_mem_stage_nbytes = want;
    }
  }
  result = libxstream_mem_stage_ptr;
  *nbytes = libxstream_mem_stage_nbytes;
#else
  LIBXS_UNUSED(nbytes);
#endif
  return result;
}


/**
 * Copy of the staging window. Parallel because a serial copy is slower than the
 * transport it replaces (see LIBXSTREAM_MEM_STAGE_NT); a caller already inside a
 * parallel region cannot open one, which is why libxstream_mem_stage_ready
 * refuses to stage there at all.
 */
static void libxstream_mem_stage_copy(void* dst, const void* src, size_t nbytes)
{
#if defined(_OPENMP)
  const int max_threads = omp_get_max_threads();
  const int want = libxstream_opencl_config.stage_nt;
  const int nthreads = (want < max_threads ? want : max_threads);
  /**
   * One block per iteration, handed out round-robin, so the grain alone decides
   * the distribution: never coarser than one share per thread, and finer than
   * that where the transfer is large enough to carry the extra iterations.
   */
  const size_t share = (nbytes + (size_t)nthreads - 1) / (size_t)nthreads;
  const size_t grain = (libxstream_opencl_config.stage_grain < share
    ? libxstream_opencl_config.stage_grain : share);
  const int nblocks = (int)(0 != grain ? ((nbytes + grain - 1) / grain) : 0);
  int i;
# pragma omp parallel for num_threads(nthreads) schedule(static, 1)
  for (i = 0; i < nblocks; ++i) {
    const size_t offset = grain * (size_t)i;
    if (offset < nbytes) {
      const size_t n = ((nbytes - offset) < grain ? (nbytes - offset) : grain);
      memcpy((char*)dst + offset, (const char*)src + offset, n);
    }
  }
#else
  memcpy(dst, src, nbytes);
#endif
}


/**
 * Whether this transfer should be staged: the mode asks for it, the range was
 * declared, the transfer is large enough to amortize a copy, and a parallel
 * copy is actually available here.
 */
static int libxstream_mem_stage_ready(const void* host_mem, size_t nbytes)
{
  int result = 0;
#if defined(LIBXSTREAM_MEM_STAGING)
  if (1 == libxstream_opencl_config.pin && (LIBXSTREAM_MEM_STAGE_MIN) <= nbytes
    && 0 == omp_in_parallel() && 0 != libxstream_opencl_config.npins)
  {
    result = libxstream_mem_pinned(host_mem, nbytes);
    if (0 != result) {
      LIBXS_ATOMIC_SIZE(LIBXS_ATOMIC_ADD_FETCH)(&libxstream_opencl_config.nstaged, 1, LIBXS_ATOMIC_RELAXED);
      LIBXS_ATOMIC_SIZE(LIBXS_ATOMIC_ADD_FETCH)(&libxstream_opencl_config.nstaged_bytes, nbytes, LIBXS_ATOMIC_RELAXED);
    }
  }
#else
  LIBXS_UNUSED(host_mem);
  LIBXS_UNUSED(nbytes);
#endif
  return result;
}


/* like libxstream_mem_copy_h2d, but apply some async workaround. */
LIBXSTREAM_API_INTERN int libxstream_opencl_mem_copy_h2d(const void* /*host_mem*/, void* /*dev_mem*/, size_t /*nbytes*/,
  cl_command_queue /*queue*/, int /*blocking*/, cl_event* /*event*/);
LIBXSTREAM_API_INTERN int libxstream_opencl_mem_copy_h2d(
  const void* host_mem, void* dev_mem, size_t nbytes, cl_command_queue queue, int blocking, cl_event* event)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
# if defined(LIBXSTREAM_ASYNC)
  const cl_bool finish = (0 != blocking || 0 == (1 & libxstream_opencl_config.async) || LIBXSTREAM_WA_UNIFIED(devinfo));
# else
  const cl_bool finish = CL_TRUE;
# endif
  int result = EXIT_SUCCESS;
  assert(NULL != host_mem && NULL != dev_mem);
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clEnqueueMemcpyINTEL) {
    result = devinfo->clEnqueueMemcpyINTEL(queue, finish, dev_mem, host_mem, nbytes, 0, NULL, event);
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm)
  {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
    /**
     * Enqueue the copy rather than mapping and copying on the host. Mapping is
     * what coarse-grain SVM requires for *host* access to the buffer, but a
     * transfer needs no host access at all: clEnqueueSVMMemcpy is a queued
     * command, so the runtime can use its copy engine and the call can be
     * asynchronous. The map/memcpy/unmap it replaces ran single-threaded inside
     * the enqueue at host store-to-device speed - 8.9 GB/s against 46.0 GB/s
     * for a 128 MB H2D on a GPU Max 1550, where the enqueued copy also
     * overlapped a concurrent kernel completely. libxstream_mem_copy_d2d
     * already took this route for the very same allocations.
     */
    result = clEnqueueSVMMemcpy(queue, finish, dev_mem, host_mem, nbytes, 0, NULL, event);
#   else
    memcpy(dev_mem, host_mem, nbytes);
#   endif
  }
  else
# endif
  {
    size_t offset = 0;
    libxstream_opencl_info_memptr_t* const info = libxstream_opencl_info_devptr_modify(
      NULL, dev_mem, 1 /*elsize*/, &nbytes, &offset);
    if (NULL != info) {
      result = clEnqueueWriteBuffer(queue, info->memory, finish, offset, nbytes, host_mem, 0, NULL, event);
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != result && !finish) { /* retry synchronously */
    int result_sync = EXIT_FAILURE;
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result_sync = devinfo->clEnqueueMemcpyINTEL(queue, CL_TRUE, dev_mem, host_mem, nbytes, 0, NULL, event);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm)
    {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
      /* blocking form of the enqueued copy (see above) */
      result_sync = clEnqueueSVMMemcpy(queue, CL_TRUE, dev_mem, host_mem, nbytes, 0, NULL, event);
#   else
      memcpy(dev_mem, host_mem, nbytes);
      result_sync = EXIT_SUCCESS;
#   endif
    }
    else
# endif
    {
      size_t offset = 0;
      libxstream_opencl_info_memptr_t* const info = libxstream_opencl_info_devptr_modify(
        NULL, dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (NULL != info) {
        result_sync = clEnqueueWriteBuffer(queue, info->memory, CL_TRUE, offset, nbytes, host_mem, 0, NULL, event);
      }
    }
    if (EXIT_SUCCESS == result_sync) {
      libxstream_opencl_config.async &= ~1; /* retract async feature */
      if (0 != libxstream_opencl_config.verbosity) {
        fprintf(stderr, "WARN ACC/OpenCL: falling back to synchronous upload (code=%i).\n", result);
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Upload through the staging window, copy of one half overlapping the transfer
 * of the other. The two must not be conflated: a half may only be refilled once
 * the transfer reading it has completed, which is what the per-half event is
 * for. Both halves are drained before returning, because the window belongs to
 * the thread and the next call would otherwise overwrite memory still in
 * flight - silent corruption rather than a visible stall.
 */
static int libxstream_mem_stage_h2d(const void* host_mem, void* dev_mem, size_t nbytes,
  void* stage, size_t window, cl_command_queue queue, cl_event* event)
{
  const size_t half = window / 2;
  cl_event pending[2];
  size_t done = 0;
  int slot = 0, i;
  int result = (NULL != stage && 0 != half) ? EXIT_SUCCESS : EXIT_FAILURE;
  pending[0] = NULL;
  pending[1] = NULL;
  while (done < nbytes && EXIT_SUCCESS == result) {
    const size_t n = ((nbytes - done) < half ? (nbytes - done) : half);
    const int last = (nbytes <= (done + n));
    char* const buffer = (char*)stage + (size_t)slot * half;
    if (NULL != pending[slot]) { /* this half is still being read by the device */
      result = clWaitForEvents(1, pending + slot);
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(pending[slot]));
      pending[slot] = NULL;
    }
    if (EXIT_SUCCESS == result) {
      libxstream_mem_stage_copy(buffer, (const char*)host_mem + done, n);
      result = libxstream_opencl_mem_copy_h2d(buffer, (char*)dev_mem + done, n, queue,
        0 /*asynchronous*/, (0 != last && NULL != event) ? event : (pending + slot));
    }
    done += n;
    slot ^= 1;
  }
  for (i = 0; i < 2; ++i) {
    if (NULL != pending[i]) {
      if (EXIT_SUCCESS == result) result = clWaitForEvents(1, pending + i);
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(pending[i]));
      pending[i] = NULL;
    }
  }
  /* the caller owns the event it asked for, so it is waited on but not released */
  if (EXIT_SUCCESS == result && NULL != event && NULL != *event) {
    result = clWaitForEvents(1, event);
  }
  if (EXIT_SUCCESS != result) { /* staging is best-effort: fall back to the direct transfer */
    result = libxstream_opencl_mem_copy_h2d(host_mem, dev_mem, nbytes, queue, 1 /*blocking*/, event);
  }
  return result;
}


LIBXSTREAM_API int libxstream_mem_copy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, libxstream_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != host_mem && NULL != dev_mem) || 0 == nbytes);
  assert(NULL != libxstream_opencl_config.device.context);
  if (
# if (0 != LIBXSTREAM_USM)
    host_mem != dev_mem && /* fast-path only sensible without offsets */
# endif
    NULL != host_mem && NULL != dev_mem && 0 != nbytes)
  {
    const cl_bool finish = (NULL != stream ? CL_FALSE : CL_TRUE);
    const libxstream_opencl_stream_t* str;
    cl_event event = NULL;
    /**
     * The staging window is acquired before the lock and before any command of
     * this transfer is enqueued. Allocating it later would map a buffer on the
     * default queue from inside a locked region with work already in flight,
     * which is a wait on the very pipeline the caller is still filling.
     */
    size_t window = nbytes;
    void* const stage = (0 != libxstream_mem_stage_ready(host_mem, nbytes)
      ? libxstream_mem_stage(&window) : NULL);
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str);
    if (NULL == stage) {
      result = libxstream_opencl_mem_copy_h2d(
        host_mem, dev_mem, nbytes, str->queue, finish, NULL == libxstream_opencl_config.hist_h2d ? NULL : &event);
    }
    else {
      result = libxstream_mem_stage_h2d(host_mem, dev_mem, nbytes, stage, window,
        str->queue, NULL == libxstream_opencl_config.hist_h2d ? NULL : &event);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) { /* libxstream_mem_copy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        void* const data = LIBXSTREAM_EVENT_DATA(nbytes, libxstream_event_kind_h2d);
        assert(NULL != libxstream_opencl_config.hist_h2d);
        if (!finish) { /* asynchronous */
          result = clSetEventCallback(event, CL_COMPLETE, libxstream_mem_copy_notify, data);
        }
        else libxstream_mem_copy_notify(event, CL_COMPLETE, data); /* synchronous */
      }
      else LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
  CL_RETURN(result, "");
}


/* like libxstream_mem_copy_d2h, but apply some async workaround. */
LIBXSTREAM_API_INTERN int libxstream_opencl_mem_copy_d2h(const void* /*dev_mem*/, void* /*host_mem*/, size_t /*offset*/,
  size_t /*nbytes*/, cl_command_queue /*queue*/, int /*blocking*/, cl_event* /*event*/);
LIBXSTREAM_API_INTERN int libxstream_opencl_mem_copy_d2h(
  const void* dev_mem, void* host_mem, size_t offset, size_t nbytes, cl_command_queue queue, int blocking, cl_event* event)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
# if defined(LIBXSTREAM_ASYNC)
  const cl_bool finish = (0 != blocking || 0 == (2 & libxstream_opencl_config.async) || LIBXSTREAM_WA_UNIFIED(devinfo));
# else
  const cl_bool finish = CL_TRUE;
# endif
  int result = EXIT_SUCCESS;
  assert(NULL != dev_mem);
# if (1 >= LIBXSTREAM_USM)
  if (NULL != devinfo->clEnqueueMemcpyINTEL) {
    result = devinfo->clEnqueueMemcpyINTEL(queue, finish, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm)
  {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
    /* enqueued copy instead of map/memcpy/unmap: see libxstream_mem_copy_h2d */
    result = clEnqueueSVMMemcpy(queue, finish, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
#   else
    memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
#   endif
  }
  else
# endif
  {
    result = clEnqueueReadBuffer(queue, (cl_mem)(uintptr_t)dev_mem, finish, offset, nbytes, host_mem, 0, NULL, event);
  }
  if (EXIT_SUCCESS != result && !finish) { /* retry synchronously */
    int result_sync = EXIT_FAILURE;
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result_sync = devinfo->clEnqueueMemcpyINTEL(queue, CL_TRUE, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm)
    {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
      /* blocking form of the enqueued copy (see libxstream_mem_copy_h2d) */
      result_sync = clEnqueueSVMMemcpy(queue, CL_TRUE, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
#   else
      memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
#   endif
    }
    else
# endif
    {
      result_sync = clEnqueueReadBuffer(queue, (cl_mem)(uintptr_t)dev_mem, CL_TRUE, offset, nbytes, host_mem, 0, NULL, event);
    }
    if (EXIT_SUCCESS == result_sync) {
      libxstream_opencl_config.async &= ~2; /* retract async feature */
      if (0 != libxstream_opencl_config.verbosity) {
        fprintf(stderr, "WARN ACC/OpenCL: falling back to synchronous readback (code=%i).\n", result);
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Download through the staging window: the read of one half overlaps the copy
 * of the other out to the caller. The order is the mirror of the upload - a
 * half is copied out only after its read completed, and the read of the next
 * half is already enqueued by then.
 */
static int libxstream_mem_stage_d2h(const void* dev_mem, void* host_mem, size_t offset, size_t nbytes,
  void* stage, size_t window, cl_command_queue queue, cl_event* event)
{
  const size_t half = window / 2;
  cl_event pending[2];
  size_t done = 0, prev_off = 0, prev_n = 0;
  int slot = 0, prev = -1, i;
  int result = (NULL != stage && 0 != half) ? EXIT_SUCCESS : EXIT_FAILURE;
  pending[0] = NULL;
  pending[1] = NULL;
  while (done < nbytes && EXIT_SUCCESS == result) {
    const size_t n = ((nbytes - done) < half ? (nbytes - done) : half);
    const int last = (nbytes <= (done + n));
    result = libxstream_opencl_mem_copy_d2h(dev_mem, (char*)stage + (size_t)slot * half, offset + done, n,
      queue, 0 /*asynchronous*/, (0 != last && NULL != event) ? event : (pending + slot));
    if (EXIT_SUCCESS == result && 0 <= prev) { /* drain the half filled before this one */
      if (NULL != pending[prev]) {
        result = clWaitForEvents(1, pending + prev);
        LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(pending[prev]));
        pending[prev] = NULL;
      }
      if (EXIT_SUCCESS == result) {
        libxstream_mem_stage_copy((char*)host_mem + prev_off, (const char*)stage + (size_t)prev * half, prev_n);
      }
    }
    prev = slot;
    prev_off = done;
    prev_n = n;
    done += n;
    slot ^= 1;
  }
  if (EXIT_SUCCESS == result && 0 <= prev) { /* the final half, whose event may be the caller's */
    if (NULL != pending[prev]) result = clWaitForEvents(1, pending + prev);
    else if (NULL != event && NULL != *event) result = clWaitForEvents(1, event);
    if (EXIT_SUCCESS == result) {
      libxstream_mem_stage_copy((char*)host_mem + prev_off, (const char*)stage + (size_t)prev * half, prev_n);
    }
  }
  for (i = 0; i < 2; ++i) {
    if (NULL != pending[i]) {
      if (EXIT_SUCCESS == result) result = clWaitForEvents(1, pending + i);
      LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(pending[i]));
      pending[i] = NULL;
    }
  }
  if (EXIT_SUCCESS != result) { /* staging is best-effort: fall back to the direct transfer */
    result = libxstream_opencl_mem_copy_d2h(dev_mem, host_mem, offset, nbytes, queue, 1 /*blocking*/, event);
  }
  return result;
}


LIBXSTREAM_API int libxstream_mem_copy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, libxstream_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != dev_mem && NULL != host_mem) || 0 == nbytes);
  if (
# if (0 != LIBXSTREAM_USM)
    host_mem != dev_mem && /* fast-path only sensible without offsets */
# endif
    NULL != host_mem && NULL != dev_mem && 0 != nbytes)
  {
    const cl_bool finish = (NULL != stream ? CL_FALSE : CL_TRUE);
    libxstream_opencl_info_memptr_t* info = NULL;
    cl_event event = NULL;
    size_t offset = 0, window = nbytes;
    void* nconst;
    const libxstream_opencl_stream_t* str;
    /* acquired before the lock, for the reason given in libxstream_mem_copy_h2d */
    void* const stage = (0 != libxstream_mem_stage_ready(host_mem, nbytes)
      ? libxstream_mem_stage(&window) : NULL);
    LIBXS_UNION_ASSIGN(void*, nconst, const void*, dev_mem);
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str);
    info = libxstream_opencl_info_devptr_modify(NULL, nconst, 1 /*elsize*/, &nbytes, &offset);
    { const void* const source = (NULL == info ? dev_mem : (const void*)info->memory);
      cl_event* const hist = (NULL == libxstream_opencl_config.hist_d2h ? NULL : &event);
      /* info_devptr_modify returns NULL for a USM-pointer, which is then its own source. */
      if (NULL == stage) {
        result = libxstream_opencl_mem_copy_d2h(source, host_mem, offset, nbytes, str->queue, finish, hist);
      }
      else {
        result = libxstream_mem_stage_d2h(source, host_mem, offset, nbytes, stage, window, str->queue, hist);
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) { /* libxstream_mem_copy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        void* const data = LIBXSTREAM_EVENT_DATA(nbytes, libxstream_event_kind_d2h);
        assert(NULL != libxstream_opencl_config.hist_d2h);
        if (!finish) { /* asynchronous */
          result = clSetEventCallback(event, CL_COMPLETE, libxstream_mem_copy_notify, data);
        }
        else libxstream_mem_copy_notify(event, CL_COMPLETE, data); /* synchronous */
      }
      else LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_copy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, libxstream_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != devmem_src && NULL != devmem_dst) || 0 == nbytes);
  if (NULL != devmem_src && NULL != devmem_dst && devmem_src != devmem_dst && 0 != nbytes) {
# if defined(LIBXSTREAM_ASYNC)
    cl_event event = NULL, *const pevent = (0 == (4 & libxstream_opencl_config.async) || NULL == stream) ? &event : NULL;
# else
    cl_event event = NULL, *const pevent = NULL;
# endif
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    void* nconst;
    const libxstream_opencl_stream_t* str;
    LIBXS_UNION_ASSIGN(void*, nconst, const void*, devmem_src);
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str && NULL != devinfo->context);
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result = devinfo->clEnqueueMemcpyINTEL(str->queue, CL_FALSE /*blocking*/, devmem_dst, devmem_src, nbytes, 0, NULL,
        NULL == libxstream_opencl_config.hist_d2d ? pevent : &event);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm)
    {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
      result = clEnqueueSVMMemcpy(str->queue, CL_FALSE /*blocking*/, devmem_dst, devmem_src, nbytes, 0, NULL,
        NULL == libxstream_opencl_config.hist_d2d ? pevent : &event);
#   else
      memcpy(devmem_dst, devmem_src, nbytes);
#   endif
    }
    else
# endif
    {
      size_t offset_src = 0, offset_dst = 0;
      libxstream_opencl_info_memptr_t* const info_src = libxstream_opencl_info_devptr_modify(
        NULL, nconst, 1 /*elsize*/, &nbytes, &offset_src);
      libxstream_opencl_info_memptr_t* const info_dst = libxstream_opencl_info_devptr_modify(
        NULL, devmem_dst, 1 /*elsize*/, &nbytes, &offset_dst);
      if (NULL != info_src && NULL != info_dst) {
        result = clEnqueueCopyBuffer(str->queue, info_src->memory, info_dst->memory, offset_src, offset_dst, nbytes, 0, NULL,
          NULL == libxstream_opencl_config.hist_d2d ? pevent : &event);
      }
      else result = EXIT_FAILURE;
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) { /* libxstream_mem_copy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        void* const data = LIBXSTREAM_EVENT_DATA(nbytes, libxstream_event_kind_d2d);
        if (NULL == pevent) { /* asynchronous */
          assert(NULL != libxstream_opencl_config.hist_d2d);
          result = clSetEventCallback(event, CL_COMPLETE, libxstream_mem_copy_notify, data);
        }
        else { /* synchronous */
          result = clWaitForEvents(1, &event);
          if (EXIT_SUCCESS == result) {
            if (NULL != libxstream_opencl_config.hist_d2d) {
              libxstream_mem_copy_notify(event, CL_COMPLETE, data);
            }
            else result = clReleaseEvent(event);
          }
          else LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
        }
      }
      else LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, libxstream_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  assert(NULL != dev_mem || 0 == nbytes);
  if (0 != nbytes) {
# if defined(LIBXSTREAM_ASYNC)
    const cl_bool wait = (0 == (8 & libxstream_opencl_config.async) || NULL == stream);
# else
    const cl_bool wait = CL_TRUE;
# endif
    /**
     * An event is needed either to wait on the fill, or to time it for the ZERO
     * histogram. Timing must not force a wait: that would serialize the fill
     * against the enqueuing thread and change what is being measured.
     */
    const int measure = (0 == value && NULL != libxstream_opencl_config.hist_zero);
    cl_event event = NULL, *const pevent = (0 != wait || 0 != measure) ? &event : NULL;
    const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
    const libxstream_opencl_stream_t* str;
    size_t base = 0, vsize = 1;
    if (0 == LIBXS_MOD2(nbytes, 4)) vsize = 4;
    else if (0 == LIBXS_MOD2(nbytes, 2)) vsize = 2;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str && NULL != devinfo->context);
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clEnqueueMemFillINTEL) {
      result = devinfo->clEnqueueMemFillINTEL(str->queue, (char*)dev_mem + offset, &value, vsize, nbytes, 0, NULL, pevent);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm)
    {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
      result = clEnqueueSVMMemFill(str->queue, (char*)dev_mem + offset, &value, vsize, nbytes, 0, NULL, pevent);
#   else
      memset((char*)dev_mem + offset, value, nbytes);
#   endif
    }
    else
# endif
    {
      const libxstream_opencl_info_memptr_t* const info = libxstream_opencl_info_devptr_modify(
        NULL, dev_mem, 1 /*elsize*/, &nbytes, &base);
      if (NULL != info) {
        result = clEnqueueFillBuffer(str->queue, info->memory, &value, vsize, base + offset, nbytes, 0, NULL, pevent);
        dev_mem = info->memptr;
      }
      else result = EXIT_FAILURE;
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) {
      if (0 != wait) {
        int result_release;
        CL_CHECK(result, clWaitForEvents(1, &event));
        if (0 != measure && EXIT_SUCCESS == result) {
          /* Completed already: account for it directly (callback would never fire later). */
          libxstream_mem_copy_notify(event, CL_COMPLETE, LIBXSTREAM_EVENT_DATA(nbytes, libxstream_event_kind_zero));
          event = NULL; /* notify released the event */
        }
        if (NULL != event) {
          result_release = clReleaseEvent(event);
          if (EXIT_SUCCESS == result) result = result_release;
        }
      }
      else { /* asynchronous: notify releases the event when the fill completes */
        assert(0 != measure);
        if (EXIT_SUCCESS == result) {
          result = clSetEventCallback(event, CL_COMPLETE, libxstream_mem_copy_notify,
            LIBXSTREAM_EVENT_DATA(nbytes, libxstream_event_kind_zero));
        }
        else LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
      }
    }
  }
  CL_RETURN(result, "");
}


LIBXSTREAM_API int libxstream_mem_zero(void* dev_mem, size_t offset, size_t nbytes, libxstream_stream_t* stream)
{
  return libxstream_opencl_memset(dev_mem, 0 /*value*/, offset, nbytes, stream);
}


LIBXSTREAM_API int libxstream_opencl_info_devmem(
  cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified)
{
  int result = EXIT_SUCCESS, unified = 0;
  size_t size_free = 0, size_total = 0, size_local = 0;
  cl_device_local_mem_type cl_local_type = CL_GLOBAL;
  cl_ulong cl_size_total = 0, cl_size_local = 0;
  cl_bool cl_unified = CL_FALSE;
# if !defined(_WIN32)
#   if defined(_SC_PAGE_SIZE)
  const long page_size = sysconf(_SC_PAGE_SIZE);
#   else
  const long page_size = 4096;
#   endif
  long pages_free = 0, pages_total = 0;
#   if defined(__linux__)
#     if defined(_SC_PHYS_PAGES)
  pages_total = sysconf(_SC_PHYS_PAGES);
#     else
  pages_total = 0;
#     endif
#     if defined(_SC_AVPHYS_PAGES)
  pages_free = sysconf(_SC_AVPHYS_PAGES);
#     else
  pages_free = pages_total;
#     endif
#   elif defined(__APPLE__) && defined(__MACH__)
  /*const*/ size_t size_pages_free = sizeof(const long), size_pages_total = sizeof(const long);
  LIBXS_EXPECT(0 == sysctlbyname("hw.memsize", &pages_total, &size_pages_total, NULL, 0));
  if (0 < page_size) pages_total /= page_size;
  if (0 != sysctlbyname("vm.page_free_count", &pages_free, &size_pages_free, NULL, 0)) {
    pages_free = pages_total;
  }
#   endif
  if (0 < page_size && 0 <= pages_free && 0 <= pages_total) {
    const size_t size_page = (size_t)page_size;
    size_total = size_page * (size_t)pages_total;
    size_free = size_page * (size_t)pages_free;
  }
# else
  MEMORYSTATUSEX mem_status;
  mem_status.dwLength = sizeof(mem_status);
  if (GlobalMemoryStatusEx(&mem_status)) {
    size_total = (size_t)mem_status.ullTotalPhys;
    size_free = (size_t)mem_status.ullAvailPhys;
  }
# endif
  CL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_total, NULL));
  CL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &cl_local_type, NULL));
  if (CL_LOCAL == cl_local_type) {
    CL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_local, NULL));
  }
  CL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &cl_unified, NULL));
  if (EXIT_SUCCESS == result) {
    if (cl_size_total < size_total) size_total = cl_size_total;
    if (size_total < size_free) size_free = size_total;
    size_local = cl_size_local;
    unified = cl_unified;
    assert(size_free <= size_total);
  }
  assert(NULL != mem_local || NULL != mem_total || NULL != mem_free || NULL != mem_unified);
  if (NULL != mem_unified) *mem_unified = unified;
  if (NULL != mem_local) *mem_local = size_local;
  if (NULL != mem_total) *mem_total = size_total;
  if (NULL != mem_free) *mem_free = size_free;
  return result;
}


LIBXSTREAM_API int libxstream_mem_info(size_t* mem_free, size_t* mem_total)
{
  const cl_device_id device_id = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
  int result;
  result = libxstream_opencl_info_devmem(device_id, mem_free, mem_total, NULL /*mem_local*/, NULL /*mem_unified*/);
  CL_RETURN(result, "");
}

#endif /*__OPENCL*/
