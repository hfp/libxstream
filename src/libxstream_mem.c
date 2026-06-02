/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
# include <libxstream_opencl.h>
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
  return (EXIT_SUCCESS == status) ? result : NULL;
}


LIBXSTREAM_API_INTERN void libxstream_mem_hst_xfree(void* pointer, const void* extra)
{
  LIBXS_UNUSED(extra);
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
    result = devinfo->clDeviceMemAllocINTEL(devinfo->context, did, NULL, size, 0, &status);
  }
  else if (NULL != devinfo->clSharedMemAllocINTEL) {
    const cl_device_id did = libxstream_opencl_config.devices[libxstream_opencl_config.device_id];
    result = devinfo->clSharedMemAllocINTEL(devinfo->context, did, NULL, size, 0, &status);
  }
  else
# endif
# if (0 != LIBXSTREAM_USM)
    if (0 != devinfo->usm) {
    const int svmflags = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)
                            ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0);
    result = clSVMAlloc(devinfo->context, (cl_svm_mem_flags)(CL_MEM_READ_WRITE | svmflags), size, 0);
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
  if (NULL != extra) {
    const uintptr_t addr = (uintptr_t)extra;
    libxstream_stream_sync((libxstream_stream_t*)addr);
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
        fprintf(stderr, "ERROR ACC/OpenCL: pointer=%p base=%p size=%llu offset=%llu info failed\n",
          memory, NULL != miss ? miss->memptr : NULL,
          (unsigned long long)(NULL != amount ? (*amount * elsize) : 0),
          (unsigned long long)(NULL != miss ? (pointer - (uintptr_t)miss->memptr) : 0));
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
  LIBXS_ASSIGN(&non_const, &memory);
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


LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_mem_copy_notify(cl_event /*event*/, cl_int /*event_status*/, void* /*data*/);
LIBXSTREAM_API_INTERN void CL_CALLBACK libxstream_mem_copy_notify(cl_event event, cl_int event_status, void* data)
{
  cl_command_type type = CL_COMMAND_SVM_MEMCPY;
  int result = EXIT_SUCCESS;
  double vals[2];
  vals[1] = libxstream_opencl_duration(event, &result) * 1E6; /* Microseconds */
  LIBXS_UNUSED(event_status);
  assert(CL_COMPLETE == event_status && NULL != data && 8 == sizeof(data));
  if (EXIT_SUCCESS == result && EXIT_SUCCESS == clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL)) {
    const size_t size = 0x3FFFFFFFFFFFFFFF & (size_t)data;
    const int kind = LIBXS_CAST_INT(((size_t)data) >> 62);
    vals[0] = 1E-6 * size; /* Megabyte */
    if (CL_COMMAND_WRITE_BUFFER != type && CL_COMMAND_READ_BUFFER != type && CL_COMMAND_COPY_BUFFER != type) {
      switch (kind) {
        case libxstream_event_kind_h2d: type = CL_COMMAND_WRITE_BUFFER; break;
        case libxstream_event_kind_d2h: type = CL_COMMAND_READ_BUFFER; break;
        case libxstream_event_kind_d2d: type = CL_COMMAND_COPY_BUFFER; break;
        default: assert(libxstream_event_kind_none == kind); /* should not happen */
      }
    }
    switch (type) {
      case CL_COMMAND_WRITE_BUFFER: {
        assert(NULL != libxstream_opencl_config.hist_h2d && libxstream_event_kind_h2d == kind);
        libxs_hist_push(libxstream_opencl_config.lock_memory, libxstream_opencl_config.hist_h2d, vals);
        if (0 > libxstream_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: H2D mb=%.1f us=%.0f\n", vals[0], vals[1]);
      } break;
      case CL_COMMAND_READ_BUFFER: {
        assert(NULL != libxstream_opencl_config.hist_d2h && libxstream_event_kind_d2h == kind);
        libxs_hist_push(libxstream_opencl_config.lock_memory, libxstream_opencl_config.hist_d2h, vals);
        if (0 > libxstream_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: D2H mb=%.1f us=%.0f\n", vals[0], vals[1]);
      } break;
      case CL_COMMAND_COPY_BUFFER: {
        assert(NULL != libxstream_opencl_config.hist_d2d && libxstream_event_kind_d2d == kind);
        libxs_hist_push(libxstream_opencl_config.lock_memory, libxstream_opencl_config.hist_d2d, vals);
        if (0 > libxstream_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: D2D mb=%.1f us=%.0f\n", vals[0], vals[1]);
      } break;
    }
  }
  if (NULL != event) LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(event));
}


LIBXSTREAM_API int libxstream_mem_allocate(void** dev_mem, size_t nbytes)
{
  /* assume no lock is needed to protect against context/device changes */
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
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
        libxstream_opencl_info_memptr_t* info = NULL;
        const size_t size = 1;
        static const char source[] = "kernel void memptr(global unsigned long* ptr) {\n"
                   "  const union { global unsigned long* p; unsigned long u; } cast = { ptr };\n"
                   "  const size_t i = get_global_id(0);\n"
                   "  ptr[i] = cast.u + i;\n"
                   "}\n";
        LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
        assert(NULL != devinfo->stream.queue && NULL != memory);
        assert(sizeof(size_t) == sizeof(cl_ulong));
        if (devinfo->context != devinfo->memptr_context) {
          if (NULL != devinfo->memptr_kernel) {
            LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseKernel(devinfo->memptr_kernel));
            devinfo->memptr_kernel = NULL;
          }
          devinfo->memptr_context = NULL;
        }
        if (NULL == devinfo->memptr_kernel) {
          result = libxstream_opencl_kernel(0 /*source_kind*/, source, "memptr" /*kernel_name*/, NULL /*build_params*/,
            NULL /*build_options*/, NULL /*try_build_options*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/,
            &devinfo->memptr_kernel);
          if (EXIT_SUCCESS == result) devinfo->memptr_context = devinfo->context;
        }
        CL_CHECK(result, clSetKernelArg(devinfo->memptr_kernel, 0, sizeof(cl_mem), &memory));
        if (EXIT_SUCCESS == result) {
          result = clEnqueueNDRangeKernel(
            devinfo->stream.queue, devinfo->memptr_kernel, 1 /*work_dim*/, NULL /*offset*/, &size, NULL /*local_work_size*/, 0, NULL, NULL);
        }
        if (EXIT_SUCCESS == result) {
          result = clEnqueueReadBuffer(
            devinfo->stream.queue, memory, CL_TRUE /*blocking*/, 0 /*offset*/, sizeof(void*), &memptr, 0, NULL, NULL /*event*/);
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


LIBXSTREAM_API int libxstream_mem_copy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, libxstream_stream_t* stream)
{
  const libxstream_opencl_device_t* const devinfo = &libxstream_opencl_config.device;
  int result = EXIT_SUCCESS;
  assert((NULL != host_mem && NULL != dev_mem) || 0 == nbytes);
  assert(NULL != devinfo->context);
  if (
# if (0 != LIBXSTREAM_USM)
    host_mem != dev_mem && /* fast-path only sensible without offsets */
# endif
    NULL != host_mem && NULL != dev_mem && 0 != nbytes)
  {
# if defined(LIBXSTREAM_ASYNC)
    const cl_bool finish = (0 == (1 & libxstream_opencl_config.async) || NULL == stream ||
                            (0 != (8 & libxstream_opencl_config.wa) && 0 != devinfo->intel && 0 != devinfo->unified));
# else
    const cl_bool finish = CL_TRUE;
# endif
    const libxstream_opencl_stream_t* str;
    cl_event event = NULL;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str);
# if (1 >= LIBXSTREAM_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result = devinfo->clEnqueueMemcpyINTEL(
        str->queue, finish, dev_mem, host_mem, nbytes, 0, NULL, NULL == libxstream_opencl_config.hist_h2d ? NULL : &event);
    }
    else
# endif
# if (0 != LIBXSTREAM_USM)
      if (0 != devinfo->usm)
    {
#   if (1 >= LIBXSTREAM_USM) || defined(LIBXSTREAM_MEM_SVM_USM)
      const int svmfine = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm));
      if (0 == svmfine) {
        cl_event unmap_event = NULL;
        result = clEnqueueSVMMap(str->queue, CL_TRUE, CL_MAP_WRITE, dev_mem, nbytes, 0, NULL, NULL);
        if (EXIT_SUCCESS == result) {
          memcpy(dev_mem, host_mem, nbytes);
          result = clEnqueueSVMUnmap(str->queue, dev_mem, 0, NULL, &unmap_event);
          if (EXIT_SUCCESS == result && finish) {
            result = clWaitForEvents(1, &unmap_event);
          }
        }
        if (NULL != unmap_event) {
          if (NULL != libxstream_opencl_config.hist_h2d) {
            event = unmap_event;
          }
          else {
            LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(unmap_event));
          }
        }
      }
      else {
        memcpy(dev_mem, host_mem, nbytes);
      }
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
        result = clEnqueueWriteBuffer(str->queue, info->memory, finish, offset, nbytes, host_mem, 0, NULL,
          NULL == libxstream_opencl_config.hist_h2d ? NULL : &event);
      }
      else result = EXIT_FAILURE;
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) { /* libxstream_mem_copy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        void* const data = (void*)(nbytes | ((size_t)libxstream_event_kind_h2d) << 62);
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
  const cl_bool finish = (0 != blocking || 0 == (2 & libxstream_opencl_config.async) ||
                          (0 != (8 & libxstream_opencl_config.wa) && 0 != devinfo->intel && 0 != devinfo->unified));
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
    const int svmfine = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm));
    union { const void* cv; void* v; } src;
    src.cv = dev_mem;
    if (0 == svmfine) {
      cl_event unmap_event = NULL;
      result = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, (char*)src.v + offset, nbytes, 0, NULL, NULL);
      if (EXIT_SUCCESS == result) {
        memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
        result = clEnqueueSVMUnmap(queue, (char*)src.v + offset, 0, NULL, &unmap_event);
        if (EXIT_SUCCESS == result && finish) {
          result = clWaitForEvents(1, &unmap_event);
        }
      }
      if (NULL != unmap_event) {
        if (NULL != event) {
          *event = unmap_event;
        }
        else {
          LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(unmap_event));
        }
      }
    }
    else {
      if (finish) result = clFinish(queue);
      if (EXIT_SUCCESS == result) memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
    }
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
      result_sync = EXIT_SUCCESS;
      if (0 == ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)) {
        cl_event unmap_event = NULL;
        union { const void* cv; void* v; } src2;
        src2.cv = dev_mem;
        result_sync = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, (char*)src2.v + offset, nbytes, 0, NULL, NULL);
        if (EXIT_SUCCESS == result_sync) {
          memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
          result_sync = clEnqueueSVMUnmap(queue, (char*)src2.v + offset, 0, NULL, &unmap_event);
          if (EXIT_SUCCESS == result_sync) {
            result_sync = clWaitForEvents(1, &unmap_event);
          }
        }
        if (NULL != unmap_event) {
          if (NULL != event) {
            *event = unmap_event;
          }
          else {
            LIBXS_EXPECT_DEBUG(EXIT_SUCCESS == clReleaseEvent(unmap_event));
          }
        }
      }
      else {
        result_sync = clFinish(queue);
        if (EXIT_SUCCESS == result_sync) {
          memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
        }
      }
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
    size_t offset = 0;
    union {
      const void* input;
      void* ptr;
    } nconst;
    const libxstream_opencl_stream_t* str;
    nconst.input = dev_mem;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    str = (NULL != stream ? stream : libxstream_opencl_stream(NULL, libxs_tid()));
    assert(NULL != str);
    info = libxstream_opencl_info_devptr_modify(NULL, nconst.ptr, 1 /*elsize*/, &nbytes, &offset);
    if (NULL == info) { /* USM-pointer: info_devptr_modify returns NULL when USM is active */
      result = libxstream_opencl_mem_copy_d2h(
        dev_mem, host_mem, offset, nbytes, str->queue, finish, NULL == libxstream_opencl_config.hist_d2h ? NULL : &event);
    }
    else {
      result = libxstream_opencl_mem_copy_d2h(
        info->memory, host_mem, offset, nbytes, str->queue, finish, NULL == libxstream_opencl_config.hist_d2h ? NULL : &event);
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, libxstream_opencl_config.lock_memory);
    if (NULL != event) { /* libxstream_mem_copy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        void* const data = (void*)(nbytes | ((size_t)libxstream_event_kind_d2h) << 62);
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
    union {
      const void* input;
      void* ptr;
    } nconst;
    const libxstream_opencl_stream_t* str;
    nconst.input = devmem_src;
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
        NULL, nconst.ptr, 1 /*elsize*/, &nbytes, &offset_src);
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
        void* const data = (void*)(nbytes | ((size_t)libxstream_event_kind_d2d) << 62);
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
    cl_event event = NULL, *const pevent = (0 == (8 & libxstream_opencl_config.async) || NULL == stream) ? &event : NULL;
# else
    cl_event event = NULL, *const pevent = NULL;
# endif
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
      int result_release;
      CL_CHECK(result, clWaitForEvents(1, &event));
      result_release = clReleaseEvent(event);
      if (EXIT_SUCCESS == result) result = result_release;
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
