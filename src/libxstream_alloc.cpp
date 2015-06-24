/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#if defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
#include "libxstream_alloc.hpp"
#include "libxstream_workitem.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <malloc.h>
#include <cstring>
#include <libxstream_end.h>

#if defined(__MKL)
# include <mkl.h>
#endif

#if defined(_WIN32)
# include <windows.h>
#else
# include <xmmintrin.h>
# include <sys/mman.h>
#endif

#define LIBXSTREAM_ALLOC_VALLOC
#if !defined(LIBXSTREAM_OFFLOAD) || (defined(__INTEL_COMPILER) && (1600 <= __INTEL_COMPILER))
# define LIBXSTREAM_ALLOC_MMAP
#endif


namespace libxstream_alloc_internal {

LIBXSTREAM_TARGET(mic) unsigned int       abs(unsigned int        a)  { return a; }
LIBXSTREAM_TARGET(mic) unsigned long      abs(unsigned long       a)  { return a; }
LIBXSTREAM_TARGET(mic) unsigned long long abs(unsigned long long  a)  { return a; }

template<typename S, typename T>
LIBXSTREAM_TARGET(mic) S linear_size(size_t dims, const T shape[], S initial_size)
{
  LIBXSTREAM_ASSERT(shape);
  S result = 0 < dims ? (std::max<S>(initial_size, 1) * static_cast<S>(shape[0])) : initial_size;
  LIBXSTREAM_PRAGMA_LOOP_COUNT(0, LIBXSTREAM_MAX_NDIMS, 2)
  for (size_t i = 1; i < dims; ++i) result *= static_cast<S>(shape[i]);
  return result;
}

static/*IPO*/ struct config_type {
  config_type() {
#if defined(__GNUC__) && !defined(__CYGWIN__) && (!defined(LIBXSTREAM_CHECK) || (2 > ((2*LIBXSTREAM_CHECK+1)/2)))
    mallopt(M_CHECK_ACTION, 0); // disable MALLOC_CHECK_
#endif
  }
  static config_type& instance() {
    static config_type config;
    return config;
  }
} &config = config_type::instance();

struct info_type {
  static void alloc_size(size_t buffer_size, size_t alignment, size_t extra_size, size_t& alloc_size, size_t& unlock_size) {
    const size_t auto_alignment = libxstream_alignment(buffer_size, alignment);
    const size_t aligned_size = libxstream_align(buffer_size, auto_alignment);
    LIBXSTREAM_ASSERT(buffer_size <= aligned_size);
    unlock_size = extra_size + sizeof(info_type);
    alloc_size = std::max(aligned_size, buffer_size + unlock_size);
  }

  void* pointer;
  size_t size;
  bool real;
};

} // namespace libxstream_alloc_internal


LIBXSTREAM_TARGET(mic) void libxstream_alloc_init()
{
#if !defined(__MIC__) // not needed
  libxstream_alloc_internal::config_type::instance();
  libxstream_sink(&libxstream_alloc_internal::config);
#endif
}


LIBXSTREAM_TARGET(mic) size_t libxstream_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXSTREAM_TARGET(mic) size_t libxstream_lcm(size_t a, size_t b)
{
  using libxstream_alloc_internal::abs;
  using std::abs;
  return abs(a * b) / libxstream_gcd(a, b);
}


LIBXSTREAM_TARGET(mic) size_t libxstream_alignment(size_t size, size_t alignment)
{
#if defined(LIBXSTREAM_OFFLOAD)
  static const size_t max_algn = ((LIBXSTREAM_MAX_ALIGN) / (LIBXSTREAM_MAX_SIMD)) * (LIBXSTREAM_MAX_SIMD);
  static const size_t max_simd = std::min(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
#else
  static const size_t max_algn = LIBXSTREAM_MAX_SIMD, max_simd = LIBXSTREAM_MAX_SIMD;
#endif
  const size_t a = 0 == alignment ? max_algn : ((LIBXSTREAM_MAX_ALIGN / alignment) * alignment);
  const size_t b = 0 == alignment ? max_simd : std::min(alignment, static_cast<size_t>(LIBXSTREAM_MAX_ALIGN));
  const size_t c = std::max(sizeof(void*), alignment);
  return a <= size ? a : (b < size ? b : c);
}


LIBXSTREAM_TARGET(mic) size_t libxstream_align(size_t size, size_t alignment)
{
  const size_t auto_alignment = libxstream_alignment(size, alignment);
  const size_t aligned = ((size + auto_alignment - 1) / auto_alignment) * auto_alignment;
  LIBXSTREAM_ASSERT(aligned == LIBXSTREAM_ALIGN(size_t, size, auto_alignment/*pot*/));
  return aligned;
}


LIBXSTREAM_TARGET(mic) void* libxstream_align(void* address, size_t alignment)
{
  LIBXSTREAM_ASSERT(0 != alignment);
  const uintptr_t aligned = ((reinterpret_cast<uintptr_t>(address) + alignment - 1) / alignment) * alignment;
  LIBXSTREAM_ASSERT(aligned == LIBXSTREAM_ALIGN(uintptr_t, address, alignment/*pot*/));
  return reinterpret_cast<void*>(aligned);
}


LIBXSTREAM_TARGET(mic) const void* libxstream_align(const void* address, size_t alignment)
{
  LIBXSTREAM_ASSERT(0 != alignment);
  const uintptr_t aligned = ((reinterpret_cast<uintptr_t>(address) + alignment - 1) / alignment) * alignment;
  LIBXSTREAM_ASSERT(aligned == LIBXSTREAM_ALIGN(uintptr_t, address, alignment/*pot*/));
  return reinterpret_cast<void*>(aligned);
}


LIBXSTREAM_TARGET(mic) size_t libxstream_linear_size(size_t dims, const size_t shape[], size_t initial_size)
{
  return libxstream_alloc_internal::linear_size(dims, shape, initial_size);
}


LIBXSTREAM_TARGET(mic) int libxstream_linear_offset(size_t dims, const int offset[], const size_t shape[])
{
  LIBXSTREAM_ASSERT(offset && shape);
  int result = 0;

  if (0 < dims) {
    size_t size = shape[0];
    result = offset[0];

    LIBXSTREAM_PRAGMA_LOOP_COUNT(1, LIBXSTREAM_MAX_NDIMS, 2)
    for (size_t i = 1; i < dims; ++i) {
      result += offset[i] * static_cast<int>(size);
      size *= shape[i];
    }
  }

  return result;
}


LIBXSTREAM_TARGET(mic) size_t libxstream_linear_address(size_t dims, const int offset[], const size_t shape[], const size_t pitch[])
{
  LIBXSTREAM_ASSERT(offset && shape && pitch);
  size_t result = 0;

  if (0 < dims) {
    size_t d = dims - 1;
    int p = static_cast<int>(pitch[0]);
    result = offset[0] * libxstream_alloc_internal::linear_size<int>(d, shape + 1, 1);

    LIBXSTREAM_PRAGMA_LOOP_COUNT(1, LIBXSTREAM_MAX_NDIMS, 2)
    for (size_t i = 1; i < dims; ++i) {
      result += libxstream_alloc_internal::linear_size(d - i, shape + i + 1, p * offset[i]);
      p *= static_cast<int>(pitch[i]);
    }
  }

  return result;
}


int libxstream_real_allocate(void** memory, size_t size, size_t alignment, const void* extra, size_t extra_size)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
      size_t alloc_size = 0, unlock_size = 0;
      libxstream_alloc_internal::info_type::alloc_size(size, alignment, extra_size, alloc_size, unlock_size);
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
      void *const buffer = new char[alloc_size];
      memset(buffer, 0, alloc_size);
#elif defined(__MKL)
      void *const buffer = mkl_malloc(alloc_size, LIBXSTREAM_MAX_SIMD);
#elif defined(_WIN32)
      void *const buffer = _aligned_malloc(alloc_size, LIBXSTREAM_MAX_SIMD);
#elif defined(__GNUC__)
      void *const buffer = _mm_malloc(alloc_size, LIBXSTREAM_MAX_SIMD);
#else
      result = (0 == posix_memalign(&memory, LIBXSTREAM_MAX_SIMD, alloc_size)) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
      LIBXSTREAM_CHECK_ERROR(result);
#endif
      if (0 != buffer) {
        char *const dst = static_cast<char*>(buffer);
        if (0 < extra_size && 0 != extra) {
          const char *const src = static_cast<const char*>(extra);
          for (size_t i = 0; i < extra_size; ++i) dst[i] = src[i];
        }
        libxstream_alloc_internal::info_type& info = *reinterpret_cast<libxstream_alloc_internal::info_type*>(dst + extra_size);
        info.pointer = buffer;
        info.size = size;
        info.real = false;
        *memory = dst + unlock_size;
      }
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_INTERNAL_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == result);
  return result;
}


int libxstream_real_deallocate(const void* memory)
{
  if (memory) {
    void* buffer = 0;
    libxstream_alloc_info(memory, 0, &buffer, 0, 0);
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
    delete[] static_cast<char*>(buffer);
#elif defined(__MKL)
    mkl_free(buffer);
#elif defined(_WIN32)
    _aligned_free(buffer);
#elif defined(__GNUC__)
    _mm_free(buffer);
#else
    free(buffer);
#endif
  }

  return LIBXSTREAM_ERROR_NONE;
}


int libxstream_virt_allocate(void** memory, size_t size, size_t alignment, const void* extra, size_t extra_size)
{
  LIBXSTREAM_CHECK_CONDITION(0 == extra_size || 0 != extra);
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
#if defined(_WIN32)
# if defined(LIBXSTREAM_ALLOC_VALLOC)
      size_t alloc_size = 0, unlock_size = 0;
      libxstream_alloc_internal::info_type::alloc_size(size, alignment, extra_size, alloc_size, unlock_size);
      void* buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE, PAGE_NOACCESS);
      if (0 != buffer) {
        buffer = VirtualAlloc(buffer, unlock_size, MEM_COMMIT, PAGE_READWRITE);
      }
      if (0 != buffer) {
        char *const dst = static_cast<char*>(buffer);
        if (0 < extra_size && 0 != extra) {
          const char *const src = static_cast<const char*>(extra);
          for (size_t i = 0; i < extra_size; ++i) dst[i] = src[i];
        }
        libxstream_alloc_internal::info_type& info = *reinterpret_cast<libxstream_alloc_internal::info_type*>(dst + extra_size);
        info.pointer = buffer;
        info.size = size;
        info.real = false;
        *memory = dst + unlock_size;
      }
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# else
      result = libxstream_real_allocate(memory, size, alignment, extra, extra_size);
# endif
#else
# if defined(LIBXSTREAM_ALLOC_MMAP)
      size_t alloc_size = 0, unlock_size = 0;
      libxstream_alloc_internal::info_type::alloc_size(size, alignment, extra_size, alloc_size, unlock_size);
#if 0 // TODO
      void* buffer = mmap(0, alloc_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (MAP_FAILED != buffer) {
        buffer = mmap(buffer, unlock_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      }
#else
      void *const buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
      if (MAP_FAILED != buffer) {
        char *const dst = static_cast<char*>(buffer);
        if (0 < extra_size && 0 != extra) {
          const char *const src = static_cast<const char*>(extra);
          for (size_t i = 0; i < extra_size; ++i) dst[i] = src[i];
        }
        libxstream_alloc_internal::info_type& info = *reinterpret_cast<libxstream_alloc_internal::info_type*>(dst + extra_size);
        info.pointer = buffer;
        info.size = size;
        info.real = false;
        *memory = dst + unlock_size;
      }
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# else
      result = libxstream_real_allocate(memory, size, alignment, extra, extra_size);
# endif
#endif
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_INTERNAL_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == result);
  return result;
}


int libxstream_virt_deallocate(const void* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
#if defined(_WIN32)
# if defined(LIBXSTREAM_ALLOC_VALLOC)
    void* buffer = 0;
    libxstream_alloc_info(memory, 0, &buffer, 0, 0);
    result = FALSE != VirtualFree(buffer, 0, MEM_RELEASE) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
# else
    result = libxstream_real_deallocate(memory);
# endif
#else
# if defined(LIBXSTREAM_ALLOC_MMAP)
    void* buffer = 0;
    size_t size = 0;
    libxstream_alloc_info(memory, &size, &buffer, 0, 0);
    result = 0 == munmap(buffer, size) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
# else
    result = libxstream_real_deallocate(memory);
# endif
#endif
  }

  LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == result);
  return result;
}


int libxstream_alloc_info(const void* memory, size_t* size, void** extra, size_t* extra_size, bool* real)
{
  LIBXSTREAM_CHECK_CONDITION(0 != size || 0 != extra || 0 != extra_size || 0 != real);

  if (0 != memory) {
    using libxstream_alloc_internal::info_type;
    const info_type& info = *reinterpret_cast<const info_type*>(static_cast<const char*>(memory) - sizeof(info_type));
    if (size) *size = info.size;
    if (real) *real = info.real;
    if (extra) *extra = info.pointer;
    if (extra_size) {
      const char *const a = reinterpret_cast<const char*>(&info);
      const char *const b = static_cast<const char*>(info.pointer);
      LIBXSTREAM_ASSERT(a >= b);
      *extra_size = a - b;
    }
  }
  else {
    LIBXSTREAM_CHECK_CONDITION(0 == real);
    if (size) *size = 0;
    if (extra) *extra = 0;
    if (extra_size) *extra_size = 0;
  }

  return LIBXSTREAM_ERROR_NONE;
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
