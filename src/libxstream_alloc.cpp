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
#include "libxstream_alloc.hpp"
#include <libxstream_begin.h>
#include <algorithm>
#include <libxstream_end.h>

#if defined(__MKL)
# include <mkl.h>
#endif

#if defined(_WIN32)
# include <windows.h>
#else
# include <xmmintrin.h>
#endif


namespace libxstream_alloc_internal {

unsigned int       abs(unsigned int        a)  { return a; }
unsigned long      abs(unsigned long       a)  { return a; }
unsigned long long abs(unsigned long long  a)  { return a; }


template<typename S, typename T>
LIBXSTREAM_TARGET(mic) S linear_size(size_t dims, const T shape[], S initial_size)
{
  LIBXSTREAM_ASSERT(shape);
  S result = 0 < dims ? (std::max<S>(initial_size, 1) * static_cast<S>(shape[0])) : initial_size;
#if defined(__INTEL_COMPILER)
# pragma loop_count min(1), max(LIBXSTREAM_MAX_NDIMS), avg(2)
#endif
  for (size_t i = 1; i < dims; ++i) result *= static_cast<S>(shape[i]);
  return result;
}


void* get_user_data(void* memory)
{
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
  return memory;
#elif defined(_WIN32)
  return memory;
#else
  return reinterpret_cast<char*>(memory) + sizeof(size_t);
#endif
}


const void* get_user_data(const void* memory)
{
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
  return memory;
#elif defined(_WIN32)
  return memory;
#else
  return reinterpret_cast<const char*>(memory) + sizeof(size_t);
#endif
}

} // namespace libxstream_alloc_internal


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
  const size_t min_a = std::min(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  const size_t max_a = libxstream_lcm(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  return 0 == alignment ? (max_a <= size ? max_a : (min_a < size ? min_a : sizeof(void*))) : std::max(alignment, static_cast<size_t>(sizeof(void*)));
}


LIBXSTREAM_TARGET(mic) size_t libxstream_align(size_t size, size_t alignment)
{
  const size_t auto_alignment = libxstream_alignment(size, alignment);
  return ((size + auto_alignment - 1) / auto_alignment) * auto_alignment;
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

#if defined(__INTEL_COMPILER)
#   pragma loop_count min(1), max(LIBXSTREAM_MAX_NDIMS), avg(2)
#endif
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

#if defined(__INTEL_COMPILER)
#   pragma loop_count min(1), max(LIBXSTREAM_MAX_NDIMS), avg(2)
#endif
    for (size_t i = 1; i < dims; ++i) {
      result += libxstream_alloc_internal::linear_size(d - i, shape + i + 1, p * offset[i]);
      p *= static_cast<int>(pitch[i]);
    }
  }

  return result;
}


int libxstream_real_allocate(void** memory, size_t size, size_t alignment)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
#if defined(LIBXSTREAM_DEBUG)
      if (char *const buffer = new char[size]) {
        std::fill_n(buffer, size, 0);
        *memory = buffer;
      }
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
#elif defined(__MKL)
      void *const buffer = mkl_malloc(size, static_cast<int>(libxstream_alignment(size, alignment)));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#elif defined(_WIN32)
      void *const buffer = _aligned_malloc(size, libxstream_alignment(size, alignment));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#elif defined(__GNUC__)
      void *const buffer = _mm_malloc(size, static_cast<int>(libxstream_alignment(size, alignment)));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#else
# if defined(LIBXSTREAM_CHECK)
      result = (0 == posix_memalign(memory, libxstream_alignment(size, alignment), size) && 0 != *memory)
# else
      result = (0 == posix_memalign(memory, libxstream_alignment(size, alignment), size))
# endif
        ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#endif
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  return result;
}


int libxstream_real_deallocate(const void* memory)
{
  if (memory) {
#if defined(LIBXSTREAM_DEBUG)
    delete[] static_cast<const char*>(memory);
#elif defined(__MKL)
    mkl_free(const_cast<void*>(memory));
#elif defined(_WIN32)
    _aligned_free(const_cast<void*>(memory));
#elif defined(__GNUC__)
    _mm_free(const_cast<void*>(memory));
#else
    free(const_cast<void*>(memory));
#endif
  }

  return LIBXSTREAM_ERROR_NONE;
}


int libxstream_virt_allocate(void** memory, size_t size, size_t alignment, const void* data, size_t data_size)
{
  LIBXSTREAM_CHECK_CONDITION(0 == data_size || 0 != data);
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
      size = std::max(size, data_size); // sanitize
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
      result = libxstream_real_allocate(memory, size, alignment);
      LIBXSTREAM_CHECK_ERROR(result);
      void *const user_data = libxstream_alloc_internal::get_user_data(*memory);
#elif defined(_WIN32)
      void *const buffer = VirtualAlloc(0, size, MEM_RESERVE, PAGE_NOACCESS);
      LIBXSTREAM_CHECK_CONDITION(0 != buffer);
      void *const user_data = libxstream_alloc_internal::get_user_data(VirtualAlloc(buffer, data_size, MEM_COMMIT, PAGE_READWRITE));
      *memory = buffer;
#else
      void *const buffer = mmap(0, size, PROT_READ | PROT_WRITE/*PROT_NONE*/, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      LIBXSTREAM_CHECK_CONDITION(MAP_FAILED != buffer);
      void *const user_data = libxstream_alloc_internal::get_user_data(buffer);
      LIBXSTREAM_ASSERT(sizeof(size) <= static_cast<char*>(user_data) - static_cast<char*>(buffer));
      *static_cast<size_t*>(buffer) = size;
      *memory = buffer;
#endif
      if (0 < data_size && 0 != data) {
        LIBXSTREAM_CHECK_CONDITION(0 != user_data);
        const char *const src = static_cast<const char*>(data);
        char *const dst = static_cast<char*>(user_data);
        for (size_t i = 0; i < data_size; ++i) dst[i] = src[i];
      }
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  return result;
}


int libxstream_virt_deallocate(const void* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
    result = libxstream_real_deallocate(memory);
#elif defined(_WIN32)
    result = FALSE != VirtualFree(const_cast<void*>(memory), 0, MEM_RELEASE) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#else
    const size_t size = *static_cast<const size_t*>(memory);
    result = 0 == munmap(const_cast<void*>(memory), size) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#endif
  }

  return result;
}
