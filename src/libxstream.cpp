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
#include <libxstream.hpp>
#include <algorithm>
#include <limits>

#if defined(LIBXSTREAM_STDFEATURES)
# include <thread>
# include <atomic>
#endif

#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(__GNUC__)
# include <pthread.h>
#endif

#if defined(LIBXSTREAM_OFFLOAD)
# pragma offload_attribute(push,target(mic))
#endif
#include <cstring>
#if !defined(_WIN32)
# include <sys/mman.h>
# include <unistd.h>
#endif
#if defined(LIBXSTREAM_OFFLOAD)
# pragma offload_attribute(pop)
#endif

#if defined(__MKL)
# include <mkl.h>
#endif

#if defined(LIBXSTREAM_OFFLOAD)
# include <offload.h>
#endif

#if defined(_WIN32)
# include <windows.h>
#else
# include <xmmintrin.h>
#endif


namespace libxstream_internal {

unsigned int       abs(unsigned int        a)  { return a; }
unsigned long      abs(unsigned long       a)  { return a; }
unsigned long long abs(unsigned long long  a)  { return a; }

template<typename T>
inline T gcd(T a, T b)
{
  while (0 != b) {
    const T r = a % b;
    a = b;
    b = r;
  }
  return a;
}


template<typename T>
inline T lcm(T a, T b)
{
  //using libxstream_internal::abs;
  using std::abs;
  return abs(a * b) / gcd(a, b);
}


template<typename T>
inline T auto_alignment(T size)
{
  const T min_a = std::min(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  const T max_a = lcm(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  return max_a <= size ? max_a : (min_a < size ? min_a : sizeof(void*));
}


int allocate_real(void** memory, size_t size)
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
      void *const buffer = mkl_malloc(size, static_cast<int>(libxstream_internal::auto_alignment(size)));
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
      void *const buffer = _aligned_malloc(size, libxstream_internal::auto_alignment(size));
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
      void *const buffer = _mm_malloc(size, static_cast<int>(libxstream_internal::auto_alignment(size)));
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
      result = (0 == posix_memalign(memory, libxstream_internal::auto_alignment(size), size) && 0 != *memory)
# else
      result = (0 == posix_memalign(memory, libxstream_internal::auto_alignment(size), size))
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


int deallocate_real(const void* memory)
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


int allocate_virtual(void** memory, size_t size, const void* data, size_t data_size = 0)
{
  LIBXSTREAM_CHECK_CONDITION(0 == data_size || (0 != data && data_size <= size));
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
      result = allocate_real(memory, size);
      LIBXSTREAM_CHECK_ERROR(result);
      void *const user_data = get_user_data(*memory);
#elif defined(_WIN32)
      void *const buffer = VirtualAlloc(0, size, MEM_RESERVE, PAGE_NOACCESS);
      LIBXSTREAM_CHECK_CONDITION(0 != buffer);
      void *const user_data = get_user_data(VirtualAlloc(buffer, data_size, MEM_COMMIT, PAGE_READWRITE));
      *memory = buffer;
#else
      void *const buffer = mmap(0, size, PROT_READ | PROT_WRITE/*PROT_NONE*/, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      LIBXSTREAM_CHECK_CONDITION(MAP_FAILED != buffer);
      void *const user_data = get_user_data(buffer);
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


int deallocate_virtual(const void* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
    result = deallocate_real(memory);
#elif defined(_WIN32)
    result = FALSE != VirtualFree(const_cast<void*>(memory), 0, MEM_RELEASE) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#else
    const size_t size = *static_cast<const size_t*>(memory);
    result = 0 == munmap(const_cast<void*>(memory), size) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#endif
  }

  return result;
}


class context_type {
public:
  context_type()
    : m_lock(libxstream_lock_create())
    , m_device(-2)
  {}

  ~context_type() {
    libxstream_lock_destroy(m_lock);
  }

public:
  libxstream_lock* lock() { return m_lock; }
  int global_device() const { return m_device; }

  void global_device(int device) {
    libxstream_lock_acquire(m_lock);
    if (-1 > m_device) {
      m_device = device;
    }
    libxstream_lock_release(m_lock);
  }

  // store the active device per host-thread
  int& device() {
    static LIBXSTREAM_TLS int instance = -2;
    return instance;
  }

private:
  libxstream_lock* m_lock;
  int m_device;
} context;


LIBXSTREAM_EXPORT void mem_info(uint64_t& memory_physical, uint64_t& memory_not_allocated)
{
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  const BOOL ok = GlobalMemoryStatusEx(&status);
  if (TRUE == ok) {
    memory_not_allocated = status.ullAvailPhys;
    memory_physical = status.ullTotalPhys;
  }
  else {
    memory_not_allocated = 0;
    memory_physical = 0;
  }
#else
  const long memory_pages_size = sysconf(_SC_PAGE_SIZE);
  const long memory_pages_phys = sysconf(_SC_PHYS_PAGES);
  const long memory_pages_avail = sysconf(_SC_AVPHYS_PAGES);
  memory_not_allocated = memory_pages_size * memory_pages_avail;
  memory_physical = memory_pages_size * memory_pages_phys;
#endif
}

} // namespace libxstream_internal


libxstream_lock* libxstream_lock_create()
{
#if defined(_OPENMP)
# if defined(LIBXSTREAM_NONRECURSIVE_LOCKS)
  omp_lock_t *const typed_lock = new omp_lock_t;
  omp_init_lock(typed_lock);
# else
  omp_nest_lock_t *const typed_lock = new omp_nest_lock_t;
  omp_init_nest_lock(typed_lock);
# endif
#elif defined(__GNUC__)
  pthread_mutexattr_t attributes;
  pthread_mutexattr_init(&attributes);
# if defined(LIBXSTREAM_NONRECURSIVE_LOCKS)
  pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_NORMAL);
# else
  pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_RECURSIVE);
# endif
  pthread_mutex_t *const typed_lock = new pthread_mutex_t;
  pthread_mutex_init(typed_lock, &attributes);
#elif defined(LIBXSTREAM_STDFEATURES)
  // std::mutex cannot be used due to same thread needed for lock/unlock
  std::atomic<int> *const typed_lock = new std::atomic<int>(0);
#endif
  return typed_lock;
}


void libxstream_lock_destroy(libxstream_lock* lock)
{
#if defined(_OPENMP)
# if defined(LIBXSTREAM_NONRECURSIVE_LOCKS)
  omp_lock_t *const typed_lock = static_cast<omp_lock_t*>(lock);
  omp_destroy_lock(typed_lock);
# else
  omp_nest_lock_t *const typed_lock = static_cast<omp_nest_lock_t*>(lock);
  omp_destroy_nest_lock(typed_lock);
# endif
#elif defined(__GNUC__)
  pthread_mutex_t *const typed_lock = static_cast<pthread_mutex_t*>(lock);
  pthread_mutex_destroy(typed_lock);
#elif defined(LIBXSTREAM_STDFEATURES)
  std::atomic<int> *const typed_lock = static_cast<std::atomic<int>*>(lock);
#endif
  delete typed_lock;
}


void libxstream_lock_acquire(libxstream_lock* lock)
{
  LIBXSTREAM_ASSERT(lock);
#if defined(_OPENMP)
# if defined(LIBXSTREAM_NONRECURSIVE_LOCKS)
  omp_lock_t *const typed_lock = static_cast<omp_lock_t*>(lock);
  omp_set_lock(typed_lock);
# else
  omp_nest_lock_t *const typed_lock = static_cast<omp_nest_lock_t*>(lock);
  omp_set_nest_lock(typed_lock);
# endif
#elif defined(__GNUC__)
  pthread_mutex_t *const typed_lock = static_cast<pthread_mutex_t*>(lock);
  pthread_mutex_lock(typed_lock);
#elif defined(LIBXSTREAM_STDFEATURES)
  std::atomic<int>& typed_lock = *static_cast<std::atomic<int>*>(lock);
  if (1 < ++typed_lock) {
    while (1 < typed_lock) {
      std::this_thread::yield();
    }
  }
#endif
}


void libxstream_lock_release(libxstream_lock* lock)
{
  LIBXSTREAM_ASSERT(lock);
#if defined(_OPENMP)
# if defined(LIBXSTREAM_NONRECURSIVE_LOCKS)
  omp_lock_t *const typed_lock = static_cast<omp_lock_t*>(lock);
  omp_unset_lock(typed_lock);
# else
  omp_nest_lock_t *const typed_lock = static_cast<omp_nest_lock_t*>(lock);
  omp_unset_nest_lock(typed_lock);
# endif
#elif defined(__GNUC__)
  pthread_mutex_t *const typed_lock = static_cast<pthread_mutex_t*>(lock);
  pthread_mutex_unlock(typed_lock);
#elif defined(LIBXSTREAM_STDFEATURES)
  std::atomic<int>& typed_lock = *static_cast<std::atomic<int>*>(lock);
  --typed_lock;
#endif
}


int this_thread_id()
{
  static LIBXSTREAM_TLS int id = -1;
  if (0 > id) {
#if defined(LIBXSTREAM_STDFEATURES)
    static std::atomic<int> num_threads(0);
    id = num_threads++;
#elif defined(_OPENMP)
    static int num_threads = 0;
#   pragma omp critical
    id = num_threads++;
#else
    static int num_threads = 0;
    libxstream_lock *const lock = libxstream_internal::context.lock();
    libxstream_lock_acquire(lock);
    id = num_threads++;
    libxstream_lock_release(lock);
#endif
  }
  return id;
}


void this_thread_yield()
{
#if defined(LIBXSTREAM_STDFEATURES)
  std::this_thread::yield();
#elif defined(__GNUC__)
  pthread_yield();
#endif
}


extern "C" int libxstream_get_ndevices(size_t* ndevices)
{
  LIBXSTREAM_CHECK_CONDITION(ndevices);

#if defined(LIBXSTREAM_OFFLOAD) && !defined(__MIC__)
  *ndevices = std::min(_Offload_number_of_devices(), LIBXSTREAM_MAX_NDEVICES);
#else
  *ndevices = 1; // host
#endif

#if defined(LIBXSTREAM_DEBUG)
  static LIBXSTREAM_TLS bool print = true;
  if (print) {
    LIBXSTREAM_PRINT_INFOCTX("ndevices=%lu", static_cast<unsigned long>(*ndevices));
    print = false;
  }
#endif

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_get_active_device(int* device)
{
  LIBXSTREAM_CHECK_CONDITION(0 != device);
  int result = LIBXSTREAM_ERROR_NONE, active_device = libxstream_internal::context.device();

  if (-1 > active_device) {
    active_device = libxstream_internal::context.global_device();

    if (-1 > active_device) {
      size_t ndevices = 0;
      result = libxstream_get_ndevices(&ndevices);
      active_device = ndevices - 1;
      libxstream_internal::context.global_device(active_device);
      libxstream_internal::context.device() = active_device;
    }

    LIBXSTREAM_PRINT_INFOCTX("device=%i (fallback) thread=%i", active_device, this_thread_id());
  }

  *device = active_device;
  return result;
}


extern "C" int libxstream_set_active_device(int device)
{
  size_t ndevices = LIBXSTREAM_MAX_NDEVICES;
  LIBXSTREAM_CHECK_CONDITION(-1 <= device && ndevices >= static_cast<size_t>(device + 1) && LIBXSTREAM_ERROR_NONE == libxstream_get_ndevices(&ndevices) && ndevices >= static_cast<size_t>(device + 1));

  if (-1 > libxstream_internal::context.global_device()) {
    libxstream_internal::context.global_device(device);
  }

  libxstream_internal::context.device() = device;
  LIBXSTREAM_PRINT_INFOCTX("device=%i thread=%i", device, this_thread_id());

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_mem_info(int device, size_t* allocatable, size_t* physical)
{
  LIBXSTREAM_CHECK_CONDITION(allocatable || physical);
  uint64_t memory_physical = 0, memory_allocatable = 0;
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_OFFLOAD_BEGIN(0, device, &memory_physical, &memory_allocatable)
  {
    uint64_t& memory_physical = *ptr<uint64_t,1>();
    uint64_t& memory_allocatable = *ptr<uint64_t,2>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
#     pragma offload target(mic:LIBXSTREAM_OFFLOAD_DEVICE) //out(memory_physical, memory_allocatable)
      {
        libxstream_internal::mem_info(memory_physical, memory_allocatable);
      }
    }
    else
#endif
    {
      libxstream_internal::mem_info(memory_physical, memory_allocatable);
    }
  }
  LIBXSTREAM_OFFLOAD_END(true);

  LIBXSTREAM_PRINT_INFOCTX("device=%i allocatable=%lu physical=%lu", device,
    static_cast<unsigned long>(memory_allocatable),
    static_cast<unsigned long>(memory_physical));
  LIBXSTREAM_CHECK_CONDITION(0 < memory_physical && 0 < memory_allocatable);

  if (allocatable) {
    *allocatable = memory_allocatable;
  }

  if (physical) {
    *physical = memory_physical;
  }

  return result;
}


extern "C" int libxstream_mem_allocate(int device, void** memory, size_t size, size_t /*TODO: alignment*/)
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_OFFLOAD_BEGIN(0, device, memory, size, &result)
  {
    void*& memory = *ptr<void*,1>();
    const size_t size = val<const size_t,2>();
    int& result = *ptr<int,3>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      const int device = LIBXSTREAM_OFFLOAD_DEVICE;
      result = libxstream_internal::allocate_virtual(&memory, size, &device, sizeof(device));

      if (LIBXSTREAM_ERROR_NONE == result && 0 != memory) {
        char *const buffer = static_cast<char*>(memory);
#       pragma offload_transfer target(mic:LIBXSTREAM_OFFLOAD_DEVICE) nocopy(buffer: length(size) alloc_if(true))
      }
    }
    else
#endif
    {
      result = libxstream_internal::allocate_real(&memory, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(true);

  LIBXSTREAM_PRINT_INFOCTX("device=%i buffer=0x%lx size=%lu", device,
    memory ? static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(memory)) : 0UL,
    static_cast<unsigned long>(size));

  return result;
}


extern "C" int libxstream_mem_deallocate(int device, const void* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    LIBXSTREAM_OFFLOAD_BEGIN(0, device, memory, &result)
    {
      const char *const memory = ptr<const char,1>();
      int& result = *ptr<int,2>();

#if defined(LIBXSTREAM_OFFLOAD)
      if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
# if defined(LIBXSTREAM_CHECK)
        const int memory_device = *static_cast<const int*>(libxstream_internal::get_user_data(memory));
        if (memory_device != LIBXSTREAM_OFFLOAD_DEVICE) {
          LIBXSTREAM_PRINT_WARNING("device %i does not match allocating device %i", LIBXSTREAM_OFFLOAD_DEVICE, memory_device);
          LIBXSTREAM_OFFLOAD_DEVICE_UPDATE(memory_device);
        }
# endif
#       pragma offload_transfer target(mic:LIBXSTREAM_OFFLOAD_DEVICE) nocopy(memory: length(0) free_if(true))
        result = libxstream_internal::deallocate_virtual(memory);
      }
      else
#endif
      {
        result = libxstream_internal::deallocate_real(memory);
      }
    }
    LIBXSTREAM_OFFLOAD_END(true);
  }

  LIBXSTREAM_PRINT_INFOCTX("device=%i buffer=0x%lx", device,
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(memory)));

  return result;
}


extern "C" int libxstream_memset_zero(void* memory, size_t size, libxstream_stream* stream)
{
  LIBXSTREAM_CHECK_CONDITION(memory && stream);

  LIBXSTREAM_OFFLOAD_BEGIN(stream, memory, size)
  {
    char* dst = ptr<char,0>();
    const size_t size = val<const size_t,1>();

    LIBXSTREAM_PRINT_INFO("libxstream_memset_zero: buffer=0x%lx size=%lu stream=0x%lx",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dst)), static_cast<unsigned long>(size),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(LIBXSTREAM_OFFLOAD_STREAM)));

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(size) out(dst: length(0) alloc_if(false) free_if(false))
        {
          memset(dst, 0, size);
        }
      }
      else {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(size) out(dst: length(0) alloc_if(false) free_if(false))
        {
          memset(dst, 0, size);
        }
      }
    }
    else
#endif
    {
      memset(dst, 0, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false);

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t size, libxstream_stream* stream)
{
  LIBXSTREAM_CHECK_CONDITION(host_mem && dev_mem && stream);

  LIBXSTREAM_OFFLOAD_BEGIN(stream, host_mem, dev_mem, size)
  {
    const char *const src = ptr<const char,0>();
    char *const dst = ptr<char,1>();
    const size_t size = val<const size_t,2>();

    LIBXSTREAM_PRINT_INFO("libxstream_memcpy_h2d: 0x%lx->0x%lx size=%lu stream=0x%lx",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(src)),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dst)), static_cast<unsigned long>(size),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(LIBXSTREAM_OFFLOAD_STREAM)));

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
      else {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
    }
    else
#endif
    {
      std::copy(src, src + size, dst);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false);

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t size, libxstream_stream* stream)
{
  LIBXSTREAM_CHECK_CONDITION(dev_mem && host_mem && stream);

  LIBXSTREAM_OFFLOAD_BEGIN(stream, dev_mem, host_mem, size)
  {
    const char* src = ptr<const char,0>();
    char *const dst = ptr<char,1>();
    const size_t size = val<const size_t,2>();

    LIBXSTREAM_PRINT_INFO("libxstream_memcpy_d2h: 0x%lx->0x%lx size=%lu stream=0x%lx",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(src)),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dst)), static_cast<unsigned long>(size),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(LIBXSTREAM_OFFLOAD_STREAM)));

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          out(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
      else {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          out(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
    }
    else
#endif
    {
      std::copy(src, src + size, dst);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false);

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_memcpy_d2d(const void* src, void* dst, size_t size, libxstream_stream* stream)
{
  LIBXSTREAM_CHECK_CONDITION(src && dst && stream);

  LIBXSTREAM_OFFLOAD_BEGIN(stream, src, dst, size)
  {
    const uint64_t *const src = ptr<const uint64_t,0>();
    uint64_t* dst = ptr<uint64_t,1>();
    const size_t size = val<const size_t,2>();

    LIBXSTREAM_PRINT_INFO("libxstream_memcpy_d2d: 0x%lx->0x%lx size=%lu stream=0x%lx",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(src)),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dst)), static_cast<unsigned long>(size),
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(LIBXSTREAM_OFFLOAD_STREAM)));

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      // TODO: implement cross-device transfer

      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(size) in(src: length(0) alloc_if(false) free_if(false)) out(dst: length(0) alloc_if(false) free_if(false))
        memcpy(dst, src, size);
      }
      else {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(size) in(src: length(0) alloc_if(false) free_if(false)) out(dst: length(0) alloc_if(false) free_if(false))
        memcpy(dst, src, size);
      }
    }
    else
#endif
    {
      memcpy(dst, src, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false);

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_priority_range(int* least, int* greatest)
{
  *least = -1;
  *greatest = -1;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_create(libxstream_stream** stream, int device, int demux, int priority, const char* name)
{
  LIBXSTREAM_CHECK_CONDITION(stream);
  libxstream_stream *const s = new libxstream_stream(device, 0 != demux, priority, name);
  LIBXSTREAM_ASSERT(s);
  *stream = s;

#if defined(LIBXSTREAM_DEBUG)
  if (name && *name) {
    LIBXSTREAM_PRINT_INFOCTX("device=%i stream=0x%lx name=\"%s\"",
      device, static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(stream)), name);
  }
  else {
    LIBXSTREAM_PRINT_INFOCTX("device=%i stream=0x%lx",
      device, static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(stream)));
  }
#endif

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_destroy(libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  if (stream) {
    const char *const name = stream->name();
    if (name && *name) {
      LIBXSTREAM_PRINT_INFOCTX("stream=0x%lx name=\"%s\"",
        static_cast<unsigned long>(reinterpret_cast<const uintptr_t>(stream)), name);
    }
    else {
      LIBXSTREAM_PRINT_INFOCTX("stream=0x%lx",
        static_cast<unsigned long>(reinterpret_cast<const uintptr_t>(stream)));
    }
  }
#endif
  delete stream;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_sync(libxstream_stream* stream)
{
  const int result = stream ? stream->wait(0) : libxstream_stream::sync();

#if defined(LIBXSTREAM_DEBUG)
  if (0 != stream) {
    const char *const name = stream->name();
    if (name && *name) {
      LIBXSTREAM_PRINT_INFOCTX("stream=0x%lx name=\"%s\"",
        static_cast<unsigned long>(reinterpret_cast<const uintptr_t>(stream)), name);
    }
    else {
      LIBXSTREAM_PRINT_INFOCTX("stream=0x%lx",
        static_cast<unsigned long>(reinterpret_cast<const uintptr_t>(stream)));
    }
  }
  else {
    LIBXSTREAM_PRINT_INFOCTX0("synchronize all streams");
  }
#endif

  return result;
}


extern "C" int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event)
{
  const int result = event ? event->wait(stream) : (stream ? stream->wait(0) : libxstream_stream::sync());

  LIBXSTREAM_PRINT_INFOCTX("event=0x%lx stream=0x%lx",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));

  return result;
}


extern "C" int libxstream_event_create(libxstream_event** event)
{
  LIBXSTREAM_CHECK_CONDITION(event);
  *event = new libxstream_event;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_destroy(libxstream_event* event)
{
  delete event;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_record(libxstream_event* event, libxstream_stream* stream)
{
  LIBXSTREAM_PRINT_INFOCTX("event=0x%lx stream=0x%lx",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));

  if (stream) {
    event->enqueue(*stream, true);
  }
  else {
    libxstream_stream::enqueue(*event);
  }

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_query(const libxstream_event* event, int* has_occured)
{
  LIBXSTREAM_PRINT_INFOCTX("event=0x%lx", static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)));
  LIBXSTREAM_CHECK_CONDITION(event && has_occured);

  bool occurred = true;
  const int result = event->query(occurred, 0);
  *has_occured = occurred ? 1 : 0;

  return result;
}


extern "C" int libxstream_event_synchronize(libxstream_event* event)
{
  LIBXSTREAM_PRINT_INFOCTX("event=0x%lx", static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)));
  return event ? event->wait(0) : libxstream_stream::sync();
}
