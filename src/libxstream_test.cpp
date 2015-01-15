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
#if defined(LIBXSTREAM_TEST) && (0 != (2*LIBXSTREAM_TEST+1)/2)

#include <libxstream_test.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <atomic>

#if defined(LIBXSTREAM_TEST_STANDALONE)
# if defined(_OPENMP)
#   include <omp.h>
# endif
#endif


namespace libxstream_test_internal {

std::atomic<size_t> lock(0);


bool wait()
{
  const bool result = 0 < --lock;
  while(0 < lock) { // spin/yield
#if defined(LIBXSTREAM_YIELD) && defined(LIBXSTREAM_MIC_STDTHREAD)
    std::this_thread::yield();
#endif
  }
  return result;
}


LIBXSTREAM_EXPORT void check(bool& result, const unsigned char* buffer, size_t size, char pattern)
{
  result = true;
  for (size_t i = 0; i < size && result; ++i) {
    result = pattern == buffer[i];
  }
}

} // namespace libxstream_test_internal


libxstream_test::libxstream_test()
  : m_return_code(LIBXSTREAM_ERROR_NONE), m_device(-1), m_stream(0), m_event(0), m_host_mem(0), m_dev_mem(0)
{
  ++libxstream_test_internal::lock;

#if defined(LIBXSTREAM_MIC_STDTHREAD)
  const std::thread::id id = std::this_thread::get_id();
  fprintf(stderr, "TST entered by thread=0x%lx\n",
    static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(&id)));
#endif

  m_return_code = libxstream_get_active_device(&m_device);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_get_active_device: failed!\n");
    return;
  }

  size_t mem_free = 0, mem_avail = 0;
  m_return_code = libxstream_mem_info(m_device, &mem_free, &mem_avail);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_dev_mem_info: failed!\n");
    return;
  }

  m_return_code = libxstream_stream_create(&m_stream, m_device, 0, "Test Stream");
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_stream_create: failed!\n");
    return;
  }

  const size_t size = 4711u * 1024u;
  const char pattern_a = 'a', pattern_b = 'b';
  LIBXSTREAM_ASSERT(pattern_a != pattern_b);
  m_return_code = libxstream_mem_allocate(-1, &m_host_mem, size, 0);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_host_mem_allocate: failed!\n");
    return;
  }

  m_return_code = libxstream_mem_allocate(m_device, &m_dev_mem, size, 0);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_dev_mem_allocate: failed!\n");
    return;
  }

  m_return_code = libxstream_mem_info(m_device, &mem_free, &mem_avail);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_dev_mem_info: failed!\n");
    return;
  }

  std::fill_n(m_host_mem, size, pattern_a);
  m_return_code = libxstream_memcpy_h2d(m_host_mem, m_dev_mem, size, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memcpy_h2d: failed!\n");
    return;
  }

  bool ok = false;
  LIBXSTREAM_OFFLOAD_BEGIN(m_stream, &ok, m_dev_mem, size, pattern_a)
  {
#if defined(LIBXSTREAM_DEBUG)
    fprintf(stderr, "TST device-side validation started\n");
#endif
    const unsigned char* dev_mem = ptr<const unsigned char,1>();
    const size_t size = val<const size_t,2>();
    const char pattern = val<const char,3>();
    bool& ok = *ptr<bool,0>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(size, pattern) in(dev_mem: length(0) alloc_if(false) free_if(false)) //out(ok)
        {
          libxstream_test_internal::check(ok, dev_mem, size, pattern);
#if defined(LIBXSTREAM_DEBUG)
          fprintf(stderr, "TST device-side validation completed\n");
#endif
        }
      }
      else {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(size, pattern) in(dev_mem: length(0) alloc_if(false) free_if(false)) //out(ok)
        {
          libxstream_test_internal::check(ok, dev_mem, size, pattern);
#if defined(LIBXSTREAM_DEBUG)
          fprintf(stderr, "TST device-side validation completed\n");
#endif
        }
      }
    }
    else
#endif
    {
      libxstream_test_internal::check(ok, dev_mem, size, pattern);
#if defined(LIBXSTREAM_DEBUG)
      fprintf(stderr, "TST device-side validation completed\n");
#endif
    }
  }
  LIBXSTREAM_OFFLOAD_END(false)

  m_return_code = libxstream_event_create(&m_event);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_create: failed!\n");
    return;
  }

  m_return_code = libxstream_event_record(m_event, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_record: failed!\n");
    return;
  }

  m_return_code = libxstream_event_synchronize(m_event);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_synchronize: failed!\n");
    return;
  }

  if (!ok) {
    fprintf(stderr, "TST libxstream_memcpy_h2d: validation failed!\n");
    m_return_code = LIBXSTREAM_ERROR_RUNTIME;
    return;
  }

  std::fill_n(m_host_mem, size, pattern_b);
  m_return_code = libxstream_memcpy_d2h(m_dev_mem, m_host_mem, size, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memcpy_d2h: failed!\n");
    return;
  }

  const size_t size2 = size / 2;
  m_return_code = libxstream_memset_zero(m_dev_mem, size2, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memset_zero: memset failed!\n");
    return;
  }

  m_return_code = libxstream_memset_zero(m_dev_mem + size2, size - size2, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memset_zero: memset (offset=%lu) failed!\n",
      static_cast<unsigned long>(size2));
    return;
  }

  m_return_code = libxstream_event_record(m_event, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_record: failed!\n");
    return;
  }

  int has_occured = 0;
  m_return_code = libxstream_event_query(m_event, &has_occured);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_query: failed!\n");
    return;
  }

  if (0 == has_occured) {
    m_return_code = libxstream_event_synchronize(m_event);
    if (LIBXSTREAM_ERROR_NONE != m_return_code) {
      fprintf(stderr, "TST libxstream_event_synchronize: failed!\n");
      return;
    }
  }

  libxstream_test_internal::check(ok, m_host_mem, size, pattern_a);
  if (!ok) {
    fprintf(stderr, "TST libxstream_memcpy_d2h: validation failed!\n");
    m_return_code = LIBXSTREAM_ERROR_RUNTIME;
    return;
  }

  m_return_code = libxstream_memcpy_d2h(m_dev_mem, m_host_mem, size2, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memcpy_d2h: failed!\n");
    return;
  }

  m_return_code = libxstream_memcpy_d2h(m_dev_mem + size2, m_host_mem + size2, size - size2, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_memcpy_d2h: use of pointer offset=%lu failed!\n",
      static_cast<unsigned long>(size2));
    return;
  }

  m_return_code = libxstream_event_record(m_event, m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_record: failed!\n");
    return;
  }

  m_return_code = libxstream_stream_sync(m_stream);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_stream_sync: failed!\n");
    return;
  }

  m_return_code = libxstream_event_query(m_event, &has_occured);
  if (LIBXSTREAM_ERROR_NONE != m_return_code) {
    fprintf(stderr, "TST libxstream_event_query: failed!\n");
    return;
  }

  if (0 == has_occured) {
    fprintf(stderr, "TST libxstream_event_query: event did not occur!\n");
    m_return_code = LIBXSTREAM_ERROR_RUNTIME;
    return;
  }

  libxstream_test_internal::check(ok, m_host_mem, size, 0);
  if (!ok) {
    fprintf(stderr, "TST libxstream_memset_zero: validation failed!\n");
    m_return_code = LIBXSTREAM_ERROR_RUNTIME;
    return;
  }
}


libxstream_test::~libxstream_test()
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (LIBXSTREAM_ERROR_NONE == result && LIBXSTREAM_ERROR_NONE != (result = libxstream_event_destroy(m_event))) {
    fprintf(stderr, "TST libxstream_event_destroy: failed!\n");
  }

  if (LIBXSTREAM_ERROR_NONE == result && LIBXSTREAM_ERROR_NONE != (result = libxstream_mem_deallocate(-1, m_host_mem))) {
    fprintf(stderr, "TST libxstream_host_mem_deallocate: failed!\n");
  }

  if (LIBXSTREAM_ERROR_NONE == result && LIBXSTREAM_ERROR_NONE != (result = libxstream_mem_deallocate(m_device, m_dev_mem))) {
    fprintf(stderr, "TST libxstream_dev_mem_deallocate: failed!\n");
  }

  if (LIBXSTREAM_ERROR_NONE == result && LIBXSTREAM_ERROR_NONE != (result = libxstream_stream_destroy(m_stream))) {
    fprintf(stderr, "TST libxstream_stream_destroy: failed!\n");
  }

  if (LIBXSTREAM_ERROR_NONE == m_return_code && LIBXSTREAM_ERROR_NONE == result) {
    fprintf(stderr, "TST successfully completed.\n");
  }
  else {
    fprintf(stderr, "TST test suite failed!\n");
#if defined(LIBXSTREAM_TEST) && (1 == (2*LIBXSTREAM_TEST+1)/2)
    if (!libxstream_test_internal::wait()) {
      fprintf(stderr, "TST terminating application.\n");
      exit(m_return_code); 
    }
#endif
  }

#if defined(LIBXSTREAM_TEST) && (2 == (2*LIBXSTREAM_TEST+1)/2)
  if (!libxstream_test_internal::wait()) {
    fprintf(stderr, "TST terminating application.\n");
    exit(result);
  }
#endif
}


#if defined(LIBXSTREAM_TEST_STANDALONE)
int main(int argc, char* argv[])
{
  try {
#if defined(_OPENMP)
    const int ntasks = std::max(1 < argc ? std::atoi(argv[1]) : omp_get_max_threads(), 1);
#else
    const int ntasks = 1;
#endif

    size_t ndevices = 0;
    if (0 == libxstream_get_ndevices(&ndevices) && 0 < ndevices) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      for (int i = 0; i < ntasks; ++i) {
#if defined(_OPENMP)
#       pragma omp single nowait
#endif
        {
#if defined(_OPENMP) && (200203 < _OPENMP)
#         pragma omp task
#endif
          libxstream_set_active_device(i % ndevices);
        }
      }
    }
    else {
      fprintf(stderr, "Error: no device found!\n");
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
#endif // LIBXSTREAM_TEST_STANDALONE

#endif // defined(LIBXSTREAM_TEST)
