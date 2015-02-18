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
#include "test.hpp"
#include <libxstream_begin.h>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <atomic>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>


namespace test_internal {

LIBXSTREAM_TARGET(mic) void check(libxstream_bool* result, const void* buffer, size_t size, char pattern)
{
  const libxstream_argument* arg = 0;
  libxstream_get_argument(buffer, &arg);
  size_t shape = 0;
  libxstream_get_shape(arg, &shape);
  bool ok = shape == size;

  const char *const values = reinterpret_cast<const char*>(buffer);
  for (size_t i = 0; i < size && ok; ++i) {
    ok = pattern == values[i];
  }
  LIBXSTREAM_ASSERT(result);
  *result = ok;
}

} // namespace test_internal


test_type::test_type(int device)
  : m_device(device), m_stream(0), m_event(0)
  , m_host_mem(0), m_dev_mem(0)
{
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_get_active_device(&m_device));

  size_t mem_free = 0, mem_avail = 0;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_info(m_device, &mem_free, &mem_avail));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_stream_create(&m_stream, m_device, 0, 0, 0));

  const size_t size = 4711u * 1024u;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_allocate(-1, &m_host_mem, size, 0));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_allocate(m_device, &m_dev_mem, size, 0));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_info(m_device, &mem_free, &mem_avail));

  const char pattern_a = 'a', pattern_b = 'b';
  LIBXSTREAM_ASSERT(pattern_a != pattern_b);
  std::fill_n(reinterpret_cast<char*>(m_host_mem), size, pattern_a);
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memcpy_h2d(m_host_mem, m_dev_mem, size, m_stream));

  libxstream_bool ok = false;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_create_signature(&m_signature, 4));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_output(m_signature, 0, &ok, libxstream_type2value<libxstream_bool>::value, 0, 0));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_input (m_signature, 1, m_dev_mem, LIBXSTREAM_TYPE_BYTE, 1, &size));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_input (m_signature, 2, &size, libxstream_type2value<size_t>::value, 0, 0));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_input (m_signature, 3, &pattern_a, libxstream_type2value<char>::value, 0, 0));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_call(reinterpret_cast<libxstream_function>(test_internal::check), m_signature, m_stream, LIBXSTREAM_CALL_DEFAULT));

  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_create(&m_event));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_record(m_event, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_synchronize(m_event));
  LIBXSTREAM_CHECK_CONDITION_RETURN(ok);

  std::fill_n(reinterpret_cast<char*>(m_host_mem), size, pattern_b);
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memcpy_d2h(m_dev_mem, m_host_mem, size, m_stream));

  const size_t size2 = size / 2;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memset_zero(m_dev_mem, size2, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memset_zero(reinterpret_cast<char*>(m_dev_mem) + size2, size - size2, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_record(m_event, m_stream));

  int has_occured = 0;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_query(m_event, &has_occured));
  if (0 == has_occured) {
    LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_synchronize(m_event));
  }

  test_internal::check(&ok, m_host_mem, size, pattern_a);
  LIBXSTREAM_CHECK_CONDITION_RETURN(ok);

  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memcpy_d2h(m_dev_mem, m_host_mem, size2, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_memcpy_d2h(reinterpret_cast<const char*>(m_dev_mem) + size2, reinterpret_cast<char*>(m_host_mem) + size2, size - size2, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_record(m_event, m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_stream_sync(m_stream));

  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_query(m_event, &has_occured));
  LIBXSTREAM_CHECK_CONDITION_RETURN(0 != has_occured);

  test_internal::check(&ok, m_host_mem, size, 0);
  LIBXSTREAM_CHECK_CONDITION_RETURN(ok);
}


test_type::~test_type()
{
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_destroy(m_event));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_deallocate(-1, m_host_mem));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_deallocate(m_device, m_dev_mem));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_stream_destroy(m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_destroy_signature(m_signature));
  fprintf(stdout, "TST successfully completed.\n");
}


int main(int argc, char* argv[])
{
  try {
#if defined(_OPENMP)
    const int ntasks = std::max(1 < argc ? std::atoi(argv[1]) : omp_get_max_threads(), 1);
#else
    const int ntasks = 1;
#endif

    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
      throw std::runtime_error("no device found!");
    }

#if defined(_OPENMP)
#   pragma omp parallel for schedule(dynamic,1)
#endif
    for (int i = 0; i < ntasks; ++i) {
      const test_type test(i % ndevices);
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
