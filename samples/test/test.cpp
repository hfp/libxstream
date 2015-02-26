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
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>


namespace test_internal {

LIBXSTREAM_TARGET(mic) void check(libxstream_bool* result, const void* buffer, size_t size, char pattern)
{
  size_t arg = 0;
  bool ok = true;
  // check function is called with using LIBXSTREAM hence introspection may not be available
  if (LIBXSTREAM_ERROR_NONE == libxstream_get_argument(buffer, &arg)) {
    size_t value = 0;
    ok = 1 == arg && LIBXSTREAM_ERROR_NONE == libxstream_get_shape(0, arg, &value) && size == value;
    ok = ok && LIBXSTREAM_ERROR_NONE == libxstream_get_dims(0, 0, &value) && 0 == value;
    ok = ok && LIBXSTREAM_ERROR_NONE == libxstream_get_dims(0, 1, &value) && 1 == value;
    ok = ok && LIBXSTREAM_ERROR_NONE == libxstream_get_dims(0, 2, &value) && 0 == value;
  }

  const char *const values = reinterpret_cast<const char*>(buffer);
  for (size_t i = 0; i < size && ok; ++i) {
    ok = pattern == values[i];
  }
  LIBXSTREAM_ASSERT(result);
  *result = ok ? LIBXSTREAM_TRUE : LIBXSTREAM_FALSE;
}

} // namespace test_internal


test_type::test_type(int device)
  : m_signature(0), m_stream(0), m_event(0)
  , m_host_mem(0), m_dev_mem1(0), m_dev_mem2(0)
{
  size_t mem_free = 0, mem_avail = 0;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_info(device, &mem_free, &mem_avail));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(&m_stream, device, 0, 0, 0));

  const size_t size = 4711u * 1024u;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, &m_host_mem, size, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(device, &m_dev_mem1, size, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(device, &m_dev_mem2, size, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_info(device, &mem_free, &mem_avail));

  const void* real = 0;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_pointer(-1, m_host_mem, &real));
  LIBXSTREAM_CHECK_CONDITION_THROW(m_host_mem == real);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_pointer(device, m_dev_mem1, &real));
#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD)
  LIBXSTREAM_CHECK_CONDITION_THROW(m_dev_mem1 != real);
#endif
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_pointer(device, m_dev_mem2, &real));
#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD)
  LIBXSTREAM_CHECK_CONDITION_THROW(m_dev_mem2 != real);
#endif

  const char pattern_a = 'a', pattern_b = 'b';
  LIBXSTREAM_ASSERT(pattern_a != pattern_b);
  std::fill_n(reinterpret_cast<char*>(m_host_mem), size, pattern_a);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memcpy_h2d(m_host_mem, m_dev_mem1, size, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memcpy_d2d(m_dev_mem1, m_dev_mem2, size, m_stream));

  libxstream_bool ok = LIBXSTREAM_FALSE;
  size_t nargs = 0, arity = 0, argsize = 0;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_create_signature(&m_signature, 4));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_nargs(m_signature, &nargs));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_arity(m_signature, &arity));
  LIBXSTREAM_CHECK_CONDITION_THROW(4 == nargs && 0 == arity);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_output(m_signature, 0, &ok, libxstream_type2value<libxstream_bool>::value(), 0, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_nargs(m_signature, &nargs));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_arity(m_signature, &arity));
  LIBXSTREAM_CHECK_CONDITION_THROW(4 == nargs && 1 == arity);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_elemsize(m_signature, 0, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(sizeof(libxstream_bool) == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_datasize(m_signature, 0, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(sizeof(libxstream_bool) == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_input (m_signature, 1, m_dev_mem1, LIBXSTREAM_TYPE_VOID, 1, &size));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_nargs(m_signature, &nargs));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_arity(m_signature, &arity));
  LIBXSTREAM_CHECK_CONDITION_THROW(4 == nargs && 2 == arity);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_elemsize(m_signature, 1, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(1 == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_datasize(m_signature, 1, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(size == argsize);
  // for testing purpose the following argument is weak-typed instead of (..., libxstream_type2value<size_t>::value(), 0, 0)
  argsize = sizeof(size_t);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_input (m_signature, 2, &size, LIBXSTREAM_TYPE_VOID, 0, &argsize));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_nargs(m_signature, &nargs));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_arity(m_signature, &arity));
  LIBXSTREAM_CHECK_CONDITION_THROW(4 == nargs && 3 == arity);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_elemsize(m_signature, 2, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(sizeof(size_t) == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_datasize(m_signature, 2, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(sizeof(size_t) == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_input (m_signature, 3, &pattern_a, libxstream_type2value<char>::value(), 0, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_nargs(m_signature, &nargs));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_arity(m_signature, &arity));
  LIBXSTREAM_CHECK_CONDITION_THROW(4 == nargs && 4 == arity);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_elemsize(m_signature, 3, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(1 == argsize);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_get_datasize(m_signature, 3, &argsize));
  LIBXSTREAM_CHECK_CONDITION_THROW(1 == argsize);
  const libxstream_function function = reinterpret_cast<libxstream_function>(test_internal::check);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_fn_call(function, m_signature, m_stream, LIBXSTREAM_CALL_DEFAULT));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_create(&m_event));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_record(m_event, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_synchronize(m_event));
  LIBXSTREAM_CHECK_CONDITION_THROW(LIBXSTREAM_FALSE != ok);

  std::fill_n(reinterpret_cast<char*>(m_host_mem), size, pattern_b);
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memcpy_d2h(m_dev_mem2, m_host_mem, size, m_stream));

  const size_t size2 = size / 2;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memset_zero(m_dev_mem1, size2, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memset_zero(reinterpret_cast<char*>(m_dev_mem1) + size2, size - size2, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_record(m_event, m_stream));

  int has_occured = 0;
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_query(m_event, &has_occured));
  if (0 == has_occured) {
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_synchronize(m_event));
  }

  test_internal::check(&ok, m_host_mem, size, pattern_a);
  LIBXSTREAM_CHECK_CONDITION_THROW(LIBXSTREAM_FALSE != ok);

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memcpy_d2h(m_dev_mem1, m_host_mem, size2, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_memcpy_d2h(reinterpret_cast<const char*>(m_dev_mem1) + size2, reinterpret_cast<char*>(m_host_mem) + size2, size - size2, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_record(m_event, m_stream));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(m_stream));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_event_query(m_event, &has_occured));
  LIBXSTREAM_CHECK_CONDITION_THROW(0 != has_occured);

  test_internal::check(&ok, m_host_mem, size, 0);
  LIBXSTREAM_CHECK_CONDITION_THROW(LIBXSTREAM_FALSE != ok);
}


test_type::~test_type()
{
  int device = -1;
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_stream_device(m_stream, &device));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_deallocate(-1, m_host_mem));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_deallocate(device, m_dev_mem1));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_mem_deallocate(device, m_dev_mem2));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_fn_destroy_signature(m_signature));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_stream_destroy(m_stream));
  LIBXSTREAM_CHECK_CALL_RETURN(libxstream_event_destroy(m_event));
  fprintf(stdout, "TST successfully completed.\n");
}


int main(int argc, char* argv[])
{
  try {
#if defined(_OPENMP)
    const int ntasks = std::max(1 < argc ? std::atoi(argv[1]) : omp_get_max_threads(), 1);
#else
    const int ntasks = std::max(1 < argc ? std::atoi(argv[1]) : 1, 1);
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
