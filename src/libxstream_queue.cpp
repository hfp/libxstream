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
#include "libxstream_queue.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <cstdio>
#if defined(LIBXSTREAM_STDFEATURES)
# include <atomic>
#endif
#include <libxstream_end.h>


libxstream_queue::libxstream_queue()
  : m_index(0)
#if defined(LIBXSTREAM_STDFEATURES)
  , m_size(new std::atomic<size_t>(0))
#else
  , m_size(new size_t(0))
#endif
{
  std::fill_n(m_buffer, LIBXSTREAM_MAX_QSIZE, entry_type(0, *this));
}


libxstream_queue::~libxstream_queue()
{
#if defined(LIBXSTREAM_DEBUG)
  size_t dangling = 0;
  for (size_t i = 0; i < LIBXSTREAM_MAX_QSIZE; ++i) {
    dangling += 0 != m_buffer[i].item() ? 1 : 0;
  }
  if (0 < dangling) {
    LIBXSTREAM_PRINT(1, "%lu work item%s dangling!", static_cast<unsigned long>(dangling), 1 < dangling ? "s are" : " is");
  }
#endif
#if defined(LIBXSTREAM_STDFEATURES)
  delete static_cast<std::atomic<size_t>*>(m_size);
#else
  delete static_cast<size_t*>(m_size);
#endif
}


size_t libxstream_queue::size() const
{
  const size_t index = m_index;
#if defined(LIBXSTREAM_STDFEATURES)
  const size_t offset = *static_cast<const std::atomic<size_t>*>(m_size);
#else
  const size_t offset = *static_cast<const size_t*>(m_size);
#endif
  const_item_type item = m_buffer[LIBXSTREAM_MOD(index, LIBXSTREAM_MAX_QSIZE)].item();
  return 0 != item ? (offset - index) : (std::max<size_t>(offset - index, 1) - 1);
}


libxstream_queue::entry_type& libxstream_queue::allocate_entry()
{
  entry_type* result = 0;
#if defined(LIBXSTREAM_STDFEATURES)
  result = m_buffer + LIBXSTREAM_MOD((*static_cast<std::atomic<size_t>*>(m_size))++, LIBXSTREAM_MAX_QSIZE);
#elif defined(_OPENMP)
  size_t size1 = 0;
  size_t& size = *static_cast<size_t*>(m_size);
# if (201107 <= _OPENMP)
#   pragma omp atomic capture
# else
#   pragma omp critical
# endif
  size1 = ++size;
  result = m_buffer + LIBXSTREAM_MOD(size1 - 1, LIBXSTREAM_MAX_QSIZE);
#else // generic
  size_t& size = *static_cast<size_t*>(m_size);
  libxstream_lock *const lock = libxstream_lock_get(this);
  libxstream_lock_acquire(lock);
  result = m_buffer + LIBXSTREAM_MOD(size++, LIBXSTREAM_MAX_QSIZE);
  libxstream_lock_release(lock);
#endif
  LIBXSTREAM_ASSERT(0 != result && result->queue() == this);

  if (0 != result->item()) {
    LIBXSTREAM_PRINT0(1, "queuing work is stalled!");
    do { // stall if capacity is exceeded
      this_thread_sleep();
    }
    while (0 != result->item());
  }

  LIBXSTREAM_ASSERT(0 == result->item());
  return *result;
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
