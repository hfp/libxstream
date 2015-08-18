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
#include "libxstream_workqueue.hpp"
#include "libxstream_workitem.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <cstdio>
#if defined(LIBXSTREAM_STDFEATURES)
# include <atomic>
#endif
#include <libxstream_end.h>


void libxstream_workqueue::entry_type::push(libxstream_workitem& workitem)
{
  delete m_dangling;
  const bool async = 0 == (LIBXSTREAM_CALL_WAIT & workitem.flags());
  const bool witem = 0 == (LIBXSTREAM_CALL_EVENT & workitem.flags());
  libxstream_workitem *const item = async ? workitem.clone() : &workitem;
  m_status = witem ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_NOT_AWORKITEM;
  m_dangling = async ? item : 0;
  m_item = item;
}


int libxstream_workqueue::entry_type::wait(bool any_status) const
{
  int result = LIBXSTREAM_ERROR_NONE;
  const libxstream_workitem* item = m_item;
  bool ok = true;
#if defined(LIBXSTREAM_SLEEP_CLIENT)
  size_t cycle = 0;
#endif

  if (any_status) {
    while (0 != item) {
#if defined(LIBXSTREAM_SLEEP_CLIENT)
      this_thread_wait(cycle);
#else
      this_thread_yield();
#endif
      if (0 == item->stream() || 0 != *item->stream()) {
        item = m_item;
      }
      else {
        ok = false;
        item = 0;
      }
    }
  }
  else {
    while (0 != item || LIBXSTREAM_ERROR_NONE != m_status) {
#if defined(LIBXSTREAM_SLEEP_CLIENT)
      this_thread_wait(cycle);
#else
      this_thread_yield();
#endif
      if (0 == item || 0 == item->stream() || 0 != *item->stream()) {
        item = m_item;
      }
      else {
        ok = false;
        item = 0;
      }
    }
  }

  if (ok && LIBXSTREAM_NOT_AWORKITEM != m_status) {
    result = m_status;
  }

  LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == result);
  return result;
}


void libxstream_workqueue::entry_type::execute()
{
  LIBXSTREAM_ASSERT(0 != m_item && (m_item == m_dangling || 0 == m_dangling));
  (*m_item)(*this);
}


void libxstream_workqueue::entry_type::pop()
{
  if (0 == (LIBXSTREAM_CALL_LOOP & m_item->flags())) {
    m_item = 0;
    LIBXSTREAM_ASSERT(0 != m_queue);
    m_queue->pop();
  }
}


libxstream_workqueue::libxstream_workqueue()
#if defined(LIBXSTREAM_STDFEATURES)
  : m_position(new std::atomic<size_t>(0))
#else
  : m_position(new size_t(0))
#endif
  , m_index(0)
{
  std::fill_n(m_buffer, LIBXSTREAM_MAX_QSIZE, entry_type(this, 0));
}


libxstream_workqueue::~libxstream_workqueue()
{
#if defined(LIBXSTREAM_INTERNAL_TRACE)
  size_t pending = 0;
#endif
  for (size_t i = 0; i < (LIBXSTREAM_MAX_QSIZE); ++i) {
#if defined(LIBXSTREAM_INTERNAL_TRACE)
    pending += 0 == m_buffer[i].item() ? 0 : 1;
#endif
    delete m_buffer[i].dangling();
  }
  LIBXSTREAM_PRINT(0 < pending ? 1 : 0, "workqueue: %lu work item%s pending!", static_cast<unsigned long>(pending), 1 < pending ? "s" : "");

#if defined(LIBXSTREAM_STDFEATURES)
  delete static_cast<std::atomic<size_t>*>(m_position);
#else
  delete static_cast<size_t*>(m_position);
#endif
}


libxstream_workqueue::entry_type& libxstream_workqueue::allocate_entry()
{
  entry_type* result = 0;
#if defined(LIBXSTREAM_STDFEATURES)
  result = m_buffer + LIBXSTREAM_MOD2((*static_cast<std::atomic<size_t>*>(m_position))++, LIBXSTREAM_MAX_QSIZE);
#elif defined(_OPENMP)
  size_t size1 = 0;
  size_t& size = *static_cast<size_t*>(m_position);
# if (201107 <= _OPENMP)
# pragma omp atomic capture
# else
# pragma omp critical
# endif
  size1 = ++size;
  result = m_buffer + LIBXSTREAM_MOD2(size1 - 1, LIBXSTREAM_MAX_QSIZE);
#else // generic
  size_t& size = *static_cast<size_t*>(m_position);
  libxstream_lock *const lock = libxstream_lock_get(this);
  libxstream_lock_acquire(lock);
  result = m_buffer + LIBXSTREAM_MOD2(size++, LIBXSTREAM_MAX_QSIZE);
  libxstream_lock_release(lock);
#endif
  LIBXSTREAM_ASSERT(0 != result && result->queue() == this);

  if (0 != result->item()) {
    LIBXSTREAM_PRINT0(1, "workqueue: queuing work is stalled!");
    do { // stall if capacity is exceeded
      this_thread_sleep();
    }
    while (0 != result->item());
  }

  return *result;
}


size_t libxstream_workqueue::position() const
{
#if defined(LIBXSTREAM_STDFEATURES)
  const size_t atomic_position = *static_cast<const std::atomic<size_t>*>(m_position);
#else
  const size_t atomic_position = *static_cast<const size_t*>(m_position);
#endif
  return atomic_position;
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
