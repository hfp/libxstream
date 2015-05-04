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
#ifndef LIBXSTREAM_QUEUE_HPP
#define LIBXSTREAM_QUEUE_HPP

#include "libxstream.hpp"

#if defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)


class libxstream_queue {
public:
  typedef void* item_type;
  typedef const void* const_item_type;

  class entry_type {
  public:
    entry_type(libxstream_queue* queue = 0, item_type item = 0): m_queue(queue), m_dangling(0), m_item(item) {}
    entry_type(libxstream_queue& queue): m_queue(&queue), m_dangling(0), m_item(reinterpret_cast<item_type>(-1)) {}

  public:
    bool valid() const { return reinterpret_cast<item_type>(-1) != m_item; }
    const libxstream_queue* queue() const { return m_queue; }
    item_type item() { return m_item; }
    const_item_type item() const { return m_item; }
    const_item_type push(item_type item, bool dangling) {
      const const_item_type result = m_dangling;
      m_dangling = dangling ? item : 0;
      m_item = item;
      return result;
    }
    void pop() {
      m_item = 0;
      LIBXSTREAM_ASSERT(0 != m_queue);
      m_queue->pop();
    }
    void wait() {
#if defined(LIBXSTREAM_SLEEP_CLIENT)
      size_t cycle = 0;
      while (0 != m_item) this_thread_wait(cycle);
#else
      while (0 != m_item) this_thread_yield();
#endif
    }

  private:
    libxstream_queue* m_queue;
    const_item_type m_dangling;
    volatile item_type m_item;
  };

public:
  libxstream_queue();
  ~libxstream_queue();

public:
  size_t size() const;

  entry_type& allocate_entry();

  entry_type& get() { // not thread-safe!
    return m_buffer[LIBXSTREAM_MOD(m_index, LIBXSTREAM_MAX_QSIZE)];
  }

  entry_type get() const { // not thread-safe!
    return m_buffer[LIBXSTREAM_MOD(m_index, LIBXSTREAM_MAX_QSIZE)];
  }

  void pop() { // not thread-safe!
    ++m_index;
  }

private:
  entry_type m_buffer[LIBXSTREAM_MAX_QSIZE];
  size_t m_index;
  void* m_size;
};

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
#endif // LIBXSTREAM_QUEUE_HPP
