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

#if defined(LIBXSTREAM_STDFEATURES)
# include <thread>
# include <atomic>
#else
// TODO: support Windows threads
# include <pthread.h>
#endif

#define LIBXSTREAM_CAPTURE_USE_QUEUE


namespace libxstream_offload_internal {

class queue_type {
public:
  typedef const libxstream_offload_region* value_type;
  queue_type()
    : m_lock(libxstream_lock_create())
    , m_terminated(false)
    , m_index(0)
#if defined(LIBXSTREAM_STDFEATURES)
    , m_thread(run, this)
#else
    , m_thread(0)
#endif
    , m_size(0)
  {
    std::fill_n(m_buffer, LIBXSTREAM_MAX_QSIZE, static_cast<value_type>(0));
#if !defined(LIBXSTREAM_STDFEATURES)
    start();
#endif
  }

  ~queue_type() {
    terminate();
    libxstream_lock_destroy(m_lock);
  }

public:
  bool start() {
    if (!m_terminated) {
#if defined(LIBXSTREAM_STDFEATURES)
      if (!m_thread.joinable()) {
        libxstream_lock_acquire(m_lock);
        if (!m_thread.joinable()) {
          std::thread(run, this).swap(m_thread);
        }
        libxstream_lock_release(m_lock);
      }
#else
      if (0 == m_thread) {
        libxstream_lock_acquire(m_lock);
        if (0 == m_thread) {
          pthread_create(&m_thread, 0, run, this);
        }
        libxstream_lock_release(m_lock);
      }
#endif
    }

    return !m_terminated;
  }

  void terminate() {
    push(terminator, false); // terminates the background thread

#if defined(LIBXSTREAM_DEBUG)
    size_t dangling = 0;
    for (size_t i = 0; i < LIBXSTREAM_MAX_QSIZE; ++i) {
      const value_type item = m_buffer[i];
      if (0 != item && terminator != item) {
        m_buffer[i] = 0;
        ++dangling;
        delete item;
      }
    }
    if (0 < dangling) {
      LIBXSTREAM_PRINT_WARNING("%lu work item%s dangling!", static_cast<unsigned long>(dangling), 1 < dangling ? "s are" : " is");
    }
#endif

#if defined(LIBXSTREAM_STDFEATURES)
    if (m_thread.joinable()) {
      m_thread.join();
    }
#else
    if (0 != m_thread) {
      pthread_join(m_thread, 0);
      m_thread = 0;
    }
#endif

    m_terminated = true;
  }

  bool empty() const {
    return 0 == get();
  }

  size_t size() const {
    const size_t offset = m_size, index = m_index;
    const value_type& entry = m_buffer[offset%LIBXSTREAM_MAX_QSIZE];
    return 0 != entry ? (offset - index) : (std::max<size_t>(offset - index, 1) - 1);
  }

  void push(const libxstream_offload_region& offload_region, bool wait = true) {
    push(&offload_region, wait);
  }

  value_type get() const { // not thread-safe!
    return m_buffer[m_index%LIBXSTREAM_MAX_QSIZE];
  }

  void pop() { // not thread-safe!
    LIBXSTREAM_ASSERT(!empty());
    m_buffer[m_index%LIBXSTREAM_MAX_QSIZE] = 0;
    ++m_index;
  }

private:
  void push(const value_type& offload_region, bool wait) {
    LIBXSTREAM_ASSERT(0 != offload_region);
    value_type* entry = 0;
#if defined(LIBXSTREAM_STDFEATURES)
    entry = m_buffer + (m_size++ % LIBXSTREAM_MAX_QSIZE);
#else
# if defined(_OPENMP)
#   pragma omp critical
    entry = m_buffer + (m_size++ % LIBXSTREAM_MAX_QSIZE);
# else
    libxstream_lock_acquire(m_lock);
    entry = m_buffer + (m_size++ % LIBXSTREAM_MAX_QSIZE);
    libxstream_lock_release(m_lock);
# endif
#endif
    LIBXSTREAM_ASSERT(0 != entry);

#if defined(LIBXSTREAM_DEBUG)
    if (0 != *entry) {
      LIBXSTREAM_PRINT_WARNING0("queuing work is stalled!");
    }
#endif
    // stall the push if LIBXSTREAM_MAX_QSIZE is exceeded
    while (0 != *entry) {
      this_thread_yield();
    }

    LIBXSTREAM_ASSERT(0 == *entry);
    *entry = terminator != offload_region ? offload_region->clone() : terminator;

    if (wait) {
      while (0 != *entry) {
        this_thread_yield();
      }
    }
  }

  static void* run(void* queue) {
    queue_type& q = *static_cast<queue_type*>(queue);
    value_type offload_region = 0;

    for (;;) {
      while (0 == (offload_region = q.get())) {
        this_thread_yield();
      }

      if (terminator != offload_region) {
        LIBXSTREAM_ASSERT(terminator != offload_region);
        (*offload_region)();
        delete offload_region;
        LIBXSTREAM_ASSERT(terminator != offload_region);
        q.pop();
      }
      else {
        LIBXSTREAM_ASSERT(terminator == offload_region);
        q.pop();
        break;
      }
    }

    return queue;
  }

private:
  static const value_type terminator;
  value_type m_buffer[LIBXSTREAM_MAX_QSIZE];
  libxstream_lock* m_lock;
  bool m_terminated;
  size_t m_index;
#if defined(LIBXSTREAM_STDFEATURES)
  std::thread m_thread;
  std::atomic<size_t> m_size;
#else
  pthread_t m_thread;
  size_t m_size;
#endif
#if defined(LIBXSTREAM_CAPTURE_USE_QUEUE)
} queue;
#else
};
#endif
/*static*/ const queue_type::value_type queue_type::terminator = reinterpret_cast<queue_type::value_type>(-1);

} // namespace libxstream_offload_internal


libxstream_offload_region::libxstream_offload_region(libxstream_stream* stream, size_t argc, const arg_type argv[])
  : m_stream(stream)
#if defined(LIBXSTREAM_DEBUG)
  , m_argc(argc)
#endif
{
  LIBXSTREAM_ASSERT(argc <= LIBXSTREAM_MAX_NARGS);
  for (size_t i = 0; i < argc; ++i) {
    m_argv[i] = argv[i];
  }
}


void libxstream_offload(const libxstream_offload_region& offload_region, bool wait)
{
#if defined(LIBXSTREAM_CAPTURE_USE_QUEUE)
  if (libxstream_offload_internal::queue.start()) {
    libxstream_offload_internal::queue.push(offload_region, wait);
  }
#else
  offload_region();
#endif
}


void libxstream_offload_shutdown()
{
#if defined(LIBXSTREAM_CAPTURE_USE_QUEUE)
  libxstream_offload_internal::queue.terminate();
#endif
}


bool libxstream_offload_busy()
{
#if defined(LIBXSTREAM_CAPTURE_USE_QUEUE)
  //return 0 < libxstream_offload_internal::queue.size();
  return !libxstream_offload_internal::queue.empty();
#else
  return false;
#endif
}
