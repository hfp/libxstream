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
#include <atomic>

#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
# include <pthread.h>
#else
# include <thread>
#endif

#define LIBXSTREAM_CAPTURE_USE_QUEUE


namespace libxstream_offload_internal {

class queue_type {
public:
  typedef const libxstream_offload_region* value_type;
  queue_type()
    : m_index(0)
    , m_size(0)
#if defined(LIBXSTREAM_DEBUG)
    , m_max_size(0)
    , m_npush(0)
    , m_npop(0)
#endif
    , m_terminated(false)
#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
    , m_thread(0)
#else
    , m_thread(run, this)
#endif
    , m_lock(libxstream_lock_create())
  {
    std::fill_n(m_buffer, LIBXSTREAM_MAX_QSIZE, static_cast<value_type>(0));
#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
    start();
#endif
  }

  ~queue_type() {
    terminate();
    libxstream_lock_destroy(m_lock);
#if defined(LIBXSTREAM_DEBUG)
    fprintf(stderr, "\tqueue: size=%lu pushes=%lu pops=%lu\n",
      static_cast<unsigned long>(m_max_size),
      static_cast<unsigned long>(m_npush),
      static_cast<unsigned long>(m_npop));
#endif
  }

public:
  bool start() {
    if (!m_terminated) {
#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
      if (0 == m_thread) {
        libxstream_lock_acquire(m_lock);
        if (0 == m_thread) {
          pthread_create(&m_thread, 0, run, this);
        }
        libxstream_lock_release(m_lock);
      }
#else
      if (!m_thread.joinable()) {
        libxstream_lock_acquire(m_lock);
        if (!m_thread.joinable()) {
          std::thread(run, this).swap(m_thread);
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
    for (size_t i = 0; i < LIBXSTREAM_MAX_QSIZE; ++i) {
      const value_type item = m_buffer[i];
      if (0 != item && terminator != item) {
        m_buffer[i] = 0;
        fprintf(stderr, "\tdangling work item in queue!\n");
        delete item;
      }
    }
#endif
#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
    if (0 != m_thread) {
      pthread_join(m_thread, 0);
      m_thread = 0;
    }
#else
    if (m_thread.joinable()) {
      m_thread.join();
    }
#endif
    m_terminated = true;
  }

  bool empty() const {
    return 0 == get();
  }

#if defined(LIBXSTREAM_DEBUG)
  size_t size() const {
    const size_t offset = m_size, index = m_index;
    const storage_type& entry = m_buffer[offset%LIBXSTREAM_MAX_QSIZE];
    return 0 != static_cast<value_type>(entry) ? (offset - index) : (std::max<size_t>(offset - index, 1) - 1);
  }
#endif

  void push(const libxstream_offload_region& offload_region, bool wait = true) {
    push(&offload_region, wait);
  }

  value_type get() const { // not thread-safe!
    return m_buffer[m_index%LIBXSTREAM_MAX_QSIZE];
  }

  void pop() { // not thread-safe!
    LIBXSTREAM_ASSERT(!empty());
#if defined(LIBXSTREAM_DEBUG)
    ++m_npop;
#endif
    m_buffer[m_index%LIBXSTREAM_MAX_QSIZE] = 0;
    ++m_index;
  }

private:
  void yield() {
    this_thread_yield();
#if defined(LIBXSTREAM_DEBUG)
    m_max_size = std::max<size_t>(m_max_size, size());
#endif
  }

  void push(const value_type& offload_region, bool wait) {
#if defined(LIBXSTREAM_DEBUG)
    ++m_npush;
#endif
    LIBXSTREAM_ASSERT(0 != offload_region);
    storage_type& entry = m_buffer[m_size++%LIBXSTREAM_MAX_QSIZE];

#if defined(LIBXSTREAM_DEBUG)
    if (0 != static_cast<value_type>(entry)) {
      fprintf(stderr, "\tqueuing work is stalled!\n");
    }
#endif
    // stall the push if LIBXSTREAM_MAX_QSIZE is exceeded
    while (0 != static_cast<value_type>(entry)) {
      yield();
    }

    LIBXSTREAM_ASSERT(0 == static_cast<value_type>(entry));
    entry = terminator != offload_region ? offload_region->clone() : terminator;

    if (wait) {
      while (0 != static_cast<value_type>(entry)) {
        yield();
      }
    }
  }

  static void* run(void* queue) {
    queue_type& q = *static_cast<queue_type*>(queue);
    value_type offload_region = 0;

    for (;;) {
      while (0 == (offload_region = q.get())) {
        q.yield();
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
#if defined(LIBXSTREAM_DEBUG)
  // avoid some false positives when detecting data races
  typedef std::atomic<value_type> storage_type;
  typedef std::atomic<size_t> counter_type;
#else // certain items are not meant to be thread-safe
  typedef value_type storage_type;
  typedef size_t counter_type;
#endif
  storage_type m_buffer[LIBXSTREAM_MAX_QSIZE];
  counter_type m_index;
  std::atomic<size_t> m_size;
#if defined(LIBXSTREAM_DEBUG)
  std::atomic<size_t> m_max_size;
  std::atomic<size_t> m_npush;
  std::atomic<size_t> m_npop;
#endif
#if (defined(LIBXSTREAM_THREAD_API) && (1 == (2*LIBXSTREAM_THREAD_API+1)/2) || !defined(LIBXSTREAM_STDTHREAD)) && !defined(_MSC_VER)
  typedef pthread_t thread_type;
#else
  typedef std::thread thread_type;
#endif
  bool m_terminated;
  thread_type m_thread;
  libxstream_lock* m_lock;
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
#if defined(LIBXSTREAM_DEBUG)
  if (!wait) {
    libxstream_stream *const stream = offload_region.stream();
    if (stream && stream->thread_id() != this_thread_id())  {
      fprintf(stderr, "\tstream is accessed by multiple threads!\n");
    }
  }
#endif
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
