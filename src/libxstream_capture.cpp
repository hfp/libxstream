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
#include "libxstream_capture.hpp"
#include "libxstream_queue.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <cstdio>
#if defined(LIBXSTREAM_STDFEATURES)
# include <thread>
# include <atomic>
#else
# if defined(__GNUC__)
#   include <pthread.h>
# else
#   include <Windows.h>
# endif
#endif
#include <libxstream_end.h>


namespace libxstream_capture_internal {

class scheduler_type {
public:
  typedef libxstream_queue::entry_type entry_type;
  typedef libxstream_queue::item_type item_type;

public:
  scheduler_type()
    : m_global_queue()
    , m_stream(0)
    , m_status(LIBXSTREAM_ERROR_NONE)
#if defined(LIBXSTREAM_STDFEATURES)
    , m_thread() // do not start here
#else
    , m_thread(0)
#endif
  {}

  ~scheduler_type() {
    if (running()) {
      // terminates the background thread
      push(entry_type(&m_global_queue).item(), true);

#if defined(LIBXSTREAM_STDFEATURES)
      m_thread.detach();
#else
# if defined(__GNUC__)
      pthread_detach(m_thread);
# else
      CloseHandle(m_thread);
# endif
#endif
    }
  }

public:
  bool running() const {
#if defined(LIBXSTREAM_STDFEATURES)
    return m_thread.joinable();
#else
    return 0 != m_thread;
#endif
  }

  void start() {
    if (!running()) {
      libxstream_lock *const lock = libxstream_lock_get(this);
      libxstream_lock_acquire(lock);

      if (!running()) {
#if defined(LIBXSTREAM_STDFEATURES)
        std::thread(run, this).swap(m_thread);
#else
# if defined(__GNUC__)
        pthread_create(&m_thread, 0, run, this);
# else
        m_thread = CreateThread(0, 0, run, this, 0, 0);
# endif
#endif
      }

      libxstream_lock_release(lock);
    }
  }

  int status(int code) {
#if defined(LIBXSTREAM_STDFEATURES)
    return std::atomic_exchange(&m_status, code);
#elif defined(_OPENMP)
    int result = 0;
#   pragma omp critical
    {
      result = m_status;
      m_status = code;
    }
    return result;
#else // generic
    int result = 0;
    libxstream_lock *const lock = libxstream_lock_get(this);
    libxstream_lock_acquire(lock);
    result = m_status;
    m_status = code;
    libxstream_lock_release(lock);
    return result;
#endif
  }

  entry_type& get() {
    entry_type* result = &m_global_queue.get();

    if (0 == result->item()) { // no item in global queue
      m_stream = libxstream_stream::schedule(m_stream);
      libxstream_queue* queue = m_stream ? m_stream->queue_begin() : 0;
      const libxstream_capture_base* item = 0;

      if (queue) {
        result = &queue->get();
        item = static_cast<const libxstream_capture_base*>(result->item());
      }

      // item in stream-local queue is a wait-item
      if (item && 0 != (item->flags() & LIBXSTREAM_CALL_WAIT)) {
        queue = m_stream->queue_next(); // next/other queue

        while (0 != queue) {
          entry_type& i = queue->get();

          if (0 != i.item()) {
            result = &i;
            queue = 0; // break
          }
          else {
            queue = m_stream->queue_next();
          }
        }
      }
    }

    return *result;
  }

  void push(libxstream_capture_base& work_item) {
    push(&work_item, 0 != (work_item.flags() & LIBXSTREAM_CALL_WAIT));
  }

private:
  void push(item_type work_item, bool wait) {
    LIBXSTREAM_ASSERT(0 != work_item);
    entry_type& entry = m_global_queue.allocate_entry();
    entry.push(work_item);

#if !defined(LIBXSTREAM_WAIT)
    if (wait)
#else
    libxstream_use_sink(&wait);
#endif
    {
      entry.wait();
    }
  }

#if defined(LIBXSTREAM_STDFEATURES) || defined(__GNUC__)
  static void* run(void* scheduler)
#else
  static DWORD WINAPI run(_In_ LPVOID scheduler)
#endif
  {
    scheduler_type& s = *static_cast<scheduler_type*>(scheduler);
    bool continue_run = true;

#if defined(LIBXSTREAM_ASYNCHOST) && (201307 <= _OPENMP)
#   pragma omp parallel
#   pragma omp master
#endif
    for (; continue_run;) {
      scheduler_type::entry_type* entry = &s.get();
      scheduler_type::item_type item = entry->item();
      bool valid = entry->valid();
      size_t cycle = 0;

      while (0 == item && valid) {
        this_thread_wait(cycle);
        entry = &s.get();
        item = entry->item();
        valid = entry->valid();
      }

      if (valid) {
        libxstream_capture_base *const work_item = static_cast<libxstream_capture_base*>(item);

        (*work_item)();
#if defined(LIBXSTREAM_ASYNCHOST) && (201307 <= _OPENMP)
#       pragma omp taskwait
#endif
        delete work_item;
      }
      else {
        continue_run = false;
      }

      entry->pop();
    }

#if defined(LIBXSTREAM_STDFEATURES) || defined(__GNUC__)
    return scheduler;
#else
    return EXIT_SUCCESS;
#endif
  }

private:
  libxstream_queue m_global_queue;
  libxstream_stream* m_stream;
#if defined(LIBXSTREAM_STDFEATURES)
  std::atomic<int> m_status;
  std::thread m_thread;
#elif defined(__GNUC__)
  int m_status;
  pthread_t m_thread;
#else
  int m_status;
  HANDLE m_thread;
#endif
};
static/*IPO*/ scheduler_type scheduler;

} // namespace libxstream_capture_internal


libxstream_capture_base::libxstream_capture_base(size_t argc, const arg_type argv[], libxstream_stream* stream, int flags)
  : m_function(0)
  , m_stream(stream)
  , m_flags(flags)
  , m_thread(this_thread_id())
{
  if (2 == argc && (argv[0].signature() || argv[1].signature())) {
    const libxstream_argument* signature = 0;
    if (argv[1].signature()) {
      m_function = *reinterpret_cast<const libxstream_function*>(argv + 0);
      signature = static_cast<const libxstream_argument*>(libxstream_get_value(argv[1]).const_pointer);
    }
    else {
      LIBXSTREAM_ASSERT(argv[0].signature());
      m_function = *reinterpret_cast<const libxstream_function*>(argv + 1);
      signature = static_cast<const libxstream_argument*>(libxstream_get_value(argv[0]).const_pointer);
    }

    size_t arity = 0;
    if (signature) {
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_get_arity(signature, &arity));
      LIBXSTREAM_PRAGMA_LOOP_COUNT(0, LIBXSTREAM_MAX_NARGS, LIBXSTREAM_MAX_NARGS/2)
      for (size_t i = 0; i < arity; ++i) m_signature[i] = signature[i];
    }
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(m_signature, arity, libxstream_argument::kind_invalid, 0, LIBXSTREAM_TYPE_INVALID, 0, 0));
  }
  else {
    LIBXSTREAM_ASSERT(argc <= (LIBXSTREAM_MAX_NARGS));
    for (size_t i = 0; i < argc; ++i) m_signature[i] = argv[i];
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(m_signature, argc, libxstream_argument::kind_invalid, 0, LIBXSTREAM_TYPE_INVALID, 0, 0));
#if defined(LIBXSTREAM_DEBUG)
    size_t arity = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_arity(m_signature, &arity) && arity == argc);
#endif
  }

  libxstream_capture_internal::scheduler.start();
}


libxstream_capture_base* libxstream_capture_base::clone() const
{
  libxstream_capture_base *const instance = virtual_clone();
  return instance;
}


void libxstream_capture_base::operator()()
{
  virtual_run();
}


int libxstream_capture_base::status(int code)
{
  return libxstream_capture_internal::scheduler.status(code);
}


void libxstream_enqueue(libxstream_capture_base& work_item, bool clone)
{
  libxstream_capture_base *const item = clone ? work_item.clone() : &work_item;
  libxstream_capture_internal::scheduler.push(*item);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
