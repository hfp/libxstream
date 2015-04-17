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

#define LIBXSTREAM_CAPTURE_TERMINATOR reinterpret_cast<const scheduler_type::value_type>(-1)


namespace libxstream_capture_internal {

class scheduler_type {
public:
  typedef libxstream_queue::value_type value_type;

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
  {
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

  ~scheduler_type() {
    // terminates the background thread
    push(LIBXSTREAM_CAPTURE_TERMINATOR, true);

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

public:
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

  volatile value_type get() {
    volatile value_type item = m_global_queue.get();
    if (0 == item) {
      m_stream = libxstream_stream::schedule(m_stream);
      if (m_stream) {
        libxstream_queue* queue = m_stream->queue();
        item = 0 != queue ? queue->get() : 0;

        if (item) {
          const libxstream_capture_base& work_item = *static_cast<const libxstream_capture_base*>(item);

          if (0 != (work_item.flags() & LIBXSTREAM_CALL_WAIT)) {
            libxstream_queue* q = m_stream->queue(queue); // next/other queue

            while (q) {
              while (!q->empty()) {
                push(q->get(), false);
                q->pop();
              }
              q = m_stream->queue(q);
              if (q == queue) q = 0; // break
            }

            if (!m_global_queue.empty()) { // work was pushed into global queue
              // let worker thread pickup the global queue just next time
              item = 0;
            }
          }
        }
      }
    }
    else {
      m_stream = 0;
    }
    return item;
  }

  void pop() {
    if (m_stream) {
      libxstream_queue *const queue = m_stream->queue();
      LIBXSTREAM_ASSERT(queue);
      queue->pop();
    }
    else {
      m_global_queue.pop();
    }
  }

  void push(libxstream_capture_base& work_item) {
    push(&work_item, 0 != (work_item.flags() & LIBXSTREAM_CALL_WAIT));
  }

private:
  void push(value_type work_item, bool wait) {
    LIBXSTREAM_ASSERT(work_item);
    volatile libxstream_queue::value_type& slot = m_global_queue.allocate_push();
    LIBXSTREAM_ASSERT(0 == slot);
    slot = work_item; // push

    if (wait) {
      while (work_item == slot) {
        this_thread_yield();
      }
    }
  }

#if defined(LIBXSTREAM_STDFEATURES) || defined(__GNUC__)
  static void* run(void* scheduler)
#else
  static DWORD WINAPI run(_In_ LPVOID scheduler)
#endif
  {
    scheduler_type& s = *static_cast<scheduler_type*>(scheduler);
    volatile scheduler_type::value_type item = 0;
    bool continue_run = true;

#if defined(LIBXSTREAM_ASYNCHOST) && (201307 <= _OPENMP)
#   pragma omp parallel
#   pragma omp master
#endif
    for (; continue_run;) {
      size_t cycle = 0;
      while (0 == (item = s.get())) {
        if ((LIBXSTREAM_WAIT_ACTIVE_CYCLES) > cycle) {
          this_thread_yield();
          ++cycle;
        }
        else {
          this_thread_sleep();
        }
      }

      if ((LIBXSTREAM_CAPTURE_TERMINATOR) != item) {
        libxstream_capture_base *const work_item = static_cast<libxstream_capture_base*>(item);

        (*work_item)();
#if defined(LIBXSTREAM_ASYNCHOST) && (201307 <= _OPENMP)
#       pragma omp taskwait
#endif
        delete work_item;
        s.pop();
      }
      else {
        while (0 != s.get()) s.pop(); // flush
        continue_run = false;
      }
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
#if defined(LIBXSTREAM_SYNCHRONOUS) && 2 == (LIBXSTREAM_SYNCHRONOUS)
};
#else
};
static/*IPO*/ scheduler_type scheduler;
#endif

} // namespace libxstream_capture_internal


libxstream_capture_base::libxstream_capture_base(size_t argc, const arg_type argv[], libxstream_stream* stream, int flags)
  : m_function(0)
  , m_stream(stream)
#if defined(LIBXSTREAM_SYNCHRONOUS) && 1 == (LIBXSTREAM_SYNCHRONOUS)
  , m_flags(flags | LIBXSTREAM_CALL_WAIT)
#else
  , m_flags(flags)
#endif
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
#if defined(LIBXSTREAM_SYNCHRONOUS) && 2 == (LIBXSTREAM_SYNCHRONOUS)
  libxstream_use_sink(&code);
  return LIBXSTREAM_ERROR_NONE;
#else
  return libxstream_capture_internal::scheduler.status(code);
#endif
}


void libxstream_enqueue(libxstream_capture_base& work_item, bool clone)
{
  libxstream_capture_base *const item = clone ? work_item.clone() : &work_item;
#if defined(LIBXSTREAM_SYNCHRONOUS) && 2 == (LIBXSTREAM_SYNCHRONOUS)
  (*item)();
  delete item;
#else
  libxstream_capture_internal::scheduler.push(*item);
#endif
}


void libxstream_enqueue(const libxstream_capture_base& work_item)
{
  libxstream_enqueue(*work_item.clone(), false);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
