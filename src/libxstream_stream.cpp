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
#include "libxstream_stream.hpp"
#include "libxstream_capture.hpp"
#include "libxstream_event.hpp"
#include "libxstream_queue.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <string>
#include <cstdio>
#if defined(LIBXSTREAM_STDFEATURES)
# include <atomic>
#endif
#include <libxstream_end.h>

// allows to wait for an event issued prior to the pending signal
//#define LIBXSTREAM_STREAM_WAIT_PAST
// check whether a signal is really pending; update internal state
#define LIBXSTREAM_STREAM_CHECK_PENDING


namespace libxstream_stream_internal {

static/*IPO*/ class registry_type {
public:
  typedef libxstream_stream* value_type;

public:
  registry_type()
    : m_istreams(0)
  {
    std::fill_n(m_signals, LIBXSTREAM_MAX_NDEVICES, 0);
    std::fill_n(m_streams, LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS, static_cast<value_type>(0));
  }

  ~registry_type() {
    const size_t n = max_nstreams();
    for (size_t i = 0; i < n; ++i) {
#if defined(LIBXSTREAM_DEBUG)
      if (0 != m_streams[i]) {
        LIBXSTREAM_PRINT(1, "dangling stream \"%s\"!", m_streams[i]->name());
      }
#endif
      libxstream_stream_destroy(m_streams[i]);
    }
  }

public:
  volatile value_type& allocate() {
#if !defined(LIBXSTREAM_STDFEATURES)
    libxstream_lock *const lock = libxstream_lock_get(this);
    libxstream_lock_acquire(lock);
#endif
    volatile value_type* i = m_streams + LIBXSTREAM_MOD(m_istreams++, (LIBXSTREAM_MAX_NDEVICES) * (LIBXSTREAM_MAX_NSTREAMS));
    while (0 != *i) i = m_streams + LIBXSTREAM_MOD(m_istreams++, (LIBXSTREAM_MAX_NDEVICES) * (LIBXSTREAM_MAX_NSTREAMS));
#if !defined(LIBXSTREAM_STDFEATURES)
    libxstream_lock_release(lock);
#endif
    return *i;
  }

  size_t max_nstreams() const {
    return std::min<size_t>(m_istreams, LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS);
  }

  size_t nstreams(int device) const {
    const size_t n = max_nstreams();
    size_t result = 0;
    for (size_t i = 0; i < n; ++i) {
      result += (0 != m_streams[i] && m_streams[i]->device() == device) ? 1 : 0;
    }
    return result;
  }

  size_t nstreams() const {
    const size_t n = max_nstreams();
    size_t result = 0;
    for (size_t i = 0; i < n; ++i) {
      result += 0 != m_streams[i] ? 1 : 0;
    }
    return result;
  }

  libxstream_signal& signal(int device) {
    LIBXSTREAM_ASSERT(-1 <= device && device <= LIBXSTREAM_MAX_NDEVICES);
    return m_signals[device+1];
  }

  volatile value_type* streams() {
    return m_streams;
  }

  int enqueue(libxstream_event& event, const libxstream_stream* exclude) {
    LIBXSTREAM_ASSERT(0 == event.expected());
    int result = LIBXSTREAM_ERROR_NONE;
    const size_t n = max_nstreams();
    bool reset = true;

    for (size_t i = 0; i < n; ++i) {
      const value_type stream = m_streams[i];

      if (stream != exclude) {
        result = event.enqueue(*stream, reset);
        LIBXSTREAM_CHECK_ERROR(result);
        reset = false;
      }
    }
    if (reset) {
      result = event.reset();
    }

    return result;
  }

  value_type schedule(const libxstream_stream* exclude) {
    const size_t n = max_nstreams();
    value_type result = 0;

    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
      result = m_streams[i];
      if (result == exclude) {
        result = 0;
        j = i;
        i = n; // break
      }
    }

    const size_t end = j + n;
    for (size_t i = j + 1; i < end; ++i) {
      const value_type stream = m_streams[/*i%n*/i<n?i:(i-n)];
      if (0 != stream) {
        result = stream;
        i = end; // break
      }
    }

    return result;
  }

  int sync(int device) {
    const size_t n = max_nstreams();
    for (size_t i = 0; i < n; ++i) {
      if (const value_type stream = m_streams[i]) {
        const int stream_device = stream->device();
        if (stream_device == device) {
          const int result = stream->wait(0);
          LIBXSTREAM_CHECK_ERROR(result);
        }
      }
    }
    return LIBXSTREAM_ERROR_NONE;
  }

  int sync() {
    const size_t n = max_nstreams();
    for (size_t i = 0; i < n; ++i) {
      if (const value_type stream = m_streams[i]) {
        const int result = stream->wait(0);
        LIBXSTREAM_CHECK_ERROR(result);
      }
    }
    return LIBXSTREAM_ERROR_NONE;
  }

private:
  // not necessary to be device-specific due to single-threaded offload
  libxstream_signal m_signals[LIBXSTREAM_MAX_NDEVICES + 1];
  volatile value_type m_streams[LIBXSTREAM_MAX_NDEVICES*LIBXSTREAM_MAX_NSTREAMS];
#if defined(LIBXSTREAM_STDFEATURES)
  std::atomic<size_t> m_istreams;
#else
  size_t m_istreams;
#endif
} registry;


template<typename A, typename E, typename D>
bool atomic_compare_exchange(A& atomic, E& expected, D desired)
{
#if defined(LIBXSTREAM_STDFEATURES)
  const bool result = std::atomic_compare_exchange_weak(&atomic, &expected, desired);
#elif defined(_OPENMP)
  bool result = false;
# pragma omp critical
  {
    result = atomic == expected;
    if (result) {
      atomic = desired;
    }
    else {
      expected = atomic;
    }
  }
#else // generic
  bool result = false;
  libxstream_lock *const lock = libxstream_lock_get(&atomic);
  libxstream_lock_acquire(lock);
  result = atomic == expected;
  if (result) {
    atomic = desired;
  }
  else {
    expected = atomic;
  }
  libxstream_lock_release(lock);
#endif
  return result;
}


template<typename A, typename T>
T atomic_store(A& atomic, T value)
{
  T result = value;
#if defined(LIBXSTREAM_STDFEATURES)
  result = std::atomic_exchange(&atomic, value);
#elif defined(_OPENMP)
# pragma omp critical
  {
    result = atomic;
    atomic = value;
  }
#else // generic
  libxstream_lock_acquire(registry.lock());
  result = atomic;
  atomic = value;
  libxstream_lock_release(registry.lock());
#endif
  return result;
}

} // namespace libxstream_stream_internal


/*static*/int libxstream_stream::enqueue(libxstream_event& event, const libxstream_stream* exclude)
{
  return libxstream_stream_internal::registry.enqueue(event, exclude);
}


/*static*/libxstream_stream* libxstream_stream::schedule(const libxstream_stream* exclude)
{
  return libxstream_stream_internal::registry.schedule(exclude);
}


/*static*/int libxstream_stream::sync(int device)
{
  return libxstream_stream_internal::registry.sync(device);
}


/*static*/int libxstream_stream::sync()
{
  return libxstream_stream_internal::registry.sync();
}


libxstream_stream::libxstream_stream(int device, int priority, const char* name)
  : m_device(device), m_priority(priority), m_thread(-1)
#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  , m_handle(0) // lazy creation
  , m_npartitions(0)
#endif
{
  std::fill_n(m_pending, LIBXSTREAM_MAX_NTHREADS, static_cast<libxstream_signal>(0));
  std::fill_n(m_queues, LIBXSTREAM_MAX_NTHREADS, static_cast<libxstream_queue*>(0));

#if defined(LIBXSTREAM_TRACE) && ((1 == ((2*LIBXSTREAM_TRACE+1)/2) && defined(LIBXSTREAM_DEBUG)) || 1 < ((2*LIBXSTREAM_TRACE+1)/2))
  if (name && 0 != *name) {
    const size_t length = std::min(std::char_traits<char>::length(name), sizeof(m_name) - 1);
    std::copy(name, name + length, m_name);
    m_name[length] = 0;
  }
  else {
    m_name[0] = 0;
  }
#else
  libxstream_use_sink(name);
#endif

  using namespace libxstream_stream_internal;
  volatile registry_type::value_type& entry = libxstream_stream_internal::registry.allocate();
  entry = this;
}


libxstream_stream::~libxstream_stream()
{
  using namespace libxstream_stream_internal;
  volatile registry_type::value_type *const end = registry.streams() + registry.max_nstreams();
  volatile registry_type::value_type *const stream = std::find(registry.streams(), end, this);
  LIBXSTREAM_ASSERT(stream != end);
  *stream = 0; // unregister stream

  const size_t nthreads = nthreads_active();
  for (size_t i = 0; i < nthreads; ++i) {
    delete m_queues[i];
  }

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && !defined(__MIC__) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  if (0 != m_handle) {
    _Offload_stream_destroy(m_device, m_handle);
  }
#endif
}


libxstream_signal libxstream_stream::signal() const
{
  return ++libxstream_stream_internal::registry.signal(m_device);
}


int libxstream_stream::wait(libxstream_signal signal)
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_ASYNC_BEGIN(this, m_pending, signal)
  {
    libxstream_signal *const pending_signals = ptr<libxstream_signal,0>();
    const libxstream_signal signal = val<const libxstream_signal,1>();

    const int nthreads = static_cast<int>(nthreads_active());
    for (int i = 0; i < nthreads; ++i) {
      const libxstream_signal pending_signal = pending_signals[i];
      if (0 != pending_signal) {
#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (0 != (2*LIBXSTREAM_ASYNC+1)/2)
        if (0 <= LIBXSTREAM_ASYNC_DEVICE) {
# if defined(LIBXSTREAM_STREAM_WAIT_PAST)
          const libxstream_signal wait_pending = 0 != signal ? signal : pending_signal;
# else
          const libxstream_signal wait_pending = pending_signal;
# endif
#         pragma offload_wait LIBXSTREAM_ASYNC_TARGET wait(wait_pending)
        }
#endif
        if (0 == signal) {
          pending_signals[i] = 0;
        }
#if defined(LIBXSTREAM_STREAM_WAIT_PAST)
        else {
          i = nthreads; // break
        }
#endif
      }
    }

    if (0 != signal) {
      for (int i = 0; i < nthreads; ++i) {
        if (signal == pending_signals[i]) {
          pending_signals[i] = 0;
        }
      }
    }
  }
  LIBXSTREAM_ASYNC_END(LIBXSTREAM_CALL_DEFAULT | LIBXSTREAM_CALL_WAIT, result);

  return result;
}


void libxstream_stream::pending(int thread, libxstream_signal signal)
{
  LIBXSTREAM_ASSERT(0 <= thread && thread < LIBXSTREAM_MAX_NTHREADS);
  m_pending[thread] = signal;
}


libxstream_signal libxstream_stream::pending(int thread) const
{
  LIBXSTREAM_ASSERT(0 <= thread && thread < LIBXSTREAM_MAX_NTHREADS);
#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && !defined(__MIC__) && defined(LIBXSTREAM_ASYNC) && (0 != (2*LIBXSTREAM_ASYNC+1)/2) && defined(LIBXSTREAM_STREAM_CHECK_PENDING)
  const libxstream_signal lookup_signal = m_pending[thread];
  libxstream_signal signal = lookup_signal;
  if (0 != lookup_signal && 0 != _Offload_signaled(m_device, reinterpret_cast<void*>(lookup_signal))) {
    m_pending[thread] = 0;
    signal = 0;
  }
#else
  const libxstream_signal signal = m_pending[thread];
#endif
  return signal;
}


void libxstream_stream::enqueue(libxstream_capture_base& work_item, bool clone)
{
  const int thread = this_thread_id();
  LIBXSTREAM_ASSERT(thread < LIBXSTREAM_MAX_NTHREADS);
  libxstream_queue *volatile q = m_queues[thread];

  if (0 == q) {
    libxstream_lock *const lock = libxstream_lock_get(m_queues + thread);
    libxstream_lock_acquire(lock);

    if (0 == q) {
      q = new libxstream_queue;
      m_queues[thread] = q;
    }

    libxstream_lock_release(lock);
  }

  LIBXSTREAM_ASSERT(0 != q);
  libxstream_queue::entry_type& entry = q->allocate_entry();
  libxstream_capture_base *const item = clone ? work_item.clone() : &work_item;
  entry.push(item);

  if (0 != (work_item.flags() & LIBXSTREAM_CALL_WAIT)) {
    entry.wait();
  }
}


void libxstream_stream::enqueue(const libxstream_capture_base& work_item)
{
  enqueue(*work_item.clone(), false);
}


libxstream_queue* libxstream_stream::queue_begin()
{
  libxstream_queue* result = 0 <= m_thread ? m_queues[m_thread] : 0;

  if (0 == result || 0 == result->get().item()) {
    const int nthreads = static_cast<int>(nthreads_active());
    size_t size = result ? result->size() : 0;

    for (int i = 0; i < nthreads; ++i) {
      libxstream_queue *const qi = m_queues[i];
      const size_t si = (0 != qi && 0 != qi->get().item()) ? qi->size() : 0;
      if (size < si) {
        m_thread = i;
        result = qi;
        size = si;
      }
    }

    if (0 == size && 0 != result && 0 != result->get().item() &&
      0 == (LIBXSTREAM_CALL_WAIT & static_cast<const libxstream_capture_base*>(result->get().item())->flags()))
    {
      result = 0 <= m_thread ? m_queues[m_thread] : 0;
    }
  }

  return result;
}


libxstream_queue* libxstream_stream::queue_next()
{
  libxstream_queue* result = 0;

  if (0 <= m_thread) {
    const int nthreads = static_cast<int>(nthreads_active());
    const int end = m_thread + nthreads;
    for (int i = m_thread + 1; i < end; ++i) {
      const int thread = /*i % nthreads*/i < nthreads ? i : (i - nthreads);
      libxstream_queue *const qi = m_queues[thread];
      if (0 != qi) {
        result = qi;
        m_thread = thread;
        i = end; // break
      }
    }
  }

  return result;
}


#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
_Offload_stream libxstream_stream::handle() const
{
  const size_t nstreams = libxstream_stream_internal::registry.nstreams(m_device);

  if (nstreams != m_npartitions) {
    if (0 != m_handle) {
      const_cast<libxstream_stream*>(this)->wait(0); // complete pending operations on old stream
      _Offload_stream_destroy(m_device, m_handle);
    }

    // TODO: implement device discovery (number of threads)
    const size_t nthreads = 224;
    // TODO: implement stream priorities (weighting)
    m_handle = _Offload_stream_create(m_device, static_cast<int>(nthreads / nstreams));
    m_npartitions = nstreams;
  }
  return m_handle;
}
#endif


#if defined(LIBXSTREAM_TRACE) && ((1 == ((2*LIBXSTREAM_TRACE+1)/2) && defined(LIBXSTREAM_DEBUG)) || 1 < ((2*LIBXSTREAM_TRACE+1)/2))
const char* libxstream_stream::name() const
{
  return m_name;
}
#endif


const libxstream_stream* cast_to_stream(const void* stream)
{
  return static_cast<const libxstream_stream*>(stream);
}


libxstream_stream* cast_to_stream(void* stream)
{
  return static_cast<libxstream_stream*>(stream);
}


const libxstream_stream* cast_to_stream(const libxstream_stream* stream)
{
  return stream;
}


libxstream_stream* cast_to_stream(libxstream_stream* stream)
{
  return stream;
}


const libxstream_stream* cast_to_stream(const libxstream_stream& stream)
{
  return &stream;
}


libxstream_stream* cast_to_stream(libxstream_stream& stream)
{
  return &stream;
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
