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
#include <string>

#if defined(LIBXSTREAM_STDFEATURES)
# include <atomic>
#endif

// allows to wait for an event issued prior to the pending signal
//#define LIBXSTREAM_STREAM_WAIT_PAST


namespace libxstream_stream_internal {

class registry_type {
public:
  registry_type()
    : m_istreams(0)
#if !defined(LIBXSTREAM_STDFEATURES)
    , m_lock(libxstream_lock_create())
#endif
  {
    std::fill_n(m_signals, LIBXSTREAM_MAX_NDEVICES, 0);
    std::fill_n(m_streams, LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS, static_cast<libxstream_stream*>(0));
  }

  ~registry_type() {
    const size_t n = max_nstreams();
    for (size_t i = 0; i < n; ++i) {
#if defined(LIBXSTREAM_DEBUG)
      if (0 != m_streams[i]) {
        LIBXSTREAM_PRINT_WARNING("dangling stream \"%s\"!", m_streams[i]->name());
      }
#endif
      libxstream_stream_destroy(m_streams[i]);
    }
#if !defined(LIBXSTREAM_STDFEATURES)
    libxstream_lock_destroy(m_lock);
#endif
  }

public:
  libxstream_stream** allocate() {
#if !defined(LIBXSTREAM_STDFEATURES)
    libxstream_lock_acquire(m_lock);
#endif
    libxstream_stream** i = m_streams + (m_istreams++ % (LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS));
    while (0 != *i) i = m_streams + (m_istreams++ % (LIBXSTREAM_MAX_NDEVICES * LIBXSTREAM_MAX_NSTREAMS));
#if !defined(LIBXSTREAM_STDFEATURES)
    libxstream_lock_release(m_lock);
#endif
    return i;
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

  libxstream_stream** streams() {
    return m_streams;
  }

  void enqueue(libxstream_event& event) {
    LIBXSTREAM_ASSERT(0 == event.expected());
    const size_t n = max_nstreams();
    bool reset = true;
    for (size_t i = 0; i < n; ++i) {
      if (libxstream_stream *const stream = m_streams[i]) {
        event.enqueue(*stream, reset);
        reset = false;
      }
    }
  }

  int sync(int device) {
    const size_t n = max_nstreams();
    for (size_t i = 0; i < n; ++i) {
      if (libxstream_stream *const stream = m_streams[i]) {
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
      if (libxstream_stream *const stream = m_streams[i]) {
        const int result = stream->wait(0);
        LIBXSTREAM_CHECK_ERROR(result);
      }
    }
    return LIBXSTREAM_ERROR_NONE;
  }

#if defined(LIBXSTREAM_LOCK_RETRY) && !defined(LIBXSTREAM_STDFEATURES) && !defined(_OPENMP)
  libxstream_lock* lock() {
    return m_lock;
  }
#endif

private:
  // not necessary to be device-specific due to single-threaded offload
  libxstream_signal m_signals[LIBXSTREAM_MAX_NDEVICES + 1];
  libxstream_stream* m_streams[LIBXSTREAM_MAX_NDEVICES*LIBXSTREAM_MAX_NSTREAMS];
#if defined(LIBXSTREAM_STDFEATURES)
  std::atomic<size_t> m_istreams;
#else
  size_t m_istreams;
  libxstream_lock* m_lock;
#endif
} registry;

} // namespace libxstream_stream_internal


/*static*/void libxstream_stream::enqueue(libxstream_event& event)
{
  libxstream_stream_internal::registry.enqueue(event);
}


/*static*/int libxstream_stream::sync(int device)
{
  return libxstream_stream_internal::registry.sync(device);
}


/*static*/int libxstream_stream::sync()
{
  return libxstream_stream_internal::registry.sync();
}


libxstream_stream::libxstream_stream(int device, bool demux, int priority, const char* name)
  : m_pending(0), m_lock(libxstream_lock_create())
#if defined(LIBXSTREAM_LOCK_RETRY)
# if defined(LIBXSTREAM_STDFEATURES)
  , m_lock_alive(new std::atomic<size_t>(0))
# else
  , m_lock_alive(new size_t(0))
# endif
#endif
  , m_demux(0 != demux), m_thread(-1)
  , m_device(device), m_priority(priority), m_status(LIBXSTREAM_ERROR_NONE)
#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  , m_handle(0) // lazy creation
  , m_npartitions(0)
#endif
{
  using namespace libxstream_stream_internal;
  libxstream_stream* *const slot = libxstream_stream_internal::registry.allocate();
  *slot = this;

#if defined(LIBXSTREAM_DEBUG)
  if (name && 0 != *name) {
    const size_t length = std::min(std::char_traits<char>::length(name), sizeof(m_name) - 1);
    std::copy(name, name + length, m_name);
    m_name[length] = 0;
  }
  else {
    m_name[0] = 0;
  }
#endif
}


libxstream_stream::~libxstream_stream()
{
  using namespace libxstream_stream_internal;
  libxstream_stream* *const end = registry.streams() + registry.max_nstreams();
  libxstream_stream* *const stream = std::find(registry.streams(), end, this);
  LIBXSTREAM_ASSERT(stream != end);
  *stream = 0; // unregister stream

  const size_t nstreams = registry.nstreams();
  if (0 == nstreams) {
    libxstream_offload_shutdown();
  }

  libxstream_lock_destroy(m_lock);
#if defined(LIBXSTREAM_LOCK_RETRY)
# if defined(LIBXSTREAM_STDFEATURES)
  std::atomic<size_t> *const lock_alive = static_cast<std::atomic<size_t>*>(m_lock_alive);
# else
  size_t *const lock_alive = static_cast<size_t*>(m_lock_alive);
# endif
  delete lock_alive;
#endif

#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  if (0 != m_handle) {
    _Offload_stream_destroy(m_device, m_handle);
  }
#endif
}


libxstream_signal libxstream_stream::signal() const
{
  return ++libxstream_stream_internal::registry.signal(m_device);
}


int libxstream_stream::wait(libxstream_signal signal) const
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_OFFLOAD_BEGIN(const_cast<libxstream_stream*>(this), &result, signal)
  {
    *ptr<int,0>() = LIBXSTREAM_OFFLOAD_STREAM->reset(); // result code

    if (0 != LIBXSTREAM_OFFLOAD_PENDING) {
      const libxstream_signal signal = val<const libxstream_signal,1>();

# if defined(LIBXSTREAM_STREAM_WAIT_PAST)
      const libxstream_signal pending = signal ? signal : LIBXSTREAM_OFFLOAD_PENDING;
# else
      const libxstream_signal pending = LIBXSTREAM_OFFLOAD_PENDING;
# endif

#if defined(LIBXSTREAM_OFFLOAD)
      if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
#       pragma offload_wait LIBXSTREAM_OFFLOAD_TARGET wait(pending)
      }
#endif

      if (0 == signal || signal == LIBXSTREAM_OFFLOAD_PENDING) {
        (LIBXSTREAM_OFFLOAD_STREAM)->pending(0);
      }
    }
  }
  LIBXSTREAM_OFFLOAD_END(true);

  return result;
}


void libxstream_stream::lock()
{
#if defined(LIBXSTREAM_LOCK_RETRY)
  LIBXSTREAM_ASSERT(m_lock_alive);
# if defined(LIBXSTREAM_STDFEATURES)
  std::atomic<size_t>& lock_alive = *static_cast<std::atomic<size_t>*>(m_lock_alive);
# else
  size_t& lock_alive = *static_cast<size_t*>(m_lock_alive);
# endif
#endif
  const int this_thread = this_thread_id();

  if (m_thread != this_thread) {
#if defined(LIBXSTREAM_LOCK_RETRY) && (0 < LIBXSTREAM_LOCK_RETRY)
    const size_t lock_alive_snapshot = lock_alive;
    size_t retry = 0;
    while (!libxstream_lock_try(m_lock)) {
      this_thread_yield();

      if ((LIBXSTREAM_LOCK_RETRY) > retry) {
        retry += lock_alive_snapshot == lock_alive;
      }
      else {
        LIBXSTREAM_PRINT_WARNING("libxstream_stream_lock: stream=0x%lx seems to be dead-locked!",
          static_cast<unsigned long>(reinterpret_cast<uintptr_t>(this)));
        wait(0);
        retry = 0;
      }
    }
#else
    libxstream_lock_acquire(m_lock);
#endif
    m_thread = this_thread; // locked
    LIBXSTREAM_PRINT_INFO("libxstream_stream_lock: stream=0x%lx acquired by thread=%i",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(this)),
      this_thread);
  }

#if defined(LIBXSTREAM_LOCK_RETRY)
# if defined(LIBXSTREAM_STDFEATURES)
  ++lock_alive;
# elif defined(_OPENMP)
# pragma omp atomic
  ++lock_alive;
# else // generic
  libxstream_lock_acquire(libxstream_stream_internal::registry.lock());
  ++lock_alive;
  libxstream_lock_release(libxstream_stream_internal::registry.lock());
# endif
#endif

  LIBXSTREAM_ASSERT(m_thread == this_thread);
}


void libxstream_stream::unlock()
{
  const int this_thread = this_thread_id();
  if (m_thread == this_thread) { // locked
    LIBXSTREAM_PRINT_INFO("libxstream_stream_unlock: stream=0x%lx released by thread=%i",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(this)),
      this_thread);
    m_thread = -1; // unlock
    libxstream_lock_release(m_lock);
  }
}


#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
_Offload_stream libxstream_stream::handle() const
{
  const size_t nstreams = libxstream_stream_internal::registry.nstreams(m_device);

  if (nstreams != m_npartitions) {
    if (0 != m_handle) {
      wait(0); // complete pending operations on old stream
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


#if defined(LIBXSTREAM_DEBUG)
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
