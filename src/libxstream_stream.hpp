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
#ifndef LIBXSTREAM_STREAM_HPP
#define LIBXSTREAM_STREAM_HPP

#include "libxstream_workqueue.hpp"

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD)
# include <offload.h>
#endif

#if defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)


class libxstream_workitem;
struct libxstream_event;


struct libxstream_stream {
public:
  static int priority_range_least();
  static int priority_range_greatest();

  static int enqueue(libxstream_event& event, const libxstream_stream* exclude = 0, bool sync = true);
  static libxstream_stream* schedule(const libxstream_stream* exclude);

  static int sync_all(bool wait, int device);
  static int sync_all(bool wait);

public:
  libxstream_stream(int device, int priority, const char* name);
  ~libxstream_stream();

public:
  int device() const    { return m_device; }
  int priority() const  { return m_priority; }
  
  libxstream_signal signal() const;
  int sync(bool wait, libxstream_signal signal);

  int wait(libxstream_event& event);
  libxstream_event* events();
  size_t nevents() const;

  void pending(int thread, libxstream_signal signal);
  libxstream_signal pending(int thread) const;
  libxstream_signal pending() const {
    return 0 <= m_thread ? pending(m_thread) : 0;
  }

  libxstream_workqueue::entry_type& enqueue(libxstream_workitem& workitem);
  libxstream_workqueue* queue(bool retry = false);

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  _Offload_stream handle() const;
#endif

#if defined(LIBXSTREAM_TRACE) && 0 != ((2*LIBXSTREAM_TRACE+1)/2) && defined(LIBXSTREAM_DEBUG)
  const char* name() const { return m_name; }
#endif

private:
  libxstream_stream(const libxstream_stream& other);
  libxstream_stream& operator=(const libxstream_stream& other);

private:
  mutable libxstream_signal m_pending[LIBXSTREAM_MAX_NTHREADS];
  libxstream_workqueue* m_queues[LIBXSTREAM_MAX_NTHREADS];
  struct slot_type {
    slot_type(): events(0), size(0) {}
    libxstream_event* events;
    size_t size;
  } m_slots[LIBXSTREAM_MAX_NTHREADS];
#if defined(LIBXSTREAM_TRACE) && 0 != ((2*LIBXSTREAM_TRACE+1)/2) && defined(LIBXSTREAM_DEBUG)
  char m_name[128];
#endif
  size_t m_retry;
  int m_device;
  int m_priority;
  int m_thread;

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  mutable _Offload_stream m_handle; // lazy creation
  mutable size_t m_npartitions;
#endif
};


const libxstream_stream* cast_to_stream(const void* stream);
libxstream_stream* cast_to_stream(void* stream);

const libxstream_stream* cast_to_stream(const libxstream_stream* stream);
libxstream_stream* cast_to_stream(libxstream_stream* stream);

const libxstream_stream* cast_to_stream(const libxstream_stream& stream);
libxstream_stream* cast_to_stream(libxstream_stream& stream);

template<typename T> libxstream_stream* cast_to_stream(T stream) {
  libxstream_use_sink(&stream);
  LIBXSTREAM_ASSERT(0 == stream);
  return static_cast<libxstream_stream*>(0);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
#endif // LIBXSTREAM_STREAM_HPP
