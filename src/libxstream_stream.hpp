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
#define LIBXSTREAM_STREAM_CACHED_REGCHECK


class libxstream_workitem;
struct libxstream_event;


struct/*!class*/ libxstream_stream {
public:
  static int priority_range_least();
  static int priority_range_greatest();

  static int device(const libxstream_stream* stream);

  static libxstream_signal signal(int device);
  static libxstream_signal pending(const libxstream_stream* stream);
  static void pending(libxstream_stream* stream, libxstream_signal signal);

  static int enqueue(libxstream_event& event, const libxstream_stream* exclude = 0);
  static libxstream_stream* schedule(const libxstream_stream* exclude);

  static int wait_all(int device, const libxstream_stream* exclude = 0);
  static int wait_all(const libxstream_stream* exclude = 0);

public:
  libxstream_stream(int device, int priority, const char* name);
  ~libxstream_stream();

public:
  const libxstream_stream*volatile& registered() const;
  libxstream_stream*volatile& registered();

  int priority() const { return m_priority; }

  const libxstream_workqueue::entry_type* work() const { return m_queue.front(); }
  libxstream_workqueue::entry_type* work() { return m_queue.front(); }

  libxstream_workqueue::entry_type& enqueue(libxstream_workitem& workitem);

  int wait();

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (3 == (2*LIBXSTREAM_ASYNC+1)/2)
  _Offload_stream handle() const;
#endif

  const char* name() const {
#if defined(LIBXSTREAM_INTERNAL_TRACE)
    return m_name;
#else
    return 0;
#endif
  }

private:
  libxstream_stream(const libxstream_stream& other);
  libxstream_stream& operator=(const libxstream_stream& other);

private:
#if defined(LIBXSTREAM_INTERNAL_TRACE)
  char m_name[128];
#endif
#if defined(LIBXSTREAM_STREAM_CACHED_REGCHECK)
  libxstream_stream*volatile* m_registered;
#endif
  libxstream_workqueue m_queue;
  libxstream_signal m_pending;
  int m_device;
  int m_priority;

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (3 == (2*LIBXSTREAM_ASYNC+1)/2)
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
  libxstream_sink(&stream);
  LIBXSTREAM_ASSERT(0 == stream);
  return static_cast<libxstream_stream*>(0);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
#endif // LIBXSTREAM_STREAM_HPP
