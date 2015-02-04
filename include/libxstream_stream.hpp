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

#include "libxstream.hpp"


struct libxstream_event;


struct libxstream_stream {
public:
  static void enqueue(libxstream_event& event);

  static int sync(int device);
  static int sync();

public:
  libxstream_stream(int device,
    /**
     * Enables "demuxing" threads and streams i.e., when multiple
     * host threads attempt to queue into the same stream.
     */
    bool demux,
    int priority, const char* name);
  ~libxstream_stream();

public:
  void pending(libxstream_signal signal)  { m_pending = signal; }
  libxstream_signal pending() const       { return m_pending; }

  bool demux() const      { return m_demux; }
  void thread(int value)  { m_thread = value; }
  int thread() const      { return m_thread; }
  int device() const      { return m_device; }
  int priority() const    { return m_priority; }

  void status(int value)  { m_status = value; }
  int status() const      { return m_status; }
  int reset() {
    const int result = m_status;
    m_status = LIBXSTREAM_ERROR_NONE;
    return result;
  }

  libxstream_signal signal() const;
  int wait(libxstream_signal signal) const;

  bool locked() const { return m_locked; }
  void lock();
  void unlock();

#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  _Offload_stream handle() const;
#endif

#if defined(LIBXSTREAM_DEBUG)
  const char* name() const;
#endif

private:
  libxstream_stream(const libxstream_stream& other);
  libxstream_stream& operator=(const libxstream_stream& other);

private:
  mutable libxstream_signal m_pending;
  libxstream_lock* m_lock;
  bool m_locked, m_demux;
  int m_thread;
  int m_device;
  int m_priority;
  int m_status;

#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (2 == (2*LIBXSTREAM_ASYNC+1)/2)
  mutable _Offload_stream m_handle; // lazy creation
  mutable size_t m_npartitions;
#endif

#if defined(LIBXSTREAM_DEBUG)
  char m_name[128];
#endif
};


const libxstream_stream* cast_to_stream(const void* stream);
libxstream_stream* cast_to_stream(void* stream);

const libxstream_stream* cast_to_stream(const libxstream_stream* stream);
libxstream_stream* cast_to_stream(libxstream_stream* stream);

const libxstream_stream* cast_to_stream(const libxstream_stream& stream);
libxstream_stream* cast_to_stream(libxstream_stream& stream);

template<typename T> libxstream_stream* cast_to_stream(T stream) {
  LIBXSTREAM_ASSERT(0 == stream);
  return static_cast<libxstream_stream*>(0);
}

#endif // LIBXSTREAM_STREAM_HPP
