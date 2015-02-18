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
#ifndef LIBXSTREAM_CAPTURE_HPP
#define LIBXSTREAM_CAPTURE_HPP

#include <libxstream_stream.hpp>

#if defined(LIBXSTREAM_EXPORTED) || defined(LIBXSTREAM_INTERNAL)

#define LIBXSTREAM_ASYNC_PENDING (capture_region_pending)
#define LIBXSTREAM_ASYNC_READY (0 == (LIBXSTREAM_ASYNC_PENDING))
#define LIBXSTREAM_ASYNC_STREAM (m_stream)
#define LIBXSTREAM_ASYNC_DEVICE (capture_region_device)
#define LIBXSTREAM_ASYNC_DEVICE_UPDATE(DEVICE) LIBXSTREAM_ASYNC_DEVICE = (DEVICE)

#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (0 != (2*LIBXSTREAM_ASYNC+1)/2)
# if (1 == (2*LIBXSTREAM_ASYNC+1)/2) // asynchronous offload
#   define LIBXSTREAM_ASYNC_DECL \
      libxstream_signal capture_region_signal_consumed = capture_region_signal
#   define LIBXSTREAM_ASYNC_TARGET target(mic:LIBXSTREAM_ASYNC_DEVICE)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET signal(capture_region_signal_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL wait(LIBXSTREAM_ASYNC_PENDING)
# elif (2 == (2*LIBXSTREAM_ASYNC+1)/2) // compiler streams
#   define LIBXSTREAM_ASYNC_DECL \
      const _Offload_stream handle_ = LIBXSTREAM_ASYNC_STREAM ? LIBXSTREAM_ASYNC_STREAM->handle() : 0; \
      libxstream_signal capture_region_signal_consumed = capture_region_signal
#   define LIBXSTREAM_ASYNC_TARGET target(mic:LIBXSTREAM_ASYNC_DEVICE) stream(handle_)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET signal(capture_region_signal_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL
# endif
#elif defined(LIBXSTREAM_OFFLOAD) // synchronous offload
# if defined(LIBXSTREAM_DEBUG)
#   define LIBXSTREAM_ASYNC_DECL const libxstream_signal capture_region_signal_consumed = capture_region_signal + 1
# else
#   define LIBXSTREAM_ASYNC_DECL const libxstream_signal capture_region_signal_consumed = capture_region_signal;
# endif
# define LIBXSTREAM_ASYNC_TARGET target(mic:LIBXSTREAM_ASYNC_DEVICE)
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET
# define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL
#else
# if defined(LIBXSTREAM_DEBUG)
#   define LIBXSTREAM_ASYNC_DECL libxstream_signal capture_region_signal_consumed = capture_region_signal + 1
# else
#   define LIBXSTREAM_ASYNC_DECL libxstream_signal capture_region_signal_consumed = capture_region_signal;
# endif
# define LIBXSTREAM_ASYNC_TARGET
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL
# define LIBXSTREAM_ASYNC_TARGET_WAIT
#endif

#define LIBXSTREAM_ASYNC_BEGIN(STREAM, ...) do { \
  libxstream_stream *const libxstream_capture_stream = cast_to_stream(STREAM); \
  const libxstream_capture_base::arg_type libxstream_capture_argv[] = { __VA_ARGS__ }; \
  const struct libxstream_capture: public libxstream_capture_base { \
    libxstream_capture(size_t argc, const arg_type argv[], libxstream_stream* stream, bool wait, bool sync = false) \
      : libxstream_capture_base(argc, argv, stream, wait, sync) \
    { \
      libxstream_offload(*this, wait); \
    } \
    libxstream_capture* virtual_clone() const { \
      return new libxstream_capture(*this); \
    } \
    void virtual_run() const { \
      const libxstream_signal LIBXSTREAM_ASYNC_PENDING = LIBXSTREAM_ASYNC_STREAM ? LIBXSTREAM_ASYNC_STREAM->pending(thread()) : 0; \
      int LIBXSTREAM_ASYNC_DEVICE = LIBXSTREAM_ASYNC_STREAM ? LIBXSTREAM_ASYNC_STREAM->device() : val<int,0>(); \
      const libxstream_signal capture_region_signal = LIBXSTREAM_ASYNC_STREAM ? LIBXSTREAM_ASYNC_STREAM->signal() : 0; \
      LIBXSTREAM_ASYNC_DECL; do
#define LIBXSTREAM_ASYNC_END(...) while(false); \
      if (LIBXSTREAM_ASYNC_STREAM && capture_region_signal != capture_region_signal_consumed) { \
        LIBXSTREAM_ASYNC_STREAM->pending(thread(), capture_region_signal); \
      } \
    } \
  } capture_region(sizeof(libxstream_capture_argv) / sizeof(*libxstream_capture_argv), \
    libxstream_capture_argv, libxstream_capture_stream, __VA_ARGS__); \
  } while(0)


struct LIBXSTREAM_EXPORT_INTERNAL libxstream_stream;


struct LIBXSTREAM_EXPORT_INTERNAL libxstream_capture_base {
public:
  class LIBXSTREAM_EXPORT_INTERNAL arg_type: public libxstream_argument {
  public:
    arg_type(): m_signature(false) {
      libxstream_construct(*this, kind_inout, 0, LIBXSTREAM_TYPE_VOID, 0, 0);
    }

    arg_type(const libxstream_argument* signature): m_signature(true) {
      const size_t size = sizeof(libxstream_argument*);
      libxstream_construct(*this, kind_input, signature, LIBXSTREAM_TYPE_BYTE, 1, &size);
    }

    template<typename T> arg_type(T arg): m_signature(false) {
      libxstream_construct(*this, kind_input, &arg, libxstream_type2value<T>::value, 0, 0);
    }

    template<typename T> arg_type(T* arg): m_signature(false) {
      const size_t size = sizeof(T*);
      libxstream_construct(*this, kind_inout, reinterpret_cast<void*>(arg), LIBXSTREAM_TYPE_BYTE, 1, &size);
    }

    template<typename T> arg_type(const T* arg): m_signature(false) {
      const size_t size = sizeof(T*);
      libxstream_construct(*this, kind_input, reinterpret_cast<const void*>(arg), LIBXSTREAM_TYPE_BYTE, 1, &size);
    }
  public:
    bool signature() const { return m_signature; }
  private:
    bool m_signature;
  };

public:
  libxstream_capture_base(size_t argc, const arg_type argv[], libxstream_stream* stream, bool wait, bool sync);
  virtual ~libxstream_capture_base();

public:
  template<typename T,size_t i> T val() const {
#if defined(LIBXSTREAM_DEBUG)
    size_t size = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_fn_arity(m_signature, &size) && i < size);
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_datasize(m_signature + i, &size) && sizeof(T) <= size);
#endif
    return *reinterpret_cast<const T*>(libxstream_address(m_signature[i]));
  }

  template<typename T,size_t i> T* ptr() const {
#if defined(LIBXSTREAM_DEBUG)
    size_t size = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_fn_arity(m_signature, &size) && i < size);
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_datasize(m_signature + i, &size) && sizeof(T*) <= size);
#endif
    return reinterpret_cast<T*>(libxstream_get_data(m_signature[i]));
  }

  libxstream_function function() const { return m_function; }
  const libxstream_argument* signature() const { return m_signature; }

  libxstream_capture_base* clone() const;
  void operator()() const;
  int thread() const;

private:
  virtual libxstream_capture_base* virtual_clone() const = 0;
  virtual void virtual_run() const = 0;

private:
  mutable libxstream_function m_function;
  libxstream_argument m_signature[(LIBXSTREAM_MAX_NARGS)+1];
  bool m_unlock, m_sync;
#if defined(LIBXSTREAM_THREADLOCAL_SIGNALS)
  int m_thread;
#endif

protected:
  libxstream_stream* m_stream;
};


LIBXSTREAM_EXPORT_INTERNAL void libxstream_offload(const libxstream_capture_base& capture_region, bool wait);

void libxstream_capture_shutdown();

#endif // defined(LIBXSTREAM_EXPORTED) || defined(LIBXSTREAM_INTERNAL)
#endif // LIBXSTREAM_CAPTURE_HPP
