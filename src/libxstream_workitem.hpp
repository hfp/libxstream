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
#ifndef LIBXSTREAM_WORKITEM_HPP
#define LIBXSTREAM_WORKITEM_HPP

#include "libxstream_argument.hpp"
#include "libxstream_stream.hpp"
#include "libxstream_workqueue.hpp"

#if defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)

#define LIBXSTREAM_OFFLOAD_ALLOC alloc_if(1) free_if(0)
#define LIBXSTREAM_OFFLOAD_FREE  alloc_if(0) free_if(1)
#define LIBXSTREAM_OFFLOAD_REUSE alloc_if(0) free_if(0)
#define LIBXSTREAM_OFFLOAD_REFRESH length(0) LIBXSTREAM_OFFLOAD_REUSE
#define LIBXSTREAM_OFFLOAD_DATA(ARG, IS_SCALAR) inout(ARG: length(((IS_SCALAR)*sizeof(libxstream_argument::data_union))) alloc_if(IS_SCALAR) free_if(IS_SCALAR))

#define LIBXSTREAM_ASYNC_PENDING workitem_pending_generated
#define LIBXSTREAM_ASYNC_STREAM (0 != m_stream ? *m_stream : 0)
#define LIBXSTREAM_ASYNC_DEVICE workitem_device
#define LIBXSTREAM_ASYNC_QENTRY workitem_qentry
#define LIBXSTREAM_ASYNC_INTERNAL(NAME) LIBXSTREAM_CONCATENATE(NAME,_internal)

#if defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (1 < (2*LIBXSTREAM_ASYNC+1)/2) // asynchronous offload
# if (2 == (2*LIBXSTREAM_ASYNC+1)/2) // asynchronous offload
#   define LIBXSTREAM_ASYNC_DECL \
      const int LIBXSTREAM_ASYNC_DEVICE = device(); \
      const libxstream_signal LIBXSTREAM_ASYNC_PENDING = libxstream_stream::pending(LIBXSTREAM_ASYNC_STREAM); \
      const libxstream_signal workitem_signal_generated = libxstream_stream::signal(LIBXSTREAM_ASYNC_DEVICE); \
      libxstream_signal workitem_pending_consumed = LIBXSTREAM_ASYNC_PENDING; \
      libxstream_signal workitem_signal_consumed = workitem_signal_generated; \
      libxstream_sink(&LIBXSTREAM_ASYNC_DEVICE); \
      libxstream_sink(&LIBXSTREAM_ASYNC_PENDING); \
      libxstream_sink(&workitem_signal_generated); \
      libxstream_sink(&workitem_pending_consumed); \
      libxstream_sink(&workitem_signal_consumed)
#   define LIBXSTREAM_ASYNC_TARGET target(mic:LIBXSTREAM_ASYNC_DEVICE)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET signal(workitem_signal_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET wait(workitem_pending_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL wait(workitem_pending_consumed++)
# elif (3 == (2*LIBXSTREAM_ASYNC+1)/2) // compiler streams
#   define LIBXSTREAM_ASYNC_DECL \
      const _Offload_stream handle_ = LIBXSTREAM_ASYNC_STREAM ? LIBXSTREAM_ASYNC_STREAM->handle() : 0; \
      const int LIBXSTREAM_ASYNC_DEVICE = device(); \
      const libxstream_signal LIBXSTREAM_ASYNC_PENDING = libxstream_stream::pending(LIBXSTREAM_ASYNC_STREAM); \
      const libxstream_signal workitem_signal_generated = libxstream_stream::signal(LIBXSTREAM_ASYNC_DEVICE); \
      libxstream_signal workitem_pending_consumed = LIBXSTREAM_ASYNC_PENDING; \
      libxstream_signal workitem_signal_consumed = workitem_signal_generated; \
      libxstream_sink(&LIBXSTREAM_ASYNC_DEVICE); \
      libxstream_sink(&LIBXSTREAM_ASYNC_PENDING); \
      libxstream_sink(&workitem_signal_generated); \
      libxstream_sink(&workitem_pending_consumed); \
      libxstream_sink(&workitem_signal_consumed)
#   define LIBXSTREAM_ASYNC_TARGET target(mic) stream(handle_)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET signal(workitem_signal_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET wait(workitem_pending_consumed++)
#   define LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL
# endif
#elif defined(LIBXSTREAM_OFFLOAD) && (0 != LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (0 < (2*LIBXSTREAM_ASYNC+1)/2) // synchronous offload
# define LIBXSTREAM_ASYNC_DECL \
    int LIBXSTREAM_ASYNC_DEVICE = device(); \
    libxstream_signal LIBXSTREAM_ASYNC_PENDING = 0, workitem_signal_generated = 0, workitem_pending_consumed = 0, workitem_signal_consumed = 0; \
    libxstream_sink(&LIBXSTREAM_ASYNC_DEVICE); \
    libxstream_sink(&LIBXSTREAM_ASYNC_PENDING); \
    libxstream_sink(&workitem_signal_generated); \
    libxstream_sink(&workitem_pending_consumed); \
    libxstream_sink(&workitem_signal_consumed)
# define LIBXSTREAM_ASYNC_TARGET target(mic:LIBXSTREAM_ASYNC_DEVICE)
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_ASYNC_TARGET
# define LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_ASYNC_TARGET
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT LIBXSTREAM_ASYNC_TARGET_SIGNAL
#else // no offload
# define LIBXSTREAM_ASYNC_DECL \
    int LIBXSTREAM_ASYNC_DEVICE = device(); \
    libxstream_signal LIBXSTREAM_ASYNC_PENDING = 0, workitem_signal_generated = 0, workitem_pending_consumed = 0, workitem_signal_consumed = 0; \
    libxstream_sink(&LIBXSTREAM_ASYNC_DEVICE); \
    libxstream_sink(&LIBXSTREAM_ASYNC_PENDING); \
    libxstream_sink(&workitem_signal_generated); \
    libxstream_sink(&workitem_pending_consumed); \
    libxstream_sink(&workitem_signal_consumed)
# define LIBXSTREAM_ASYNC_TARGET
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL
# define LIBXSTREAM_ASYNC_TARGET_WAIT
# define LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT
#endif

#define LIBXSTREAM_ASYNC_BEGIN \
  typedef struct LIBXSTREAM_UNIQUE(workitem_type): public libxstream_workitem { \
    LIBXSTREAM_UNIQUE(workitem_type)(libxstream_stream* stream, int flags, size_t argc, const arg_type argv[], const char* name) \
      : libxstream_workitem(stream, flags, argc, argv, name) \
    {} \
    LIBXSTREAM_UNIQUE(workitem_type)* virtual_clone() const { \
      return new LIBXSTREAM_UNIQUE(workitem_type)(*this); \
    } \
    void virtual_run(libxstream_workqueue::entry_type& LIBXSTREAM_ASYNC_QENTRY) { \
      LIBXSTREAM_ASYNC_DECL; libxstream_sink(&LIBXSTREAM_ASYNC_QENTRY); do
#define LIBXSTREAM_ASYNC_END(STREAM, FLAGS, NAME, ...) while(libxstream_nonconst(LIBXSTREAM_FALSE)); \
      LIBXSTREAM_ASSERT(workitem_pending_generated == workitem_pending_consumed || (workitem_pending_generated + 1) == workitem_pending_consumed); \
      LIBXSTREAM_ASSERT(workitem_signal_generated == workitem_signal_consumed || (workitem_signal_generated + 1) == workitem_signal_consumed); \
      if (workitem_signal_generated != workitem_signal_consumed) { \
        libxstream_stream::pending(LIBXSTREAM_ASYNC_STREAM, workitem_signal_generated); \
      } \
      else if (workitem_pending_generated != workitem_pending_consumed) { \
        libxstream_stream::pending(LIBXSTREAM_ASYNC_STREAM, 0); \
      } \
    } \
  } LIBXSTREAM_UNIQUE(workitem_type); \
  const libxstream_workitem::arg_type LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,argv))[] = { libxstream_workitem::arg_type(), __VA_ARGS__ }; \
  LIBXSTREAM_UNIQUE(workitem_type) LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,item))(cast_to_stream(STREAM), FLAGS, \
    sizeof(LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,argv))) / sizeof(*LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,argv))) - 1, \
    LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,argv)) + 1, __FUNCTION__); \
  libxstream_workqueue::entry_type& LIBXSTREAM_ASYNC_INTERNAL(NAME) = LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,item)).stream() \
    ? (*LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,item)).stream())->enqueue(LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,item))) \
    : libxstream_enqueue(&LIBXSTREAM_UNIQUE(LIBXSTREAM_CONCATENATE(NAME,item))); \
  const libxstream_workqueue::entry_type& NAME = LIBXSTREAM_ASYNC_INTERNAL(NAME); \
  libxstream_sink(&NAME)


class libxstream_workitem {
public:
  class arg_type: public libxstream_argument {
  public:
    arg_type(): m_signature(false) {
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_inout, 0, LIBXSTREAM_TYPE_INVALID, 0, 0));
    }
    arg_type(libxstream_function function): m_signature(false) {
      const size_t size = sizeof(void*);
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_input, &function, LIBXSTREAM_TYPE_VOID, 0, &size));
    }
    arg_type(const libxstream_argument* signature): m_signature(true) {
      const size_t size = sizeof(libxstream_argument*);
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_input, signature, LIBXSTREAM_TYPE_VOID, 1, &size));
    }
    template<typename T> arg_type(T arg): m_signature(false) {
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_input, &arg, libxstream_map_to<T>::type(), 0, 0));
    }
    template<typename T> arg_type(T* arg): m_signature(false) {
      const size_t unknown = 0;
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_inout, reinterpret_cast<void*>(arg), libxstream_map_to<T>::type(), 1, &unknown));
    }
    template<typename T> arg_type(const T* arg): m_signature(false) {
      const size_t unknown = 0;
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_construct(this, 0, kind_input, reinterpret_cast<const void*>(arg), libxstream_map_to<T>::type(), 1, &unknown));
    }
  public:
    bool signature() const { return m_signature; }
  private:
    bool m_signature;
  };

public:
  libxstream_workitem(libxstream_stream* stream, int flags, size_t argc, const arg_type argv[], const char* name);
  virtual ~libxstream_workitem();

public:
  template<typename T,size_t i> T& val() {
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
    size_t arity = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_arity(m_signature, &arity) && i < arity);
#endif
    return *reinterpret_cast<T*>(&m_signature[i]);
  }

  template<typename T,size_t i> T val() const {
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
    size_t arity = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_arity(m_signature, &arity) && i < arity);
#endif
    return *reinterpret_cast<const T*>(&m_signature[i]);
  }

  template<typename T,size_t i> T* ptr() {
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
    size_t arity = 0;
    LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == libxstream_get_arity(m_signature, &arity) && i < arity);
#endif
    return *reinterpret_cast<T**>(&m_signature[i]);
  }

  libxstream_stream*volatile* stream() { return m_stream; }
  const libxstream_stream*volatile* stream() const {
    return const_cast<const libxstream_stream*volatile*>(m_stream);
  }

  int thread() const { return m_thread; }
  int device() const {
    return 0 == (LIBXSTREAM_CALL_DEVICE & m_flags) ? libxstream_stream::device(LIBXSTREAM_ASYNC_STREAM) : val<int,0>();
  }

  void flags(int value) { m_flags = value; }
  int flags() const { return m_flags; }

  const libxstream_event* event() const { return m_event; }
  void event(const libxstream_event* value) { m_event = value; }

  libxstream_workitem* clone() const;
  void operator()(libxstream_workqueue::entry_type& entry);

private:
  virtual libxstream_workitem* virtual_clone() const = 0;
  virtual void virtual_run(libxstream_workqueue::entry_type& entry) = 0;

protected:
  libxstream_stream*volatile* m_stream;
  libxstream_argument m_signature[(LIBXSTREAM_MAX_NARGS)+1];
  libxstream_function m_function;

private:
  const libxstream_event* m_event;
  int m_thread, m_flags;
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
  const char* m_name;
#endif
};


libxstream_workqueue::entry_type& libxstream_enqueue(libxstream_workitem* workitem);

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
#endif // LIBXSTREAM_WORKITEM_HPP
