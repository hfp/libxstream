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
#ifndef LIBXSTREAM_MACROS_H
#define LIBXSTREAM_MACROS_H

#include "libxstream_config.h"

#if (1600 < _MSC_VER) || !defined(_WIN32)
# if !defined(LIBXSTREAM_STDTHREAD)
#   define LIBXSTREAM_STDTHREAD
# endif
# if !defined(_GLIBCXX_HAS_GTHREADS)
#   define _GLIBCXX_HAS_GTHREADS
# endif
# if !defined(_GLIBCXX_USE_C99_STDINT_TR1)
#   define _GLIBCXX_USE_C99_STDINT_TR1
# endif
# if !defined(_GLIBCXX_USE_SCHED_YIELD)
#   define _GLIBCXX_USE_SCHED_YIELD
# endif
#endif

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSTREAM_ATTRIBUTE(A) __declspec(A)
# define LIBXSTREAM_ALIGNED(DECL, N) LIBXSTREAM_ATTRIBUTE(align(N)) DECL
#elif defined(__GNUC__)
# define LIBXSTREAM_ATTRIBUTE(A) __attribute__((A))
# define LIBXSTREAM_ALIGNED(DECL, N) DECL LIBXSTREAM_ATTRIBUTE(aligned(N))
#endif

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSTREAM_TLS LIBXSTREAM_ATTRIBUTE(thread)
#elif defined(__GNUC__)
# define LIBXSTREAM_TLS __thread
#endif
#if !defined(LIBXSTREAM_TLS)
# define LIBXSTREAM_TLS
#endif

#define LIBXSTREAM_TOSTRING_AUX(SYMBOL) #SYMBOL
#define LIBXSTREAM_TOSTRING(SYMBOL) LIBXSTREAM_TOSTRING_AUX(SYMBOL)

#if defined(__INTEL_COMPILER)
# define LIBXSTREAM_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
# define LIBXSTREAM_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
#elif (199901L <= __STDC_VERSION__)
# define LIBXSTREAM_ASSUME_ALIGNED(A, N)
# define LIBXSTREAM_PRAGMA(DIRECTIVE) _Pragma(LIBXSTREAM_STRINGIFY(DIRECTIVE))
#elif defined(_MSC_VER)
# define LIBXSTREAM_ASSUME_ALIGNED(A, N)
# define LIBXSTREAM_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
#else
# define LIBXSTREAM_ASSUME_ALIGNED(A, N)
# define LIBXSTREAM_PRAGMA(DIRECTIVE)
#endif

#if defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXSTREAM_OFFLOAD
# define LIBXSTREAM_TARGET(A) LIBXSTREAM_ATTRIBUTE(target(A))
#else
# define LIBXSTREAM_TARGET(A)
#endif

#if defined(__cplusplus)
# define LIBXSTREAM_EXTERN_C extern "C"
#else
# define LIBXSTREAM_EXTERN_C
#endif // __cplusplus

#define LIBXSTREAM_EXPORT LIBXSTREAM_TARGET(mic)

#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN32__)
# define LIBXSTREAM_RESTRICT __restrict__
#elif defined(_MSC_VER)
# define LIBXSTREAM_RESTRICT __restrict
#else
# define LIBXSTREAM_RESTRICT
#endif

#if (defined(LIBXSTREAM_ERROR_DEBUG) || defined(_DEBUG)) && !defined(NDEBUG) && !defined(LIBXSTREAM_DEBUG)
# define LIBXSTREAM_DEBUG
#endif

#if defined(LIBXSTREAM_ERROR_CHECK) && !defined(LIBXSTREAM_CHECK)
# define LIBXSTREAM_CHECK
#endif

#if defined(LIBXSTREAM_ASYNC) && ((1 == ((2*LIBXSTREAM_ASYNC+1)/2) && defined(LIBXSTREAM_DEBUG)) || 1 < ((2*LIBXSTREAM_ASYNC+1)/2))
# define LIBXSTREAM_ASSERT(A) assert(A)
# define LIBXSTREAM_PRINT_INFO(MESSAGE, ...) fprintf(stderr, "DBG " MESSAGE "\n", __VA_ARGS__)
# define LIBXSTREAM_PRINT_INFOCTX(MESSAGE, ...) fprintf(stderr, "DBG %s: " MESSAGE "\n", __FUNCTION__, __VA_ARGS__)
# define LIBXSTREAM_PRINT_WARNING(MESSAGE, ...) fprintf(stderr, "WRN " MESSAGE "\n", __VA_ARGS__)
# define LIBXSTREAM_PRINT_WARNING0(MESSAGE) fprintf(stderr, "WRN " MESSAGE "\n")
#else
# define LIBXSTREAM_ASSERT(A)
# define LIBXSTREAM_PRINT_INFO(MESSAGE, ...)
# define LIBXSTREAM_PRINT_INFOCTX(MESSAGE, ...)
# define LIBXSTREAM_PRINT_WARNING(MESSAGE, ...)
# define LIBXSTREAM_PRINT_WARNING0(MESSAGE)
#endif

#define LIBXSTREAM_ERROR_NONE       0
#define LIBXSTREAM_ERROR_RUNTIME   -1
#define LIBXSTREAM_ERROR_CONDITION -2

#if defined(_MSC_VER)
# define LIBXSTREAM_SNPRINTF(S, N, F, ...) _snprintf_s(S, N, _TRUNCATE, F, __VA_ARGS__)
#else
# define LIBXSTREAM_SNPRINTF(S, N, F, ...) snprintf(S, N, F, __VA_ARGS__)
#endif

#define LIBXSTREAM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSTREAM_MAX(A, B) ((A) < (B) ? (B) : (A))

#if defined(LIBXSTREAM_CHECK)
# define LIBXSTREAM_CHECK_ERROR(RETURN_VALUE) if (LIBXSTREAM_ERROR_NONE != (RETURN_VALUE)) return RETURN_VALUE;
# define LIBXSTREAM_CHECK_CONDITION(CONDITION) if (!(CONDITION)) return LIBXSTREAM_ERROR_CONDITION;
# define LIBXSTREAM_CHECK_CONDITION_RETURN(CONDITION) if (!(CONDITION)) return;
# define LIBXSTREAM_CHECK_CALL_RETURN(EXPRESSION) do { int result = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != result) return; } while(0)
# ifdef __cplusplus
#   define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) do { int result = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != result) throw std::runtime_error(LIBXSTREAM_TOSTRING(EXPRESSION) " at " __FILE__ ":" LIBXSTREAM_TOSTRING(__LINE__)); } while(0)
# else
#   define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) do { int result = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != result) abort(result); } while(0)
# endif
# if defined(_OPENMP)
#   define LIBXSTREAM_CHECK_CALL(EXPRESSION) LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION)
# else
#   define LIBXSTREAM_CHECK_CALL(EXPRESSION) do { int result = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != result) return result; } while(0)
# endif
#else
# define LIBXSTREAM_CHECK_ERROR(RETURN_VALUE) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (RETURN_VALUE));
# define LIBXSTREAM_CHECK_CONDITION(CONDITION) LIBXSTREAM_ASSERT(CONDITION);
# define LIBXSTREAM_CHECK_CONDITION_RETURN(CONDITION) LIBXSTREAM_ASSERT(CONDITION);
# define LIBXSTREAM_CHECK_CALL_RETURN(EXPRESSION) EXPRESSION
# define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) EXPRESSION
# define LIBXSTREAM_CHECK_CALL(EXPRESSION) EXPRESSION
#endif

#define LIBXSTREAM_OFFLOAD_PENDING (offload_region_pending)
#define LIBXSTREAM_OFFLOAD_READY (0 == (LIBXSTREAM_OFFLOAD_PENDING))
#define LIBXSTREAM_OFFLOAD_STREAM (this->stream())
#define LIBXSTREAM_OFFLOAD_DEVICE (offload_region_device)
#define LIBXSTREAM_OFFLOAD_DEVICE_UPDATE(DEVICE) LIBXSTREAM_OFFLOAD_DEVICE = (DEVICE)

#if defined(LIBXSTREAM_OFFLOAD) && defined(LIBXSTREAM_ASYNC) && (0 != (2*LIBXSTREAM_ASYNC+1)/2)
# if (1 == (2*LIBXSTREAM_ASYNC+1)/2) // asynchronous offload
#   define LIBXSTREAM_OFFLOAD_DECL \
      int LIBXSTREAM_OFFLOAD_DEVICE = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->device() : val<int,0>(); \
      const libxstream_signal LIBXSTREAM_OFFLOAD_PENDING = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->pending() : 0
#   define LIBXSTREAM_OFFLOAD_TARGET target(mic:LIBXSTREAM_OFFLOAD_DEVICE)
#   define LIBXSTREAM_OFFLOAD_TARGET_SIGNAL LIBXSTREAM_OFFLOAD_TARGET signal(offload_region_signal_consumed++)
#   define LIBXSTREAM_OFFLOAD_TARGET_WAIT LIBXSTREAM_OFFLOAD_TARGET_SIGNAL wait(LIBXSTREAM_OFFLOAD_PENDING)
# elif (2 == (2*LIBXSTREAM_ASYNC+1)/2) // compiler streams
#   define LIBXSTREAM_OFFLOAD_DECL \
      int LIBXSTREAM_OFFLOAD_DEVICE = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->device() : val<int,0>(); \
      const libxstream_signal LIBXSTREAM_OFFLOAD_PENDING = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->pending() : 0; \
      const _Offload_stream handle_ = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->handle() : 0
#   define LIBXSTREAM_OFFLOAD_TARGET target(mic:LIBXSTREAM_OFFLOAD_DEVICE) stream(handle_)
#   define LIBXSTREAM_OFFLOAD_TARGET_SIGNAL LIBXSTREAM_OFFLOAD_TARGET signal(offload_region_signal_consumed++)
#   define LIBXSTREAM_OFFLOAD_TARGET_WAIT LIBXSTREAM_OFFLOAD_TARGET_SIGNAL
# endif
#else // synchronous offload
# define LIBXSTREAM_OFFLOAD_DECL \
    int LIBXSTREAM_OFFLOAD_DEVICE = LIBXSTREAM_OFFLOAD_STREAM ? LIBXSTREAM_OFFLOAD_STREAM->device() : val<int,0>(); \
    const libxstream_signal LIBXSTREAM_OFFLOAD_PENDING = 0
# define LIBXSTREAM_OFFLOAD_TARGET target(mic:LIBXSTREAM_OFFLOAD_DEVICE)
# define LIBXSTREAM_OFFLOAD_TARGET_SIGNAL LIBXSTREAM_OFFLOAD_TARGET
# define LIBXSTREAM_OFFLOAD_TARGET_WAIT LIBXSTREAM_OFFLOAD_TARGET_SIGNAL
#endif

#define LIBXSTREAM_OFFLOAD_BEGIN(STREAM, ARG, ...) do { \
  libxstream_stream *const libxstream_offload_region_stream = cast_to_stream(STREAM); \
  const libxstream_offload_region::arg_type libxstream_offload_region_argv[] = { ARG, __VA_ARGS__ }; \
  const struct offload_region_type: public libxstream_offload_region { \
    bool m_offload_region_wait; \
    offload_region_type(libxstream_stream* stream, size_t argc, const arg_type argv[], bool wait) \
      : libxstream_offload_region(stream, argc, argv) \
      , m_offload_region_wait(wait) \
    { \
      if (!wait && stream && stream->demux()) { \
        const int this_thread = this_thread_id(); \
        while (this_thread != stream->thread()) { \
          if (libxstream_offload_busy()) { \
            this_thread_yield(); \
          } \
          else { \
            stream->lock(); \
            const int thread = stream->thread(); \
            if (this_thread != thread) { \
              if (0 <= thread) stream->wait(0); \
              stream->thread(this_thread); \
              LIBXSTREAM_PRINT_INFO("demux: thread=%i acquired stream=0x%lx", this_thread, \
                static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream))); \
            } \
            stream->unlock(); \
          } \
        } \
      } \
      libxstream_offload(*this, wait); \
    } \
    offload_region_type* clone() const { return new offload_region_type(*this); } \
    void operator()() const { LIBXSTREAM_OFFLOAD_DECL; \
      libxstream_signal offload_region_signal = 0; \
      if (LIBXSTREAM_OFFLOAD_STREAM) { \
        offload_region_signal = LIBXSTREAM_OFFLOAD_STREAM->signal(); \
        if (m_offload_region_wait && LIBXSTREAM_OFFLOAD_STREAM->demux()) { \
          LIBXSTREAM_OFFLOAD_STREAM->thread(-1); \
          LIBXSTREAM_PRINT_INFO("demux: thread=%i released stream=0x%lx", LIBXSTREAM_OFFLOAD_STREAM->thread(), \
            static_cast<unsigned long>(reinterpret_cast<uintptr_t>(LIBXSTREAM_OFFLOAD_STREAM))); \
        } \
      } \
      libxstream_signal offload_region_signal_consumed = offload_region_signal; do
#define LIBXSTREAM_OFFLOAD_END(WAIT) while(false); \
      if (LIBXSTREAM_OFFLOAD_STREAM && offload_region_signal != offload_region_signal_consumed) { \
        LIBXSTREAM_OFFLOAD_STREAM->pending(offload_region_signal); \
      } \
    } \
  } offload_region(libxstream_offload_region_stream, \
    sizeof(libxstream_offload_region_argv) / sizeof(*libxstream_offload_region_argv), \
    libxstream_offload_region_argv, (WAIT)); \
  } while(0)

#endif // LIBXSTREAM_MACROS_H
