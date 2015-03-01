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
#include "libxstream_offload.hpp"
#include "libxstream_argument.hpp"
#include "libxstream_capture.hpp"
#include "libxstream_context.hpp"
#include <algorithm>


namespace libxstream_offload_internal {

LIBXSTREAM_TARGET(mic) void call(libxstream_function function, libxstream_argument signature[], char* translation[], size_t arity, int flags)
{
  const struct LIBXSTREAM_TARGET(mic) argument_type {
    libxstream_argument* m_signature;
    explicit argument_type(libxstream_argument* signature): m_signature(signature) {}
    void* operator[](int i) const { return *reinterpret_cast<void**>(m_signature+ i); }
  } a(signature);

  if (signature && translation) {
    size_t np = 0;
#if defined(__INTEL_COMPILER)
#   pragma loop_count min(0), max(LIBXSTREAM_MAX_NARGS), avg(LIBXSTREAM_MAX_NARGS/2)
#endif
    for (size_t i = 0; i < arity; ++i) {
      if (0 < signature[i].dims) {
#if defined(__INTEL_COMPILER)
#       pragma forceinline recursive
#endif
        libxstream_set_data(signature[i], translation[np]);
        ++np;
      }
    }
  }

  libxstream_context& context = libxstream_context::instance(signature, flags);

  switch (arity) {
    case  0: function(); break;
    case  1: function(a[0]); break;
    case  2: function(a[0], a[1]); break;
    case  3: function(a[0], a[1], a[2]); break;
    case  4: function(a[0], a[1], a[2], a[3]); break;
    case  5: function(a[0], a[1], a[2], a[3], a[4]); break;
    case  6: function(a[0], a[1], a[2], a[3], a[4], a[5]); break;
    case  7: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
    case  8: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
    case  9: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); break;
    case 10: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); break;
    case 11: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]); break;
    case 12: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); break;
    case 13: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]); break;
    case 14: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]); break;
    case 15: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]); break;
    case 16: function(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]); break;
    default: {
      LIBXSTREAM_ASSERT(false);
    }
  }

  // mark context as invalid
  context.flags = LIBXSTREAM_CALL_INVALID;
  context.signature = 0;
}

} // namespace libxstream_offload_internal


int libxstream_offload(libxstream_function function, const libxstream_argument signature[], libxstream_stream* stream, int flags)
{
  LIBXSTREAM_ASSERT(0 == (LIBXSTREAM_CALL_INVALID & flags));
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_ASYNC_BEGIN(stream, function, signature) {
    LIBXSTREAM_TARGET(mic) /*const*/ libxstream_function fhybrid = 0 == (m_flags & LIBXSTREAM_CALL_NATIVE) ? m_function : 0;
    const void *const fnative = reinterpret_cast<const void*>(m_function);
    libxstream_argument *const signature = m_signature;
    const int flags = m_flags;
    size_t arity = 0;
    libxstream_fn_arity(m_signature, &arity);

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_ASYNC_DEVICE) {
      char* p[(LIBXSTREAM_MAX_NARGS)];
      size_t np = 0;
#if defined(__INTEL_COMPILER)
#     pragma loop_count min(0), max(LIBXSTREAM_MAX_NARGS), avg(LIBXSTREAM_MAX_NARGS/2)
#endif
      for (size_t i = 0; i < arity; ++i) {
        if (0 < m_signature[i].dims) {
          p[np] = libxstream_get_data(m_signature[i]);
          ++np;
        }
      }
# if defined(LIBXSTREAM_DEBUG)
      for (size_t i = np; i < (LIBXSTREAM_MAX_NARGS); ++i) p[i] = 0;
# endif
      switch (np) {
        case 0: {
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1))
            {
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, 0, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1))
            {
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, 0, arity, flags);
            }
          }
        } break;
        case 1: {
          char *a0 = p[0];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 2: {
          char *a0 = p[0], *a1 = p[1];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 3: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 4: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 5: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 6: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 7: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 8: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 9: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 10: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8], *a9 = p[9];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 11: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8], *a9 = p[9], *a10 = p[10];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 12: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 13: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 14: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 15: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5],   *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13], *a14 = p[14];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        case 16: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5],   *a6 = p[6],   *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13], *a14 = p[14], *a15 = p[15];
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT in(fhybrid, fnative, arity, flags) in(signature: length(arity + 1)) \
              inout(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, flags);
            }
          }
        } break;
        default: {
          LIBXSTREAM_CHECK_CALL_ASSERT(status(LIBXSTREAM_ERROR_CONDITION));
        }
      }
    }
    else
#endif
    {
      libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, 0, arity, flags);
    }
  }
  LIBXSTREAM_ASYNC_END(flags, result);

  return result;
}
