/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
#include "libxstream_offload.hpp"
#include "libxstream_argument.hpp"
#include "libxstream_workitem.hpp"
#include "libxstream_context.hpp"
#include "libxstream_stream.hpp"

#include <libxstream_begin.h>
#include <algorithm>
#include <libxstream_end.h>


namespace libxstream_offload_internal {

LIBXSTREAM_RETARGETABLE void call(libxstream_function function, libxstream_argument arguments[], char* translation[], size_t arity, int flags)
{
  const struct LIBXSTREAM_RETARGETABLE argument_type {
    libxstream_argument* m_arguments;
    explicit argument_type(libxstream_argument arguments[]): m_arguments(arguments) {}
    const void* operator[](size_t i) const { return libxstream_get_value(m_arguments[i]).const_pointer; }
  } a(arguments);

  if (arguments && translation) {
    size_t np = 0;
    LIBXSTREAM_PRAGMA_LOOP_COUNT(0, LIBXSTREAM_MAX_NARGS, LIBXSTREAM_MAX_NARGS/2)
    for (size_t i = 0; i < arity; ++i) {
      if (0 != arguments[i].data.pointer && (0 != arguments[i].dims || 0 != (libxstream_argument::kind_output & arguments[i].kind))) {
#if defined(__INTEL_COMPILER)
#       pragma forceinline recursive
#endif
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_set_value(arguments[i], translation[np++]));
      }
    }
  }

  libxstream_context& context = libxstream_context::instance(arguments, flags);

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
  context.flags = LIBXSTREAM_CALL_EXTERNAL;
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
  context.signature = 0;
#endif
}

} // namespace libxstream_offload_internal


libxstream_workqueue::entry_type& libxstream_offload(libxstream_function function, const libxstream_argument signature[], libxstream_stream* stream, int flags)
{
  LIBXSTREAM_ASSERT(0 == (LIBXSTREAM_CALL_EXTERNAL & flags));
  LIBXSTREAM_ASYNC_BEGIN {
    LIBXSTREAM_RETARGETABLE /*const*/ libxstream_function fhybrid = 0 == (flags() & LIBXSTREAM_CALL_NATIVE) ? m_function : 0;
    const void *const fnative = reinterpret_cast<const void*>(m_function);
    libxstream_argument *const signature = m_signature;
    const int cflags = flags();
    size_t arity = 0;
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_get_arity(m_signature, &arity));

    LIBXSTREAM_PRINT(2, "fn_call: stream=0x%llx fn=0x%llx args=0x%llx (%lu) flags=%i",
      reinterpret_cast<unsigned long long>(LIBXSTREAM_ASYNC_STREAM),
      reinterpret_cast<unsigned long long>(fhybrid ? fhybrid : ((libxstream_function)fnative)),
      reinterpret_cast<unsigned long long>(signature), static_cast<unsigned long>(arity), cflags);
    LIBXSTREAM_PRINT0(3, "************************************************************************");

#if defined(LIBXSTREAM_OFFLOAD_BUILD)
    if (0 <= LIBXSTREAM_ASYNC_DEVICE) {
      char* p[(LIBXSTREAM_MAX_NARGS)];
      unsigned int s = 0;
      LIBXSTREAM_ASSERT(LIBXSTREAM_MAX_NARGS <= 8 * sizeof(s));
      size_t np = 0;
      LIBXSTREAM_PRAGMA_LOOP_COUNT(0, LIBXSTREAM_MAX_NARGS, LIBXSTREAM_MAX_NARGS/2)
      for (size_t i = 0; i < arity; ++i) {
        if (char *const pointer = reinterpret_cast<char*>(m_signature[i].data.pointer)) {
          if (0 != m_signature[i].dims) {
            p[np] = pointer;
            ++np;
          }
          else if (0 != (libxstream_argument::kind_output & m_signature[i].kind)) {
            p[np] = pointer;
            s |= ((2 << np) >> 1);
            ++np;
          }
        }
      }
# if defined(LIBXSTREAM_INTERNAL_DEBUG)
      for (size_t i = np; i < (LIBXSTREAM_MAX_NARGS); ++i) p[i] = 0;
# endif

      switch (np) {
        case 0: {
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1))
            libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, 0, arity, cflags);
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1))
            libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, 0, arity, cflags);
          }
        } break;
        case 1: {
          char *a0 = p[0];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)
            {
              char* translation[] = { a0 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)
            {
              char* translation[] = { a0 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 2: {
          char *a0 = p[0], *a1 = p[1];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2)
            {
              char* translation[] = { a0, a1 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2)
            {
              char* translation[] = { a0, a1 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 3: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4)
            {
              char* translation[] = { a0, a1, a2 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4)
            {
              char* translation[] = { a0, a1, a2 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 4: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8)
            {
              char* translation[] = { a0, a1, a2, a3 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8)
            {
              char* translation[] = { a0, a1, a2, a3 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 5: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)
            {
              char* translation[] = { a0, a1, a2, a3, a4 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1) LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)
            {
              char* translation[] = { a0, a1, a2, a3, a4 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 6: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2) LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 7: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4) LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 8: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8) \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 9: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)  LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16) LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 10: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8], *a9 = p[9];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)  LIBXSTREAM_OFFLOAD_DATA(a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32) LIBXSTREAM_OFFLOAD_DATA(a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 11: {
          char *a0 = p[0], *a1 = p[1], *a2 = p[2], *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7], *a8 = p[8], *a9 = p[9], *a10 = p[10];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)  LIBXSTREAM_OFFLOAD_DATA(a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64) LIBXSTREAM_OFFLOAD_DATA(a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 12: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3], *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA(a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)   \
              LIBXSTREAM_OFFLOAD_DATA(a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128) \
              LIBXSTREAM_OFFLOAD_DATA(a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 13: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4], *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)   LIBXSTREAM_OFFLOAD_DATA(a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)  LIBXSTREAM_OFFLOAD_DATA(a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256) LIBXSTREAM_OFFLOAD_DATA(a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 14: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5], *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)   LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)  LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512) LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 15: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5],   *a6 = p[6], *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13], *a14 = p[14];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)    LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)   LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512)  LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192) LIBXSTREAM_OFFLOAD_DATA(a14, s&16384)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)    LIBXSTREAM_OFFLOAD_DATA( a2, s&4)    LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)   LIBXSTREAM_OFFLOAD_DATA( a6, s&64)   LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512)  LIBXSTREAM_OFFLOAD_DATA(a10, s&1024) LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192) LIBXSTREAM_OFFLOAD_DATA(a14, s&16384)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        case 16: {
          char *a0 = p[0], *a1 = p[1],  *a2 = p[2],   *a3 = p[3],   *a4 = p[4],   *a5 = p[5],   *a6 = p[6],   *a7 = p[7];
          char *a8 = p[8], *a9 = p[9], *a10 = p[10], *a11 = p[11], *a12 = p[12], *a13 = p[13], *a14 = p[14], *a15 = p[15];
          if (0 == (LIBXSTREAM_ASYNC_PENDING)) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)    LIBXSTREAM_OFFLOAD_DATA( a2, s&4)     LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)   LIBXSTREAM_OFFLOAD_DATA( a6, s&64)    LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512)  LIBXSTREAM_OFFLOAD_DATA(a10, s&1024)  LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192) LIBXSTREAM_OFFLOAD_DATA(a14, s&16384) LIBXSTREAM_OFFLOAD_DATA(a15, s&32768)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL_WAIT in(fhybrid, fnative, arity, cflags) in(signature: length(arity + 1)) \
              LIBXSTREAM_OFFLOAD_DATA( a0, s&1)    LIBXSTREAM_OFFLOAD_DATA( a1, s&2)    LIBXSTREAM_OFFLOAD_DATA( a2, s&4)     LIBXSTREAM_OFFLOAD_DATA( a3, s&8)    \
              LIBXSTREAM_OFFLOAD_DATA( a4, s&16)   LIBXSTREAM_OFFLOAD_DATA( a5, s&32)   LIBXSTREAM_OFFLOAD_DATA( a6, s&64)    LIBXSTREAM_OFFLOAD_DATA( a7, s&128)  \
              LIBXSTREAM_OFFLOAD_DATA( a8, s&256)  LIBXSTREAM_OFFLOAD_DATA( a9, s&512)  LIBXSTREAM_OFFLOAD_DATA(a10, s&1024)  LIBXSTREAM_OFFLOAD_DATA(a11, s&2048) \
              LIBXSTREAM_OFFLOAD_DATA(a12, s&4096) LIBXSTREAM_OFFLOAD_DATA(a13, s&8192) LIBXSTREAM_OFFLOAD_DATA(a14, s&16384) LIBXSTREAM_OFFLOAD_DATA(a15, s&32768)
            {
              char* translation[] = { a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
              libxstream_offload_internal::call(fhybrid ? fhybrid : reinterpret_cast<libxstream_function>(fnative), signature, translation, arity, cflags);
            }
          }
        } break;
        default: {
          LIBXSTREAM_ASYNC_QENTRY.status() = LIBXSTREAM_ERROR_CONDITION;
        }
      }
    }
    else
#endif
    {
      libxstream_offload_internal::call(fhybrid ? fhybrid : ((libxstream_function)fnative), signature, 0, arity, cflags);
    }
  }
  LIBXSTREAM_ASYNC_END(stream, flags & ~LIBXSTREAM_CALL_WAIT, work, function, signature);

  return LIBXSTREAM_ASYNC_INTERNAL(work);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM)
