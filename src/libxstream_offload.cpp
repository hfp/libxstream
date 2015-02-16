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
#include "libxstream_context.hpp"


namespace libxstream_offload_internal {

// the following construct helps to inline libxstream_get_value within an offload region
LIBXSTREAM_TARGET(mic) void get_arg_value(size_t arity, const libxstream_argument* signature, char* arg[]) {
#if defined(__INTEL_COMPILER)
# pragma loop_count min(0), max(LIBXSTREAM_MAX_NARGS), avg(LIBXSTREAM_MAX_NARGS/2)
#endif
  for (size_t i = 0; i < arity; ++i) {
#if defined(__INTEL_COMPILER)
#   pragma forceinline recursive
#endif
    arg[i] = libxstream_get_value(signature[i]);
  }
}

} // namespace libxstream_offload_internal


#define LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN { \
  libxstream_context& context = libxstream_context::instance(); \
  context.signature = signature; \
  context.stream = stream; \
  char* arg[LIBXSTREAM_MAX_NARGS]; \
  libxstream_offload_internal::get_arg_value(arity, signature, arg)

#define LIBXSTREAM_OFFLOAD_CONTEXT_END \
  context.signature = 0; \
  context.stream = 0; \
}


void libxstream_offload(libxstream_function function, const libxstream_argument* signature, const libxstream_stream* stream)
{
  LIBXSTREAM_ASSERT(function);
  LIBXSTREAM_TARGET(mic) void (*fn)(LIBXSTREAM_VARIADIC) = function;

  size_t arity = 0;
  libxstream_get_arity(signature, &arity);

  switch (arity) {
    case 0: {
#if defined(LIBXSTREAM_OFFLOAD)
//#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn();
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
#if 0
    case 1: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 2: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
#endif
#if 0
    case 3: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 4: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 5: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 6: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 7: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 8: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 9: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 10: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 11: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 12: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10], arg[11]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 13: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10], arg[11], arg[12]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 14: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10], arg[11], arg[12], arg[13]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 15: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10], arg[11], arg[12], arg[13], arg[14]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
    case 16: {
#if defined(LIBXSTREAM_OFFLOAD)
#     pragma offload target(mic:device) if(0 <= device) in(signature: length(arity))
#endif
      LIBXSTREAM_OFFLOAD_CONTEXT_BEGIN;
      fn(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], arg[9], arg[10], arg[11], arg[12], arg[13], arg[14], arg[15]);
      LIBXSTREAM_OFFLOAD_CONTEXT_END;
    } break;
#endif
    default: LIBXSTREAM_ASSERT(false);
  }
}
