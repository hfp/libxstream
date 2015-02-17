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

#define LIBXSTREAM_OFFLOAD_SIGNATURE(SIGNATURE, ARITY) in(ARITY) in(SIGNATURE: length(ARITY))
#define LIBXSTREAM_OFFLOAD_REFRESH length(0) alloc_if(false) free_if(false)


int libxstream_offload(libxstream_function function, const libxstream_argument* signature, libxstream_stream& stream, bool wait)
{
  LIBXSTREAM_ASYNC_BEGIN(stream, function, signature) {
    LIBXSTREAM_TARGET(mic) const libxstream_function fun = function();
    const libxstream_argument *const sig = signature();
    LIBXSTREAM_ASSERT(fun && sig);

    size_t arity = 0;
    libxstream_get_arity(sig, &arity);

    switch (arity) {
      case 0: {
#if defined(LIBXSTREAM_OFFLOAD)
        if (0 <= LIBXSTREAM_ASYNC_DEVICE) {
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_OFFLOAD_SIGNATURE(sig, arity)
            {
              libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
              fun();
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_OFFLOAD_SIGNATURE(sig, arity)
            {
              libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
              fun();
            }
          }
        }
        else
#endif
        {
          libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
          fun();
        }
      } break;
      case 1: {
        char *a0 = libxstream_get_data(sig[0]);
#if defined(LIBXSTREAM_OFFLOAD)
        if (0 <= LIBXSTREAM_ASYNC_DEVICE) {
          if (LIBXSTREAM_ASYNC_READY) {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_SIGNAL LIBXSTREAM_OFFLOAD_SIGNATURE(sig, arity) \
              inout(a0: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              libxstream_context& context = libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
              libxstream_set_data(context.signature[0], a0);
              fun(a0);
            }
          }
          else {
#           pragma offload LIBXSTREAM_ASYNC_TARGET_WAIT LIBXSTREAM_OFFLOAD_SIGNATURE(sig, arity) \
              inout(a0: LIBXSTREAM_OFFLOAD_REFRESH)
            {
              libxstream_context& context = libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
              fun(a0);
            }
          }
        }
        else
#endif
        {
          libxstream_context& context = libxstream_context::instance(sig, sig + arity, LIBXSTREAM_ASYNC_STREAM);
          fun(a0);
        }
      } break;
      default: {
        LIBXSTREAM_ASSERT(false);
      }
    }
  }
  LIBXSTREAM_ASYNC_END(wait);

  return LIBXSTREAM_ERROR_NONE;
}
