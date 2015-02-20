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
#include "libxstream_context.hpp"


libxstream_context& libxstream_context::instance()
{
  static LIBXSTREAM_TLS libxstream_context context;
  return context;
}


libxstream_context& libxstream_context::instance(libxstream_function function, const libxstream_argument arguments[], size_t arity, int flags_)
{
  libxstream_context& context = instance();

  for (size_t i = 0; i <= arity; ++i) {
    context.signature[i] = arguments[i];
  }
  context.fhybrid = function;
  context.fnative = reinterpret_cast<const void*>(function);
  context.flags = flags_;

  return context;
}


LIBXSTREAM_TARGET(mic) const libxstream_argument* libxstream_find(const libxstream_context& context, const void* variable)
{
  const libxstream_argument* argument = 0;
  if (context.signature) {
    for (const libxstream_argument* argi = context.signature; LIBXSTREAM_TYPE_VOID != argi->type; ++argi) {
      LIBXSTREAM_ASSERT(libxstream_argument::kind_invalid != argi->kind);
      if (variable == libxstream_get_data(*argi)) {
        argument = argi;
        break;
      }
    }
  }
  return argument;
}
