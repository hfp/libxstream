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
#include "libxstream_argument.hpp"
#include <algorithm>
#include <cstdio>


LIBXSTREAM_EXPORT_INTERNAL int libxstream_construct(libxstream_argument arguments[], size_t arg, libxstream_argument::kind_type kind, const void* value, libxstream_type type, size_t dims, const size_t shape[])
{
  size_t size = 0;
  LIBXSTREAM_CHECK_CONDITION((((libxstream_argument::kind_invalid == kind || libxstream_argument::kind_inout == kind) && LIBXSTREAM_TYPE_INVALID == type) || LIBXSTREAM_TYPE_INVALID > type)
    && ((0 == dims && 0 == shape) || (0 == dims && 0 != shape && (LIBXSTREAM_TYPE_VOID == type || (LIBXSTREAM_ERROR_NONE == libxstream_get_typesize(type, &size) && 1 == size))) || (0 < dims))
    && (LIBXSTREAM_MAX_NDIMS) >= dims);

  LIBXSTREAM_ASSERT((LIBXSTREAM_MAX_NARGS) >= arg);
  libxstream_argument& argument = arguments[arg];

#if defined(LIBXSTREAM_DEBUG)
  std::fill_n(reinterpret_cast<char*>(&argument), sizeof(libxstream_argument), 0); // avoid false pos. with mem. analysis
#endif
  argument.kind = kind;
  argument.dims = dims;

  if (shape) {
    if (0 < dims || LIBXSTREAM_TYPE_VOID != type) {
      argument.type = type;
    }
    else {
      int i = -1;
      do {
        const libxstream_type autotype = static_cast<libxstream_type>(i);
        LIBXSTREAM_CHECK_CALL(libxstream_get_typesize(autotype, &size));
        if (*shape != size) {
          ++i;
        }
        else {
#if defined(LIBXSTREAM_PRINT)
          const char* context[] = { "", "input", "output", "inout" };
          LIBXSTREAM_PRINT_WARN("libxstream_fn_%s: signature=0x%llx arg=%lu is weakly typed (no elemental type)!",
            context[kind], reinterpret_cast<unsigned long long>(arguments), static_cast<unsigned long>(arg));
#endif
          argument.type = autotype;
          i = LIBXSTREAM_TYPE_VOID; // break
        }
      }
      while (i < LIBXSTREAM_TYPE_VOID);
      LIBXSTREAM_CHECK_CONDITION(i != LIBXSTREAM_TYPE_VOID);
    }

    size_t *const dst = argument.shape;
#if defined(__INTEL_COMPILER)
#   pragma loop_count min(0), max(LIBXSTREAM_MAX_NDIMS), avg(2)
#endif
    for (size_t i = 0; i < dims; ++i) dst[i] = shape[i];
  }
  else {
    std::fill_n(argument.shape, dims, 0);
    argument.type = type;
  }

  return libxstream_argument::kind_invalid != kind ? libxstream_set_data(argument, value) : LIBXSTREAM_ERROR_NONE;
}


int libxstream_construct(libxstream_argument* signature, size_t nargs)
{
  LIBXSTREAM_CHECK_CONDITION((0 != signature || 0 == nargs) && (LIBXSTREAM_MAX_NARGS) >= nargs);

  if (0 != signature) {
#if defined(__INTEL_COMPILER)
#   pragma loop_count min(0), max(LIBXSTREAM_MAX_NARGS), avg(LIBXSTREAM_MAX_NARGS/2)
#endif
    for (size_t i = 0; i < nargs; ++i) {
      LIBXSTREAM_CHECK_CALL(libxstream_construct(signature, i, libxstream_argument::kind_inout, 0, LIBXSTREAM_TYPE_INVALID, 0, 0));
    }
    LIBXSTREAM_CHECK_CALL(libxstream_construct(signature, nargs, libxstream_argument::kind_invalid, 0, LIBXSTREAM_TYPE_INVALID, 0, 0));
  }

  return LIBXSTREAM_ERROR_NONE;
}


LIBXSTREAM_EXPORT_INTERNAL LIBXSTREAM_TARGET(mic) char* libxstream_get_data(const libxstream_argument& arg)
{
  char* data = 0;

  const char *const src = libxstream_address(arg);
  if (0 != arg.dims || 0 != (libxstream_argument::kind_output & arg.kind)) {
    char *const dst = reinterpret_cast<char*>(&data);
    for (size_t i = 0; i < sizeof(void*); ++i) dst[i] = src[i];
  }
  else {
#if defined(LIBXSTREAM_PASS_BY_VALUE)
    char *const dst = reinterpret_cast<char*>(&data);
    for (size_t i = 0; i < sizeof(void*); ++i) dst[i] = src[i];
#else
    data = const_cast<char*>(src);
#endif
  }

  return data;
}


LIBXSTREAM_TARGET(mic) int libxstream_set_data(libxstream_argument& arg, const void* data)
{
  char *const dst = libxstream_address(arg);
  if (0 != arg.dims || 0 != (libxstream_argument::kind_output & arg.kind)) {
    const char *const src = reinterpret_cast<const char*>(&data);
    for (size_t i = 0; i < sizeof(void*); ++i) dst[i] = src[i];
  }
  else {
    size_t size = 0;
    LIBXSTREAM_CHECK_CALL(libxstream_get_typesize(arg.type, &size));

    if (data) {
      const char *const src = static_cast<const char*>(data);
      for (size_t i = 0; i < size; ++i) dst[i] = src[i];
    }
    else {
      for (size_t i = 0; i < size; ++i) dst[i] = 0;
    }
  }
  return LIBXSTREAM_ERROR_NONE;
}
