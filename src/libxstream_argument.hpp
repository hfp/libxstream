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
#ifndef LIBXSTREAM_ARGUMENT_HPP
#define LIBXSTREAM_ARGUMENT_HPP

#include <libxstream.h>

#if defined(LIBXSTREAM_EXPORTED) || defined(__LIBXSTREAM) || defined(LIBXSTREAM_INTERNAL)


extern "C" struct LIBXSTREAM_TARGET(mic) libxstream_argument {
  enum kind_type {
    kind_invalid  = 0,
    kind_input    = 1,
    kind_output   = 2,
    kind_inout    = kind_output | kind_input,
  };

  // This data member *must* be the first!
  union element_union {
    char data[sizeof(void*)];
    void* pointer;
    signed char i8;
    unsigned char u8;
    short i16;
    unsigned u16;
    int i32;
    unsigned int u32;
    long long i64;
    unsigned long long u64;
    float f32;
    double f64;
    float c32[2];
    double c64[2];
    char c;
  } data;

  size_t shape[LIBXSTREAM_MAX_NDIMS];
  kind_type kind;
  libxstream_type type;
  size_t dims;
};


LIBXSTREAM_EXPORT_INTERNAL int libxstream_construct(libxstream_argument& arg, libxstream_argument::kind_type kind, const void* value, libxstream_type type, size_t dims, const size_t shape[]);
int libxstream_construct(libxstream_argument* signature, size_t nargs);

LIBXSTREAM_EXPORT_INTERNAL LIBXSTREAM_TARGET(mic) char* libxstream_get_data(const libxstream_argument& arg);
LIBXSTREAM_TARGET(mic) int libxstream_set_data(libxstream_argument& arg, const void* data);

LIBXSTREAM_TARGET(mic) inline char* libxstream_address(libxstream_argument& arg) {
  return reinterpret_cast<char*>(&arg);
}

LIBXSTREAM_TARGET(mic) inline const char* libxstream_address(const libxstream_argument& arg) {
  return reinterpret_cast<const char*>(&arg);
}

#endif // defined(LIBXSTREAM_EXPORTED) || defined(LIBXSTREAM_INTERNAL)
#endif // LIBXSTREAM_ARGUMENT_HPP
