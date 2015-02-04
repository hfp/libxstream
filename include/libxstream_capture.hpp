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


struct libxstream_stream;


struct libxstream_offload_region {
public:
  struct arg_type {
    union { void* p; double d; } value;
#if defined(LIBXSTREAM_DEBUG)
    size_t size;
#endif

    arg_type()
#if defined(LIBXSTREAM_DEBUG)
      : size(0)
#endif
    {
      value.p = 0; value.d = 0;
    }

    template<typename T> arg_type(T arg)
#if defined(LIBXSTREAM_DEBUG)
      : size(sizeof(T))
#endif
    {
      const unsigned char *const src = reinterpret_cast<const unsigned char*>(&arg);
      unsigned char *const dst = reinterpret_cast<unsigned char*>(&value);
      for (size_t i = 0; i < sizeof(T); ++i) dst[i] = src[i];
      for (size_t i = sizeof(T); i < sizeof(value); ++i) dst[i] = 0;
    }
  };

public:
  libxstream_offload_region(libxstream_stream* stream, size_t argc, const arg_type argv[]);
  virtual ~libxstream_offload_region() {}

public:
  libxstream_stream* stream() const { return m_stream; }

  template<typename T,size_t i> T* ptr() const {
    LIBXSTREAM_ASSERT(i < m_argc && sizeof(T*) <= m_argv[i].size);
    return static_cast<T*>(m_argv[i].value.p);
  }

  template<typename T,size_t i> T val() const {
    LIBXSTREAM_ASSERT(i < m_argc && sizeof(T) <= m_argv[i].size);
    return *reinterpret_cast<const T*>(m_argv + i);
  }

  virtual libxstream_offload_region* clone() const = 0;
  virtual void operator()() const = 0;

private:
  arg_type m_argv[LIBXSTREAM_MAX_NARGS];
  libxstream_stream* m_stream;
#if defined(LIBXSTREAM_DEBUG)
  size_t m_argc;
#endif
};


void libxstream_offload(const libxstream_offload_region& offload_region, bool wait = true);
void libxstream_offload_shutdown();
bool libxstream_offload_busy();

#endif // LIBXSTREAM_CAPTURE_HPP
