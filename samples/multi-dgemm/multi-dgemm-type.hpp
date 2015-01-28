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
#ifndef MULTI_DGEMM_TYPE_HPP
#define MULTI_DGEMM_TYPE_HPP

#include <libxstream.hpp>


class multi_dgemm_type {
public:
  typedef LIBXSTREAM_EXPORT void (*process_fn_type)(int, int,
    const size_t*, const double*, const double*, double*);

  class host_data_type {
  public:
    host_data_type(int size, const int split[]);
    ~host_data_type();
  public:
    bool ready() const;
    int size() const;
    const double* adata() const;
    const double* bdata() const;
    double* cdata();
    const size_t* index() const;
    size_t bytes() const;
    size_t flops() const;
  private:
    int m_size;
    double *m_adata, *m_bdata, *m_cdata;
    size_t *m_index, m_flops;
  };

public:
  multi_dgemm_type();
  ~multi_dgemm_type();

public:
  bool ready() const;
  int init(host_data_type& host_data, int device, int max_batch);
  int operator()(libxstream_stream& stream, process_fn_type process_fn, int index, int size);

private:
  host_data_type* m_host_data;
  int m_device;

  double *m_adata, *m_bdata, *m_cdata;
  size_t *m_index;
};

#endif // MULTI_DGEMM_TYPE_HPP
