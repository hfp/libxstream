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
#include "multi-dgemm-type.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

#define MULTI_DGEMM_TYPE_USE_DEMUX


multi_dgemm_type::host_data_type::host_data_type(int size, const int split[])
  : m_size(size)
  , m_adata(0), m_bdata(0), m_cdata(0)
  , m_idata(0), m_flops(0)
{
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_idata), sizeof(size_t) * (size + 1), 0));

  int isize = split[0];
  size_t msize = 0, n = 100, nn = n * n;
  for (int i = 0; i < isize; ++i) {
    m_flops += nn * (2 * n + 1);
    m_idata[i] = msize;
    msize += nn;
  }
  isize += split[1];
  n = 600, nn = n * n;
  for (int i = split[0]; i < isize; ++i) {
    m_flops += nn * (2 * n + 1);
    m_idata[i] = msize;
    msize += nn;
  }
  n = 1000, nn = n * n;
  for (int i = isize; i < size; ++i) {
    m_flops += nn * (2 * n + 1);
    m_idata[i] = msize;
    msize += nn;
  }
  m_idata[size] = msize;

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_adata), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_bdata), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_cdata), sizeof(double) * msize, 0));

  static const double scale = 1.0 / RAND_MAX;
  for (size_t i = 0; i < msize; ++i) {
    m_adata[i] = scale * (2 * std::rand() - RAND_MAX);
    m_bdata[i] = scale * (2 * std::rand() - RAND_MAX);
    m_cdata[i] = 0;
  }
}


multi_dgemm_type::host_data_type::~host_data_type()
{
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_adata));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_bdata));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_cdata));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_idata));
}


bool multi_dgemm_type::host_data_type::ready() const
{
  return m_adata && m_bdata && m_cdata && m_idata;
}


int multi_dgemm_type::host_data_type::size() const
{
  return m_size;
}


const double* multi_dgemm_type::host_data_type::adata() const
{
  return m_adata;
}


const double* multi_dgemm_type::host_data_type::bdata() const
{
  return m_bdata;
}


double* multi_dgemm_type::host_data_type::cdata()
{
  return m_cdata;
}


const size_t* multi_dgemm_type::host_data_type::idata() const
{
  return m_idata;
}


size_t multi_dgemm_type::host_data_type::max_matrix_size() const
{
  LIBXSTREAM_ASSERT(0 == m_size || 0 == m_idata[0]);
  size_t result = 0, i0 = 0;
  for (int i = 0; i < m_size; ++i) {
    const size_t i1 = m_idata[i+1];
    result = std::max(result, i1 - i0);
    i0 = i1;
  }
  return result;
}


size_t multi_dgemm_type::host_data_type::bytes() const
{
  return sizeof(double) * m_idata[m_size] * 3 + sizeof(size_t) * m_size;
}


size_t multi_dgemm_type::host_data_type::flops() const
{
  return m_flops;
}


multi_dgemm_type::multi_dgemm_type()
  : m_stream(0), m_host_data(0)
  , m_adata(0), m_bdata(0), m_cdata(0)
  , m_idata(0)
{}


multi_dgemm_type::~multi_dgemm_type()
{
  if (m_stream) {
    const int device = m_stream->device();
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_destroy(m_stream));
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, m_adata));
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, m_bdata));
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, m_cdata));
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, m_idata));
  }
}


bool multi_dgemm_type::ready() const
{
  return m_stream && m_host_data && m_adata && m_bdata && m_cdata && m_idata;
}


int multi_dgemm_type::init(host_data_type& host_data, int device, int max_batch)
{
  LIBXSTREAM_ASSERT(!ready());
  m_host_data = &host_data;

  char name[128];
  LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream 0x%lx", this);
#if defined(MULTI_DGEMM_TYPE_USE_DEMUX)
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(&m_stream, device, 1, 0, name));
#else
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(&m_stream, device, 0, 0, name));
#endif

  const int max_msize = max_batch * host_data.max_matrix_size();
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_adata), sizeof(double) * max_msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_bdata), sizeof(double) * max_msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_cdata), sizeof(double) * max_msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_idata), sizeof(size_t) * max_batch, 0));

  return LIBXSTREAM_ERROR_NONE;
}


int multi_dgemm_type::operator()(process_fn_type process_fn, int index, int size)
{
  LIBXSTREAM_CHECK_CONDITION(ready() && process_fn && (index + size) <= m_host_data->size());

  if (0 < size) {
#if !defined(MULTI_DGEMM_TYPE_USE_DEMUX)
    // This manual synchronization prevents multiple threads from queuing work into the *same* stream (at the same time).
    // This is only needed if the stream was created without demux support in order to implement manual synchronization.
    if (!m_stream->demux()) m_stream->lock();
#endif
    const size_t i0 = m_host_data->idata()[index], i1 = m_host_data->idata()[index+size];
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_host_data->adata() + i0, m_adata, sizeof(double) * (i1 - i0), m_stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_host_data->bdata() + i0, m_bdata, sizeof(double) * (i1 - i0), m_stream));
    // transferring cdata is part of the benchmark; since it is all zeros we could do better with libxstream_memset_zero
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_host_data->cdata() + i0, m_cdata, sizeof(double) * (i1 - i0), m_stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_host_data->idata() + index, m_idata, sizeof(size_t) * size, m_stream));

    LIBXSTREAM_OFFLOAD_BEGIN(m_stream, process_fn,
      size, i1 - m_host_data->idata()[index+size-1],
      m_adata, m_bdata, m_cdata, m_idata)
    {
      LIBXSTREAM_EXPORT process_fn_type process_fn = val<process_fn_type,0>();
      const int size = val<const int,1>();
      const int nn = val<const int,2>();
      const double *const a = ptr<const double,3>();
      const double *const b = ptr<const double,4>();
      double* c = ptr<double,5>();
      const size_t *const i = ptr<const size_t,6>();

#if defined(LIBXSTREAM_OFFLOAD)
      if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
        if (LIBXSTREAM_OFFLOAD_READY) {
#         pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL in(size, nn) \
            in(i: length(0) alloc_if(false) free_if(false)) \
            in(a: length(0) alloc_if(false) free_if(false)) \
            in(b: length(0) alloc_if(false) free_if(false)) \
            inout(c: length(0) alloc_if(false) free_if(false))
          {
            process_fn(size, nn, i, a, b, c);
          }
        }
        else {
#         pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT in(size, nn) \
            in(i: length(0) alloc_if(false) free_if(false)) \
            in(a: length(0) alloc_if(false) free_if(false)) \
            in(b: length(0) alloc_if(false) free_if(false)) \
            inout(c: length(0) alloc_if(false) free_if(false))
          {
            process_fn(size, nn, i, a, b, c);
          }
        }
      }
      else
#endif
      {
        process_fn(size, nn, i, a, b, c);
      }
    }
    LIBXSTREAM_OFFLOAD_END(false);

    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_d2h(m_cdata, m_host_data->cdata() + i0, sizeof(double) * (i1 - i0), m_stream));
#if !defined(MULTI_DGEMM_TYPE_USE_DEMUX)
    if (!m_stream->demux()) m_stream->unlock();
#endif
  }

  return LIBXSTREAM_ERROR_NONE;
}
