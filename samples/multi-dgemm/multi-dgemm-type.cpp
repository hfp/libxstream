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
#include <cstdlib>


multi_dgemm_type::multi_dgemm_type()
  : m_device(-1)
  , m_index_hst(0), m_index_dev(0)
  , m_adata_hst(0), m_bdata_hst(0), m_cdata_hst(0)
  , m_adata_dev(0), m_bdata_dev(0), m_cdata_dev(0)
  , m_process_fn(0)
  , m_flops(0)
{}


multi_dgemm_type::~multi_dgemm_type()
{
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_index_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_adata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_bdata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_cdata_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_index_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_adata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_bdata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_cdata_hst));
}


bool multi_dgemm_type::ready() const
{
  return m_index_hst && m_index_dev
      && m_adata_hst && m_bdata_hst && m_cdata_hst
      && m_adata_dev && m_bdata_dev && m_cdata_dev;
}


size_t multi_dgemm_type::flops() const
{
  return m_flops;
}


int multi_dgemm_type::init(process_fn_type process_fn, int device, int size, const int split[])
{
  LIBXSTREAM_ASSERT(!ready());
  m_process_fn = process_fn;
  m_device = device;

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_index_hst), sizeof(size_t) * (size + 1), 0));

  int isize = split[0];
  size_t msize = 0, n = 100, nn = n * n;
  for (int i = 0; i < isize; ++i) {
    m_flops += nn * (2 * n + 1);
    m_index_hst[i] = msize;
    msize += nn;
  }
  isize += split[1];
  n = 600, nn = n * n;
  for (int i = split[0]; i < isize; ++i) {
    m_flops += nn * (2 * n + 1);
    m_index_hst[i] = msize;
    msize += nn;
  }
  n = 1000, nn = n * n;
  for (int i = isize; i < size; ++i) {
    m_flops += nn * (2 * n + 1);
    m_index_hst[i] = msize;
    msize += nn;
  }
  m_index_hst[size] = msize;

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_adata_hst), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_bdata_hst), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_cdata_hst), sizeof(double) * msize, 0));

  static const double scale = 1.0 / RAND_MAX;
  for (size_t i = 0; i < msize; ++i) {
    m_adata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
    m_bdata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
    m_cdata_hst[i] = 0;
  }

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_index_dev), sizeof(size_t) * (size + 1), 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_adata_dev), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_bdata_dev), sizeof(double) * msize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_cdata_dev), sizeof(double) * msize, 0));

  return LIBXSTREAM_ERROR_NONE;
}


int multi_dgemm_type::operator()(libxstream_stream& stream, int index, int size)
{
  LIBXSTREAM_ASSERT(0 != m_process_fn);

  if (0 < size) {
    const size_t i0 = m_index_hst[index], i1 = m_index_hst[index+size];
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_index_hst + index, m_index_dev + index, sizeof(size_t) * size, &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_adata_hst + i0, m_adata_dev + i0, sizeof(double) * (i1 - i0), &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_bdata_hst + i0, m_bdata_dev + i0, sizeof(double) * (i1 - i0), &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_cdata_hst + i0, m_cdata_dev + i0, sizeof(double) * (i1 - i0), &stream));

    LIBXSTREAM_OFFLOAD_BEGIN(stream, size, i1 - m_index_hst[index+size-1],
      m_index_dev + index, m_adata_dev, m_bdata_dev, m_cdata_dev, m_process_fn)
    {
      const int size = val<const int,0>();
      const int nn = val<const int,1>();
      const size_t *const i = ptr<const size_t,2>();
      const double *const a = ptr<const double,3>();
      const double *const b = ptr<const double,4>();
      double* c = ptr<double,5>();

      LIBXSTREAM_EXPORT process_fn_type process_fn = val<process_fn_type,6>();

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
    LIBXSTREAM_OFFLOAD_END(false)

    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_d2h(m_cdata_dev + i0, m_cdata_hst + i0, sizeof(double) * (i1 - i0), &stream));
  }

  return LIBXSTREAM_ERROR_NONE;
}
