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
  , m_aindex_hst(0), m_bindex_hst(0), m_cindex_hst(0)
  , m_aindex_dev(0), m_bindex_dev(0), m_cindex_dev(0)
  , m_adata_hst(0), m_bdata_hst(0), m_cdata_hst(0)
  , m_adata_dev(0), m_bdata_dev(0), m_cdata_dev(0)
  , m_process_fn(0)
{}


multi_dgemm_type::~multi_dgemm_type()
{
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_aindex_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_bindex_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_cindex_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_adata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_bdata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(m_device, m_cdata_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_aindex_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_bindex_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_cindex_hst));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_adata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_bdata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, m_cdata_hst));
}


bool multi_dgemm_type::ready() const
{
  return m_aindex_hst && m_bindex_hst && m_cindex_hst
      && m_aindex_dev && m_bindex_dev && m_cindex_dev
      && m_adata_hst && m_bdata_hst && m_cdata_hst
      && m_adata_dev && m_bdata_dev && m_cdata_dev;
}


int multi_dgemm_type::init(process_fn_type process_fn, int device, int size, const int split[])
{
  LIBXSTREAM_ASSERT(!ready());
  m_process_fn = process_fn;
  m_device = device;

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_aindex_hst), sizeof(size_t) * (size + 1), 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_bindex_hst), sizeof(size_t) * (size + 1), 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_cindex_hst), sizeof(size_t) * (size + 1), 0));

  size_t asize = 0, bsize = 0, csize = 0;
  int isize = split[0];
  for (int i = 0; i < isize; ++i) {
    const int m = 100, n = 100, k = 100;
    m_aindex_hst[i] = asize;
    m_bindex_hst[i] = bsize;
    m_cindex_hst[i] = csize;
    asize += m * k;
    bsize += k * n;
    csize += m * n;
  }
  isize += split[1];
  for (int i = split[0]; i < isize; ++i) {
    const int m = 600, n = 600, k = 600;
    m_aindex_hst[i] = asize;
    m_bindex_hst[i] = bsize;
    m_cindex_hst[i] = csize;
    asize += m * k;
    bsize += k * n;
    csize += m * n;
  }
  for (int i = isize; i < size; ++i) {
    const int m = 1000, n = 1000, k = 1000;
    m_aindex_hst[i] = asize;
    m_bindex_hst[i] = bsize;
    m_cindex_hst[i] = csize;
    asize += m * k;
    bsize += k * n;
    csize += m * n;
  }
  m_aindex_hst[size] = asize;
  m_bindex_hst[size] = bsize;
  m_cindex_hst[size] = csize;

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_adata_hst), sizeof(double) * asize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_bdata_hst), sizeof(double) * bsize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&m_cdata_hst), sizeof(double) * csize, 0));

  static const double scale = 1.0 / RAND_MAX;
  for (int i = 0; i < asize; ++i) {
    m_adata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
  }
  for (int i = 0; i < bsize; ++i) {
    m_bdata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
  }
  for (int i = 0; i < csize; ++i) {
    m_cdata_hst[i] = 0;
  }

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_aindex_dev), sizeof(size_t) * (size + 1), 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_bindex_dev), sizeof(size_t) * (size + 1), 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_cindex_dev), sizeof(size_t) * (size + 1), 0));

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_adata_dev), sizeof(double) * asize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_bdata_dev), sizeof(double) * bsize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&m_cdata_dev), sizeof(double) * csize, 0));

  return LIBXSTREAM_ERROR_NONE;
}


int multi_dgemm_type::operator()(libxstream_stream& stream, int index, int size)
{
  LIBXSTREAM_ASSERT(0 != m_process_fn);

  if (0 < size) {
    const size_t a0 = m_aindex_hst[index], a1 = m_aindex_hst[index+size];
    const size_t b0 = m_bindex_hst[index], b1 = m_bindex_hst[index+size];
    const size_t c0 = m_cindex_hst[index], c1 = m_cindex_hst[index+size];

    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_aindex_hst + index, m_aindex_dev + index, sizeof(size_t) * size, &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_bindex_hst + index, m_bindex_dev + index, sizeof(size_t) * size, &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_cindex_hst + index, m_cindex_dev + index, sizeof(size_t) * size, &stream));

    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_adata_hst + a0, m_adata_dev + a0, sizeof(double) * (a1 - a0), &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_bdata_hst + b0, m_bdata_dev + b0, sizeof(double) * (b1 - b0), &stream));
    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_h2d(m_cdata_hst + c0, m_cdata_dev + c0, sizeof(double) * (c1 - c0), &stream));

    m_process_fn(size,
      a1 - m_aindex_hst[index+size-1],
      b1 - m_bindex_hst[index+size-1],
      c1 - m_cindex_hst[index+size-1],
      m_aindex_dev, m_bindex_dev, m_cindex_dev,
      m_adata_dev, m_bdata_dev, m_cdata_dev);

    LIBXSTREAM_CHECK_CALL(libxstream_memcpy_d2h(m_cdata_dev + c0, m_cdata_hst + c0, sizeof(double) * (c1 - c0), &stream));
  }

  return LIBXSTREAM_ERROR_NONE;
}
