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
#include "multi-dgemm-setup.hpp"
#include <libxstream.hpp>
#include <cstdlib>


setup_type::setup_type()
  : device(-1)
  , hdata_hst(0), idata_hst(0), jdata_hst(0)
  , hdata_dev(0), idata_dev(0), jdata_dev(0)
  , adata_hst(0), bdata_hst(0), cdata_hst(0)
  , adata_dev(0), bdata_dev(0), cdata_dev(0)
  , mdata_hst(0), ndata_hst(0), kdata_hst(0)
  , mdata_dev(0), ndata_dev(0), kdata_dev(0)
{}


setup_type::~setup_type()
{
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, hdata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, idata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, jdata_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, adata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, bdata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, cdata_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, mdata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, ndata_dev));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(device, kdata_dev));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, hdata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, idata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, jdata_hst));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, adata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, bdata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, cdata_hst));

  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, mdata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, ndata_hst));
  LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_deallocate(-1, kdata_hst));
}


bool setup_type::ready() const
{
  return adata_hst && bdata_hst && cdata_hst
      && adata_dev && bdata_dev && cdata_dev
      && hdata_hst && idata_hst && jdata_hst
      && hdata_dev && idata_dev && jdata_dev
      && mdata_hst && ndata_hst && kdata_hst
      && mdata_dev && ndata_dev && kdata_dev;
}


int setup_type::operator()(int device, int size, const int split[])
{
  return ready() ? LIBXSTREAM_ERROR_NONE : init(device, size, split);
}


int setup_type::init(int device, int size, const int split[])
{
  LIBXSTREAM_ASSERT(!ready());
  this->device = device;

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&hdata_hst), sizeof(size_t) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&idata_hst), sizeof(size_t) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&jdata_hst), sizeof(size_t) * size, 0));

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&mdata_hst), sizeof(int) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&ndata_hst), sizeof(int) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&kdata_hst), sizeof(int) * size, 0));

  size_t asize = 0, bsize = 0, csize = 0;
  int isize = split[0];
  for (int i = 0; i < isize; ++i) {
    mdata_hst[i] = ndata_hst[i] = kdata_hst[i] = 100;
    hdata_hst[i] = asize; idata_hst[i] = bsize; jdata_hst[i] = csize;
    asize += mdata_hst[i] * kdata_hst[i];
    bsize += kdata_hst[i] * ndata_hst[i];
    csize += mdata_hst[i] * ndata_hst[i];
  }
  isize += split[1];
  for (int i = split[0]; i < isize; ++i) {
    mdata_hst[i] = ndata_hst[i] = kdata_hst[i] = 600;
    hdata_hst[i] = asize; idata_hst[i] = bsize; jdata_hst[i] = csize;
    asize += mdata_hst[i] * kdata_hst[i];
    bsize += kdata_hst[i] * ndata_hst[i];
    csize += mdata_hst[i] * ndata_hst[i];
  }
  for (int i = isize; i < size; ++i) {
    mdata_hst[i] = ndata_hst[i] = kdata_hst[i] = 1000;
    hdata_hst[i] = asize; idata_hst[i] = bsize; jdata_hst[i] = csize;
    asize += mdata_hst[i] * kdata_hst[i];
    bsize += kdata_hst[i] * ndata_hst[i];
    csize += mdata_hst[i] * ndata_hst[i];
  }

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&adata_hst), sizeof(double) * asize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&bdata_hst), sizeof(double) * bsize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(-1, reinterpret_cast<void**>(&cdata_hst), sizeof(double) * csize, 0));

  static const double scale = 1.0 / RAND_MAX;
  for (int i = 0; i < asize; ++i) {
    adata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
  }
  for (int i = 0; i < bsize; ++i) {
    bdata_hst[i] = scale * (2 * std::rand() - RAND_MAX);
  }

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&hdata_dev), sizeof(size_t) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&idata_dev), sizeof(size_t) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&jdata_dev), sizeof(size_t) * size, 0));

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&adata_dev), sizeof(double) * asize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&bdata_dev), sizeof(double) * bsize, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&cdata_dev), sizeof(double) * csize, 0));

  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&mdata_dev), sizeof(int) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&ndata_dev), sizeof(int) * size, 0));
  LIBXSTREAM_CHECK_CALL(libxstream_mem_allocate(device, reinterpret_cast<void**>(&kdata_dev), sizeof(int) * size, 0));

  return LIBXSTREAM_ERROR_NONE;
}
