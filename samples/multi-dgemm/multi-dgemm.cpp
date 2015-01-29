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
#include <libxstream.hpp>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#if defined(_OPENMP)
# include <omp.h>
#endif

//#define MULTI_DGEMM_USE_NESTED
#define MULTI_DGEMM_USE_CHECK
#define MULTI_DGEMM_USE_ISYNC

#define DGEMM dgemm_


LIBXSTREAM_EXTERN_C LIBXSTREAM_EXPORT void DGEMM(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);


LIBXSTREAM_EXPORT void process(int size, int nn, const size_t* idata,
  const double* adata, const double* bdata, double* cdata)
{
  if (0 < size) {
#if defined(LIBXSTREAM_DEBUG)
    fprintf(stdout, "processing offset %lu\n", static_cast<unsigned long>(idata[0]));
#endif
    static const double alpha = 1, beta = 1;
    static const char trans = 'N';
    const size_t base = idata[0];

#if defined(_OPENMP) && defined(MULTI_DGEMM_USE_NESTED)
    const int nthreads = omp_get_max_threads() / size;
    const int dynamic = omp_get_dynamic(), nested = omp_get_nested();
    omp_set_dynamic(0);
    omp_set_nested(1);
#   pragma omp parallel for schedule(dynamic,1) num_threads(size)
#endif
    for (int i = 0; i < size; ++i) {
#if defined(_OPENMP) && defined(MULTI_DGEMM_USE_NESTED)
      omp_set_num_threads(nthreads);
#endif
      LIBXSTREAM_ASSERT(base <= idata[i]);
      const size_t i0 = idata[i], i1 = (i + 1) < size ? idata[i+1] : (i0 + nn), n2 = i1 - i0, offset = i0 - base;
      const int n = static_cast<int>(std::sqrt(static_cast<double>(n2)) + 0.5);
      DGEMM(&trans, &trans, &n, &n, &n, &alpha, adata + offset, &n, bdata + offset, &n, &beta, cdata + offset, &n);
    }
  }

#if defined(_OPENMP) && defined(MULTI_DGEMM_USE_NESTED)
    omp_set_dynamic(dynamic);
    omp_set_nested(nested);
#endif
}


int main(int argc, char* argv[])
{
  try {
    const int nitems = std::max(1 < argc ? std::atoi(argv[1]) : 32, 0);
    const int nbatch = std::max(2 < argc ? std::atoi(argv[2]) : 4, 1);
    const int nstreams = std::min(std::max(3 < argc ? std::atoi(argv[3]) : 2, 0), LIBXSTREAM_MAX_STREAMS);

    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
      throw std::runtime_error("no device found!");
    }
#if !defined(_OPENMP)
    fprintf(stdout, "Warning: OpenMP support needed for timing results.\n");
#endif

    fprintf(stdout, "Initializing host data...");
    const int split[] = { int(nitems * 18.0 / 250.0 + 0.5), int(nitems * 74.0 / 250.0 + 0.5) };
    multi_dgemm_type::host_data_type host_data(nitems, split);
    fprintf(stdout, " %.1f MB.\n", host_data.bytes() * 1E-6);

    fprintf(stdout, "Preparing %i stream%s per device...\n", nstreams, 1 < nstreams ? "s" : "");
    const int idevices = static_cast<int>(ndevices), istreams = idevices * nstreams;
    multi_dgemm_type multi_dgemm[LIBXSTREAM_MAX_DEVICES];
    libxstream_stream* streams[LIBXSTREAM_MAX_STREAMS*LIBXSTREAM_MAX_DEVICES];
    std::fill_n(streams, LIBXSTREAM_MAX_STREAMS * LIBXSTREAM_MAX_DEVICES, static_cast<libxstream_stream*>(0));

    fprintf(stdout, "Allocating memory for %i device%s...\n", idevices, 1 == idevices ? "" : "s");
    for (int device = 0; device < idevices; ++device) {
      LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[device].init(host_data, device, nbatch));
      for (int i = 0; i < nstreams; ++i) {
        const int stream = device * idevices + i;
        LIBXSTREAM_ASSERT(stream <= istreams);
        char name[128];
        LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %d", stream + 1);
        LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(streams + stream, device, 0, name));
      }
    }

    const int nbatches = (nitems + nbatch - 1) / nbatch;
    fprintf(stdout, "Running %i batch%s of %i item%s...\n", nbatches,
      1 < nbatches ? "es" : "", std::min(nbatch, nitems),
      1 < nbatch ? "s" : "");

#if defined(_OPENMP)
# if !defined(LIBXSTREAM_OFFLOAD)
    omp_set_dynamic(0);
    omp_set_nested(1);
# endif
    const double start = omp_get_wtime();
#   pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < nitems; i += nbatch) {
      const int istream = i % istreams;
#if defined(MULTI_DGEMM_USE_ISYNC)
      if (0 < i && 0 == istream) {
        LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(0));
      }
#endif
      libxstream_stream *const stream = streams[istream];
      LIBXSTREAM_ASSERT(stream); // initialized above
      LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[stream->device()](*stream, &process, i, std::min(nbatch, nitems - i)));
    }

    // sync all streams to complete any pending work
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(0));

#if defined(_OPENMP)
    const double duration = omp_get_wtime() - start;
    fprintf(stdout, "Performance: %.1f GFLOPS/s\n", host_data.flops() * 1E-9 / duration);
    fprintf(stdout, "Duration: %.1f s\n", duration);
#endif

    std::for_each(streams, streams + istreams, std::ptr_fun(libxstream_stream_destroy));

#if defined(MULTI_DGEMM_USE_CHECK)
    std::vector<double> expected(host_data.max_matrix_size());
    double max_error = 0;
    size_t i0 = 0, i1 = 0;
    for (int i = 0; i < nitems; ++i) {
      i1 = host_data.idata()[i+1];
      const int nn = i1 - i0;
      std::fill_n(&expected[0], nn, 0.0);
      process(1, nn, host_data.idata() + i, host_data.adata() + i0, host_data.bdata() + i0, &expected[0]);
      for (int n = 0; n < nn; ++n) max_error = std::max(max_error, std::abs(expected[n] - host_data.cdata()[i0+n]));
      i0 = i1;
    }
    fprintf(stdout, "Error: %f\n", max_error);
#endif
    fprintf(stdout, "Finished.\n");
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
