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
#include <cmath>

#if defined(_OPENMP)
# include <omp.h>
#endif

//#define MULTI_DGEMM_USE_NESTED
#define DGEMM dgemm_


LIBXSTREAM_EXTERN_C LIBXSTREAM_EXPORT void DGEMM(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);


LIBXSTREAM_EXPORT void process(int size, int nn, const size_t* indexes,
  const double* adata, const double* bdata, double* cdata)
{
  static const double alpha = 1, beta = 1;
  static const char trans = 'N';

#if defined(_OPENMP) && defined(MULTI_DGEMM_USE_NESTED)
  const int nthreads = omp_get_max_threads() / size;
  omp_set_dynamic(0);
  omp_set_nested(1);
# pragma omp parallel for schedule(dynamic,1) num_threads(size)
#endif
  for (int i = 0; i < size; ++i) {
#if defined(_OPENMP) && defined(MULTI_DGEMM_USE_NESTED)
    omp_set_num_threads(nthreads);
#endif
    const size_t i0 = indexes[i], i1 = (i + 1) < size ? indexes[i+1] : (nn + i0), n2 = i1 - i0;
    const int n = static_cast<int>(std::sqrt(static_cast<double>(n2)) + 0.5);
    DGEMM(&trans, &trans, &n, &n, &n, &alpha, adata + i0, &n, bdata + i0, &n, &beta, cdata + i0, &n);
  }
}


int main(int argc, char* argv[])
{
  try {
    const int nitems = std::max(1 < argc ? std::atoi(argv[1]) : 32, 0);
    const int nbatch = std::max(2 < argc ? std::atoi(argv[2]) : 4, 1);
    const int nstreams = std::min(std::max(3 < argc ? std::atoi(argv[3]) : 2, 0), LIBXSTREAM_MAX_STREAMS);

    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE == libxstream_get_ndevices(&ndevices) && 0 < ndevices) {
      fprintf(stdout, "Initializing host data...");
      const int split[] = { int(nitems * 18.0 / 250.0 + 0.5), int(nitems * 74.0 / 250.0 + 0.5) };
      multi_dgemm_type::host_data_type host_data(nitems, split);
      fprintf(stdout, " %.1f MB.\n", host_data.bytes() * 1E-6);

      fprintf(stdout, "Preparing %i stream%s per device...\n", nstreams, 1 < nstreams ? "s" : "");
      const int idevices = static_cast<int>(ndevices);
      multi_dgemm_type multi_dgemm[LIBXSTREAM_MAX_DEVICES];
      libxstream_stream* streams[LIBXSTREAM_MAX_STREAMS*LIBXSTREAM_MAX_DEVICES];
      std::fill_n(streams, LIBXSTREAM_MAX_STREAMS * LIBXSTREAM_MAX_DEVICES, static_cast<libxstream_stream*>(0));
      fprintf(stdout, "Allocating memory for %i device%s...\n", idevices, 1 == idevices ? "" : "s");
      for (int device = 0; device < idevices; ++device) {
        LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[device].init(host_data, device));
        for (int i = 0; i < nstreams; ++i) {
          const int stream = device * idevices + i;
          char name[128];
          LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %d", stream + 1);
          LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(streams + stream, device, 0, name));
        }
      }

      const int nbatches = (nitems + nbatch - 1) / nbatch;
      fprintf(stdout, "Running %i batch%s of %i item%s...\n", nbatches, 1 < nbatches ? "es" : "", std::min(nbatch, nitems), 1 < nbatch ? "s" : "");
#if defined(_OPENMP)
      const double start = omp_get_wtime();
#     pragma omp parallel
#endif
      for (int i = 0; i < nitems; i += nbatch) {
#if defined(_OPENMP)
#       pragma omp single nowait
#endif
        {
#if defined(_OPENMP) && (200203 < _OPENMP)
#         pragma omp task
#endif
          {
            const int stream = i % (idevices * nstreams), device = streams[stream]->device();
            LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[device](*streams[stream], process, i, std::min(nbatch, nitems - i)));
          }
        }
      }

      // sync all streams to complete any pending work
      LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(0));

#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      fprintf(stdout, "Performance: %.1f GFLOPS/s\n", host_data.flops() * 1E-9 / duration);
      fprintf(stdout, "Duration: %.1f s\n", duration);
#endif

      std::for_each(streams, streams + LIBXSTREAM_MAX_STREAMS * LIBXSTREAM_MAX_DEVICES,
        std::ptr_fun(libxstream_stream_destroy));
      fprintf(stdout, "Finished.\n");
    }
    else {
      fprintf(stderr, "Error: no device found!\n");
    }
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
