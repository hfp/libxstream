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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cassert>

#if defined(_OPENMP)
# include <omp.h>
#endif


LIBXSTREAM_EXPORT void process(int size, int mk, int kn, int mn,
  const size_t* aindex, const size_t* bindex, const size_t* cindex,
  const double* adata, const double* bdata, double* cdata)
{
  fprintf(stderr, "Running:");
  for (int i = 0; i < size; ++i) {
    const size_t mki = (i + 1) < size ? (aindex[i+1] - aindex[i]) : mk;
    const size_t kni = (i + 1) < size ? (bindex[i+1] - bindex[i]) : kn;
    const size_t mni = (i + 1) < size ? (cindex[i+1] - cindex[i]) : mn;
    const int m = static_cast<int>(std::sqrt(static_cast<double>(mki * mni) / kni + 0.5));
    const int n = static_cast<int>(static_cast<double>(mni) / m + 0.5);
    const int k = static_cast<int>(static_cast<double>(mki) / m + 0.5);
    fprintf(stderr, " %ix%ix%i", m, n, k);
  }
  fprintf(stderr, "\n");
}


int main(int argc, char* argv[])
{
  try {
    const int nstreams = std::min(std::max(1 < argc ? std::atoi(argv[1]) : 2, 0), LIBXSTREAM_MAX_STREAMS);
    const int nitems = std::max(2 < argc ? std::atoi(argv[2]) : 250, 0);
    const int nbatch = std::max(3 < argc ? std::atoi(argv[3]) : 1, 1);

    const int split[] = { int(nitems * 18.0 / 250.0 + 0.5), int(nitems * 74.0 / 250.0 + 0.5) };
    size_t ndevices = 0;

    if (LIBXSTREAM_ERROR_NONE == libxstream_get_ndevices(&ndevices) && 0 < ndevices) {
      fprintf(stderr, "Running %i items in batches of %i item(s)...\n", nitems, nbatch);
      multi_dgemm_type multi_dgemm[LIBXSTREAM_MAX_DEVICES];

      libxstream_stream* streams[LIBXSTREAM_MAX_STREAMS];
      std::fill_n(streams, LIBXSTREAM_MAX_STREAMS, static_cast<libxstream_stream*>(0));

#if defined(_OPENMP)
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
            const int stream = i % nstreams, device = stream % ndevices;

            if (0 == streams[stream]) {
#if defined(_OPENMP)
              const void *const id = static_cast<char*>(0) + ndevices + stream;
#             pragma omp critical(id)
#endif
              if (0 == streams[stream]) {
                if (!multi_dgemm[device].ready()) {
                  LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[device].init(process, device, nitems, split));
                }
                LIBXSTREAM_ASSERT(multi_dgemm[device].ready());

                char name[128];
                LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %d", stream + 1);
                LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(streams + stream, device, 0, name));
              }
            }

            LIBXSTREAM_ASSERT(0 != streams[stream]);
            multi_dgemm[device](*streams[stream], i, std::min(nbatch, nitems - i));
          }
        }
      }

      std::for_each(streams, streams + LIBXSTREAM_MAX_STREAMS, std::ptr_fun(libxstream_stream_destroy));
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
