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
#include <libxstream_begin.h>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>

#define COPY_ISYNC


int main(int argc, char* argv[])
{
  try {
#if defined(_OPENMP)
    const int nthreads = std::min(std::max(1 < argc ? std::atoi(argv[1]) : 1, 1), omp_get_max_threads());
#else
    const int nthreads = std::min(std::max(1 < argc ? std::atoi(argv[1]) : 1, 1), 1);
    LIBXSTREAM_PRINT0(1, "OpenMP support needed for performance results!");
    libxstream_use_sink(&nthreads);
#endif
    const int nstreams = std::min(std::max(2 < argc ? std::atoi(argv[2]) : 1, 1), LIBXSTREAM_MAX_NSTREAMS);
    const size_t maxsize = static_cast<size_t>(std::min(std::max(3 < argc ? std::atoi(argv[3]) : 2048, 1), 8192)) * (1 << 20), minsize = 8;
    int nrepeat = std::min(std::max(4 < argc ? std::atoi(argv[4]) : 7, 3), 100);

    const size_t stride = 2;
    for (size_t size = minsize, n = 1; size <= maxsize; size <<= 1, ++n) {
      if (0 == (n % stride)) {
        nrepeat <<= 1;
      }
    }

    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
      throw std::runtime_error("no device found!");
    }

    void *mem_hst = 0, *mem_dev = 0;
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1, &mem_hst, maxsize, 0));
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate( 0, &mem_dev, maxsize, 0));

    libxstream_stream* stream[LIBXSTREAM_MAX_NSTREAMS];
    for (int i = 0; i < nstreams; ++i) {
      char name[128];
      LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %i", i + 1);
      LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(stream + i, 0, 0, name));
    }

    for (size_t size = minsize, n = 1; size <= maxsize; size <<= 1, ++n) {
      if (0 == (n % stride)) {
        nrepeat >>= 1;
      }

#if defined(_OPENMP)
      const double start = omp_get_wtime();
#     pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
      for (int i = 0; i < nrepeat; ++i) {
        const int j = i % nstreams;
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memcpy_h2d(mem_hst, mem_dev, size, stream[j]));

#if defined(COPY_ISYNC)
        const int k = (j + 1) % nstreams;
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_sync(stream[k]));
#endif
      }

      // sync all streams to complete any pending work
      LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(0));

#if defined(_OPENMP)
      const double duration = omp_get_wtime() - start;
      LIBXSTREAM_FLOCK(stdout);
      fprintf(stdout, "%lu Byte x %i: ", static_cast<unsigned long>(size), nrepeat);
      if (0 < duration) {
        fprintf(stdout, "%.1f MB/s\n", (1.0 * size * nrepeat) / ((1ul << 20) * duration));
      }
      else {
        fprintf(stdout, "-\n");
      }
      LIBXSTREAM_FUNLOCK(stdout);
#endif
    }

    fprintf(stdout, "Finished\n");
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
