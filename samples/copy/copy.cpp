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
#include <cstring>
#include <cstdio>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>

//#define COPY_NO_SYNC


int main(int argc, char* argv[])
{
  try {
    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
      LIBXSTREAM_PRINT0(1, "No device found!");
    }
    const int device = static_cast<int>(ndevices) - 1;
    size_t allocatable = 0;
    libxstream_mem_info(device, &allocatable, 0);
    allocatable >>= 20; // MB

    const bool copyin = 1 < argc ? ('o' != *argv[1]) : true;
#if defined(_OPENMP)
    const int nthreads = std::min(std::max(2 < argc ? std::atoi(argv[2]) : 1, 1), omp_get_max_threads());
#else
    //const int nthreads = std::min(std::max(2 < argc ? std::atoi(argv[2]) : 1, 1), 1);
    LIBXSTREAM_PRINT0(1, "OpenMP support needed for performance results!");
#endif
    const int nstreams = std::min(std::max(3 < argc ? std::atoi(argv[3]) : 1, 1), LIBXSTREAM_MAX_NSTREAMS);
#if defined(LIBXSTREAM_OFFLOAD)
    const size_t nreserved = nstreams + 1;
#else
    const size_t nreserved = nstreams + 2;
#endif
    const size_t minsize = 8, maxsize = static_cast<size_t>(std::min(std::max(4 < argc ? std::atoi(argv[4]) : static_cast<int>(allocatable / nreserved), 1), 8192)) * (1 << 20);
    const int minrepeat = std::min(std::max(5 < argc ? std::atoi(argv[5]) :    8, 4), 2048);
    const int maxrepeat = std::min(std::max(6 < argc ? std::atoi(argv[6]) : 4096, minrepeat), 32768);

    int steps_repeat = 0, steps_size = 0;
    for (int nrepeat = minrepeat; nrepeat <= maxrepeat; nrepeat <<= 1) ++steps_repeat;
    for (size_t size = minsize; size <= maxsize; size <<= 1) ++steps_size;
    const int stride = 0 < steps_repeat ? (steps_size / steps_repeat + 1) : 1;

    struct {
      libxstream_stream* stream;
      void *mem_hst, *mem_dev;
    } copy[LIBXSTREAM_MAX_NSTREAMS];
    memset(copy, 0, sizeof(copy)); // some initialization (avoid false positives with tools)
    for (int i = 0; i < nstreams; ++i) {
      char name[128];
      LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %i", i + 1);
      LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_create(&copy[i].stream, device, 0, name));
      if (copyin) {
        if (0 == i) {
          LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1/*host*/, &copy[i].mem_hst, maxsize, 0));
          memset(copy[i].mem_hst, -1, maxsize); // some initialization (avoid false positives with tools)
        }
        else {
          copy[i].mem_hst = copy[0].mem_hst;
        }
        LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(device, &copy[i].mem_dev, maxsize, 0));
      }
      else { // copy-out
        LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(-1/*host*/, &copy[i].mem_hst, maxsize, 0));
        memset(copy[i].mem_hst, -1, maxsize); // some initialization (avoid false positives with tools)
        if (0 == i) {
          LIBXSTREAM_CHECK_CALL_THROW(libxstream_mem_allocate(device, &copy[i].mem_dev, maxsize, 0));
        }
        else {
          copy[i].mem_dev = copy[0].mem_dev;
        }
      }
    }

    int n = 0, nrepeat = maxrepeat;
    double maxval = 0, sumval = 0, runlns = 0, duration = 0;
    for (size_t size = minsize; size <= maxsize; size <<= 1, ++n) {
      if (0 < n && 0 == (n % stride)) {
        nrepeat >>= 1;
      }

#if defined(_OPENMP)
      fprintf(stdout, "%lu Byte x %i: ", static_cast<unsigned long>(size), nrepeat);
      fflush(stdout);
      const double start = omp_get_wtime();
#     pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
      for (int i = 0; i < nrepeat; ++i) {
        const int j = i % nstreams;
        if (copyin) {
          LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memcpy_h2d(copy[j].mem_hst, copy[j].mem_dev, size, copy[j].stream));
        }
        else {
          LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memcpy_d2h(copy[j].mem_dev, copy[j].mem_hst, size, copy[j].stream));
        }

#if !defined(COPY_NO_SYNC)
        const int k = (j + 1) % nstreams;
        LIBXSTREAM_ASSERT(0 <= k && k < LIBXSTREAM_MAX_NSTREAMS);
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_sync(copy[k].stream));
#endif
      }

      // sync all streams to complete any pending work
      LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_sync(0));

#if defined(_OPENMP)
      const double iduration = omp_get_wtime() - start;
      if (0 < iduration) {
        const double bandwidth = (1.0 * size * nrepeat) / ((1ul << 20) * iduration);
        fprintf(stdout, "%.1f MB/s\n", bandwidth);
        maxval = std::max(maxval, bandwidth);
        sumval += bandwidth;
        runlns = (runlns + std::log(bandwidth)) * (0 < n ? 0.5 : 1.0);
        duration += iduration;
      }
      else {
        fprintf(stdout, "-\n");
      }
#endif
    }

    fprintf(stdout, "\n");
    if (0 < n) {
      fprintf(stdout, "Finished after %.0f s\n", duration);
      fprintf(stdout, "max: %.0f MB/s\n", maxval);
      fprintf(stdout, "rgm: %.0f MB/s\n", std::exp(runlns));
      fprintf(stdout, "avg: %.0f MB/s\n", sumval / n);
    }
    else {
      fprintf(stdout, "Finished\n");
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
