/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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

#include <libxstream_begin.h>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>

#define SYNCMETHOD 2
//#define CHECK

#define DGEMM dgemm_


LIBXSTREAM_EXTERN_C LIBXSTREAM_RETARGETABLE void DGEMM(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);


LIBXSTREAM_RETARGETABLE void process(LIBXSTREAM_INVAL(size_t) size, LIBXSTREAM_INVAL(size_t) nn, const size_t* idata,
  const double* adata, const double* bdata, double* cdata)
{
  if (0 < LIBXSTREAM_GETVAL(size)) {
    static const double alpha = 1, beta = 1;
    static const char trans = 'N';
    const int isize = static_cast<int>(size);
    const size_t base = idata[0];

    for (int i = 0; i < isize; ++i) {
      LIBXSTREAM_ASSERT(base <= idata[i]);
      const size_t i0 = idata[i], i1 = (i + 1) < isize ? idata[i+1] : (i0 + LIBXSTREAM_GETVAL(nn)), n2 = i1 - i0, offset = i0 - base;
      const int n = static_cast<int>(std::sqrt(static_cast<double>(n2)) + 0.5);
      DGEMM(&trans, &trans, &n, &n, &n, &alpha, adata + offset, &n, bdata + offset, &n, &beta, cdata + offset, &n);
    }
  }
}


int main(int argc, char* argv[])
{
  try {
    const size_t nitems = std::max(1 < argc ? std::atoi(argv[1]) : 60, 0);
    const size_t nbatch = std::max(2 < argc ? std::atoi(argv[2]) : 5, 1);
    const size_t mstreams = std::min(std::max(3 < argc ? std::atoi(argv[3]) : 2, 1), LIBXSTREAM_MAX_NSTREAMS);
#if !defined(_OPENMP)
    LIBXSTREAM_PRINT0(1, "OpenMP support needed for performance results!");
#endif

    size_t ndevices = 0;
    if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
      LIBXSTREAM_PRINT0(2, "No device found or device not ready!");
    }

    fprintf(stdout, "Initializing %i device%s and host data...", static_cast<int>(ndevices), 1 == ndevices ? "" : "s");
    const size_t split[] = { size_t(nitems * 18.0 / 250.0 + 0.5), size_t(nitems * 74.0 / 250.0 + 0.5) };
    multi_dgemm_type::host_data_type host_data(reinterpret_cast<libxstream_function>(&process), nitems, split);
    fprintf(stdout, " %.1f MB\n", host_data.bytes() * 1E-6);

    fprintf(stdout, "Initializing %i stream%s per device...", static_cast<int>(mstreams), 1 < mstreams ? "s" : "");
    const size_t nstreams = LIBXSTREAM_MAX(mstreams, 1) * LIBXSTREAM_MAX(ndevices, 1);
    multi_dgemm_type multi_dgemm[LIBXSTREAM_MAX_NSTREAMS];
    for (size_t i = 0; i < nstreams; ++i) {
      char name[128];
      LIBXSTREAM_SNPRINTF(name, sizeof(name), "Stream %i", static_cast<int>(i + 1));
      LIBXSTREAM_CHECK_CALL_THROW(multi_dgemm[i].init(name, host_data, 0 < ndevices ? static_cast<int>(i % ndevices) : -1, nbatch));
    }
    if (0 < nstreams) {
      fprintf(stdout, " %.1f MB\n", mstreams * multi_dgemm[0].bytes() * 1E-6);
    }

    // start benchmark with no pending work
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_wait(0));

    const size_t nbatches = (nitems + nbatch - 1) / nbatch;
    fprintf(stdout, "Running %i batch%s of %i item%s...\n", static_cast<int>(nbatches),
      1 < nbatches ? "es" : "", static_cast<int>(std::min(nbatch, nitems)),
      1 < nbatch ? "s" : "");

    const int end = static_cast<int>(nitems), ninc = static_cast<int>(nbatch * nstreams);
#if defined(_OPENMP)
    const double start = omp_get_wtime();
#endif
    for (int i = 0; i < end; i += ninc) {
      const size_t n = std::min<size_t>(nstreams, end - i);

      for (size_t j = 0; j < n; ++j) { // enqueue work into streams
        const size_t batch = j * nbatch, base = i + batch, size = base < nitems ? std::min(nbatch, nitems - base) : 0;
        multi_dgemm_type& call = multi_dgemm[j];
        LIBXSTREAM_CHECK_CALL_ASSERT(call(base, size));
#if defined(SYNCMETHOD) && (2 <= SYNCMETHOD) // record event
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_event_record(call.event(), call.stream()));
#endif
      }

#if defined(SYNCMETHOD)
      for (size_t j = 0; j < n; ++j) { // synchronize streams
        const size_t k = n - j - 1; // j-reverse
# if (3 <= (SYNCMETHOD))
        // wait for an event within a stream
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_wait_event(multi_dgemm[k].stream(), multi_dgemm[(j+nstreams-1)%n].event()));
# elif (2 <= (SYNCMETHOD))
        // wait for an event on the host
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_event_wait(multi_dgemm[k].event()));
# else
        LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_wait(multi_dgemm[k].stream()));
# endif
      }
#endif
    }

    // wait for all streams to complete pending work
    LIBXSTREAM_CHECK_CALL_THROW(libxstream_stream_wait(0));

#if defined(_OPENMP)
    const double duration = omp_get_wtime() - start;
    if (0 < duration) {
      fprintf(stdout, "Performance: %.1f GFLOPS/s\n", host_data.flops() * 1E-9 / duration);
    }
    fprintf(stdout, "Duration: %.1f s\n", duration);
#endif

#if !defined(CHECK)
    const char *const check_env = getenv("CHECK");
    if (check_env && *check_env && 0 != atoi(check_env))
#endif
    {
      std::vector<double> expected(host_data.max_matrix_size());
      const size_t testbatchsize = 1;
      double max_error = 0;
      size_t i0 = 0;
      for (size_t i = 0; i < nitems; ++i) {
        const size_t i1 = host_data.idata()[i+1];
        const int nn = static_cast<int>(i1 - i0);
        std::fill_n(&expected[0], nn, 0.0);
        process(LIBXSTREAM_SETVAL(testbatchsize), LIBXSTREAM_SETVAL(nn), host_data.idata() + i, host_data.adata() + i0, host_data.bdata() + i0, &expected[0]);
        for (int n = 0; n < nn; ++n) max_error = std::max(max_error, std::abs(expected[n] - host_data.cdata()[i0+n]));
        i0 = i1;
      }
      fprintf(stdout, "Error: %g\n", max_error);
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
