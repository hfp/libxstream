#include <libxstream_begin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <libxstream_end.h>


LIBXSTREAM_TARGET(mic) void histogram(const char* data, size_t* histogram)
{
  static const size_t maxint = (size_t)(((unsigned int)-1) >> 1);
  int i, j, m;
  size_t size;
  LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_get_shape(0/*current context*/, 0/*data*/, &size));

  m = (int)((size + maxint - 1) / maxint);
  for (i = 0; i < m; ++i) { /*OpenMP 2: size is broken down to integer space*/
    const size_t base = i * maxint;
    const int n = (int)LIBXSTREAM_MIN(size - base, maxint);
#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (j = 0; j < n; ++j) {
      const int k = (unsigned char)data[base+j];
#if defined(_OPENMP)
#     pragma omp atomic
#endif
      ++histogram[k];
    }
  }
}


FILE* fileopen(const char* name, const char* mode, size_t* size)
{
  FILE *const file = (name && *name) ? fopen(name, mode) : 0;
  long lsize = -1;

  if (0 != file) {
    if (0 == fseek(file, 0L, SEEK_END)) {
      lsize = ftell(file);
      rewind(file);
    }
  }

  if (0 != size && 0 <= lsize) {
    *size = lsize;
  }

  return 0 <= lsize ? file : 0;
}


int main(int argc, char* argv[])
{
  size_t ndevices = 0;
  if (LIBXSTREAM_ERROR_NONE != libxstream_get_ndevices(&ndevices) || 0 == ndevices) {
    LIBXSTREAM_PRINT0(1, "No device found or device not ready!");
  }

  size_t filesize = 0;
  FILE *const file = 1 < argc ? fileopen(argv[1], "rb", &filesize) : 0;
  const size_t nitems = (1 < argc && 0 == filesize && 0 < atoi(argv[1])) ? (atoi(argv[1]) * (1 << 20)/*MB*/) : (0 < filesize ? filesize : (512 << 20));
  const size_t mbatch = LIBXSTREAM_MIN(2 < argc ? strtoul(argv[2], 0, 10) : 0/*auto*/, nitems);
  const int nstreams = LIBXSTREAM_MIN(LIBXSTREAM_MAX(3 < argc ? atoi(argv[3]) : 2, 1), LIBXSTREAM_MAX_NSTREAMS) * (int)ndevices;
#if defined(_OPENMP)
  const int nthreads = LIBXSTREAM_MIN(LIBXSTREAM_MAX(4 < argc ? atoi(argv[4]) : 2, 1), omp_get_max_threads());
#else
  LIBXSTREAM_PRINT0(1, "OpenMP support needed for performance results!");
#endif
  const size_t nbatch = (0 == mbatch) ? (nitems / nstreams) : mbatch;

  char* data;
  { /*allocate and initialize host memory*/
    size_t i;
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_allocate(-1/*host*/, (void**)&data, nitems, 0));
    if (0 == filesize || nitems > fread(data, 1, filesize, file)) {
      for (i = 0; i < nitems; ++i) data[i] = LIBXSTREAM_MOD(rand(), 256/*POT*/);
    }
  }

  struct {
    libxstream_stream* handle;
    size_t* histogram;
    char* data;
  } stream[(LIBXSTREAM_MAX_NDEVICES)*(LIBXSTREAM_MAX_NSTREAMS)];

  { /*allocate and initialize streams and device memory*/
    int i;
    for (i = 0; i < nstreams; ++i) {
#if defined(NDEBUG) /*no name*/
      const char *const name = 0;
#else
      char name[128];
      LIBXSTREAM_SNPRINTF(name, sizeof(name), "stream %i", i + 1);
#endif
      const int device = (0 < ndevices) ? ((int)(i % ndevices)) : -1;
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_create(&stream[i].handle, device, 0, name));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_allocate(device, (void**)&stream[i].data, nbatch, 0));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_allocate(device, (void**)&stream[i].histogram, 256 * sizeof(size_t), 0));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memset_zero(stream[i].histogram, 256 * sizeof(size_t), stream[i].handle));
    }
  }

  /*process data in chunks of size nbatch*/
  const size_t hsize = 256;
  const int end = (int)((nitems + nbatch - 1) / nbatch);
  int batch;
  libxstream_type sizetype = LIBXSTREAM_TYPE_U32;
  LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_get_autotype(sizeof(size_t), sizetype, &sizetype));
#if defined(_OPENMP)
  const double start = omp_get_wtime();
# pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
  for (batch = 0; batch < end; ++batch) {
    libxstream_argument* signature;
    const int i = batch % nstreams; /*stream index*/
    const size_t j = batch * nbatch, size = LIBXSTREAM_MIN(nbatch, nitems - j);
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memcpy_h2d(data + j, stream[i].data, size, stream[i].handle));
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_fn_signature(&signature));
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_fn_input(signature, 0, stream[i].data, LIBXSTREAM_TYPE_CHAR, 1, &size));
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_fn_output(signature, 1, stream[i].histogram, sizetype, 1, &hsize));
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_fn_call((libxstream_function)histogram, signature, stream[i].handle, LIBXSTREAM_CALL_DEFAULT));
  }

  size_t histogram[256];
  memset(histogram, 0, sizeof(histogram));
  { /*reduce stream-local histograms*/
    LIBXSTREAM_ALIGNED(size_t local[256], LIBXSTREAM_MAX_SIMD);
    int i, j;
    for (i = 0; i < nstreams; ++i) {
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_memcpy_d2h(stream[i].histogram, local, sizeof(local), stream[i].handle));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_wait(stream[i].handle)); /*wait for pending work*/
      for (j = 0; j < 256; ++j) histogram[j] += local[j];
    }
  }

  const double kilo = 1.0 / (1 << 10), mega = 1.0 / (1 << 20);
  double entropy = 0;
  { /*calculate entropy*/
    const double log2_nitems = log2((double)nitems);
    int i;
    for (i = 0; i < 256; ++i) {
      const double h = (double)histogram[i];
      entropy -= h * (log2(h) - log2_nitems);
    }
    entropy /= nitems;
  }
  if ((1 << 20) <= nitems) { // mega
    fprintf(stdout, "Compression %.2fx (%0.f%%): %.1f -> %.1f MB", 8.0 / entropy, 100.0 - 12.5 * entropy, mega * nitems, mega * entropy * nitems / 8.0);
  }
  else if ((1 << 10) <= nitems) { // kilo
    fprintf(stdout, "Compression %.2fx (%0.f%%): %.1f -> %.1f KB", 8.0 / entropy, 100.0 - 12.5 * entropy, kilo * nitems, kilo * entropy * nitems / 8.0);
  }
  else  {
    fprintf(stdout, "Compression %.2fx (%0.f%%): %.0f -> %0.f B", 8.0 / entropy, 100.0 - 12.5 * entropy, 1.0 * nitems, entropy * nitems / 8.0);
  }
  fprintf(stdout, " (entropy of %.0f bit)\n", entropy);

#if defined(_OPENMP)
  const double duration = omp_get_wtime() - start;
  if (0 < duration) {
    fprintf(stdout, "Finished after %.1f s\n", duration);
  }
  else {
    fprintf(stdout, "Finished\n");
  }
#endif

  { /*release resources*/
    int i;
    for (i = 0; i < nstreams; ++i) {
      int device = -1;
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_device(stream[i].handle, &device));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_deallocate(device, stream[i].histogram));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_deallocate(device, stream[i].data));
      LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_stream_destroy(stream[i].handle));
    }
    LIBXSTREAM_CHECK_CALL_ASSERT(libxstream_mem_deallocate(-1/*host*/, data));
  }

  return EXIT_SUCCESS;
}
