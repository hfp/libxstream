/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxstream/libxstream_blas.h>
#include <libxstream/libxstream_opencl.h>
#include "libxstream_blas_plugin.h"
#if defined(LIBXS_INTERCEPT_DYNAMIC)
# include <unistd.h>
#endif

/* the plugin: beside the library, in the directory it was built into (static), or by the dynamic loader */
#if !defined(LIBXSTREAM_BLAS_PLUGIN_FILE)
# define LIBXSTREAM_BLAS_PLUGIN_FILE "libxstream_mkl.so"
#endif


/* The plugin's table, and its state: 0 = not yet tried, 1 = loading, 2 = loaded, -1 = none. */
static libxstream_blas_plugin_t libxstream_blas_table;
static volatile int libxstream_blas_state;
/* Context the plugin was asked about, and whether it serves it: 0 = not yet asked, 1 = asking, 2 = yes, -1 = no. */
static void* volatile libxstream_blas_context;
static volatile int libxstream_blas_serves;


/* Loads the plugin (LIBXSTREAM_BLAS: 0 disables it, a path selects it), once per process. */
static void libxstream_blas_load(void);
static void libxstream_blas_load(void)
{
  int state = -1;
#if defined(LIBXS_INTERCEPT_DYNAMIC)
  const char* const env = getenv("LIBXSTREAM_BLAS");
  if (NULL == env || '0' != env[0] || '\0' != env[1]) {
    char path[PATH_MAX];
    union { void (*fn)(void); const void* ptr; } self;
    union { libxstream_blas_plugin_fn fn; void* ptr; } entry;
    void* handle = NULL;
    Dl_info info;
    path[0] = '\0';
    self.fn = libxstream_blas_load;
    if (NULL != env && '\0' != *env && ('1' != env[0] || '\0' != env[1])) {
      LIBXS_SNPRINTF(path, sizeof(path), "%s", env);
    }
    else if (0 != dladdr(self.ptr, &info) && NULL != info.dli_fname) { /* beside the library */
      const char* const slash = strrchr(info.dli_fname, '/');
      if (NULL != slash && 0 < LIBXS_SNPRINTF(path, sizeof(path), "%.*s/%s", (int)(slash - info.dli_fname),
        info.dli_fname, LIBXSTREAM_BLAS_PLUGIN_FILE) && 0 != access(path, R_OK))
      {
        path[0] = '\0';
      }
    }
#if defined(LIBXSTREAM_BLAS_PLUGIN_DIR)
    if ('\0' == path[0] && (0 >= LIBXS_SNPRINTF(path, sizeof(path), "%s/%s", LIBXSTREAM_BLAS_PLUGIN_DIR,
      LIBXSTREAM_BLAS_PLUGIN_FILE) || 0 != access(path, R_OK)))
    {
      path[0] = '\0';
    }
#endif
    if ('\0' == path[0]) LIBXS_SNPRINTF(path, sizeof(path), "%s", LIBXSTREAM_BLAS_PLUGIN_FILE);
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL); /* its runtime stays out of the global scope */
    if (NULL != handle) {
      entry.ptr = dlsym(handle, LIBXSTREAM_BLAS_PLUGIN_ENTRY);
      LIBXS_MEMZERO(&libxstream_blas_table);
      libxstream_blas_table.version = LIBXSTREAM_BLAS_PLUGIN_VERSION;
      if (NULL != entry.ptr && 0 == entry.fn(&libxstream_blas_table)
        && LIBXSTREAM_BLAS_PLUGIN_VERSION == libxstream_blas_table.version && NULL != libxstream_blas_table.name)
      {
        state = 2; /* never unloaded (see libxstream_blas_plugin_t) */
      }
      else dlclose(handle);
    }
    if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
      LIBXS_STDIO_ACQUIRE();
      if (2 == state) fprintf(stderr, "INFO LIBXSTREAM: device BLAS %s (%s)\n", libxstream_blas_table.name, path);
      else fprintf(stderr, "INFO LIBXSTREAM: no device BLAS (%s)\n", path);
      LIBXS_STDIO_RELEASE();
    }
  }
#endif
  LIBXS_ATOMIC_STORE(&libxstream_blas_state, state, LIBXS_ATOMIC_SEQ_CST);
}


/* Kind of memory libxstream_mem_allocate returns (LIBXSTREAM_BLAS_MEM_*), as the device pool prefers it. */
static int libxstream_blas_memkind(void);
static int libxstream_blas_memkind(void)
{
  const libxstream_opencl_device_t* const device = &libxstream_opencl_config.device;
  return (0 == (2 & libxstream_mem_pooled()) ? LIBXSTREAM_BLAS_MEM_BUFFER
    : ((NULL != device->clDeviceMemAllocINTEL || NULL != device->clSharedMemAllocINTEL) ? LIBXSTREAM_BLAS_MEM_USM
      : LIBXSTREAM_BLAS_MEM_SVM));
}


/**
 * C := 2*A*B + C (3x2, with A at an offset into memory of libxstream_mem_allocate) against the host,
 * in each precision the plugin accepts, on LIBXSTREAM's own stream: it checks the memory as callers
 * pass it (an address within an allocation) rather than the plugin's own. Small integers make it exact.
 */
static int libxstream_blas_check(const libxstream_blas_plugin_t* plugin);
static int libxstream_blas_check(const libxstream_blas_plugin_t* plugin)
{
  enum { m = 3, n = 2, k = 4, offset = 5, size = offset + m * k + k * n + m * n };
  libxstream_stream_t* const stream = &libxstream_opencl_config.device.stream;
  double hd[size], ref[m * n], cd[m * n];
  float hf[size], cf[m * n];
  char* dev = NULL;
  int result = libxstream_mem_allocate((void**)&dev, sizeof(hd)), nserved = 0, f64, i, j, p;
  for (i = 0; i < size; ++i) hf[i] = (float)(hd[i] = (double)(i % 7 - 3));
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      double c = hd[offset + m * k + k * n + i + j * m];
      for (p = 0; p < k; ++p) c += 2 * hd[offset + i + p * m] * hd[offset + m * k + p + j * k];
      ref[i + j * m] = c;
    }
  }
  for (f64 = 0; f64 < 2 && EXIT_SUCCESS == result; ++f64) {
    const size_t es = (0 != f64 ? sizeof(double) : sizeof(float));
    void* const host = (0 != f64 ? (void*)hd : (void*)hf);
    result = libxstream_mem_copy_h2d(host, dev, es * size, stream);
    if (EXIT_SUCCESS == result) {
      char* const a = dev + es * offset;
      if (0 == plugin->gemm(libxstream_opencl_config.device.context, stream->queue, f64, 'N', 'N', m, n, k, 2.0, a, m,
        a + es * m * k, k, 1.0, a + es * (m * k + k * n), m))
      {
        result = libxstream_mem_copy_d2h(a + es * (m * k + k * n), 0 != f64 ? (void*)cd : (void*)cf, es * m * n, stream);
        if (EXIT_SUCCESS == result) result = libxstream_stream_sync(stream);
        for (i = 0; i < m * n && EXIT_SUCCESS == result; ++i) {
          if ((0 != f64 ? cd[i] : (double)cf[i]) != ref[i]) result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result) ++nserved;
      }
      else result = libxstream_stream_sync(stream); /* precision declined */
    }
  }
  if (NULL != dev) libxstream_mem_deallocate(dev);
  return (EXIT_SUCCESS == result && 0 < nserved) ? EXIT_SUCCESS : EXIT_FAILURE;
}


/* The plugin if it serves the device (probe and check, once per context), NULL otherwise. */
static const libxstream_blas_plugin_t* libxstream_blas_find(const libxstream_stream_t* stream);
static const libxstream_blas_plugin_t* libxstream_blas_find(const libxstream_stream_t* stream)
{
  const libxstream_blas_plugin_t* result = NULL;
  int state = LIBXS_ATOMIC_LOAD(&libxstream_blas_state, LIBXS_ATOMIC_SEQ_CST);
  if (0 == state) {
    int untried = 0;
    if (LIBXS_ATOMIC_CMPSWP(&libxstream_blas_state, untried, 1, LIBXS_ATOMIC_SEQ_CST)) libxstream_blas_load();
    while (1 == (state = LIBXS_ATOMIC_LOAD(&libxstream_blas_state, LIBXS_ATOMIC_SEQ_CST))) LIBXS_SYNC_PAUSE;
  }
  if (2 == state && NULL != stream && NULL != stream->queue) {
    void* const context = libxstream_opencl_config.device.context;
    int serves = LIBXS_ATOMIC_LOAD(&libxstream_blas_serves, LIBXS_ATOMIC_SEQ_CST);
    if (1 != serves && context != libxstream_blas_context /* another context (device): ask again */
      && LIBXS_ATOMIC_CMPSWP(&libxstream_blas_serves, serves, 1, LIBXS_ATOMIC_SEQ_CST))
    {
      const int kind = libxstream_blas_memkind();
      const int served = (0 == libxstream_blas_table.probe(context, libxstream_opencl_config.device.stream.queue, kind)
        && EXIT_SUCCESS == libxstream_blas_check(&libxstream_blas_table));
      if (2 <= libxstream_opencl_config.verbosity || 0 > libxstream_opencl_config.verbosity) {
        static const char* const kinds[] = {"", "Intel USM", "SVM", "buffers"};
        LIBXS_STDIO_ACQUIRE();
        fprintf(stderr, "INFO LIBXSTREAM: device BLAS %s %s with %s\n", libxstream_blas_table.name,
          0 != served ? "serves" : "declines", kinds[kind]);
        LIBXS_STDIO_RELEASE();
      }
      libxstream_blas_context = context;
      LIBXS_ATOMIC_STORE(&libxstream_blas_serves, 0 != served ? 2 : -1, LIBXS_ATOMIC_SEQ_CST);
    }
    while (1 == (serves = LIBXS_ATOMIC_LOAD(&libxstream_blas_serves, LIBXS_ATOMIC_SEQ_CST))) LIBXS_SYNC_PAUSE;
    if (2 == serves) result = &libxstream_blas_table;
  }
  return result;
}


/* Single or double precision: 1 or 0, and -1 for any other datatype. */
static int libxstream_blas_f64(libxs_data_t datatype);
static int libxstream_blas_f64(libxs_data_t datatype)
{
  return (LIBXS_DATATYPE_F64 == datatype ? 1 : (LIBXS_DATATYPE_F32 == datatype ? 0 : -1));
}


LIBXSTREAM_API const char* libxstream_blas_query(libxstream_stream_t* stream)
{
  const libxstream_blas_plugin_t* const plugin = libxstream_blas_find(stream);
  return (NULL != plugin ? plugin->name : NULL);
}


LIBXSTREAM_API int libxstream_blas_gemm(libxstream_stream_t* stream, libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  const libxstream_blas_plugin_t* const plugin = libxstream_blas_find(stream);
  const int f64 = libxstream_blas_f64(datatype);
  int result = EXIT_FAILURE;
  if (NULL != plugin && NULL != plugin->gemm && 0 <= f64
    && 0 == plugin->gemm(libxstream_opencl_config.device.context, stream->queue, f64, transa, transb, m, n, k, alpha, a,
      lda, b, ldb, beta, c, ldc))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXSTREAM_API int libxstream_blas_syrk(libxstream_stream_t* stream, libxs_data_t datatype, char uplo, char trans,
  int n, int k, double alpha, const void* a, int lda, const void* b, int ldb, double beta, void* c, int ldc)
{
  const libxstream_blas_plugin_t* const plugin = libxstream_blas_find(stream);
  const int f64 = libxstream_blas_f64(datatype);
  int result = EXIT_FAILURE;
  if (NULL != plugin && NULL != plugin->syrk && 0 <= f64
    && 0 == plugin->syrk(libxstream_opencl_config.device.context, stream->queue, f64, uplo, trans, n, k, alpha, a, lda,
      b, ldb, beta, c, ldc))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXSTREAM_API int libxstream_blas_dgmm(libxstream_stream_t* stream, libxs_data_t datatype, char side, int m, int n,
  const void* a, int lda, const void* x, int incx, void* c, int ldc)
{
  const libxstream_blas_plugin_t* const plugin = libxstream_blas_find(stream);
  const int f64 = libxstream_blas_f64(datatype);
  int result = EXIT_FAILURE;
  if (NULL != plugin && NULL != plugin->dgmm && 0 <= f64
    && 0 == plugin->dgmm(libxstream_opencl_config.device.context, stream->queue, f64, side, m, n, a, lda, x, incx, c,
      ldc))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}
