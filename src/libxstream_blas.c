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


/* The plugin if it serves the stream's device, NULL otherwise. */
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
    const libxstream_opencl_device_t* const device = &libxstream_opencl_config.device;
    /* oneMKL addresses Intel USM and SVM, which libxstream_mem_allocate returns if either is in use */
    if (0 == libxstream_blas_table.xmem
      || (0 != device->intel && (NULL != device->clDeviceMemAllocINTEL || 0 != device->usm)))
    {
      result = &libxstream_blas_table;
    }
  }
  return result;
}


/* Single or double precision: 1 or 0, and -1 for any other datatype. */
static int libxstream_blas_f64(libxs_data_t datatype);
static int libxstream_blas_f64(libxs_data_t datatype)
{
  return (LIBXS_DATATYPE_F64 == datatype ? 1 : (LIBXS_DATATYPE_F32 == datatype ? 0 : -1));
}


LIBXSTREAM_API const char* libxstream_blas_query(libxstream_stream_t* stream, int* xmem)
{
  const libxstream_blas_plugin_t* plugin = libxstream_blas_find(stream);
  /* a plugin which fails its self-test refuses every call, hence it is not reported */
  if (NULL != plugin && NULL != plugin->probe && 0 != plugin->probe(libxstream_opencl_config.device.context, stream->queue)) {
    plugin = NULL;
  }
  if (NULL != xmem) *xmem = (NULL != plugin ? plugin->xmem : 0);
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
