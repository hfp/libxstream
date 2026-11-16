/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki_opencl.h"
#include <libxs/libxs_mem.h>
#include <libxs/libxs.h>

/**
 * Splices the warp-group MMA into PTX that carries the kernel's markers, as the
 * sample does in-process (ozaki_wgmma_splice), for a runtime that does not hand
 * out the PTX it built or to assemble the result with ptxas. Reads the input
 * completely before writing, so the output may be the same file.
 */


int main(int argc, char* argv[])
{
  const char* const entry = (3 < argc) ? argv[3] : "gemm_crt_fused";
  const int maxnreg = (4 < argc) ? atoi(argv[4]) : 0;
  FILE* const file = (1 < argc) ? fopen(argv[1], "rb") : NULL;
  char* text = NULL;
  char* patched = NULL;
  long size = -1;
  int result = (NULL != file) ? EXIT_SUCCESS : EXIT_FAILURE;
  libxs_init();
  if (EXIT_SUCCESS == result && 0 == fseek(file, 0, SEEK_END)) size = ftell(file);
  if (0 < size && 0 == fseek(file, 0, SEEK_SET)) {
    text = (char*)libxs_malloc(NULL, (size_t)size + 1, 0 /*auto-align*/);
  }
  if (NULL != text && (size_t)size == fread(text, 1, (size_t)size, file)) {
    text[size] = '\0';
    patched = ozaki_wgmma_splice(text, (size_t)size, entry, maxnreg);
  }
  if (NULL != file) fclose(file);
  if (NULL != patched) {
    FILE* const out = (2 < argc && 0 != strcmp("-", argv[2])) ? fopen(argv[2], "wb") : stdout;
    const size_t n = strlen(patched);
    if (NULL == out || n != fwrite(patched, 1, n, out)) result = EXIT_FAILURE;
    if (NULL != out && stdout != out && 0 != fclose(out)) result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result) {
    fprintf(stderr, "ERROR: %s is unreadable, spliced already, or carries no marker of %s\n", argv[1], entry);
    result = EXIT_FAILURE;
  }
  else {
    fprintf(stderr, "Usage: %s PTX [OUT|- [ENTRY [MAXNREG]]]\n", argv[0]);
    fprintf(stderr, "       ENTRY defaults to gemm_crt_fused, OUT to the standard output.\n");
  }
  if (NULL != patched) libxs_free(patched);
  if (NULL != text) libxs_free(text);
  libxs_finalize();
  return result;
}
