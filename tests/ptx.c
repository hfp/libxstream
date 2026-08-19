/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__OPENCL)
# if defined(LIBXSTREAM_SOURCE)
#   include <libxstream/libxstream_source.h>
# else
#   include <libxstream/libxstream_opencl.h>
# endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__OPENCL)

/**
 * Selecting NVIDIA's accelerated target is what makes warp-group MMA reachable
 * from OpenCL at all (the front-end emits the plain target and no build option
 * changes it), so the rewrite is covered here rather than only in a sample.
 * The default test build carries no OpenCL backend and skips; run "make OCL=1"
 * to exercise this.
 */
static int check(const char text[], const char expected[])
{
  const size_t size = strlen(text);
  char* out = NULL;
  size_t size_new = 0;
  int result = libxstream_opencl_retarget_ptx(text, size, &out, &size_new);
  if (NULL == expected) { /* rewrite must be refused */
    if (EXIT_SUCCESS == result || NULL != out) {
      fprintf(stderr, "ERROR: expected refusal for \"%s\"\n", text);
      result = EXIT_FAILURE;
    }
    else result = EXIT_SUCCESS;
  }
  else if (EXIT_SUCCESS != result) {
    fprintf(stderr, "ERROR: expected a rewrite of \"%s\"\n", text);
  }
  else if (0 != strcmp(out, expected)) {
    fprintf(stderr, "ERROR: \"%s\" -> \"%s\" (expected \"%s\")\n", text, out, expected);
    result = EXIT_FAILURE;
  }
  else if (size_new != size + 1 || size_new != strlen(out)) {
    fprintf(stderr, "ERROR: size %i (expected %i)\n", (int)size_new, (int)(size + 1));
    result = EXIT_FAILURE;
  }
  if (NULL != out) libxs_free(out);
  return result;
}


int main(void)
{
  int result = check(".version 8.3\n.target sm_90, texmode_independent\n.address_size 64\n",
    ".version 8.3\n.target sm_90a, texmode_independent\n.address_size 64\n");
  if (EXIT_SUCCESS == result) { /* the whole tail must survive, terminator included */
    result = check(".target sm_120\nret;\n", ".target sm_120a\nret;\n");
  }
  if (EXIT_SUCCESS == result) { /* an accelerated target is left alone */
    result = check(".target sm_90a, texmode_independent\n", NULL);
  }
  if (EXIT_SUCCESS == result) { /* nothing to rewrite */
    result = check(".version 8.3\n", NULL);
  }
  if (EXIT_SUCCESS == result) { /* a truncated target is not a target */
    result = check(".target sm_\n", NULL);
  }
  if (EXIT_SUCCESS == result) printf("ptx: OK\n");
  return result;
}

#else

int main(void)
{
  printf("ptx: skipped (OpenCL backend not compiled in)\n");
  return EXIT_SUCCESS;
}

#endif
