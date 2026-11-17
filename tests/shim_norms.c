/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * The norms kernel (samples/smm) through the host shim, launched the way
 * c_calculate_norms launches it (a work-group per block), and checked against
 * the squares summed on the host. The elements are small integers, so every sum
 * is exact whatever the order of the reduction: a mismatch is a kernel defect.
 */

#include <libxs/libxs_macros.h>
#include <stdio.h>

/* a work-group is an OpenMP team, the barriers need its threads */
#if defined(_OPENMP)
#include <omp.h>

#define LIBXSTREAM_CPU_TEAM 1
#include <libxstream/opencl/libxstream_cpu_begin.h>

#define WG 8
#define FN shim_norms
/* blocks below, at, and above the work-group size, 1x1 through 20x20 */
#define SHIM_NORMS_NBLKS 10
#define SHIM_NORMS_GAP 3
#include "../samples/smm/kernels/norms.cl"

static void shim_norms_launch(const double* mat, const int* offsets, const int* nelems, float* norms, int nblks)
{
  int g;
  LIBXSTREAM_CPU_GRID(nblks, 1, 1);
  for (g = 0; g < nblks; ++g) {
#pragma omp parallel num_threads(WG)
    {
      LIBXSTREAM_CPU_WORKITEM(g, 0, 0, omp_get_thread_num(), 0, WG, 1);
      FN(mat, offsets, nelems, norms);
    }
  }
}

#include <libxstream/opencl/libxstream_cpu_end.h>


int main(void)
{
  const int nelems[SHIM_NORMS_NBLKS] = {1, 3, WG, WG + 1, 25, 63, 64, 65, 169, 400};
  int offsets[SHIM_NORMS_NBLKS], total = 0, result = EXIT_SUCCESS, i, j;
  float norms[SHIM_NORMS_NBLKS];
  double* mat;
  /* the blocks in reverse order and apart, so that an offset the kernel ignores shows */
  for (i = SHIM_NORMS_NBLKS - 1; 0 <= i; --i) {
    offsets[i] = total;
    total += nelems[i] + SHIM_NORMS_GAP;
  }
  mat = (double*)malloc(sizeof(double) * total);
  if (NULL != mat) {
    for (i = 0; i < total; ++i) mat[i] = (double)((i * 7 + 3) % 19 - 9);
    /* a norm the kernel does not write shows as well */
    for (i = 0; i < SHIM_NORMS_NBLKS; ++i) norms[i] = -1.f;
    shim_norms_launch(mat, offsets, nelems, norms, SHIM_NORMS_NBLKS);
    for (i = 0; i < SHIM_NORMS_NBLKS; ++i) {
      double sum = 0;
      for (j = 0; j < nelems[i]; ++j) sum += mat[offsets[i] + j] * mat[offsets[i] + j];
      if ((float)sum != norms[i]) {
        fprintf(stderr, "shim: norms (wg=%i): block %i (%i elements) = %g, expected %g\n", WG, i, nelems[i],
          (double)norms[i], sum);
        result = EXIT_FAILURE;
      }
    }
    free(mat);
  }
  else {
    fprintf(stderr, "shim: norms out of memory\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) printf("shim: norms (wg=%i), %i blocks bit-exact\n", WG, SHIM_NORMS_NBLKS);
  return result;
}

#else

int main(void)
{
  /* stated rather than skipped, as a pass would be indistinguishable */
  printf("shim: norms NOT ATTEMPTED (work-group barriers need OpenMP)\n");
  return EXIT_SUCCESS;
}

#endif
