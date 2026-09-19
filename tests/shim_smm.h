/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SHIM_SMM_H
#define SHIM_SMM_H

/**
 * The SMM kernels (samples/smm) on the host through libxstream_cpu_begin.h, driven
 * the way smm_kernel.c and smm_trans.c launch them and checked the way acc_bench
 * checks them: C += A * B for every entry of a stack whose C blocks come in runs,
 * so that work-groups accumulate into the same block and the atomics are real.
 *
 * A flavor is one translation unit per element type, because libxstream_atomics.h
 * settles its atomic add on the first T it sees; it states each instance's build
 * string as the host emitted it and includes shim_smm_instance.h (multiply) or
 * shim_smm_trans_instance.h (transpose) once per instance.
 *
 * Every instance runs as LIBXSTREAM_CPU_TEAM=1, a work-group being an OpenMP team
 * of WG lanes: the kernels stage through local memory behind barriers, and even
 * where they do not, the lanes of one group accumulate into C at the same time.
 * The device part of each build string is a device without sub-groups (SG=0): with
 * a sub-group covering the work-group the kernel drops its barriers and relies on
 * lockstep, which a team of threads does not have.
 *
 * The operands are small integers, so every product and sum is exact in either
 * precision whatever the order the atomics apply them in: a mismatch is a kernel
 * defect, not rounding.
 */

#include <libxs/libxs_macros.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define SHIM_SMM_STACK 60
/* distinct A and B blocks, and C blocks, the latter in runs of about STACK/NC */
#define SHIM_SMM_NAB 7
#define SHIM_SMM_NC 8

/**
 * What an instance tells the runner: its shape, the batch it was compiled for, and
 * a launcher of the host's geometry. The multiply launcher takes B already
 * transposed, as the device receives it from the transpose kernel.
 */
typedef void (*shim_smm_launch_t)(SHIM_SMM_ELEM* c, const SHIM_SMM_ELEM* a, const SHIM_SMM_ELEM* b,
  const int* stack, int stack_size);
typedef void (*shim_trans_launch_t)(const int* stack, void* matrix, int stack_size);
typedef struct {
  const char* name;
  int m, n, k, bs, wg;
  shim_smm_launch_t launch;
} shim_smm_desc_t;
typedef struct {
  const char* name;
  int m, n, bs, wg, inplace;
  shim_trans_launch_t launch;
} shim_trans_desc_t;


/* the runner exists where the flavors do, which is where OpenMP provides the team */
#if defined(_OPENMP)
static unsigned int shim_smm_next(unsigned int* seed);
static void shim_smm_stack(int stack[], int stack_size, int mk, int kn, int mn, unsigned int* seed);
static int shim_smm_multiply(const shim_smm_desc_t* desc, unsigned int seed);
static int shim_smm_transpose(const shim_trans_desc_t* desc, unsigned int seed);
static int shim_smm(const shim_smm_desc_t* const smm[], int nsmm, const shim_trans_desc_t* const trans[], int ntrans,
  const char flavor[]);


static unsigned int shim_smm_next(unsigned int* seed)
{
  *seed = *seed * 1103515245U + 12345U;
  return *seed >> 16;
}


/**
 * The stack acc_bench builds (INIT_STACK): one-based element offsets, A and B blocks
 * drawn at random, and C blocks in consecutive runs of varying length, which is what
 * TRACK_C relies on and what makes several work-groups meet on one block.
 */
static void shim_smm_stack(int stack[], int stack_size, int mk, int kn, int mn, unsigned int* seed)
{
  const int navg = stack_size / SHIM_SMM_NC, nimb = LIBXS_MAX(1, navg - 4);
  int i = 0, c = 0, ntop = 0;
  while (i < stack_size) {
    ntop += navg + (int)(shim_smm_next(seed) % (unsigned int)(2 * nimb)) - nimb;
    if (stack_size < ntop) ntop = stack_size;
    for (; i < ntop; ++i) {
      stack[3 * i + 0] = (int)(shim_smm_next(seed) % SHIM_SMM_NAB) * mk + 1;
      stack[3 * i + 1] = (int)(shim_smm_next(seed) % SHIM_SMM_NAB) * kn + 1;
      stack[3 * i + 2] = c * mn + 1;
    }
    if (c + 1 < SHIM_SMM_NC) ++c;
  }
}


static int shim_smm_multiply(const shim_smm_desc_t* desc, unsigned int seed)
{
  const int m = desc->m, n = desc->n, k = desc->k, mk = m * k, kn = k * n, mn = m * n;
  const size_t na = (size_t)SHIM_SMM_NAB * mk, nb = (size_t)SHIM_SMM_NAB * kn, nc = (size_t)SHIM_SMM_NC * mn;
  SHIM_SMM_ELEM *const a = (SHIM_SMM_ELEM*)malloc(sizeof(SHIM_SMM_ELEM) * na);
  SHIM_SMM_ELEM *const b = (SHIM_SMM_ELEM*)malloc(sizeof(SHIM_SMM_ELEM) * nb);
  SHIM_SMM_ELEM *const bt = (SHIM_SMM_ELEM*)malloc(sizeof(SHIM_SMM_ELEM) * nb);
  SHIM_SMM_ELEM *const c = (SHIM_SMM_ELEM*)calloc(nc, sizeof(SHIM_SMM_ELEM));
  SHIM_SMM_ELEM *const ref = (SHIM_SMM_ELEM*)calloc(nc, sizeof(SHIM_SMM_ELEM));
  int* const stack = (int*)malloc(sizeof(int) * 3 * SHIM_SMM_STACK);
  int result = EXIT_SUCCESS, s, i, j, l;
  if (NULL == a || NULL == b || NULL == bt || NULL == c || NULL == ref || NULL == stack) {
    fprintf(stderr, "shim: smm out of memory\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    size_t e;
    for (e = 0; e < na; ++e) a[e] = (SHIM_SMM_ELEM)((int)(shim_smm_next(&seed) % 7) - 3);
    for (e = 0; e < nb; ++e) b[e] = (SHIM_SMM_ELEM)((int)(shim_smm_next(&seed) % 7) - 3);
    /* each B block column-major k x n, handed over as its transpose n x k */
    for (s = 0; s < SHIM_SMM_NAB; ++s) {
      for (j = 0; j < n; ++j) {
        for (l = 0; l < k; ++l) bt[(size_t)s * kn + (size_t)l * n + j] = b[(size_t)s * kn + (size_t)j * k + l];
      }
    }
    shim_smm_stack(stack, SHIM_SMM_STACK, mk, kn, mn, &seed);
    /* reference: acc_bench's, i.e. C += A * B per entry with A m x k and B k x n */
    for (s = 0; s < SHIM_SMM_STACK; ++s) {
      const int a0 = stack[3 * s] - 1, b0 = stack[3 * s + 1] - 1, c0 = stack[3 * s + 2] - 1;
      for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
          SHIM_SMM_ELEM sum = 0;
          for (l = 0; l < k; ++l) sum += a[a0 + l * m + i] * b[b0 + j * k + l];
          ref[c0 + j * m + i] += sum;
        }
      }
    }
    desc->launch(c, a, bt, stack, SHIM_SMM_STACK);
    for (e = 0; e < nc && EXIT_SUCCESS == result; ++e) {
      if (c[e] != ref[e]) {
        fprintf(stderr, "shim: smm %s (%ix%ix%i bs=%i wg=%i): c[%i] = %g, expected %g\n", desc->name, m, n, k,
          desc->bs, desc->wg, (int)e, (double)c[e], (double)ref[e]);
        result = EXIT_FAILURE;
      }
    }
  }
  free(a);
  free(b);
  free(bt);
  free(c);
  free(ref);
  free(stack);
  return result;
}


/* The transpose in place per stack entry, m x n column-major into n x m. */
static int shim_smm_transpose(const shim_trans_desc_t* desc, unsigned int seed)
{
  const int m = desc->m, n = desc->n, mn = m * n, nmat = SHIM_SMM_NAB;
  SHIM_SMM_ELEM *const mat = (SHIM_SMM_ELEM*)malloc(sizeof(SHIM_SMM_ELEM) * nmat * mn);
  SHIM_SMM_ELEM *const org = (SHIM_SMM_ELEM*)malloc(sizeof(SHIM_SMM_ELEM) * nmat * mn);
  int* const stack = (int*)malloc(sizeof(int) * nmat);
  int result = EXIT_SUCCESS, s, i, j;
  if (NULL == mat || NULL == org || NULL == stack) {
    fprintf(stderr, "shim: smm out of memory\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < nmat * mn; ++i) org[i] = mat[i] = (SHIM_SMM_ELEM)(int)(shim_smm_next(&seed) % 1000);
    /* the blocks in reverse order, so that an offset the kernel ignores shows */
    for (s = 0; s < nmat; ++s) stack[s] = (nmat - 1 - s) * mn;
    desc->launch(stack, mat, nmat);
    for (s = 0; s < nmat && EXIT_SUCCESS == result; ++s) {
      for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
        for (i = 0; i < m && EXIT_SUCCESS == result; ++i) {
          if (mat[s * mn + i * n + j] != org[s * mn + j * m + i]) {
            fprintf(stderr, "shim: trans %s (%ix%i bs=%i wg=%i%s): block %i (%i,%i) = %g, expected %g\n", desc->name, m, n,
              desc->bs, desc->wg, 0 != desc->inplace ? " inplace" : "", s, i, j, (double)mat[s * mn + i * n + j],
              (double)org[s * mn + j * m + i]);
            result = EXIT_FAILURE;
          }
        }
      }
    }
  }
  free(mat);
  free(org);
  free(stack);
  return result;
}


static int shim_smm(const shim_smm_desc_t* const smm[], int nsmm, const shim_trans_desc_t* const trans[], int ntrans,
  const char flavor[])
{
  int result = EXIT_SUCCESS, i;
  for (i = 0; i < nsmm && EXIT_SUCCESS == result; ++i) result = shim_smm_multiply(smm[i], (unsigned int)(1 + i));
  for (i = 0; i < ntrans && EXIT_SUCCESS == result; ++i) result = shim_smm_transpose(trans[i], (unsigned int)(1 + i));
  if (EXIT_SUCCESS == result) {
    printf("shim: smm %s, %i multiply and %i transpose instances bit-exact\n", flavor, nsmm, ntrans);
  }
  return result;
}
#endif /*defined(_OPENMP)*/

#endif /*SHIM_SMM_H*/
