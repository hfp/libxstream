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
 * Compiles every OpenCL kernel, which is otherwise never compiled at all: CI has
 * no GPU and no OpenCL platform, so about 4200 lines of kernel go unchecked.
 *
 * libxstream_opencl_dump instantiates each template without a device, and an
 * OpenCL C compiler then checks the result. Levels are named rather than numbered
 * because the version does not order the feature space: OpenCL 3.0 rebased on 1.2
 * and made the 2.x collectives optional, so a higher number can guarantee less.
 *
 * The vendor paths (DPAS builtins, inline PTX) are out of reach for any compiler
 * but the vendor's own, so they are not attempted here.
 */
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

#if !defined(KERNELS_CC)
# define KERNELS_CC "clang"
#endif
#define KERNELS_MAXSRC (1 << 20)

/**
 * Root kernels with the parameters their host supplies: shape and configuration
 * are mandatory and have no defaults, so every level must provide them, and the
 * levels below vary only the language version and the optional features.  Taken
 * from the build strings in samples/ozaki/ozaki_opencl.c rather than invented.
 *
 * INTEL and NV are held at 0 throughout: the vendor paths need DPAS builtins and
 * inline PTX that no compiler but the vendor's own accepts.
 */
#define KERNELS_OZAKI_BASE \
  "-DBK=32 -DKU=2 -DRC=8 -DSG=16 -DINTEL=0 -DNV=0 -DBM_PRE=16 -DBN_PRE=16" \
  " -DBK_PRE=32 -DOZAKI_SB=1 -DCONSTANT=global -DBM=64 -DBN=64 -DRTM=2 -DRTN=2" \
  " -DOZAKI_CUTOFF=14 -DLU=0 -DKGROUPS=0 -DPB=1"
#define KERNELS_OZAKI_FP64 \
  KERNELS_OZAKI_BASE " -DNSLICES=8 -DUSE_DOUBLE=1 -DMANT_BITS=53" \
  " -DBIAS_PLUS_MANT=1075 -DOZAKI_HIER=1 -DOZAKI_TRI=0 -DOZAKI_SYM=0"
#define KERNELS_OZAKI_FP32 \
  KERNELS_OZAKI_BASE " -DNSLICES=4 -DUSE_DOUBLE=0 -DMANT_BITS=23" \
  " -DBIAS_PLUS_MANT=150 -DOZAKI_HIER=1 -DOZAKI_TRI=1 -DOZAKI_SYM=1"
#define KERNELS_OZAKI_FLAT \
  KERNELS_OZAKI_BASE " -DNSLICES=8 -DUSE_DOUBLE=1 -DMANT_BITS=53" \
  " -DBIAS_PLUS_MANT=1075 -DOZAKI_HIER=0 -DOZAKI_TRI=0 -DOZAKI_SYM=0"
#define KERNELS_OZAKI_SYM \
  KERNELS_OZAKI_BASE " -DNSLICES=8 -DUSE_DOUBLE=1 -DMANT_BITS=53" \
  " -DBIAS_PLUS_MANT=1075 -DOZAKI_HIER=1 -DOZAKI_TRI=1 -DOZAKI_SYM=1"


/**
 * A flavor is a variant of one kernel's parameters, not of the level: the level
 * says what the language provides, the flavor what the host asked to emit.
 */
typedef struct { const char* path; const char* flavor; const char* params; } kernels_file_t;
static const kernels_file_t kernel_files[] = {
  { "../samples/ozaki/kernels/gemm3m.cl", "", "" },
  { "../samples/ozaki/kernels/ozaki1_int8.cl", "fp64", KERNELS_OZAKI_FP64 },
  { "../samples/ozaki/kernels/ozaki1_int8.cl", "fp32", KERNELS_OZAKI_FP32 },
  { "../samples/ozaki/kernels/ozaki1_int8.cl", "sym", KERNELS_OZAKI_SYM },
  { "../samples/ozaki/kernels/ozaki2_int8.cl", "fp64", KERNELS_OZAKI_FP64 },
  { "../samples/ozaki/kernels/ozaki2_int8.cl", "fp32", KERNELS_OZAKI_FP32 },
  { "../samples/ozaki/kernels/ozaki2_int8.cl", "flat", KERNELS_OZAKI_FLAT },
  { "../samples/smm/kernels/transpose.cl", "",
    "-DT=float -DSM=32 -DSN=32 -DWG=32 -DCONSTANT=global" /* WG must equal SM */ },
  { "../samples/stencil/kernels/stencil_int8.cl", "", "" },
  { "../samples/stencil/kernels/stencil_fp32.cl", "", "" }
};

/**
 * Not covered yet, and named so that the gap is visible rather than implied:
 * these need values this test would have to invent (multiply.cl wants a
 * param_format type, stencil_bf16.cl the conversion pair), and inventing them
 * risks reporting a fault that is the test's rather than the kernel's.
 */
static const char* const kernel_pending[] = {
  "../samples/smm/kernels/multiply.cl", "../samples/stencil/kernels/stencil_bf16.cl"
};

/* A level names what it selects, and states the language its artifact is in. */
typedef struct { const char* name; const char* defines; const char* std; } kernels_level_t;
static const kernels_level_t kernel_levels[] = {
  { "floor", "-DLIBXSTREAM_OCLVER_C=120 -DLIBXSTREAM_OCLVER=120", "CL1.2" },
  { "cl3_bare", "-DLIBXSTREAM_OCLVER_C=300 -DLIBXSTREAM_OCLVER=300 -DGPU=1", "CL3.0" },
  { "cl3_subgroups",
    "-DLIBXSTREAM_OCLVER_C=300 -DLIBXSTREAM_OCLVER=300 -DGPU=1 -D__opencl_c_subgroups",
    "CL3.0" }
};


static const char* kernels_cc(void);
static char* loads(const char path[]);
static void dirname_of(const char path[], char buffer[], size_t size);
static int compiles(const char artifact[], const char std[]);
static int one(const kernels_file_t* file, const kernels_level_t* level);


int main(void)
{
  const int nfiles = (int)(sizeof(kernel_files) / sizeof(*kernel_files));
  const int nlevels = (int)(sizeof(kernel_levels) / sizeof(*kernel_levels));
  int result = EXIT_SUCCESS, i, j, n = 0;
  /* Required, not optional: a lint that skips itself reports success on every
     machine that cannot run it, which is indistinguishable from passing. */
  {
    char probe[512];
    if (0 >= LIBXS_SNPRINTF(probe, sizeof(probe), "%s --version >/dev/null 2>&1", kernels_cc())
      || EXIT_SUCCESS != system(probe))
    {
      fprintf(stderr, "kernels: %s is required to compile the kernels\n", kernels_cc());
      return EXIT_FAILURE;
    }
  }
  for (i = 0; i < nlevels && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < nfiles && EXIT_SUCCESS == result; ++j) {
      result = one(kernel_files + j, kernel_levels + i);
      if (EXIT_SUCCESS == result) ++n;
    }
  }
  if (EXIT_SUCCESS == result) {
    const int npending = (int)(sizeof(kernel_pending) / sizeof(*kernel_pending));
    printf("kernels: %d compiled at -Wall -Wextra -pedantic -Werror\n", n);
    for (i = 0; i < npending; ++i) printf("kernels: NOT COVERED %s\n", kernel_pending[i]);
  }
  return result;
}


/**
 * KERNELS_CC in the environment overrides the default, which is how a platform
 * names a compiler that can build OpenCL C (Homebrew LLVM on macOS) and how icx
 * is asked for the Intel dialect instead.
 */
static const char* kernels_cc(void)
{
  const char* const env = getenv("KERNELS_CC");
  return (NULL != env && '\0' != *env) ? env : KERNELS_CC;
}


static char* loads(const char path[])
{
  char* result = NULL;
  FILE* const file = fopen(path, "rb");
  if (NULL != file) {
    result = (char*)malloc(KERNELS_MAXSRC);
    if (NULL != result) {
      const size_t n = fread(result, 1, KERNELS_MAXSRC - 1, file);
      result[n] = '\0';
      if (0 == n) {
        free(result);
        result = NULL;
      }
    }
    fclose(file);
  }
  return result;
}


/* Directory of a path, so cpp can resolve the kernel's relative includes. */
static void dirname_of(const char path[], char buffer[], size_t size)
{
  const char* const slash = strrchr(path, '/');
  const size_t n = (NULL != slash) ? (size_t)(slash - path) : 0;
  if (0 != n && size > n) {
    memcpy(buffer, path, n);
    buffer[n] = '\0';
  }
  else LIBXS_EXPECT(0 < LIBXS_SNPRINTF(buffer, size, "%s", "."));
}


static int compiles(const char artifact[], const char std[])
{
  char command[1024];
  int result = EXIT_FAILURE;
  if (0 < LIBXS_SNPRINTF(command, sizeof(command),
        "%s -x cl -cl-std=%s -fsyntax-only"
        /* -Wno-unused-parameter as the library's own C build does: a kernel
         * signature is fixed by the host argument list, so an unused parameter
         * is a interface constraint and not a defect. */
        " -Wall -Wextra -pedantic -Wno-unused-parameter -Werror"
        " -Xclang -finclude-default-header %s", kernels_cc(), std, artifact))
  {
    result = (EXIT_SUCCESS == system(command) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  return result;
}


static int one(const kernels_file_t* file, const kernels_level_t* level)
{
  char name[256], defines[1024], dir[512], artifact[512], stem[128];
  const char* const path = file->path;
  char* source = loads(path);
  int result = (NULL != source ? EXIT_SUCCESS : EXIT_FAILURE);
  const char* base;
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "kernels: cannot read %s\n", path);
    return result;
  }
  base = strrchr(path, '/');
  base = (NULL != base ? base + 1 : path);
  { /* strip the .cl so the artifact is not named twice over */
    const char* const dot = strrchr(base, '.');
    if (NULL != dot && 0 == strcmp(dot, ".cl")) {
      LIBXS_EXPECT(0 < LIBXS_SNPRINTF(stem, sizeof(stem), "%.*s", (int)(dot - base), base));
      base = stem;
    }
  }
  dirname_of(path, dir, sizeof(dir));
  if (0 >= LIBXS_SNPRINTF(name, sizeof(name), "k_%s_%s%s%s", level->name, base,
        ('\0' != *file->flavor) ? "_" : "", file->flavor)
    || 0 >= LIBXS_SNPRINTF(defines, sizeof(defines), "-I%s %s %s", dir, level->defines, file->params)
    || 0 >= LIBXS_SNPRINTF(artifact, sizeof(artifact), "%s.cl", name))
  {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = libxstream_opencl_dump(source, 0 /*strlen*/, name, defines, 0 /*nv*/, "", NULL);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "kernels: %s did not instantiate [%s %s]\n", path, level->name, file->flavor);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = compiles(artifact, level->std);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "kernels: %s failed to compile [%s %s]\n", path, level->name, file->flavor);
    }
    else remove(artifact);
  }
  free(source);
  return result;
}

#else

int main(void)
{
  /* no kernels to instantiate without __OPENCL; say so rather than exit zero mutely */
  fprintf(stderr, "kernels: skipped, built without __OPENCL\n");
  return EXIT_SUCCESS;
}

#endif
