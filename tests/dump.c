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

/* A template standing in for a kernel: vendor branch, a parameter, an #if. */
static const char template_source[] =
  "#if defined(INTEL) && (0 != INTEL)\n"
  "# define WHICH_VENDOR intel_path\n"
  "#elif defined(NV) && (0 != NV)\n"
  "# define WHICH_VENDOR nvidia_path\n"
  "#else\n"
  "# define WHICH_VENDOR portable_path\n"
  "#endif\n"
  "kernel void WHICH_VENDOR(global float* c) { c[0] = (float)BM; }\n";


static int reads(const char path[], char buffer[], size_t size)
{
  int result = EXIT_FAILURE;
  FILE* const file = fopen(path, "r");
  buffer[0] = '\0';
  if (NULL != file) {
    const size_t n = fread(buffer, 1, size - 1, file);
    buffer[n] = '\0';
    if (0 != n) result = EXIT_SUCCESS;
    fclose(file);
  }
  return result;
}


/* One configuration: the artifact must name the expected path and no other. */
static int check(const char name[], const char params[], int nv, const char expect[], const char reject[])
{
  char path[256], text[4096];
  char* instanced = NULL;
  int result = libxstream_opencl_dump(template_source, 0 /*strlen*/, name, params, nv, "-cl-std=CL2.0", &instanced);
  if (EXIT_SUCCESS == result) {
    if (0 >= LIBXS_SNPRINTF(path, sizeof(path), "%s.cl", name)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = reads(path, text, sizeof(text));
  if (EXIT_SUCCESS == result) {
    /* the selected branch survived and the other did not */
    if (NULL == strstr(text, expect) || NULL != strstr(text, reject)) result = EXIT_FAILURE;
    /* the template was instanced: no directive and no parameter name remains */
    if (NULL != strstr(text, "#if") || NULL != strstr(text, "WHICH_VENDOR")
      || NULL != strstr(text, "BM"))
    {
      result = EXIT_FAILURE;
    }
    /* the std flag is recorded, and the returned text matches the file */
    if (NULL == strstr(text, "-cl-std=CL2.0")) result = EXIT_FAILURE;
    if (NULL == instanced || 0 != strcmp(instanced, text)) result = EXIT_FAILURE;
  }
  if (NULL != instanced) libxs_free(instanced);
  if (EXIT_SUCCESS != result) fprintf(stderr, "dump: %s [%s] is wrong\n", name, params);
  else remove(path);
  return result;
}


/* Includes the shipped header, so the levels check its guards and not a copy. */
static const char header_probe[] =
  "#include \"libxstream_common.h\"\n"
  "#if defined(BCST_SG)\n"
  "kernel void has_sg_broadcast(void) {}\n"
  "#else\n"
  "kernel void no_sg_broadcast(void) {}\n"
  "#endif\n";


/* One named level: the artifact must select the branch the level implies. */
static int level(const char name[], const char params[], const char expect[], const char reject[])
{
  char path[256], text[8192];
  char defines[512];
  int result;
  if (0 >= LIBXS_SNPRINTF(defines, sizeof(defines), "-I../libxstream/opencl %s", params)) return EXIT_FAILURE;
  result = libxstream_opencl_dump(header_probe, 0, name, defines, 0, "", NULL);
  if (EXIT_SUCCESS == result && 0 >= LIBXS_SNPRINTF(path, sizeof(path), "%s.cl", name)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = reads(path, text, sizeof(text));
  if (EXIT_SUCCESS == result) {
    if (NULL == strstr(text, expect) || NULL != strstr(text, reject)) result = EXIT_FAILURE;
    if (NULL != strstr(text, "#include")) result = EXIT_FAILURE; /* fused */
  }
  if (EXIT_SUCCESS != result) fprintf(stderr, "dump: level %s [%s] is wrong\n", name, params);
  else remove(path);
  return result;
}


int main(void)
{
  int result;
  /* no context, no device, no platform: nothing below touches an OpenCL call */
  result = check("dump_portable", "-DLIBXSTREAM_OCLVER=300 -DBM=64", 0, "portable_path", "intel_path");
  if (EXIT_SUCCESS == result) {
    result = check("dump_intel", "-DLIBXSTREAM_OCLVER=300 -DINTEL=1 -DBM=64", 0, "intel_path", "portable_path");
  }
  if (EXIT_SUCCESS == result) {
    result = check("dump_nvidia", "-DLIBXSTREAM_OCLVER=300 -DNV=1 -DBM=128", 1, "nvidia_path", "portable_path");
  }
  /* the parameter reaches the body rather than only the directives */
  if (EXIT_SUCCESS == result) {
    char text[4096];
    char* instanced = NULL;
    if (EXIT_SUCCESS == libxstream_opencl_dump(template_source, 0, "dump_param", "-DLIBXSTREAM_OCLVER=300 -DBM=4096", 0, "", &instanced)) {
      if (EXIT_SUCCESS == reads("dump_param.cl", text, sizeof(text))) {
        if (NULL == strstr(text, "4096")) result = EXIT_FAILURE;
      }
      else result = EXIT_FAILURE;
      remove("dump_param.cl");
    }
    else result = EXIT_FAILURE;
    if (NULL != instanced) libxs_free(instanced);
    if (EXIT_SUCCESS != result) fprintf(stderr, "dump: parameter did not reach the body\n");
  }

  /* Named, not numbered: 3.0 rebased on 1.2, so a higher number guarantees less. */
  if (EXIT_SUCCESS == result) { /* floor: nothing defined */
    result = level("lvl_floor", "", "no_sg_broadcast", "has_sg_broadcast");
  }
  if (EXIT_SUCCESS == result) { /* a 3.0 device without the optional feature */
    result = level("lvl_cl3_bare", "-DGPU=1 -DSG=32 -DLIBXSTREAM_OCLVER_C=300",
      "no_sg_broadcast", "has_sg_broadcast");
  }
  if (EXIT_SUCCESS == result) { /* the same device that does advertise it */
    result = level("lvl_cl3_subgroups", "-DGPU=1 -DSG=32 -DLIBXSTREAM_OCLVER_C=300 -D__opencl_c_subgroups",
      "has_sg_broadcast", "no_sg_broadcast");
  }
  if (EXIT_SUCCESS == result) { /* a 2.0 device, where the version is the answer */
    result = level("lvl_cl2", "-DGPU=1 -DSG=32 -DLIBXSTREAM_OCLVER_C=200",
      "has_sg_broadcast", "no_sg_broadcast");
  }

  /* refuses what it cannot write rather than reporting success */
  if (EXIT_SUCCESS == result) {
    if (EXIT_SUCCESS == libxstream_opencl_dump(NULL, 0, "dump_null", "", 0, "", NULL)) {
      fprintf(stderr, "dump: accepted a NULL source\n");
      result = EXIT_FAILURE;
    }
  }
  return result;
}

#else

int main(void)
{
  /* nothing to instantiate without __OPENCL; say so rather than exit zero mutely */
  fprintf(stderr, "dump: skipped, built without __OPENCL\n");
  return EXIT_SUCCESS;
}

#endif
