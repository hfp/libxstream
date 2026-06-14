/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxstream/libxstream_opencl.h>
#include <libxs/libxs_str.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  if (3 <= argc) {
    char a[256], b[256];
    int count = 0, order = 0;
    int n, dist;
    snprintf(a, sizeof(a), "%s", argv[1]);
    snprintf(b, sizeof(b), "%s", argv[2]);
    n = libxs_strimatch(a, b, NULL, &count);
    dist = libxs_strisimilar(a, b, NULL, LIBXS_STRISIMILAR_DEFAULT, &order);
    if (0 != n && 0 != count) {
      fprintf(stdout, "strimatch: match=%d count=%d score=%.4f\n", n, count, (double)n / count);
    }
    else {
      fprintf(stdout, "strimatch: no match\n");
    }
    fprintf(stdout, "strisimilar: distance=%d order=%d\n", dist, order);
    libxstream_opencl_device_name_cleanup(a);
    libxstream_opencl_device_name_cleanup(b);
    fprintf(stdout, "cleaned: \"%s\" vs \"%s\"\n", a, b);
    count = 0;
    order = 0;
    n = libxs_strimatch(a, b, NULL, &count);
    dist = libxs_strisimilar(a, b, NULL, LIBXS_STRISIMILAR_DEFAULT, &order);
    if (0 != n && 0 != count) {
      fprintf(stdout, "strimatch (cleaned): match=%d count=%d score=%.4f\n", n, count, (double)n / count);
    }
    else {
      fprintf(stdout, "strimatch (cleaned): no match\n");
    }
    fprintf(stdout, "strisimilar (cleaned): distance=%d order=%d\n", dist, order);
  }
  else {
    fprintf(stderr, "Usage: %s \"device-name\" \"parameter-name\"\n", argv[0]);
    result = EXIT_FAILURE;
  }
  return result;
}
