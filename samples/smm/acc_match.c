/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_str.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#if defined(_WIN32)
# define strncasecmp _strnicmp
#else
# include <strings.h>
#endif


static void opencl_libsmm_devname_cleanup(char* name);


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
    opencl_libsmm_devname_cleanup(a);
    opencl_libsmm_devname_cleanup(b);
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


static void opencl_libsmm_devname_cleanup(char* name)
{
  char* dst = name;
  char* src = name;
  while ('\0' != *src) {
    if ('[' == *src && '0' == src[1] && 'x' == src[2]) {
      while ('\0' != *src && ']' != *src) ++src;
      if (']' == *src) ++src;
    }
    else if ('(' == *src && (('R' == src[1] && ')' == src[2]) ||
      ('T' == src[1] && 'M' == src[2] && ')' == src[3])))
    {
      src += ('T' == src[1]) ? 4 : 3;
    }
    else if ('-' == *src &&
      (0 == strncasecmp(src + 1, "PCIe", 4) || 0 == strncasecmp(src + 1, "PCIE", 4)))
    {
      src += 5;
      while ('\0' != *src && ' ' != *src) ++src;
    }
    else if ((' ' == *src || src == name) &&
      (0 == strncasecmp(src + (' ' == *src ? 1 : 0), "PCIe", 4) ||
       0 == strncasecmp(src + (' ' == *src ? 1 : 0), "PCIE", 4)))
    {
      char* p = src + (' ' == *src ? 1 : 0) + 4;
      if ('\0' == *p || ' ' == *p || '-' == *p) {
        src = p;
        if ('-' == *src) {
          ++src;
          while ('\0' != *src && ' ' != *src) ++src;
        }
      }
      else {
        *dst++ = *src++;
      }
    }
    else if ('-' == *src && 0 != isdigit((unsigned char)src[1])) {
      char* p = src + 1;
      while (0 != isdigit((unsigned char)*p)) ++p;
      if (('G' == *p || 'g' == *p) && ('B' == p[1] || 'b' == p[1]) &&
          ('\0' == p[2] || ' ' == p[2]))
      {
        src = p + 2;
      }
      else {
        *dst++ = *src++;
      }
    }
    else if ((' ' == *src || src == name) && 0 != isdigit((unsigned char)src[' ' == *src ? 1 : 0])) {
      char* p = src + (' ' == *src ? 1 : 0);
      while (0 != isdigit((unsigned char)*p)) ++p;
      if (('G' == *p || 'g' == *p) && ('B' == p[1] || 'b' == p[1]) &&
          ('\0' == p[2] || ' ' == p[2] || '-' == p[2]))
      {
        src = p + 2;
      }
      else {
        *dst++ = *src++;
      }
    }
    else if ((' ' == *src || src == name) &&
      (0 == strncasecmp(src + (' ' == *src ? 1 : 0), "HBM", 3) ||
       0 == strncasecmp(src + (' ' == *src ? 1 : 0), "SXM", 3)))
    {
      char* p = src + (' ' == *src ? 1 : 0) + 3;
      while (0 != isdigit((unsigned char)*p)) ++p;
      if ('\0' == *p || ' ' == *p) {
        src = p;
      }
      else {
        *dst++ = *src++;
      }
    }
    else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';
  while (dst > name && ' ' == dst[-1]) *--dst = '\0';
  dst = name;
  src = name;
  while ('\0' != *src) {
    if (' ' == *src && ' ' == src[1]) {
      ++src;
    }
    else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';
  if (' ' == *name) {
    src = name + 1;
    dst = name;
    while ('\0' != *src) *dst++ = *src++;
    *dst = '\0';
  }
}
