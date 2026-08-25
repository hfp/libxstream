/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_PINMAP_H
#define LIBXSTREAM_PINMAP_H

#include <stddef.h>

/**
 * Matches an OpenCL device to a vendor-runtime ordinal by PCI address.
 *
 * Separated from the OpenCL and CUDA calls around it so that it can be tested
 * where CI has neither a GPU nor an OpenCL platform: the caller supplies the
 * address it read from the OpenCL device and a callback standing in for the
 * runtime's device query, and what remains here is integer comparison.
 *
 * The callback is the vendor query: it fills its first argument with the value
 * of attribute (second argument) for ordinal (third argument), returning
 * non-zero once the ordinal is out of range. Attribute codes are the caller's;
 * this file does not name them, so it stays free of vendor headers.
 *
 * Returns the first matching ordinal, or -1. Writes the number of ordinals the
 * callback answered to ndevices when that is not NULL, which is how a caller
 * distinguishes "no such device" from "the runtime enumerated nothing".
 */
static int libxstream_pin_match(unsigned int bus, unsigned int slot, unsigned int domain,
  int (*attr)(int*, int, int), int attr_bus, int attr_slot, int attr_domain, int maxdevs,
  int* ndevices)
{
  int result = -1, n = 0, stop = 0;
  if (NULL != attr) {
    while (0 == stop && n < maxdevs) {
      int b = -1, d = -1, m = 0;
      if (0 != attr(&b, attr_bus, n)) {
        stop = 1;
      }
      else {
        /* unreported slot stays -1 and cannot match; unreported domain is 0 */
        if (0 != attr(&d, attr_slot, n)) d = -1;
        if (0 != attr(&m, attr_domain, n)) m = 0;
        if (0 > result && (unsigned int)b == bus && (unsigned int)d == slot
          && (unsigned int)m == domain)
        {
          result = n;
        }
        ++n;
      }
    }
  }
  if (NULL != ndevices) *ndevices = n;
  return result;
}

#endif /*LIBXSTREAM_PINMAP_H*/
