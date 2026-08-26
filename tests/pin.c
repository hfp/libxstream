/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "../src/libxstream_pinmap.h"
#include <stdio.h>
#include <stdlib.h>

#define ATTR_BUS 33
#define ATTR_SLOT 34
#define ATTR_DOMAIN 50
#define MAXDEVS 64

/* Synthetic device table standing in for the vendor runtime. */
static int table_ndevs = 0;
static int table_bus[8], table_slot[8], table_domain[8];
static int table_no_slot = 0, table_no_domain = 0;


static int fake_attr(int* value, int attribute, int ordinal);
static int endless_attr(int* value, int attribute, int ordinal);
static int match(unsigned int bus, unsigned int slot, unsigned int domain, int* ndevices);
static void setup(int ndevs);


int main(void)
{
  int result = EXIT_SUCCESS, n = -1;

  /* the ordinal of a device that is present */
  setup(4);
  if (2 != match(0x12, 2, 0, &n) || 4 != n) result = EXIT_FAILURE;

  /* first and last, so the loop bounds are not off by one */
  if (EXIT_SUCCESS == result) {
    if (0 != match(0x10, 0, 0, NULL) || 3 != match(0x13, 3, 0, NULL)) result = EXIT_FAILURE;
  }

  /* absent device: no ordinal, but the count still reports the enumeration */
  if (EXIT_SUCCESS == result) {
    n = -1;
    if (-1 != match(0x99, 0, 0, &n) || 4 != n) result = EXIT_FAILURE;
  }

  /* a matching bus is not a match on its own: slot has to agree */
  if (EXIT_SUCCESS == result && -1 != match(0x12, 3, 0, NULL)) result = EXIT_FAILURE;

  /* nor does a matching bus and slot in another domain */
  if (EXIT_SUCCESS == result && -1 != match(0x12, 2, 1, NULL)) result = EXIT_FAILURE;

  /* an unreported domain reads as domain 0, which is the single-domain answer */
  if (EXIT_SUCCESS == result) {
    setup(4);
    table_no_domain = 1;
    if (2 != match(0x12, 2, 0, NULL)) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result && -1 != match(0x12, 2, 1, NULL)) result = EXIT_FAILURE;
  }

  /* an unreported slot must not match anything, not even slot 0 */
  if (EXIT_SUCCESS == result) {
    setup(4);
    table_no_slot = 1;
    if (-1 != match(0x10, 0, 0, NULL)) result = EXIT_FAILURE;
  }

  /* the runtime enumerated nothing: distinguishable from "not found" by n */
  if (EXIT_SUCCESS == result) {
    setup(0);
    n = -1;
    if (-1 != match(0x10, 0, 0, &n) || 0 != n) result = EXIT_FAILURE;
  }

  /* the first of two devices sharing an address wins, and it is not overwritten */
  if (EXIT_SUCCESS == result) {
    setup(4);
    table_bus[3] = table_bus[1];
    table_slot[3] = table_slot[1];
    if (1 != match((unsigned int)table_bus[1], (unsigned int)table_slot[1], 0, NULL)) {
      result = EXIT_FAILURE;
    }
  }

  /* a runtime that never stops is bounded, rather than scanned forever */
  if (EXIT_SUCCESS == result) {
    n = -1;
    if (-1 != libxstream_pin_match(0xFFFF, 0xFFFF, 0, endless_attr,
      ATTR_BUS, ATTR_SLOT, ATTR_DOMAIN, MAXDEVS, &n) || MAXDEVS != n)
    {
      result = EXIT_FAILURE;
    }
  }

  /* no callback at all is not a match */
  if (EXIT_SUCCESS == result) {
    n = -1;
    if (-1 != libxstream_pin_match(0x10, 0, 0, NULL,
      ATTR_BUS, ATTR_SLOT, ATTR_DOMAIN, MAXDEVS, &n) || 0 != n)
    {
      result = EXIT_FAILURE;
    }
  }

  if (EXIT_SUCCESS != result) fprintf(stderr, "pin: device mapping is wrong\n");
  return result;
}


static int fake_attr(int* value, int attribute, int ordinal)
{
  int result = EXIT_FAILURE;
  if (NULL != value && 0 <= ordinal && ordinal < table_ndevs) {
    if (ATTR_BUS == attribute) {
      *value = table_bus[ordinal];
      result = EXIT_SUCCESS;
    }
    else if (ATTR_SLOT == attribute && 0 == table_no_slot) {
      *value = table_slot[ordinal];
      result = EXIT_SUCCESS;
    }
    else if (ATTR_DOMAIN == attribute && 0 == table_no_domain) {
      *value = table_domain[ordinal];
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int endless_attr(int* value, int attribute, int ordinal)
{
  if (NULL != value) *value = (ATTR_BUS == attribute ? ordinal : 0);
  return EXIT_SUCCESS;
}


static int match(unsigned int bus, unsigned int slot, unsigned int domain, int* ndevices)
{
  return libxstream_pin_match(bus, slot, domain, fake_attr,
    ATTR_BUS, ATTR_SLOT, ATTR_DOMAIN, MAXDEVS, ndevices);
}


static void setup(int ndevs)
{
  int i;
  table_ndevs = ndevs;
  table_no_slot = table_no_domain = 0;
  for (i = 0; i < ndevs; ++i) {
    table_bus[i] = 0x10 + i;
    table_slot[i] = i;
    table_domain[i] = 0;
  }
}
