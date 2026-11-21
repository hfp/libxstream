/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SHIM_PACK_H
#define SHIM_PACK_H

/* The kernel of shim_pack.c, in OpenCL C: included between libxstream_cpu_begin.h and libxstream_cpu_end.h */
#include <libxstream/opencl/libxstream_pack.h>


/* Values [first, first + count) of a stream, from group first / 8 at bit position pos, per lane and per group */
kernel void shim_pack_read(global const ulong* restrict stream, const int n, const int first, const int count,
  const ulong pos, const int is_signed, global long* restrict out)
{
  global const uchar* const head = (global const uchar*)stream;
  global const ulong* const payload = stream + VUNPACK_NHEAD(n);
  ulong p = pos;
  int g = first / 8, w = vunpack_width(head, g), i;
  vunpack_group_t group = vunpack_group(payload, p, w);
  for (i = first; i < first + count; ++i) {
    const int l = i % 8;
    if (0 == l && i != first) { /* the next group */
      p += (ulong)w;
      w = vunpack_width(head, ++g);
      group = vunpack_group(payload, p, w);
    }
    { const ulong code = vunpack_code(payload, p, w, l), lane = vunpack_lane(&group, l);
      out[i - first] = (code != lane ? -1 : (0 != is_signed ? vunpack_signed(code) : (long)code));
    }
  }
}

#endif /*SHIM_PACK_H*/
