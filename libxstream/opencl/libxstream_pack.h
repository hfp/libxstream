/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_OPENCL_PACK_H
#define LIBXSTREAM_OPENCL_PACK_H

/**
 * Decodes a stream that libxs_vpack writes at level 1 for integer types, whose
 * format LIBXS documents (libxs_pack.md, "Format of Level 1"): a header of a
 * width per group of eight values, then eight interleaved streams of 64-bit
 * words, lane l of every group in stream l at the group's bit position P(g).
 *
 * A reader keeps P(g) and adds the width of each group it passes, hence groups
 * can be read in order from any group whose position is known (the host takes
 * the prefix sum over the header once). Nothing here depends on the work-item:
 * every work-item that reads a value reads the same words, which suits a
 * work-group walking a list together.
 *
 * VUNPACK_NHEAD(n) is the number of header words for n values: a byte of the
 * group size, then a width per group of eight.
 */
#define VUNPACK_NHEAD(N) ((1 + ((N) + 7) / 8 + 7) / 8)

/* a group taken apart once for its eight lanes (vunpack_lane): its words, offset, width, and mask */
typedef struct {
  global const ulong* words;
  ulong mask;
  int offset, width;
} vunpack_group_t;


/* Width of group g: the bits of its largest code */
inline int vunpack_width(global const uchar* restrict head, const int g)
{
  return (int)head[1 + g];
}


/* Code of lane l (0 to 7) of a group of width w at bit position pos of the payload */
inline ulong vunpack_code(global const ulong* restrict payload, const ulong pos, const int w, const int l)
{
  ulong result = 0;
  if (0 < w) {
    const ulong k = pos >> 6;
    const int o = (int)(pos & 63);
    result = payload[8 * k + l] >> o;
    if (64 < o + w) result |= payload[8 * k + 8 + l] << (64 - o); /* continues in the stream's next word */
    if (64 > w) result &= (((ulong)1) << w) - 1;
  }
  return result;
}


/* Group of width w at bit position pos of the payload, for reading its lanes (vunpack_lane) */
inline vunpack_group_t vunpack_group(global const ulong* restrict payload, const ulong pos, const int w)
{
  vunpack_group_t group;
  group.words = payload + 8 * (pos >> 6);
  group.mask = (64 > w ? ((((ulong)1) << w) - 1) : ~(ulong)0);
  group.offset = (int)(pos & 63);
  group.width = w;
  return group;
}


/* Code of lane l (0 to 7) of a group (vunpack_group): the same as vunpack_code, with the group's words found once */
inline ulong vunpack_lane(const vunpack_group_t* group, const int l)
{
  ulong result = 0;
  if (0 < group->width) {
    result = group->words[l] >> group->offset;
    if (64 < group->offset + group->width) result |= group->words[8 + l] << (64 - group->offset);
    result &= group->mask;
  }
  return result;
}


/* Value of a signed type from its code (zig-zag: 0, 1, 2, 3, ... become 0, -1, 1, -2, ...) */
inline long vunpack_signed(const ulong code)
{
  return (long)(code >> 1) ^ -(long)(code & 1);
}

#endif /*LIBXSTREAM_OPENCL_PACK_H*/
