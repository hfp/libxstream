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
 * The decoder of libxs_vpack's level-1 streams (opencl/libxstream_pack.h) through
 * the host shim: a kernel reads a range of values from any group whose position it
 * is given, per lane (vunpack_code) and per group (vunpack_lane), and the result is
 * checked against the values packed (and against libxs_vunpack). Widths 0 to 64
 * occur, codes cross word boundaries, the last group is partial, and signed values
 * check the zig-zag fold.
 */

#include <libxs/libxs_pack.h>
#include <stdio.h>

#include <libxstream/opencl/libxstream_cpu_begin.h>
#include <libxstream/opencl/libxstream_pack.h>

/* values [first, first + count) of a stream, reading from group first / 8 at bit position pos */
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


static void shim_pack_launch(const uint64_t* stream, int n, int first, int count, uint64_t pos, int is_signed,
  int64_t* out)
{
  LIBXSTREAM_CPU_GRID(1, 1, 1);
  LIBXSTREAM_CPU_WORKITEM(0, 0, 0, 0, 0, 1, 1);
  shim_pack_read((const ulong*)stream, n, first, count, pos, is_signed, (long*)out);
}

#include <libxstream/opencl/libxstream_cpu_end.h>


/* bit position of group g: the prefix sum of the widths in the header */
static uint64_t shim_pack_position(const uint64_t* stream, int g)
{
  const unsigned char* const head = (const unsigned char*)stream;
  uint64_t pos = 0;
  int i;
  for (i = 0; i < g; ++i) pos += head[1 + i];
  return pos;
}


static int shim_pack_case(int n, int maxbits, int is_signed, unsigned int seed)
{
  const size_t bound = libxs_vpack_bound((size_t)n, maxbits);
  uint64_t* const stream = (uint64_t*)malloc(sizeof(uint64_t) * bound);
  int64_t *const values = (int64_t*)malloc(sizeof(int64_t) * n), *const out = (int64_t*)malloc(sizeof(int64_t) * n);
  int result = EXIT_SUCCESS, i, k;
  if (NULL != stream && NULL != values && NULL != out) {
    uint64_t state = seed;
    size_t nwords;
    for (i = 0; i < n; ++i) { /* widths vary per group: each group draws its own magnitude */
      int bits;
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      bits = (int)((state >> 33) % (unsigned int)(maxbits + 1)) * (0 == (i / 8) % 5 ? 0 : 1);
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      { const uint64_t mask = (64 > bits ? ((((uint64_t)1) << bits) - 1) : ~(uint64_t)0);
        uint64_t v = (state ^ (state >> 29)) & mask;
        if (0 != is_signed) { /* magnitudes that fit maxbits with a sign */
          const int s = (int)(state >> 63);
          v >>= (0 < bits ? 1 : 0);
          values[i] = (0 != s ? -(int64_t)v - 1 : (int64_t)v);
          if (maxbits < 64) {
            const int64_t lo = -(((int64_t)1) << (maxbits - 1)), hi = (((int64_t)1) << (maxbits - 1)) - 1;
            values[i] = LIBXS_MAX(lo, LIBXS_MIN(hi, values[i]));
          }
        }
        else values[i] = (int64_t)v;
      }
    }
    nwords = libxs_vpack(stream, values, 0 != is_signed ? LIBXS_DATATYPE_I64 : LIBXS_DATATYPE_U64, (size_t)n, maxbits,
      1 /*level*/);
    if (0 == nwords || 8 != ((const unsigned char*)stream)[0]) {
      fprintf(stderr, "shim: pack n=%i maxbits=%i: not a level-1 stream\n", n, maxbits);
      result = EXIT_FAILURE;
    }
    /* from every group to the end, and the whole stream */
    for (k = 0; k < n && EXIT_SUCCESS == result; k += 8) {
      const int count = n - k;
      shim_pack_launch(stream, n, k, count, shim_pack_position(stream, k / 8), is_signed, out);
      for (i = 0; i < count; ++i) {
        if (out[i] != values[k + i]) {
          fprintf(stderr, "shim: pack n=%i maxbits=%i signed=%i: value %i from group %i is %lli, expected %lli\n", n,
            maxbits, is_signed, k + i, k / 8, (long long)out[i], (long long)values[k + i]);
          result = EXIT_FAILURE;
          break;
        }
      }
    }
    if (EXIT_SUCCESS == result) { /* the library reads the same stream */
      libxs_vunpack(out, stream, nwords, 0 != is_signed ? LIBXS_DATATYPE_I64 : LIBXS_DATATYPE_U64, (size_t)n);
      for (i = 0; i < n; ++i) {
        if (out[i] != values[i]) {
          fprintf(stderr, "shim: pack n=%i maxbits=%i: libxs_vunpack disagrees at %i\n", n, maxbits, i);
          result = EXIT_FAILURE;
          break;
        }
      }
    }
  }
  else {
    fprintf(stderr, "shim: pack out of memory\n");
    result = EXIT_FAILURE;
  }
  free(stream);
  free(values);
  free(out);
  return result;
}


int main(void)
{
  const int sizes[] = {1, 7, 8, 9, 63, 64, 65, 1000};
  const int widths[] = {1, 7, 18, 31, 32, 33, 57, 63, 64};
  int result = EXIT_SUCCESS, ncases = 0, i, j, s;
  for (i = 0; i < (int)(sizeof(sizes) / sizeof(*sizes)); ++i) {
    for (j = 0; j < (int)(sizeof(widths) / sizeof(*widths)); ++j) {
      for (s = 0; s < 2; ++s) {
        if (EXIT_SUCCESS != shim_pack_case(sizes[i], widths[j], s, (unsigned int)(1 + i * 31 + j * 7 + s))) {
          result = EXIT_FAILURE;
        }
        ++ncases;
      }
    }
  }
  if (EXIT_SUCCESS == result) printf("shim: pack (level 1), %i cases bit-exact\n", ncases);
  return result;
}
