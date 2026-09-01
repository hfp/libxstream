/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef STENCIL_HOSTMEM_H
#define STENCIL_HOSTMEM_H

#include <stddef.h>

/**
 * Stands in for the parts of LIBXSTREAM the sample uses when the kernel runs
 * on the host (make CPU=1). Host and device share one address space there, so
 * a device allocation is an allocation and a transfer is a copy that vanishes
 * whenever the two pointers already name the same memory.
 *
 * The redirections are macros rather than functions of the same name: the
 * sample must not define symbols that belong to LIBXSTREAM, and the call sites
 * stay recognizable as the host path.
 */

/* Opaque stream handle: nothing on the host is asynchronous. */
typedef struct libxstream_stream_t libxstream_stream_t;

typedef enum libxstream_opencl_mem_hint_t {
  libxstream_opencl_mem_hint_compress = 0
} libxstream_opencl_mem_hint_t;

int stencil_host_allocate(void** ptr, size_t nbytes);
int stencil_host_deallocate(void* ptr);
int stencil_host_copy(void* dst, const void* src, size_t nbytes);
int stencil_host_zero(void* ptr, size_t offset, size_t nbytes);

#define libxstream_init() EXIT_SUCCESS
#define libxstream_finalize() ((void)0)
#define libxstream_device_count(NDEVICES) (*(NDEVICES) = 1, EXIT_SUCCESS)
#define libxstream_device_set_active(DEVICE) ((void)(DEVICE), EXIT_SUCCESS)
#define libxstream_stream_sync(STREAM) ((void)(STREAM), EXIT_SUCCESS)

#define libxstream_mem_host_allocate(PTR, NBYTES, STREAM) \
  stencil_host_allocate(PTR, NBYTES)
#define libxstream_mem_host_deallocate(PTR, STREAM) \
  stencil_host_deallocate(PTR)
#define libxstream_mem_dev_allocate_hint(PTR, NBYTES, HINT) \
  stencil_host_allocate(PTR, NBYTES)
#define libxstream_mem_dev_deallocate_hint(PTR) \
  stencil_host_deallocate(PTR)
#define libxstream_mem_copy_h2d(SRC, DST, NBYTES, STREAM) \
  stencil_host_copy(DST, SRC, NBYTES)
#define libxstream_mem_copy_d2h(SRC, DST, NBYTES, STREAM) \
  stencil_host_copy(DST, SRC, NBYTES)
#define libxstream_mem_zero(PTR, OFFSET, NBYTES, STREAM) \
  stencil_host_zero(PTR, OFFSET, NBYTES)

#endif /*STENCIL_HOSTMEM_H*/
