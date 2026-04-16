/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_SOURCE_H
#define LIBXSTREAM_SOURCE_H

#if defined(LIBXSTREAM_MACROS_H)
# error Please do not include any LIBXSTREAM header other than libxstream_source.h!
#endif
#if defined(LIBXSTREAM_BUILD)
# error LIBXSTREAM_BUILD cannot be defined for the header-only LIBXSTREAM!
#endif

/**
 * This header is intentionally called "libxstream_source.h" since the followings block
 * includes *internal* files, and thereby exposes LIBXSTREAM's implementation.
 * The so-called "header-only" usage model gives up the clearly defined binary interface
 * (including support for hot-fixes after deployment), and requires to rebuild client
 * code for every (internal) change of LIBXSTREAM. Please make sure to only rely on the
 * public interface as the internal implementation may change without notice.
 */
#include "libxstream.h"
#include "../src/libxstream.c"
#include "../src/libxstream_cp2k.c"
#include "../src/libxstream_dbcsr.c"
#include "../src/libxstream_event.c"
#include "../src/libxstream_mem.c"
#include "../src/libxstream_stream.c"

#endif /*LIBXSTREAM_SOURCE_H*/
