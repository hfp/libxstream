/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXSTREAM_MACROS_H
#define LIBXSTREAM_MACROS_H

/* LIBXSTREAM header-only implies LIBXS header-only */
#if defined(LIBXSTREAM_SOURCE) && !defined(LIBXS_SOURCE)
# define LIBXS_SOURCE
#endif

#if !defined(LIBXS_MACROS_H)
# if defined(__LIBXS) || defined(LIBXS_BUILD)
#   include <libxs_macros.h>
# else /* header-only: libxs_source.h must come first */
#   include <libxs_source.h>
# endif
#endif
#if defined(LIBXSTREAM_BUILD)
# include "libxstream_version.h"
#endif

/**
 * Build-kind selection for LIBXSTREAM's own API decoration.
 * Uses the parameterized API macros from libxs_macros.h:
 *  - LIBXSTREAM_SOURCE: header-only / inline build
 *  - LIBXSTREAM_BUILD:  building the library (export symbols)
 *  - neither:           consuming the library (import symbols)
 */
#if defined(LIBXSTREAM_SOURCE) || defined(LIBXSTREAM_SOURCE_H)
# define LIBXSTREAM_BUILD_KIND LIBXS_APIKIND_INLINE
#elif defined(LIBXSTREAM_BUILD)
# define LIBXSTREAM_BUILD_KIND LIBXS_APIKIND_EXPORT
#else
# define LIBXSTREAM_BUILD_KIND LIBXS_APIKIND_IMPORT
#endif

/** Function decoration for public LIBXSTREAM functions. */
#define LIBXSTREAM_API LIBXS_API_DECL(LIBXSTREAM_BUILD_KIND)
/** Function decoration for internal LIBXSTREAM functions. */
#define LIBXSTREAM_API_INTERN LIBXS_API_DECL_INTERN(LIBXSTREAM_BUILD_KIND)

/** Public variable declaration (header). */
#define LIBXSTREAM_APIVAR_PUBLIC(DECL) LIBXS_APIVAR_DECL_PUBLIC(DECL, LIBXSTREAM_BUILD_KIND)
/** Private variable declaration (header). */
#define LIBXSTREAM_APIVAR_PRIVATE(DECL) LIBXS_APIVAR_DECL_PRIVATE(DECL, LIBXSTREAM_BUILD_KIND)

/**
 * Variable definition helper — expands alignment + visibility without
 * an extern linkage specifier.  Avoids the empty macro argument that
 * LIBXS_APIVAR_DECL_ALIGNED passes to LIBXS_APIVAR_DECL (C90 §6.8.3).
 */
#define LIBXSTREAM_APIVAR_DEF(DECL, VIS, KIND) \
  LIBXS_ALIGNED(LIBXS_APIKIND_COMMON(KIND) LIBXS_APIKIND_VIS(VIS, KIND) DECL, LIBXS_ALIGNMENT)

/** Public variable definition (source). */
#define LIBXSTREAM_APIVAR_PUBLIC_DEF(DECL) LIBXSTREAM_APIVAR_DEF(DECL, EXPORT, LIBXSTREAM_BUILD_KIND)
/** Private variable definition (source). */
#define LIBXSTREAM_APIVAR_PRIVATE_DEF(DECL) LIBXSTREAM_APIVAR_DEF(DECL, INTERN, LIBXSTREAM_BUILD_KIND)
/** Private variable declaration+definition (source). */
#define LIBXSTREAM_APIVAR_DEFINE(DECL) \
  LIBXSTREAM_APIVAR_PRIVATE(DECL); \
  LIBXSTREAM_APIVAR_PRIVATE_DEF(DECL)

/* header-only: include implementation when not building or linking the library.
 * Skip when inside libxstream_opencl.h, libxstream_dbcsr.h, or libxstream_cp2k.h
 * (deferred to the end of libxstream_opencl.h, after all types are defined). */
#if defined(LIBXSTREAM_SOURCE) && !defined(LIBXSTREAM_SOURCE_H) \
 && !defined(LIBXSTREAM_OPENCL_H) && !defined(DBCSR_ACC_H) && !defined(LIBXSTREAM_CP2K_H)
# include "libxstream_source.h"
#endif

#endif /*LIBXSTREAM_MACROS_H*/
