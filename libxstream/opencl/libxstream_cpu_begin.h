/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__LIBXS) || defined(LIBXS_SOURCE)
# include <libxs/libxs_macros.h>
# include <libxs/libxs_math.h>
#else
# include <stddef.h>
# include <stdlib.h>
# include <math.h>
#endif

/**
 * OpenCL C to host C shim: lets an ordinary C compiler translate a kernel that
 * was written for a device. Unlike its neighbours here, this header is included
 * by host sources rather than by OpenCL sources.
 *
 * Include it after every other header, then the kernel source, then
 * libxstream_cpu_end.h. The bracket is not cosmetic: the OpenCL keywords are
 * empty macros, and gcc expands macros in a pragma line, so an in-scope
 * "private" silently deletes an OpenMP private clause rather than failing.
 *
 * The work-item state is file-static, so the launcher belongs to the same
 * translation unit as the kernel it launches.
 *
 * Besides the keywords, the shim covers the built-ins a kernel reaches for that
 * the host has under another name or not at all: the as_* reinterpretations,
 * the integer atomics and compare-exchange, clz, mul_hi, min, fma, and the
 * work-item queries. A built-in the host already spells the same way (abs,
 * floor) is left to stdlib.h and math.h, which is why they are included here
 * rather than by whoever brackets a kernel. With LIBXS they come from
 * libxs_macros.h, which places them after its feature-test macros; a system
 * header ahead of those would have glibc settle on a surface without them. The
 * vector types are not here but in libxstream_vectors.h, which the kernel
 * reaches through libxstream_common.h, and which a kernel using them therefore
 * includes.
 *
 * Work-group model, selected by LIBXSTREAM_CPU_TEAM:
 * 0 (default) A work-item runs to completion, hence barrier() is a no-op and
 *   local memory is an automatic array. The kernel must then carry no
 *   cross-lane dependency, which a kernel arranges either by turning its lanes
 *   into loops or by being launched with a 1x1 work-group. Work-groups stay
 *   independent, so the launcher may run them in parallel.
 * 1 Lanes form an OpenMP team of one thread each and barrier() is an orphaned
 *   OpenMP barrier that binds to it. Local memory becomes static, hence one
 *   work-group at a time. Reproduces the on-device local memory tiling and is a
 *   debugging aid rather than a fast path.
 *
 * Host long is 32-bit on LLP64 targets whereas OpenCL long is 64-bit, so a host
 * build indexes smaller buffers there than the device does.
 *
 * Plain char is the one type the shim cannot carry over: OpenCL defines it as
 * signed, a host ABI may not, and redefining it would rewrite "unsigned char" as
 * well. A kernel therefore spells signed char wherever the sign of a byte is read
 * (plain char stays fine for addressing bytes); otherwise the host reads other
 * numbers and nothing fails.
 */

#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM) && !defined(_OPENMP)
# error LIBXSTREAM_CPU_TEAM needs OpenMP to implement barrier()
#endif

/**
 * Says that the host translates what follows, which libxstream_vectors.h needs
 * to know before libxstream_common.h reaches it: the alternative, an absent
 * OpenCL predefine, also describes the host preprocessor a kernel passes through
 * on its way to a device, and would map the vectors for the wrong translator.
 * Survives libxstream_cpu_end.h, because the types it selected do.
 */
#define LIBXSTREAM_CPU 1

/* Address spaces: a host build has only the one. */
#define global
#define private
#define constant const
#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM)
# define local static
#else
# define local
#endif

/* Kernels become ordinary functions that the launcher calls directly. */
#define kernel static

/**
 * A device-side helper becomes file-local, otherwise an external inline
 * definition refers to the static conversion helpers below. LIBXS already spells
 * "inline" for C89, hence the undef before the redefinition.
 */
#undef inline
#define inline static
/**
 * What the bracket admits that the host dialect would reject: a helper the kernel
 * at hand does not call (the shim neutralizes __attribute__, so the attribute
 * route to silence it is not available), a directive only an OpenCL compiler
 * knows ("#pragma OPENCL EXTENSION"), a declaration after a statement, which
 * OpenCL C has from C99 and a C89 host build does not, and a nested array given
 * a flat initializer, as a host emits its tables and an OpenCL compiler takes
 * them. A C99 for-declaration is an error rather than a warning there and stays
 * out of reach of any pragma.
 */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif

/* Kernel attributes carry no meaning on the host. */
#if !defined(__attribute__)
# define __attribute__(A)
#endif

/* C89 has no restrict, yet GNU compilers still honor the alias promise. */
#if !defined(__STDC_VERSION__) || (199901L > __STDC_VERSION__)
# if defined(__GNUC__) || defined(__clang__)
#   define restrict __restrict__
# else
#   define restrict
# endif
#endif

/**
 * Scalar type names that OpenCL C provides as built-ins. ulong is the one that
 * cannot be spelled as the host's own: OpenCL's is 64-bit, and a kernel reaches
 * it to hold the bit pattern of a double, where 32 bits would not be a smaller
 * range but a different number. Hence the typedef below rather than "unsigned
 * long", which is 32-bit on LLP64 targets.
 */
#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ulong libxstream_cpu_ulong_t

/* Fence flags: barrier() has nothing to order on a host build. */
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2

/**
 * The kernel's unroll hints carry over: libxstream_common.h keeps whatever is
 * defined here rather than dropping the hints off-device. Unrolling the short
 * constant-trip loops matters more on the host, because it is what leaves a
 * lane loop innermost with a straight-line body for the vectorizer.
 */
#if defined(LIBXS_PRAGMA_UNROLL_N)
# define UNROLL_FORCE(N) LIBXS_PRAGMA_UNROLL_N(N)
# define UNROLL_AUTO LIBXS_PRAGMA_UNROLL
#else
# define UNROLL_FORCE(N)
# define UNROLL_AUTO
#endif

/* Vectorize a lane loop nest of N levels. */
#if defined(LIBXS_PRAGMA_SIMD_COLLAPSE)
# define SIMD_COLLAPSE(N) LIBXS_PRAGMA_SIMD_COLLAPSE(N)
#else
# define SIMD_COLLAPSE(N)
#endif

/**
 * Storage conversions: OpenCL C reaches FP16 through the half built-ins and a
 * device may convert BF16 in hardware, neither of which a host build has. LIBXS
 * carries the host implementations, and a consumer that does not have LIBXS
 * defines the four names before this header rather than growing the shim.
 */
#if defined(__LIBXS) || defined(LIBXS_SOURCE)
# if !defined(ROUND_TO_BF16)
#   define ROUND_TO_BF16(X) libxs_round_bf16_f32(X)
# endif
# if !defined(BF16_TO_F32)
#   define BF16_TO_F32(X) libxs_bf16_to_f32(X)
# endif
# if !defined(ROUND_TO_F16)
#   define ROUND_TO_F16(X) libxs_round_f16_f32(X)
# endif
# if !defined(F16_TO_F32)
#   define F16_TO_F32(X) libxs_f16_to_f32(X)
# endif
#endif

/**
 * Reinterpretation. OpenCL selects on the type of the argument and C cannot, so
 * each name takes the one type the kernels pass it, and a caller that passes
 * another gets a conversion instead of a reinterpretation: as_uint(D) on a
 * double yields the bits of a float. The cast at the call site is what keeps
 * them apart, which is why AS_UINT in libxstream_common.h picks the name by
 * real_t rather than leaving it to overloading that is not there.
 */
#define as_uint(X) libxstream_cpu_as_uint(X)
#define as_int(X) libxstream_cpu_as_int(X)
#define as_float(X) libxstream_cpu_as_float(X)
#define as_ulong(X) libxstream_cpu_as_ulong(X)
#define as_double(X) libxstream_cpu_as_double(X)

/**
 * Integer built-ins. clz takes the width from the argument, hence one host
 * implementation serves the 32-bit and the 64-bit kernel alike; mul_hi is the
 * 64-bit form, and a 32-bit argument would widen and answer zero rather than
 * its own high half.
 */
#define clz(X) libxstream_cpu_clz(X, (int)(8 * sizeof(X)))
#define mul_hi(A, B) libxstream_cpu_mul_hi(A, B)

/**
 * fma is exactness and not convenience: an error-free transformation takes the
 * error of a product from it, and a rounded-twice a*b+c would answer a wrong
 * error rather than a less precise one, which is also why fast-math invalidates
 * the kernel that asks for it. The builtin is taken where there is one, because
 * a strict-C89 host has no declaration for fma in math.h and would reach for an
 * implicit one that returns int.
 *
 * OpenCL overloads fma on its argument type and C cannot, so the width is taken
 * from the type the three arguments have in common: a float fma formed in double
 * rounds its sum once in double and again to float, which is a different number.
 * The result is double either way, and holds the float one exactly; assigning it
 * or combining it in one more operation rounds as the device does, whereas a
 * chain of several operations on it would be evaluated in double first.
 */
#if defined(__GNUC__) || defined(__clang__)
# define fma(A, B, C) (sizeof(float) == sizeof((A) + (B) + (C)) \
    ? (double)__builtin_fmaf((float)(A), (float)(B), (float)(C)) \
    : __builtin_fma(A, B, C))
#endif

/**
 * Atomics return the value they replaced, as the device ones do. The launcher
 * may run work-groups in parallel, hence they are atomic against each other
 * and not merely read-modify-write.
 */
#define atomic_max(A, B) libxstream_cpu_atomic_max(A, B)
#define atomic_or(A, B) libxstream_cpu_atomic_or(A, B)
/**
 * Compare-exchange, from which a device without float atomics builds its atomic
 * add (libxstream_atomics.h, CMPXCHG): 32-bit under the core name, 64-bit under
 * the spelling of the atom extension. A team runs lanes at the same time, so these
 * exclude each other like the ones above rather than merely reading and writing.
 */
#define atomic_cmpxchg(A, E, V) libxstream_cpu_cmpxchg32((volatile int*)(A), E, V)
#define atom_cmpxchg(A, E, V) libxstream_cpu_cmpxchg64((volatile long*)(A), E, V)

/* The integer built-in, over the type its operands have in common. */
#define min(A, B) ((A) < (B) ? (A) : (B))

#define get_group_id(D) ((size_t)libxstream_cpu_gid[D])
#define get_local_id(D) ((size_t)libxstream_cpu_lid[D])
#define get_local_size(D) ((size_t)libxstream_cpu_lsz[D])
#define get_global_id(D) \
  ((size_t)(libxstream_cpu_gid[D] * libxstream_cpu_lsz[D] + libxstream_cpu_lid[D]))
/**
 * The grid is not part of the work-item state, so a kernel that asks for it
 * needs LIBXSTREAM_CPU_GRID as well, and answers zero until it gets it: a grid
 * the launcher never stated is better read as no work than as one work-group.
 */
#define get_num_groups(D) ((size_t)libxstream_cpu_nwg[D])
#define get_global_size(D) ((size_t)(libxstream_cpu_nwg[D] * libxstream_cpu_lsz[D]))

/* Publish the work-group count once per launch; the work-items follow. */
#define LIBXSTREAM_CPU_GRID(N0, N1, N2) libxstream_cpu_grid(N0, N1, N2)

/* Publish one work-item; the third dimension is degenerate. Survives _end.h. */
#define LIBXSTREAM_CPU_WORKITEM(G0, G1, G2, L0, L1, S0, S1) do { \
  libxstream_cpu_gid[0] = (G0); \
  libxstream_cpu_gid[1] = (G1); \
  libxstream_cpu_gid[2] = (G2); \
  libxstream_cpu_lid[0] = (L0); \
  libxstream_cpu_lid[1] = (L1); \
  libxstream_cpu_lid[2] = 0; \
  libxstream_cpu_lsz[0] = (S0); \
  libxstream_cpu_lsz[1] = (S1); \
  libxstream_cpu_lsz[2] = 1; \
} while (0)

#define barrier(FLAGS) libxstream_cpu_barrier()

/**
 * Defined once even where a translation unit brackets several kernels, whereas
 * the spellings above are re-established by every bracket because
 * libxstream_cpu_end.h retires them.
 *
 * Work-item coordinates the launcher publishes before each kernel call.
 * Threadprivate because one thread runs one whole work-group (default) or one
 * work-item of a work-group (LIBXSTREAM_CPU_TEAM) at a time.
 */
#if !defined(LIBXSTREAM_CPU_STATE)
#define LIBXSTREAM_CPU_STATE
/**
 * OpenCL's 64-bit ulong. LIBXS has already settled which spelling the target
 * has; without it the host's own is taken, and a kernel that reinterprets a
 * double then needs a target where it is 64 bits wide.
 */
#if defined(__LIBXS) || defined(LIBXS_SOURCE)
typedef uint64_t libxstream_cpu_ulong_t;
#else
typedef unsigned long libxstream_cpu_ulong_t;
#endif

static int libxstream_cpu_gid[3];
static int libxstream_cpu_lid[3];
static int libxstream_cpu_lsz[3];
#if defined(_OPENMP)
# pragma omp threadprivate(libxstream_cpu_gid, libxstream_cpu_lid, libxstream_cpu_lsz)
#endif
/* The grid is one per launch rather than per work-item, hence shared. */
static int libxstream_cpu_nwg[3];


/**
 * A function rather than the assignments in the macro, because a launcher that
 * never asks for the grid would leave the state above unreferenced: the variable
 * would then draw an unused warning that a function does not, the bracket
 * exempting unused functions and the shim having neutralized __attribute__.
 */
static void libxstream_cpu_grid(int n0, int n1, int n2)
{
  libxstream_cpu_nwg[0] = n0;
  libxstream_cpu_nwg[1] = n1;
  libxstream_cpu_nwg[2] = n2;
}


/**
 * Work-group barrier. An orphaned OpenMP barrier binds to the team the launcher
 * opened for the work-group, which spells it without the _Pragma operator that
 * C89 lacks.
 */
static void libxstream_cpu_barrier(void)
{
#if defined(LIBXSTREAM_CPU_TEAM) && (0 != LIBXSTREAM_CPU_TEAM)
# pragma omp barrier
#endif
}


/**
 * The as_* family. A union rather than a pointer cast: the cast would promise
 * the compiler two types never alias, which is exactly what is being done here.
 */
static unsigned int libxstream_cpu_as_uint(float value)
{
  union { float f; unsigned int u; } pun;
  pun.f = value;
  return pun.u;
}


static int libxstream_cpu_as_int(float value)
{
  union { float f; int i; } pun;
  pun.f = value;
  return pun.i;
}


static float libxstream_cpu_as_float(int value)
{
  union { float f; int i; } pun;
  pun.i = value;
  return pun.f;
}


static libxstream_cpu_ulong_t libxstream_cpu_as_ulong(double value)
{
  union { double d; libxstream_cpu_ulong_t u; } pun;
  pun.d = value;
  return pun.u;
}


static double libxstream_cpu_as_double(libxstream_cpu_ulong_t value)
{
  union { double d; libxstream_cpu_ulong_t u; } pun;
  pun.u = value;
  return pun.d;
}


/* Leading zeros within BITS, which the macro takes from the argument. */
static int libxstream_cpu_clz(libxstream_cpu_ulong_t value, int bits)
{
  int result = 0;
  while (result < bits && 0 == (value >> (bits - 1 - result))) ++result;
  return result;
}


/**
 * High half of a 64x64 product, assembled from 32-bit partials: the host may
 * have no wider integer, and the one it does have is not spelled portably.
 */
static libxstream_cpu_ulong_t libxstream_cpu_mul_hi(
  libxstream_cpu_ulong_t a, libxstream_cpu_ulong_t b)
{
  const libxstream_cpu_ulong_t mask = (libxstream_cpu_ulong_t)0xFFFFFFFF;
  const libxstream_cpu_ulong_t alo = a & mask, ahi = a >> 32;
  const libxstream_cpu_ulong_t blo = b & mask, bhi = b >> 32;
  const libxstream_cpu_ulong_t ll = alo * blo;
  const libxstream_cpu_ulong_t lh = alo * bhi;
  const libxstream_cpu_ulong_t hl = ahi * blo;
  /* The carry out of the low half is what the two cross terms contribute. */
  const libxstream_cpu_ulong_t carry = (ll >> 32) + (lh & mask) + (hl & mask);
  return ahi * bhi + (lh >> 32) + (hl >> 32) + (carry >> 32);
}


static int libxstream_cpu_atomic_max(int* address, int value)
{
  int result;
#if defined(_OPENMP)
# pragma omp critical(libxstream_cpu_atomic)
#endif
  {
    result = *address;
    if (result < value) *address = value;
  }
  return result;
}


static int libxstream_cpu_atomic_or(int* address, int value)
{
  int result;
#if defined(_OPENMP)
# pragma omp critical(libxstream_cpu_atomic)
#endif
  {
    result = *address;
    *address = result | value;
  }
  return result;
}


static int libxstream_cpu_cmpxchg32(volatile int* address, int expected, int desired)
{
  int result;
#if defined(_OPENMP)
# pragma omp critical(libxstream_cpu_atomic)
#endif
  {
    result = *address;
    if (result == expected) *address = desired;
  }
  return result;
}


static long libxstream_cpu_cmpxchg64(volatile long* address, long expected, long desired)
{
  long result;
#if defined(_OPENMP)
# pragma omp critical(libxstream_cpu_atomic)
#endif
  {
    result = *address;
    if (result == expected) *address = desired;
  }
  return result;
}

#endif /*LIBXSTREAM_CPU_STATE*/
