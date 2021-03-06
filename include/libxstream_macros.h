/******************************************************************************
** Copyright (c) 2013-2016, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSTREAM_MACROS_H
#define LIBXSTREAM_MACROS_H

#include "libxstream_config.h"

#define LIBXSTREAM_STRINGIFY(SYMBOL) #SYMBOL
#define LIBXSTREAM_TOSTRING(SYMBOL) LIBXSTREAM_STRINGIFY(SYMBOL)
#define LIBXSTREAM_CONCATENATE2(A, B) A##B
#define LIBXSTREAM_CONCATENATE(A, B) LIBXSTREAM_CONCATENATE2(A, B)
#define LIBXSTREAM_FSYMBOL(SYMBOL) LIBXSTREAM_CONCATENATE2(SYMBOL, _)
#define LIBXSTREAM_UNIQUE(NAME) LIBXSTREAM_CONCATENATE(NAME, __LINE__)

#if defined(__cplusplus)
# define LIBXSTREAM_EXTERN_C extern "C"
# define LIBXSTREAM_INLINE inline
# define LIBXSTREAM_VARIADIC ...
# define LIBXSTREAM_EXPORT_C LIBXSTREAM_EXTERN_C LIBXSTREAM_EXPORT
#else
# define LIBXSTREAM_EXTERN_C
# define LIBXSTREAM_VARIADIC
# define LIBXSTREAM_EXPORT_C LIBXSTREAM_EXPORT
# if defined(__STDC_VERSION__) && (199901L <= (__STDC_VERSION__))
#   define LIBXSTREAM_PRAGMA(DIRECTIVE) _Pragma(LIBXSTREAM_STRINGIFY(DIRECTIVE))
#   define LIBXSTREAM_RESTRICT restrict
#   define LIBXSTREAM_INLINE static inline
# elif defined(_MSC_VER)
#   define LIBXSTREAM_INLINE static __inline
# else
#   define LIBXSTREAM_INLINE static
# endif /*C99*/
#endif /*__cplusplus*/

#if !defined(LIBXSTREAM_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(__INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXSTREAM_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#   define LIBXSTREAM_RESTRICT __restrict
# else
#   define LIBXSTREAM_RESTRICT
# endif
#endif /*LIBXSTREAM_RESTRICT*/

#if !defined(LIBXSTREAM_PRAGMA)
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXSTREAM_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
# else
#   define LIBXSTREAM_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXSTREAM_PRAGMA*/

#if defined(_MSC_VER)
# define LIBXSTREAM_MESSAGE(MSG) LIBXSTREAM_PRAGMA(message(MSG))
#elif (40400 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
# define LIBXSTREAM_MESSAGE(MSG) LIBXSTREAM_PRAGMA(message MSG)
#else
# define LIBXSTREAM_MESSAGE(MSG)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSTREAM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSTREAM_PRAGMA(simd reduction(EXPRESSION))
# define LIBXSTREAM_PRAGMA_SIMD_COLLAPSE(N) LIBXSTREAM_PRAGMA(simd collapse(N))
# define LIBXSTREAM_PRAGMA_SIMD_PRIVATE(...) LIBXSTREAM_PRAGMA(simd private(__VA_ARGS__))
# define LIBXSTREAM_PRAGMA_SIMD LIBXSTREAM_PRAGMA(simd)
# define LIBXSTREAM_PRAGMA_NOVECTOR LIBXSTREAM_PRAGMA(novector)
#elif defined(_OPENMP) && (201307 <= _OPENMP) /*OpenMP 4.0*/
# define LIBXSTREAM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSTREAM_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXSTREAM_PRAGMA_SIMD_COLLAPSE(N) LIBXSTREAM_PRAGMA(omp simd collapse(N))
# define LIBXSTREAM_PRAGMA_SIMD_PRIVATE(...) LIBXSTREAM_PRAGMA(omp simd private(__VA_ARGS__))
# define LIBXSTREAM_PRAGMA_SIMD LIBXSTREAM_PRAGMA(omp simd)
# define LIBXSTREAM_PRAGMA_NOVECTOR
#else
# define LIBXSTREAM_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXSTREAM_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXSTREAM_PRAGMA_SIMD_PRIVATE(...)
# define LIBXSTREAM_PRAGMA_SIMD
# define LIBXSTREAM_PRAGMA_NOVECTOR
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSTREAM_PRAGMA_FORCEINLINE LIBXSTREAM_PRAGMA(forceinline recursive)
# define LIBXSTREAM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXSTREAM_PRAGMA(loop_count min(MIN) max(MAX) avg(AVG))
# define LIBXSTREAM_PRAGMA_UNROLL_N(N) LIBXSTREAM_PRAGMA(unroll(N))
# define LIBXSTREAM_PRAGMA_UNROLL LIBXSTREAM_PRAGMA(unroll)
/*# define LIBXSTREAM_UNUSED(VARIABLE) LIBXSTREAM_PRAGMA(unused(VARIABLE))*/
#else
# define LIBXSTREAM_PRAGMA_FORCEINLINE
# define LIBXSTREAM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXSTREAM_PRAGMA_UNROLL_N(N)
# define LIBXSTREAM_PRAGMA_UNROLL
#endif

#if !defined(LIBXSTREAM_UNUSED)
# if 0 /*defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)*/
#   define LIBXSTREAM_UNUSED(VARIABLE) LIBXSTREAM_PRAGMA(unused(VARIABLE))
# else
#   define LIBXSTREAM_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif

#if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# define LIBXSTREAM_UNUSED_ARG LIBXSTREAM_ATTRIBUTE(unused)
#else
# define LIBXSTREAM_UNUSED_ARG
#endif

/*Based on Stackoverflow's NBITSx macro.*/
#define LIBXSTREAM_NBITS02(N) (0 != ((N) & 2/*0b10*/) ? 1 : 0)
#define LIBXSTREAM_NBITS04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 + LIBXSTREAM_NBITS02((N) >> 2)) : LIBXSTREAM_NBITS02(N))
#define LIBXSTREAM_NBITS08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 + LIBXSTREAM_NBITS04((N) >> 4)) : LIBXSTREAM_NBITS04(N))
#define LIBXSTREAM_NBITS16(N) (0 != ((N) & 0xFF00) ? (8 + LIBXSTREAM_NBITS08((N) >> 8)) : LIBXSTREAM_NBITS08(N))
#define LIBXSTREAM_NBITS32(N) (0 != ((N) & 0xFFFF0000) ? (16 + LIBXSTREAM_NBITS16((N) >> 16)) : LIBXSTREAM_NBITS16(N))
#define LIBXSTREAM_NBITS64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 + LIBXSTREAM_NBITS32((N) >> 32)) : LIBXSTREAM_NBITS32(N))
#define LIBXSTREAM_NBITS(N) (0 != (N) ? (LIBXSTREAM_NBITS64((uint64_t)(N)) + 1) : 1)

#define LIBXSTREAM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSTREAM_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXSTREAM_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXSTREAM_MUL2(N, NPOT) ((N) << (LIBXSTREAM_NBITS(NPOT) - 1))
#define LIBXSTREAM_DIV2(N, NPOT) ((N) >> (LIBXSTREAM_NBITS(NPOT) - 1))
#define LIBXSTREAM_UP2(N, NPOT) LIBXSTREAM_MUL2(LIBXSTREAM_DIV2((N) + (NPOT) - 1, NPOT), NPOT)
#define LIBXSTREAM_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSTREAM_ATTRIBUTE(...) __declspec(__VA_ARGS__)
# if defined(__cplusplus)
#   define LIBXSTREAM_INLINE_ALWAYS __forceinline
# else
#   define LIBXSTREAM_INLINE_ALWAYS static __forceinline
# endif
# define LIBXSTREAM_ALIGNED(DECL, N) LIBXSTREAM_ATTRIBUTE(align(N)) DECL
# define LIBXSTREAM_CDECL __cdecl
#elif defined(__GNUC__)
# define LIBXSTREAM_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
# define LIBXSTREAM_INLINE_ALWAYS LIBXSTREAM_ATTRIBUTE(always_inline) LIBXSTREAM_INLINE
# define LIBXSTREAM_ALIGNED(DECL, N) DECL LIBXSTREAM_ATTRIBUTE(aligned(N))
# define LIBXSTREAM_CDECL LIBXSTREAM_ATTRIBUTE(cdecl)
#else
# define LIBXSTREAM_ATTRIBUTE(...)
# define LIBXSTREAM_INLINE_ALWAYS LIBXSTREAM_INLINE
# define LIBXSTREAM_ALIGNED(DECL, N)
# define LIBXSTREAM_CDECL
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSTREAM_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
# define LIBXSTREAM_ASSUME(EXPRESSION) __assume(EXPRESSION)
#else
# define LIBXSTREAM_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXSTREAM_ASSUME(EXPRESSION) __assume(EXPRESSION)
# elif (40500 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#   define LIBXSTREAM_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
# else
#   define LIBXSTREAM_ASSUME(EXPRESSION)
# endif
#endif
#define LIBXSTREAM_ALIGN_VALUE(N, TYPESIZE, ALIGNMENT/*POT*/) (LIBXSTREAM_UP2((N) * (TYPESIZE), ALIGNMENT) / (TYPESIZE))
#define LIBXSTREAM_ALIGN_VALUE2(N, POTSIZE, ALIGNMENT/*POT*/) LIBXSTREAM_DIV2(LIBXSTREAM_UP2(LIBXSTREAM_MUL2(N, POTSIZE), ALIGNMENT), POTSIZE)
#define LIBXSTREAM_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXSTREAM_ALIGN_VALUE((uintptr_t)(POINTER), 1, ALIGNMENT) - ((uintptr_t)(POINTER))) / sizeof(*(POINTER)))
#define LIBXSTREAM_ALIGN2(POINTPOT, ALIGNMENT/*POT*/) ((POINTPOT) + LIBXSTREAM_DIV2(LIBXSTREAM_ALIGN_VALUE2((uintptr_t)(POINTPOT), 1, ALIGNMENT) - ((uintptr_t)(POINTPOT)), sizeof(*(POINTPOT))))

#define LIBXSTREAM_HASH_VALUE(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXSTREAM_HASH2(POINTER, ALIGNMENT/*POT*/, NPOT) LIBXSTREAM_MOD2(LIBXSTREAM_HASH_VALUE(LIBXSTREAM_DIV2((uintptr_t)(POINTER), ALIGNMENT)), NPOT)

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSTREAM_TLS LIBXSTREAM_ATTRIBUTE(thread)
#elif defined(__GNUC__) || defined(__clang__)
# define LIBXSTREAM_TLS __thread
#elif defined(__cplusplus)
# define LIBXSTREAM_TLS thread_local
#endif

#if defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXSTREAM_OFFLOAD_BUILD 1
# define LIBXSTREAM_OFFLOAD(A) LIBXSTREAM_ATTRIBUTE(target(A))
#else
/*# define LIBXSTREAM_OFFLOAD_BUILD 0*/
# define LIBXSTREAM_OFFLOAD(A)
#endif
#if !defined(LIBXSTREAM_OFFLOAD_TARGET)
# define LIBXSTREAM_OFFLOAD_TARGET mic
#endif
#define LIBXSTREAM_RETARGETABLE LIBXSTREAM_OFFLOAD(LIBXSTREAM_OFFLOAD_TARGET)

/** Execute the CPUID, and receive results (EAX, EBX, ECX, EDX) for requested FUNCTION. */
#if defined(__GNUC__)
# define LIBXSTREAM_CPUID(FUNCTION, EAX, EBX, ECX, EDX) \
    __asm__ __volatile__ ("cpuid" : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) : "a"(FUNCTION))
#else
# define LIBXSTREAM_CPUID(FUNCTION, EAX, EBX, ECX, EDX) { \
    int libxsmm_cpuid_[4]; \
    __cpuid(libxsmm_cpuid_, FUNCTION); \
    EAX = libxsmm_cpuid_[0]; \
    EBX = libxsmm_cpuid_[1]; \
    ECX = libxsmm_cpuid_[2]; \
    EDX = libxsmm_cpuid_[3]; \
  }
#endif

/** Execute the XGETBV, and receive results (EAX, EDX) for req. eXtended Control Register (XCR). */
#if defined(__GNUC__)
# define LIBXSTREAM_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
    "xgetbv" : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
  )
#else
# define LIBXSTREAM_XGETBV(XCR, EAX, EDX) { \
    unsigned long long libxsmm_xgetbv_ = _xgetbv(XCR); \
    EAX = (int)libxsmm_xgetbv_; \
    EDX = (int)(libxsmm_xgetbv_ >> 32); \
  }
#endif

/**
 * Below group of preprocessor symbols are used to fixup some platform specifics.
 */
#if !defined(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
# define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif
#if !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE 1
#endif
#if !defined(_USE_MATH_DEFINES)
# define _USE_MATH_DEFINES 1
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN 1
#endif
#if !defined(NOMINMAX)
# define NOMINMAX 1
#endif
#if defined(_WIN32)
# define LIBXSTREAM_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
# define LIBXSTREAM_FLOCK(FILE) _lock_file(FILE)
# define LIBXSTREAM_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if defined(__STDC_VERSION__) && (199901L <= (__STDC_VERSION__))
#   define LIBXSTREAM_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
# else
#   define LIBXSTREAM_SNPRINTF(S, N, ...) sprintf(S, __VA_ARGS__); LIBXSTREAM_UNUSED(N)
# endif
# if !defined(__CYGWIN__)
#   define LIBXSTREAM_FLOCK(FILE) flockfile(FILE)
#   define LIBXSTREAM_FUNLOCK(FILE) funlockfile(FILE)
# else /* Only available with __CYGWIN__ *and* C++0x. */
#   define LIBXSTREAM_FLOCK(FILE)
#   define LIBXSTREAM_FUNLOCK(FILE)
# endif
#endif

#if defined(__GNUC__)
# if defined(LIBXSTREAM_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXSTREAM_OFFLOAD_TARGET))
#   include <pthread.h>
#   pragma offload_attribute(pop)
# else
#   include <pthread.h>
# endif
# define LIBXSTREAM_LOCK_TYPE pthread_mutex_t
# define LIBXSTREAM_LOCK_CONSTRUCT PTHREAD_MUTEX_INITIALIZER
# define LIBXSTREAM_LOCK_DESTROY(LOCK) pthread_mutex_destroy(&(LOCK))
# define LIBXSTREAM_LOCK_ACQUIRE(LOCK) pthread_mutex_lock(&(LOCK))
# define LIBXSTREAM_LOCK_RELEASE(LOCK) pthread_mutex_unlock(&(LOCK))
#else /*TODO: Windows*/
# define LIBXSTREAM_LOCK_TYPE HANDLE
# define LIBXSTREAM_LOCK_CONSTRUCT 0
# define LIBXSTREAM_LOCK_DESTROY(LOCK) CloseHandle(LOCK)
# define LIBXSTREAM_LOCK_ACQUIRE(LOCK) WaitForSingleObject(LOCK, INFINITE)
# define LIBXSTREAM_LOCK_RELEASE(LOCK) ReleaseMutex(LOCK)
#endif

/**
 * Below group of preprocessor symbols are used to configure the DEBUG, CHECK, and TRACE properties.
 */
#if defined(LIBXSTREAM_DEBUG) && (2 <= ((2*LIBXSTREAM_DEBUG+1)/2) || (1 == ((2*LIBXSTREAM_DEBUG+1)/2) && !defined(NDEBUG)) || defined(_DEBUG)) && !defined(LIBXSTREAM_INTERNAL_DEBUG)
# define LIBXSTREAM_INTERNAL_DEBUG LIBXSTREAM_DEBUG
#endif
#if defined(LIBXSTREAM_CHECK) && (1 <= ((2*LIBXSTREAM_CHECK+1)/2)) && !defined(LIBXSTREAM_INTERNAL_CHECK)
# define LIBXSTREAM_INTERNAL_CHECK LIBXSTREAM_CHECK
#endif
#if defined(LIBXSTREAM_TRACE) && (1 == ((2*LIBXSTREAM_TRACE+1)/2) || (2 <= ((2*LIBXSTREAM_TRACE+1)/2) && !defined(NDEBUG)) || defined(LIBXSTREAM_INTERNAL_DEBUG)) && !defined(LIBXSTREAM_INTERNAL_TRACE)
# define LIBXSTREAM_INTERNAL_TRACE LIBXSTREAM_TRACE
#endif

#define LIBXSTREAM_IMPORT_DLL __declspec(dllimport)
#if defined(_WINDLL) && defined(_WIN32)
# if defined(LIBXSTREAM_EXPORTED)
#   define LIBXSTREAM_EXPORT __declspec(dllexport)
#else
#   define LIBXSTREAM_EXPORT LIBXSTREAM_IMPORT_DLL
#endif
#else
# define LIBXSTREAM_EXPORT
#endif

LIBXSTREAM_EXPORT_C LIBXSTREAM_RETARGETABLE int libxstream_nonconst(int value);

#if defined(LIBXSTREAM_INTERNAL_DEBUG)
# define LIBXSTREAM_ASSERT(A) assert(A)
# include "libxstream_begin.h"
# include <assert.h>
# include "libxstream_end.h"
#else
# define LIBXSTREAM_ASSERT(A)
#endif

#if (defined(LIBXSTREAM_PREFER_CPP11) || !defined(_OPENMP))
# if (201103L <= __cplusplus)
#   if !defined(LIBXSTREAM_STDFEATURES)
#     define LIBXSTREAM_STDFEATURES
#   endif
#   if !defined(LIBXSTREAM_STDFEATURES_THREADX) && !defined(__MIC__)
#     if defined(_MSC_VER)
#       define LIBXSTREAM_STDFEATURES_THREADX
#     elif (40800 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#       define LIBXSTREAM_STDFEATURES_THREADX
#     endif
#   endif
# elif (1600 < _MSC_VER)
#   if !defined(LIBXSTREAM_STDFEATURES)
#     define LIBXSTREAM_STDFEATURES
#   endif
#   if !defined(LIBXSTREAM_STDFEATURES_THREADX)
#     define LIBXSTREAM_STDFEATURES_THREADX
#   endif
# elif (((40500 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)) && (1L == __cplusplus)) \
    || (defined(__INTEL_COMPILER) && defined(__GXX_EXPERIMENTAL_CXX0X__)))
#   if !defined(LIBXSTREAM_STDFEATURES)
#     define LIBXSTREAM_STDFEATURES
#   endif
# endif
#endif

#if defined(LIBXSTREAM_STDFEATURES) && defined(__CYGWIN__) && defined(__STRICT_ANSI__)
# undef __STRICT_ANSI__
#endif

#define LIBXSTREAM_TRUE  1
#define LIBXSTREAM_FALSE 0

#define LIBXSTREAM_NOT_AWORKITEM     2
#define LIBXSTREAM_NOT_SUPPORTED     1
#define LIBXSTREAM_ERROR_NONE        0
#define LIBXSTREAM_ERROR_RUNTIME    -1
#define LIBXSTREAM_ERROR_CONDITION  -2

#if defined(LIBXSTREAM_INTERNAL_TRACE)
# define LIBXSTREAM_PRINT(VERBOSITY, MESSAGE, ...) libxstream_print(VERBOSITY, "LIBXSTREAM " MESSAGE "\n", __VA_ARGS__)
# define LIBXSTREAM_PRINT0(VERBOSITY, MESSAGE) libxstream_print(VERBOSITY, "LIBXSTREAM " MESSAGE "\n")
#else
# define LIBXSTREAM_PRINT(VERBOSITY, MESSAGE, ...)
# define LIBXSTREAM_PRINT0(VERBOSITY, MESSAGE)
#endif

#if defined(__MIC__)
# define LIBXSTREAM_DEVICE_NAME "LIBXSTREAM_OFFLOAD_TARGET"
#else
# define LIBXSTREAM_DEVICE_NAME "host"
#endif

#if defined(LIBXSTREAM_INTERNAL_CHECK)
# define LIBXSTREAM_CHECK_ERROR(RETURN_VALUE) if (LIBXSTREAM_ERROR_NONE != (RETURN_VALUE)) return RETURN_VALUE
# define LIBXSTREAM_CHECK_CONDITION(CONDITION) if (!(CONDITION)) return LIBXSTREAM_ERROR_CONDITION
# define LIBXSTREAM_CHECK_CONDITION_RETURN(CONDITION) if (!(CONDITION)) return
# define LIBXSTREAM_CHECK_CALL_RETURN(EXPRESSION) if (LIBXSTREAM_ERROR_NONE != (EXPRESSION)) return
# if defined(__cplusplus)
#   define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) if (LIBXSTREAM_ERROR_NONE != (EXPRESSION)) throw std::runtime_error(LIBXSTREAM_TOSTRING(EXPRESSION) " at " __FILE__ ":" LIBXSTREAM_TOSTRING(__LINE__))
#   define LIBXSTREAM_CHECK_CONDITION_THROW(CONDITION) if (!(CONDITION)) throw std::runtime_error(LIBXSTREAM_TOSTRING(CONDITION) " at " __FILE__ ":" LIBXSTREAM_TOSTRING(__LINE__))
# else
#   define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) do { int libxstream_result_ = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != libxstream_result_) abort(libxstream_result_); } while(libxstream_nonconst(LIBXSTREAM_FALSE))
#   define LIBXSTREAM_CHECK_CONDITION_THROW(CONDITION) if (!(CONDITION)) abort(1)
# endif
# if defined(_OPENMP)
#   if defined(LIBXSTREAM_INTERNAL_DEBUG)
#     define LIBXSTREAM_CHECK_CALL(EXPRESSION) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (EXPRESSION))
#   else
#     define LIBXSTREAM_CHECK_CALL(EXPRESSION) (EXPRESSION)
#   endif
# else
#   define LIBXSTREAM_CHECK_CALL(EXPRESSION) do { int libxstream_result_ = (EXPRESSION); if (LIBXSTREAM_ERROR_NONE != libxstream_result_) return libxstream_result_; } while(libxstream_nonconst(LIBXSTREAM_FALSE))
# endif
#elif defined(LIBXSTREAM_INTERNAL_DEBUG)
# define LIBXSTREAM_CHECK_ERROR(RETURN_VALUE) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (RETURN_VALUE))
# define LIBXSTREAM_CHECK_CONDITION(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CONDITION_RETURN(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CONDITION_THROW(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CALL_RETURN(EXPRESSION) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (EXPRESSION))
# define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (EXPRESSION))
# define LIBXSTREAM_CHECK_CALL(EXPRESSION) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (EXPRESSION))
#else
# define LIBXSTREAM_CHECK_ERROR(RETURN_VALUE) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (RETURN_VALUE))
# define LIBXSTREAM_CHECK_CONDITION(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CONDITION_RETURN(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CONDITION_THROW(CONDITION) LIBXSTREAM_ASSERT(CONDITION)
# define LIBXSTREAM_CHECK_CALL_RETURN(EXPRESSION) EXPRESSION
# define LIBXSTREAM_CHECK_CALL_THROW(EXPRESSION) EXPRESSION
# define LIBXSTREAM_CHECK_CALL(EXPRESSION) EXPRESSION
#endif
#if defined(LIBXSTREAM_INTERNAL_DEBUG)
# define LIBXSTREAM_CHECK_CALL_ASSERT(EXPRESSION) LIBXSTREAM_ASSERT(LIBXSTREAM_ERROR_NONE == (EXPRESSION))
#else
# define LIBXSTREAM_CHECK_CALL_ASSERT(EXPRESSION) EXPRESSION
#endif

#if defined(__cplusplus)
# define LIBXSTREAM_INVAL(TYPE) const TYPE&
# define LIBXSTREAM_GETVAL(VALUE) VALUE
# define LIBXSTREAM_SETVAL(VALUE) VALUE
#else
# define LIBXSTREAM_INVAL(TYPE) const TYPE*
# define LIBXSTREAM_GETVAL(VALUE) *VALUE
# define LIBXSTREAM_SETVAL(VALUE) &VALUE
#endif

#endif /*LIBXSTREAM_MACROS_H*/
