/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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
#ifndef LIBXSTREAM_H
#define LIBXSTREAM_H

#include "libxstream_macros.h"
#include <stdint.h>
#include <stddef.h>


/** Boolean state. */
typedef int libxstream_bool;
/** Stream type. */
LIBXSTREAM_EXPORT_C typedef struct libxstream_stream libxstream_stream;
/** Event type. */
LIBXSTREAM_EXPORT_C typedef struct libxstream_event libxstream_event;
/** Enumeration of elemental "scalar" types. */
LIBXSTREAM_EXPORT_C typedef enum libxstream_type {
  LIBXSTREAM_TYPE_VOID,
  LIBXSTREAM_TYPE_CHAR,
  LIBXSTREAM_TYPE_I8, LIBXSTREAM_TYPE_U8,
  LIBXSTREAM_TYPE_I16, LIBXSTREAM_TYPE_U16,
  LIBXSTREAM_TYPE_I32, LIBXSTREAM_TYPE_U32,
  LIBXSTREAM_TYPE_I64, LIBXSTREAM_TYPE_U64,
  LIBXSTREAM_TYPE_F32, LIBXSTREAM_TYPE_F64,
  LIBXSTREAM_TYPE_C32, LIBXSTREAM_TYPE_C64,
  LIBXSTREAM_TYPE_BYTE = LIBXSTREAM_TYPE_U8,
  /** Must match libxstream_bool. */
  LIBXSTREAM_TYPE_BOOL = LIBXSTREAM_TYPE_I32
} libxstream_type;
/** Flags to adjust function call behavior (valid for binary combination). */
LIBXSTREAM_EXPORT_C typedef enum libxstream_call_flags {
  LIBXSTREAM_CALL_DEFAULT = 0,
  LIBXSTREAM_CALL_WAIT    = 1,
  LIBXSTREAM_CALL_SYNCHRONOUS = LIBXSTREAM_CALL_WAIT
} libxstream_call_flags;
/** Function argument type. */
LIBXSTREAM_EXPORT_C typedef struct LIBXSTREAM_TARGET(mic) libxstream_argument libxstream_argument;
/** Function type of an offloadable function. */
typedef LIBXSTREAM_TARGET(mic) void (/*LIBXSTREAM_CDECL*/*libxstream_function)(LIBXSTREAM_VARIADIC);

/** Query the number of available devices. */
LIBXSTREAM_EXPORT_C int libxstream_get_ndevices(size_t* ndevices);
/** Query the device set active for this thread. */
LIBXSTREAM_EXPORT_C int libxstream_get_active_device(int* device);
/** Set the active device for this thread. */
LIBXSTREAM_EXPORT_C int libxstream_set_active_device(int device);

/** Query the memory metrics of the device (valid to pass one NULL pointer). */
LIBXSTREAM_EXPORT_C int libxstream_mem_info(int device, size_t* allocatable, size_t* physical);
/** Allocate aligned memory (0: automatic) on the device. */
LIBXSTREAM_EXPORT_C int libxstream_mem_allocate(int device, void** memory, size_t size, size_t alignment);
/** Deallocate memory; shall match the device where the memory was allocated. */
LIBXSTREAM_EXPORT_C int libxstream_mem_deallocate(int device, const void* memory);
/** Fill memory with zeros; allocated memory can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memset_zero(void* memory, size_t size, libxstream_stream* stream);
/** Copy memory from the host to the device; addresses can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t size, libxstream_stream* stream);
/** Copy memory from the device to the host; addresses can carry an offset. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t size, libxstream_stream* stream);
/** Copy memory from device to device; cross-device copies are allowed as well. */
LIBXSTREAM_EXPORT_C int libxstream_memcpy_d2d(const void* src, void* dst, size_t size, libxstream_stream* stream);

/** Query the range of valid priorities (inclusive bounds). */
LIBXSTREAM_EXPORT_C int libxstream_stream_priority_range(int* least, int* greatest);
/** Create a stream on a device (demux<0: auto-locks, 0: manual, demux>0: sync.). */
LIBXSTREAM_EXPORT_C int libxstream_stream_create(libxstream_stream** stream, int device, int demux, int priority, const char* name);
/** Destroy a stream; pending work must be completed if results are needed. */
LIBXSTREAM_EXPORT_C int libxstream_stream_destroy(libxstream_stream* stream);
/** Wait for a stream to complete pending work; NULL to synchronize all streams. */
LIBXSTREAM_EXPORT_C int libxstream_stream_sync(libxstream_stream* stream);
/** Wait for an event recorded earlier; NULL increases the match accordingly. */
LIBXSTREAM_EXPORT_C int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event);
/** Lock a stream such that the caller thread can safely enqueue work. */
LIBXSTREAM_EXPORT_C int libxstream_stream_lock(libxstream_stream* stream);
/** Unlock a stream such that another thread can acquire the stream. */
LIBXSTREAM_EXPORT_C int libxstream_stream_unlock(libxstream_stream* stream);
/** Query the device the given stream is constructed for. */
LIBXSTREAM_EXPORT_C int libxstream_stream_device(const libxstream_stream* stream, int* device);
/** Query the device the given stream is constructed for. */
LIBXSTREAM_EXPORT_C int libxstream_stream_demux(const libxstream_stream* stream, int* demux);

/** Create an event; can be used multiple times to record an event. */
LIBXSTREAM_EXPORT_C int libxstream_event_create(libxstream_event** event);
/** Destroy an event; does not implicitly waits for the completion of the event. */
LIBXSTREAM_EXPORT_C int libxstream_event_destroy(libxstream_event* event);
/** Record an event; an event can be re-recorded multiple times. */
LIBXSTREAM_EXPORT_C int libxstream_event_record(libxstream_event* event, libxstream_stream* stream);
/** Check whether an event has occurred or not (non-blocking). */
LIBXSTREAM_EXPORT_C int libxstream_event_query(const libxstream_event* event, libxstream_bool* occured);
/** Wait for an event to complete i.e., work queued prior to recording the event. */
LIBXSTREAM_EXPORT_C int libxstream_event_synchronize(libxstream_event* event);

/** Create a function signature with a certain arity (number of arguments). */
LIBXSTREAM_EXPORT_C int libxstream_fn_create_signature(libxstream_argument** signature, size_t nargs);
/** Destroy a function signature; does not release the bound data. */
LIBXSTREAM_EXPORT_C int libxstream_fn_destroy_signature(const libxstream_argument* signature);
/** Construct an input argument; takes the device data, dimensionality, and shape. */
LIBXSTREAM_EXPORT_C int libxstream_fn_input(libxstream_argument* signature, size_t arg, const void* in, libxstream_type type, size_t dims, const size_t shape[]);
/** Construct an output argument; takes the device data, dimensionality, and shape. */
LIBXSTREAM_EXPORT_C int libxstream_fn_output(libxstream_argument* signature, size_t arg, void* out, libxstream_type type, size_t dims, const size_t shape[]);
/** Construct an in-out argument; takes the device data, dimensionality, and shape. */
LIBXSTREAM_EXPORT_C int libxstream_fn_inout(libxstream_argument* signature, size_t arg, void* inout, libxstream_type type, size_t dims, const size_t shape[]);
/** Query the maximum number of arguments that can be covered by the function signature. */
LIBXSTREAM_EXPORT_C int libxstream_fn_nargs(const libxstream_argument* signature, size_t* nargs);
/** Query the arity of the function signature (actual number of arguments). */
LIBXSTREAM_EXPORT_C int libxstream_fn_arity(const libxstream_argument* signature, size_t* arity);
/** Call a user function along with the signature; wait in case of a synchronous call. */
LIBXSTREAM_EXPORT_C int libxstream_fn_call(libxstream_function function, const libxstream_argument* signature, libxstream_stream* stream, int flags);

/** Query the size of the elemental type (Byte). */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_typesize(libxstream_type type, size_t* size);
/** Query the name of the elemental type (string does not need to be buffered). */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_typename(libxstream_type type, const char** name);
/** Query the argument corresponding to the given pointer variable; this does not work with by-value variables. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_argument(const void* variable, const libxstream_argument** arg);
/** Query a textual value of the argument (valid until next call); thread safe. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_value(const libxstream_argument* arg, const char** value);
/** Query the elemental type of the argument. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_type(const libxstream_argument* arg, libxstream_type* type);
/** Query the dimensionality of the argument; an elemental argument is 0-dimensional. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_dims(const libxstream_argument* arg, size_t* dims);
/** Query the extent of the argument; an elemental argument has an 0-extent. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_shape(const libxstream_argument* arg, size_t shape[]);
/** Query the number of elements of the argument. */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_size(const libxstream_argument* arg, size_t* size);
/** Query the data size of the argument (Byte). */
LIBXSTREAM_EXPORT_C LIBXSTREAM_TARGET(mic) int libxstream_get_datasize(const libxstream_argument* arg, size_t* size);

#if defined(__cplusplus)
template<typename TYPE> struct libxstream_type2value  { /*static const libxstream_type value = LIBXSTREAM_TYPE_VOID;*/ };
template<> struct libxstream_type2value<bool>         { /*static const libxstream_type value = LIBXSTREAM_TYPE_BOOL;*/ };
template<> struct libxstream_type2value<char>         { static const libxstream_type value = LIBXSTREAM_TYPE_CHAR; };
template<> struct libxstream_type2value<int8_t>       { static const libxstream_type value = LIBXSTREAM_TYPE_I8; };
template<> struct libxstream_type2value<uint8_t>      { static const libxstream_type value = LIBXSTREAM_TYPE_U8; };
template<> struct libxstream_type2value<int16_t>      { static const libxstream_type value = LIBXSTREAM_TYPE_I16; };
template<> struct libxstream_type2value<uint16_t>     { static const libxstream_type value = LIBXSTREAM_TYPE_U16; };
template<> struct libxstream_type2value<int32_t>      { static const libxstream_type value = LIBXSTREAM_TYPE_I32; };
template<> struct libxstream_type2value<uint32_t>     { static const libxstream_type value = LIBXSTREAM_TYPE_U32; };
template<> struct libxstream_type2value<int64_t>      { static const libxstream_type value = LIBXSTREAM_TYPE_I64; };
template<> struct libxstream_type2value<uint64_t>     { static const libxstream_type value = LIBXSTREAM_TYPE_U64; };
template<> struct libxstream_type2value<float>        { static const libxstream_type value = LIBXSTREAM_TYPE_F32; };
template<> struct libxstream_type2value<double>       { static const libxstream_type value = LIBXSTREAM_TYPE_F64; };

template<libxstream_type VALUE> struct libxstream_value2type  { typedef void type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_CHAR> { typedef char type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_I8>   { typedef signed char type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_U8>   { typedef unsigned char type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_I16>  { typedef signed short type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_U16>  { typedef unsigned short type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_I32>  { typedef int type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_U32>  { typedef unsigned int type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_I64>  { typedef long long type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_U64>  { typedef unsigned long long type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_F32>  { typedef float type; };
template<> struct libxstream_value2type<LIBXSTREAM_TYPE_F64>  { typedef double type; };
#endif // __cplusplus

#endif // LIBXSTREAM_H
