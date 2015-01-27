LIBXSTREAM
==========
Library to work with streams, events, and code regions that are able to run asynchronous while preserving the usual stream conditions. The library is targeting Intel Architecture (x86) and helps to offload work to an Intel Xeon Phi coprocessor (an instance of the Intel Many Integrated Core Architecture "MIC"). For example, using two streams may be an alternative to the usual double-buffering approach which can be used to hide buffer transfer time behind compute.

Interface
=========
The library's [C API](include/libxstream.h) completely seals the implementation and only forward declares some types. Beside of some minor syntactical sugar, the C++ API allows to make use of the [stream](include/libxstream_stream.hpp) and [event](inlcude/libxstream_event.hpp) types directly. The C++ API is currently required for own code to be enqueued into a stream. However, a future release will allow to only rely on a function pointer and a plain C interface. A future release may also provide a native FORTRAN interface.

**Data Types**

```C
/** Data type representing a signal. */
typedef uintptr_t libxstream_signal;
/** Forward declaration of the stream type (C++ API includes the definition). */
typedef struct libxstream_stream libxstream_stream;
/** Forward declaration of the event type (C++ API includes the definition). */
typedef struct libxstream_event libxstream_event;
```

**Device Interface**

```C
/** Query the number of available devices. */
int libxstream_get_ndevices(size_t* ndevices);
/** Query the device set active for this thread. */
int libxstream_get_active_device(int* device);
/** Sets the active device for this thread. */
int libxstream_set_active_device(int device);
```

**Memory Interface**

```C
/** Query the memory metrics of the given device (it is valid to pass one NULL pointer). */
int libxstream_mem_info(int device, size_t* allocatable, size_t* physical);
/** Allocates memory with alignment (0: automatic) on the given device. */
int libxstream_mem_allocate(int device, void** memory, size_t size, size_t alignment);
/** Deallocates the memory; shall match the device where the memory was allocated. */
int libxstream_mem_deallocate(int device, const void* memory);
/** Fills the memory with zeros; allocated memory can carry an offset and a smaller size. */
int libxstream_memset_zero(void* memory, size_t size, libxstream_stream* stream);
/** Copies memory from the host to the device; addresses can carry an offset. */
int libxstream_memcpy_h2d(const void* mem, void* dev_mem, size_t size, libxstream_stream*);
/** Copies memory from the device to the host; addresses can carry an offset. */
int libxstream_memcpy_d2h(const void* dev_mem, void* mem, size_t size, libxstream_stream*);
/** Copies memory from device to device; cross-device copies are allowed as well. */
int libxstream_memcpy_d2d(const void* src, void* dst, size_t size, libxstream_stream*);
```

**Stream Interface**

```C
/** Query the range of valid priorities (inclusive). */
int libxstream_stream_priority_range(int* least, int* greatest);
/** Create a stream on a given device; priority shall be within the queried bounds. */
int libxstream_stream_create(libxstream_stream**, int device, int priority, const char* name);
/** Destroy a stream; pending work must be completed explicitly if results are needed. */
int libxstream_stream_destroy(libxstream_stream* stream);
/** Wait for a stream to complete pending work; NULL to synchronize all streams. */
int libxstream_stream_sync(libxstream_stream* stream);
/** Wait for an event recorded earlier. Passing NULL increases the match accordingly. */
int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event);
```

**Event Interface**

```C
/** Create an event; can be re-used multiple times by re-recording the event. */
int libxstream_event_create(libxstream_event** event);
/** Destroy an event; does not implicitly waits for the completion of the event. */
int libxstream_event_destroy(libxstream_event* event);
/** Record an event; an event can be re-recorded multiple times. */
int libxstream_event_record(libxstream_event* event, libxstream_stream* stream);
/** Check whether an event has occured or not (non-blocking). */
int libxstream_event_query(const libxstream_event* event, int* has_occured);
/** Wait for an event to complete i.e., any work enqueued prior to recording the event. */
int libxstream_event_synchronize(libxstream_event* event);
```

Implementation
==============
The library's implementation allows enqueuing work from multiple host threads in a thread-safe manner and without oversubscribing the device. The actual implementation vehicle can be configured using a [configuration header](include/libxstream_config.h). Currently Intel's Language Extensions for Offload (LEO) are used to perform asynchronous execution and data transfers using signal/wait clauses. Other mechanism could used e.g., hStreams or COI (both are part of the Intel Manycore Platform Software Stack), or the OpenMP 4.0 offload directives.
