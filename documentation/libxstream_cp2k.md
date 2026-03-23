# CP2K Offload Interface

[`include/libxstream_cp2k.h`](https://github.com/hfp/libxstream/blob/main/include/libxstream_cp2k.h) implements CP2K's [offload runtime interface](https://github.com/cp2k/cp2k/blob/master/src/offload/offload_runtime.h) — the hardware-abstraction layer that CP2K uses for GPU-accelerated operations beyond DBCSR (grid integration, PW operations, etc.). The original interface provides `static inline` wrappers for CUDA, HIP, and OpenCL; LIBXSTREAM replaces the OpenCL path with a dedicated translation unit ([`src/libxstream_cp2k.c`](https://github.com/hfp/libxstream/blob/main/src/libxstream_cp2k.c)) that routes directly through LIBXSTREAM's internal API.

## Relationship to the DBCSR Interface

CP2K's code has two accelerator interfaces:

| Interface | Header | Purpose |
|---|---|---|
| **DBCSR ACC** | [`libxstream_dbcsr.h`](https://github.com/hfp/libxstream/blob/main/include/libxstream_dbcsr.h) | Sparse matrix operations (DBCSR library) |
| **Offload Runtime** | [`libxstream_cp2k.h`](https://github.com/hfp/libxstream/blob/main/include/libxstream_cp2k.h) | General offload (memory, streams, events, synchronization) |

The DBCSR adapter ([`src/libxstream_dbcsr.c`](https://github.com/hfp/libxstream/blob/main/src/libxstream_dbcsr.c)) uses opaque `void*` handles and translates to LIBXSTREAM's typed API. The offload runtime adapter does the same but uses CP2K's `offloadStream_t`/`offloadEvent_t` typedefs (also `void*`). Both share the underlying LIBXSTREAM implementation.

## API

The header is self-contained (C99, no LIBXSTREAM headers required) and provides opaque handle types, an error-checking macro, and functions covering five domains.

### Types and Constants

```C
typedef void* offloadStream_t;
typedef void* offloadEvent_t;
typedef int   offloadError_t;

#define offloadSuccess EXIT_SUCCESS
```

### Error Handling

```C
const char* offloadGetErrorName(offloadError_t error);
offloadError_t offloadGetLastError(void);
```

`offloadGetErrorName` maps error codes to OpenCL error strings via `libxstream_opencl_strerror`. `offloadGetLastError` consumes and clears the last recorded error.

The `OFFLOAD_CHECK` macro aborts on failure after printing the error name and source location:

```C
OFFLOAD_CHECK(offloadMalloc(&ptr, nbytes));
```

### Streams

```C
void offloadStreamCreate(offloadStream_t* stream);
void offloadStreamDestroy(offloadStream_t stream);
void offloadStreamSynchronize(offloadStream_t stream);
void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event);
```

`offloadStreamCreate` creates a stream with default priority (`LIBXSTREAM_STREAM_DEFAULT`).

### Events

```C
void offloadEventCreate(offloadEvent_t* event);
void offloadEventDestroy(offloadEvent_t event);
void offloadEventRecord(offloadEvent_t event, offloadStream_t stream);
void offloadEventSynchronize(offloadEvent_t event);
bool offloadEventQuery(offloadEvent_t event);
```

### Memory

```C
void offloadMalloc(void** ptr, size_t size);
void offloadFree(void* ptr);
void offloadMallocHost(void** ptr, size_t size);
void offloadFreeHost(void* ptr);
```

### Transfers

```C
void offloadMemcpyAsyncHtoD(void* ptr_dev, const void* ptr_hst, size_t size, offloadStream_t stream);
void offloadMemcpyAsyncDtoH(void* ptr_hst, const void* ptr_dev, size_t size, offloadStream_t stream);
void offloadMemcpyAsyncDtoD(void* dst, const void* src, size_t size, offloadStream_t stream);
void offloadMemcpyHtoD(void* ptr_dev, const void* ptr_hst, size_t size);
void offloadMemcpyDtoH(void* ptr_hst, const void* ptr_dev, size_t size);
void offloadMemsetAsync(void* ptr, int val, size_t size, offloadStream_t stream);
void offloadMemset(void* ptr, int val, size_t size);
```

The synchronous variants (`offloadMemcpyHtoD`, `offloadMemcpyDtoH`, `offloadMemset`) pass a NULL stream. `offloadMemsetAsync` supports arbitrary fill values via `libxstream_opencl_memset`.

### Device

```C
void offloadDeviceSynchronize(void);
```

### Stubs

```C
void offloadMemcpyToSymbol(const void* symbol, const void* src, size_t count);
void offloadEnsureMallocHeapSize(size_t required_size);
```

These are CUDA-specific operations (constant-memory writes and device-heap sizing) that have no direct OpenCL equivalent. They are currently stubs guarded by assertions. CP2K's OpenCL path disables the GPU grid subsystem (`__NO_OFFLOAD_GRID`) that would call them.

## See Also

* LIBXSTREAM API (`include/libxstream.h`) — the underlying OpenCL backend API
* [DBCSR ACC Interface](libxstream_dbcsr.md) — the DBCSR adapter layer
* [CP2K offload_runtime.h](https://github.com/cp2k/cp2k/blob/master/src/offload/offload_runtime.h) — upstream interface definition
