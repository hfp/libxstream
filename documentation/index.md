# LIBXSTREAM

LIBXSTREAM is an OpenCL-based accelerator backend library providing
streams, events, and device memory management for GPU offloading. It
implements the [DBCSR ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/),
making it a drop-in OpenCL backend for
[CP2K](https://cp2k.org/)/[DBCSR](https://dbcsr.cp2k.org/).

Targets OpenCL devices across vendors (Intel, AMD, NVIDIA). Depends
on [LIBXS](https://github.com/hfp/libxs) for utility infrastructure.

## Build

LIBXS must be a sibling directory (controlled by `LIBXSROOT`).
An OpenCL SDK must be available.

**Library** -- build both LIBXS and LIBXSTREAM as separate
libraries:

```bash
git clone https://github.com/hfp/libxs.git
git clone https://github.com/hfp/libxstream.git

cd libxs && make GNU=1 -j $(nproc)
cd ../libxstream && make GNU=1 -j $(nproc)
```

This produces `lib/libxstream.a` and `lib/libxstream.so`.

**Header-only** (explicit) -- include `libxstream_source.h` (no
separate library needed for either LIBXSTREAM or LIBXS). Safe
to include from multiple translation units:

```c
#include <libxstream_source.h>
```

**Header-only** (implicit) -- compile with `-DLIBXSTREAM_SOURCE`.
Any LIBXSTREAM or LIBXS public header then automatically includes
the implementation. No special include order is required.
`-DLIBXSTREAM_SOURCE` implies `-DLIBXS_SOURCE`; LIBXS can also
be made header-only independently with `-DLIBXS_SOURCE`.

The library is compiled for SSE4.2 by default but dynamically
dispatches to the best ISA available at runtime (up to AVX-512).
Use `SSE=0` to compile natively for the build host.

| Variable   | Default   | Description                                     |
|------------|-----------|-------------------------------------------------|
| GNU        | 0         | Use GNU GCC-compatible compiler                 |
| DBG        | 0         | Debug build                                     |
| SYM        | 0         | Include debug symbols (-g)                      |
| SSE        | 1         | x86 baseline: 0=native, 1=SSE4.2 (portable)     |

pkg-config support: `lib/libxstream.pc`.

## Installation

Install into a chosen prefix (LIBXS must be built first):

```bash
make GNU=1 -j $(nproc) install PREFIX=$HOME/libxstream
```

This installs headers, the static and shared libraries, and the
header-only source tree under `PREFIX`.

Out-of-tree builds are also supported:

```bash
mkdir /tmp/libxstream-build && cd /tmp/libxstream-build
make -j $(nproc) -f /path/to/libxstream/Makefile
```

## API

The public C API is declared in `include/libxstream.h`. All
implementation details are sealed behind opaque types.

### Devices

```c
int libxstream_init(void);
int libxstream_finalize(void);
int libxstream_device_count(int* ndevices);
int libxstream_device_set_active(int device_id);
int libxstream_device_sync(void);
```

### Streams

```c
int libxstream_stream_create(libxstream_stream_t** stream_p,
                             const char* name, int priority);
int libxstream_stream_destroy(libxstream_stream_t* stream);
int libxstream_stream_sync(libxstream_stream_t* stream);
int libxstream_stream_wait_event(libxstream_stream_t* stream,
                                 libxstream_event_t* event);
```

### Events

```c
int libxstream_event_create(libxstream_event_t** event_p);
int libxstream_event_destroy(libxstream_event_t* event);
int libxstream_event_record(libxstream_event_t* event,
                            libxstream_stream_t* stream);
int libxstream_event_query(libxstream_event_t* event,
                           libxstream_bool_t* has_occurred);
int libxstream_event_sync(libxstream_event_t* event);
```

### Memory

Device and host memory allocation, transfers (H2D, D2H, D2D), and
initialization:

```c
int libxstream_mem_allocate(void** dev_mem, size_t nbytes);
int libxstream_mem_deallocate(void* dev_mem);
int libxstream_mem_host_allocate(void** host_mem, size_t nbytes,
                                 libxstream_stream_t* stream);
int libxstream_mem_host_deallocate(void* host_mem,
                                   libxstream_stream_t* stream);
int libxstream_mem_copy_h2d(const void* host_mem, void* dev_mem,
                            size_t nbytes, libxstream_stream_t* stream);
int libxstream_mem_copy_d2h(const void* dev_mem, void* host_mem,
                            size_t nbytes, libxstream_stream_t* stream);
int libxstream_mem_copy_d2d(const void* src, void* dst,
                            size_t nbytes, libxstream_stream_t* stream);
int libxstream_mem_zero(void* dev_mem, size_t offset, size_t nbytes,
                        libxstream_stream_t* stream);
```

### DBCSR Compatibility

The header `include/libxstream_dbcsr.h` provides the `c_dbcsr_acc_*`
symbols expected by DBCSR, allowing LIBXSTREAM to serve as a drop-in
accelerator backend.

### CP2K Offload Interface

The header `include/libxstream_cp2k.h` implements CP2K's offload
runtime interface -- the `offload*` functions for general GPU
operations (memory, streams, events, synchronization).

## Samples

Each sample has its own Makefile under `samples/`:

```bash
cd samples/smm && make GNU=1
cd samples/ozaki && make GNU=1
```

### SMM -- Small Matrix Multiplication

Implements the ACC LIBSMM interface for batched small matrix multiply
and transpose on OpenCL devices. Includes an auto-tuning framework
and pre-tuned parameter sets for A100, BMG, GH200, H100, Mi250, P100,
PVC, and V100. See [samples/smm/README.md](samples/smm/README.md).

### Ozaki -- High-Precision GEMM

Ozaki scheme for high-precision GEMM emulation, fully offloaded to
OpenCL. Two schemes (mantissa slicing and CRT) with automatic
detection of Intel XMX matrix engines. See
[samples/ozaki/README.md](samples/ozaki/README.md). The CPU-side
GEMM wrapper is part of
[LIBXS Ozaki](https://github.com/hfp/libxs/tree/main/samples/ozaki).

## License

[BSD 3-Clause](LICENSE.md)
