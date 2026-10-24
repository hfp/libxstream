# LIBXSTREAM

LIBXSTREAM is an OpenCL-based accelerator backend library providing
streams, events, and device memory management for GPU offloading. It
implements the [DBCSR ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/),
making it a drop-in OpenCL backend for
[CP2K](https://cp2k.org/)/[DBCSR](https://dbcsr.cp2k.org/).

Targets OpenCL devices across vendors (Intel, AMD, NVIDIA). Depends
on [LIBXS](https://github.com/hfp/libxs) for utility infrastructure.

## Build

LIBXS is taken from `LIBXSROOT` if given, otherwise from a sibling
directory (`../libxs`), otherwise from where pkg-config finds it
(`libxs.pc`, with `share/libxs/Makefile.inc`). An OpenCL SDK must be
available.

**Library** — build both LIBXS and LIBXSTREAM as separate
libraries:

```bash
git clone https://github.com/hfp/libxs.git
git clone https://github.com/hfp/libxstream.git

cd libxs && make -j $(nproc)
cd ../libxstream && make -j $(nproc)
```

This produces `lib/libxstream.a` and `lib/libxstream.so`, and with the
shared library the [OpenCL-MPS](libxstream_opencl.md#opencl-mps)
server `lib/libxstream_mpsd` and its ICD `lib/libxstream_icd.so`.

**Header-only** (explicit) — include `libxstream_source.h` (no
separate library needed for either LIBXSTREAM or LIBXS). Safe
to include from multiple translation units:

```c
#include <libxstream/libxstream_source.h>
```

**Header-only** (implicit) — compile with `-DLIBXSTREAM_SOURCE`.
Any LIBXSTREAM or LIBXS public header then automatically includes
the implementation. No special include order is required.
`-DLIBXSTREAM_SOURCE` implies `-DLIBXS_SOURCE`; LIBXS can also
be made header-only independently with `-DLIBXS_SOURCE`.

Either header-only variant needs `-D__OPENCL` (implied by
`-D__OFFLOAD_OPENCL`), otherwise the implementation compiles to
nothing, and the OpenCL library at link time.

The library is compiled for SSE4.2 by default but dynamically
dispatches to the best ISA available at runtime (up to AVX-512).
Use `SSE=0` to compile natively for the build host.
The library is built into `lib/<target>`, e.g., `lib/x86_64` by default or
`lib/sapphirerapids` with `SSE=0`, and `lib/` refers to the latest build.
This lets a shared file system serve different systems.
LIBXS is used from its matching `lib/<target>` if available.

| Variable   | Default   | Description                                     |
|------------|-----------|-------------------------------------------------|
| GNU        | 1         | GNU GCC-compatible compiler (0: Intel)          |
| DBG        | 0         | Debug build                                     |
| SYM        | 0         | Include debug symbols (-g)                      |
| SSE        | 1         | x86 baseline: 0=native, 1=SSE4.2 (portable)     |
| LIBARCH    | (auto)    | Name of the build folder (`lib/<target>`)       |
| STATIC     | (unset)   | Unset: static and shared, 0: shared, 1: static  |
| OMP        | 1         | OpenMP (sizes per-thread resources)             |
| OCL        | 2         | 1: `-D__OPENCL`, 2: also `-D__OFFLOAD_OPENCL`   |
| USM        | 0         | 1: `-D__OFFLOAD_UNIFIED_MEMORY` (SVM only)      |

pkg-config support: `lib/pkgconfig/libxstream.pc` (`libxstream-static.pc` and
`libxstream-shared.pc` select the kind). CMake: `lib/cmake/libxstream`.

Tests and samples:

```bash
make tests     # build and run tests/
make samples   # build every sample
```

## Installation

Install into a chosen prefix (LIBXS must be built first):

```bash
make -j $(nproc) install PREFIX=$HOME/libxstream
```

This installs headers, the libraries, the OpenCL-MPS server and ICD,
the pkg-config and CMake files, the sample support files
(`share/libxstream/samples`), and the documentation (`share/doc/libxstream`)
under `PREFIX`. The header-only source tree is optional; add `HEADER_ONLY=1`
to install it. `CLEAN=1` removes an existing `PREFIX` first.

| Target            | Installs in addition                                       |
|-------------------|------------------------------------------------------------|
| `install-all`     | The tests (`share/libxstream/tests`)                       |
| `install-dev`     | `tool_checkenvars.sh` (`PREFIX/scripts`)                   |
| `install-realall` | Both of the above, and the sample executables              |

Out-of-tree builds are also supported:

```bash
mkdir /tmp/libxstream-build && cd /tmp/libxstream-build
make -j $(nproc) -f /path/to/libxstream/Makefile
```

## CMake

```bash
cmake -S libxstream -B build -DCMAKE_INSTALL_PREFIX=$HOME/libxstream
cmake --build build -j $(nproc) && cmake --install build
```

LIBXS is found with `find_package(libxs)` (also in `../libxs`), otherwise
fetched from its repository (FetchContent), with `LIBXS_SAMPLES_OMP` set to
`LIBXSTREAM_OMP`. OpenCL is required. OpenMP is required when `LIBXSTREAM_OMP`
is ON. On macOS, Homebrew's `libomp` is detected via `brew --prefix libomp`;
use `OpenMP_ROOT` for another installation.

| Option                           | Default | Description                                                          |
|----------------------------------|---------|----------------------------------------------------------------------|
| `LIBXSTREAM_OMP`                 | ON      | OpenMP for per-thread resources                                      |
| `LIBXSTREAM_INSTALL_HEADER_ONLY` | OFF     | Install the header-only sources                                      |
| `BUILD_TESTING`                  | OFF     | Build the tests                                                      |
| `LIBXSTREAM_BUILD_SAMPLES`       | `BUILD_TESTING` | Build the sample executables; requires BLAS                     |
| `BUILD_SHARED_LIBS`              | OFF     | Shared library, and with it the OpenCL-MPS server and ICD            |

A consumer uses `find_package(libxstream)` and links `libxstream::libxstream`,
which also works for a tree built and installed with `make`.

Enable `LIBXSTREAM_BUILD_SAMPLES` to build the SMM, DBM, Ozaki, and stencil
benchmarks under `build/samples/`. With `BUILD_TESTING`, the device-selection
test also runs using the CMake-built `acc_match.x`; it needs no GPU.

## API

The public C API is declared in `libxstream/libxstream.h`. All
implementation details are sealed behind opaque types.

### Initialization

```c
int libxstream_init(void);
int libxstream_init_config(const libxstream_init_config_t* cfg);
void libxstream_init_config_default(libxstream_init_config_t* cfg);
int libxstream_finalize(void);
```

`libxstream_init()` initializes with environment/defaults.
`libxstream_init_config()` allows explicit control over USM level,
device selection, verbosity, and sub-buffers before the runtime starts:

```c
libxstream_init_config_t cfg;
libxstream_init_config_default(&cfg);  /* -1 = use env/default */
cfg.usm = 1;      /* 0:off, 1:Intel USM, 2:SVM coarse, 3:SVM caps */
cfg.device = 0;   /* device index (-1: env/first) */
cfg.verbosity = 2;
cfg.subbuffer = 0; /* sub-buffers for offsets (no effect with USM) */
libxstream_init_config(&cfg);
```

Passing NULL to `libxstream_init_config` is equivalent to calling
`libxstream_init`.

### Devices

```c
int libxstream_device_count(int* ndevices);
int libxstream_device_set_active(int device_id);
int libxstream_device_sync(void);
```

### Streams

```c
int libxstream_stream_create(libxstream_stream_t** stream_p,
                             const char* name, int flags);
int libxstream_stream_destroy(libxstream_stream_t* stream);
int libxstream_stream_sync(libxstream_stream_t* stream);
int libxstream_stream_wait_event(libxstream_stream_t* stream,
                                 libxstream_event_t* event);
int libxstream_stream_set_profiling(libxstream_stream_t* stream);
```

The `flags` are `LIBXSTREAM_STREAM_DEFAULT`, `LIBXSTREAM_STREAM_LOW`,
`LIBXSTREAM_STREAM_HIGH`, or `LIBXSTREAM_STREAM_PROFILING`
(`libxstream_stream_flags_t`).

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
int libxstream_mem_offset(void** dev_mem, void* other, size_t offset);
int libxstream_mem_info(size_t* mem_free, size_t* mem_total);
int libxstream_mem_host_device_accessible(void);
int libxstream_mem_host_pin(void* host_mem, size_t nbytes);
int libxstream_mem_host_unpin(void* host_mem);
```

`libxstream_stream_set_profiling` enables event timestamps on a stream.
`libxstream_mem_offset` yields device memory `offset` bytes past `other`.
`libxstream_mem_info` reports free and total memory of the active device.
`libxstream_mem_host_device_accessible` tells whether a kernel may read
memory from `libxstream_mem_host_allocate` directly (requires an active
device). `libxstream_mem_host_pin` declares a host range the application
allocated for faster transfers, see
[CUDA Host Registration](libxstream_opencl.md#cuda-host-registration).

The `_hint` variants take `LIBXSTREAM_MEM_POOLED` or `LIBXSTREAM_MEM_UNPOOLED`
(`libxstream_mem_pool_t`), optionally combined with a
`libxstream_opencl_mem_hint_t`. Unpooled memory suits a caller that pools on
its own; releasing it synchronizes the device first. Deallocation takes the
hint given at allocation:

```c
int libxstream_mem_dev_allocate_hint(void** dev_mem, size_t nbytes, int hint);
int libxstream_mem_dev_deallocate_hint(void* dev_mem, int hint);
int libxstream_mem_host_allocate_hint(void** host_mem, size_t nbytes, int hint);
int libxstream_mem_host_deallocate_hint(void* host_mem, int hint);
```

### DBCSR Compatibility

The header `libxstream/libxstream_dbcsr.h` provides the `c_dbcsr_acc_*`
symbols expected by DBCSR, allowing LIBXSTREAM to serve as a drop-in
accelerator backend.

### CP2K Offload Interface

The header `libxstream/libxstream_cp2k.h` implements CP2K's offload
runtime interface — the `offload*` functions for general GPU
operations (memory, streams, events, synchronization).

## Samples

Each sample has its own Makefile under `samples/`:

```bash
make -C samples/smm
make -C samples/ozaki
```

### SMM — Small Matrix Multiplication

Implements the ACC LIBSMM interface for batched small matrix multiply
and transpose on OpenCL devices. Includes an auto-tuning framework
and pre-tuned parameter sets for A100, BMG, GH200, H100, Mi250, P100,
PVC, and V100. See [SMM][smm].

### Ozaki — High-Precision GEMM

Ozaki scheme for high-precision GEMM emulation, fully offloaded to
OpenCL. Two schemes (mantissa slicing and CRT) with automatic
detection of matrix engines (Intel XMX, NVIDIA MMA). See [Ozaki][ozaki].
The CPU-side GEMM wrapper is part of
[LIBXS Ozaki](https://github.com/hfp/libxs/tree/main/samples/ozaki).

### Stencil — Finite Difference

Seismic wave propagation (RTM/FWI) via dimension-split stencils.
Kernel paths: FP32 (default), BF16-DPAS (`STENCIL_BF16=1`), and
INT8-DPAS (`STENCIL_INT8=1`). Supports isotropic and TTI operators.
See [Stencil][stencil].

### DBM — Distributed Block-sparse Matrices

OpenCL backend of CP2K's DBM, which processes batches of small
matrix multiplications. See [DBM][dbm].

[smm]: samples/libxstream_smm.md
[ozaki]: samples/libxstream_ozaki.md
[stencil]: samples/libxstream_stencil.md
[dbm]: samples/libxstream_dbm.md

## License

[BSD 3-Clause](LICENSE.md)
