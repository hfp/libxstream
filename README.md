# LIBXSTREAM

LIBXSTREAM is an OpenCL-based accelerator backend library that provides streams, events, and device memory management for offloading compute work to GPUs. The library implements the [DBCSR ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/), making it a drop-in OpenCL backend for [CP2K](https://cp2k.org/)/[DBCSR](https://dbcsr.cp2k.org/).

LIBXSTREAM targets OpenCL devices across vendors (Intel, AMD, NVIDIA) and depends on [LIBXS](https://github.com/hfp/libxs) for utility infrastructure (memory pools, timers, synchronization, etc.).

## API

The public C API is declared in [`include/libxstream.h`](include/libxstream.h). All implementation details are sealed behind opaque types.

```C
typedef int libxstream_bool_t;
typedef struct libxstream_stream_t libxstream_stream_t;
typedef struct libxstream_event_t libxstream_event_t;
```

### Devices

```C
int libxstream_init(void);
int libxstream_finalize(void);
int libxstream_get_ndevices(int* ndevices);
int libxstream_set_active_device(int device_id);
int libxstream_device_synchronize(void);
```

### Streams

```C
int libxstream_stream_create(libxstream_stream_t** stream_p, const char* name, int priority);
int libxstream_stream_destroy(libxstream_stream_t* stream);
int libxstream_stream_sync(libxstream_stream_t* stream);
int libxstream_stream_wait_event(libxstream_stream_t* stream, libxstream_event_t* event);
```

### Events

```C
int libxstream_event_create(libxstream_event_t** event_p);
int libxstream_event_destroy(libxstream_event_t* event);
int libxstream_event_record(libxstream_event_t* event, libxstream_stream_t* stream);
int libxstream_event_query(libxstream_event_t* event, libxstream_bool_t* has_occurred);
int libxstream_event_synchronize(libxstream_event_t* event);
```

### Memory

Device and host memory allocation, transfers (H2D, D2H, D2D), and initialization. Memory pointers remain untyped (`void*`); stream parameters use the opaque stream type.

```C
void* libxstream_memdev_allocate(size_t nbytes);
void libxstream_memdev_deallocate(void* dev_mem);
int libxstream_memhst_allocate(void** host_mem, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memhst_deallocate(void* host_mem, libxstream_stream_t* stream);
int libxstream_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, libxstream_stream_t* stream);
int libxstream_memset_zero(void* dev_mem, size_t offset, size_t nbytes, libxstream_stream_t* stream);
```

### DBCSR Compatibility

The header [`include/libxstream_dbcsr.h`](include/libxstream_dbcsr.h) provides the full set of `c_dbcsr_acc_*` symbols expected by DBCSR, allowing LIBXSTREAM to serve as a drop-in accelerator backend. The DBCSR interface retains `void*` handles for streams and events; casts to the opaque types are confined to the adapter layer ([`src/libxstream_dbcsr.c`](src/libxstream_dbcsr.c)).

## Building

LIBXSTREAM depends on [LIBXS](https://github.com/hfp/libxs), which is included at compile time via source inclusion (`libxs_source.h`). The Makefiles expect LIBXS to reside as a sibling directory (`../libxs` relative to the LIBXSTREAM root). This is controlled by the `LIBXSROOT` variable in both the top-level [Makefile](Makefile) and the sample Makefiles.

```sh
# clone both repositories side by side
git clone https://github.com/hfp/libxs.git
git clone https://github.com/hfp/libxstream.git
```

```
parent/
  libxs/          <-- LIBXS (required)
  libxstream/     <-- this repository
```

An OpenCL SDK must also be available.

```sh
cd libxstream
make          # builds lib/libxstream.a and lib/libxstream.so
```

Key Makefile variables:

| Variable    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `OCL`       | `2`     | OpenCL requirement level             |
| `STATIC`    | `1`     | `0`: shared only, `1`: both, `2`: static only |
| `THREADS`   | `1`     | Thread-safe library                  |
| `OMP`       | `0`     | OpenMP support                       |
| `DBG`       | `0`     | Debug build                          |
| `CACHELINE` | `64`    | Cacheline size (bytes)               |

pkg-config support is available via `lib/libxstream.pc`.

### Samples

Each sample has its own Makefile under `samples/`:

```sh
cd samples/smm && make
cd samples/ozaki && make
```

## Samples

### SMM — Small Matrix Multiplication

[`samples/smm/`](samples/smm/) implements the [ACC LIBSMM interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/acc_libsmm.h) for batched small matrix multiply and transpose on OpenCL devices. Includes an auto-tuning framework (`tune_multiply.py`) and pre-tuned parameter sets for A100, BMG, GH200, H100, Mi250, P100, PVC, and V100. See [`samples/smm/README.md`](samples/smm/README.md) for details.

### Ozaki — High-Precision GEMM via Mantissa Slicing

[`samples/ozaki/`](samples/ozaki/) demonstrates Ozaki Scheme 1 for high-precision GEMM emulation, fully offloaded to OpenCL. Two kernel variants (int8 and bf16) with automatic detection of Intel XMX matrix engines. See [`samples/ozaki/README.md`](samples/ozaki/README.md) for details. The CPU-side GEMM wrapper that drives this GPU implementation is part of [LIBXS Ozaki](https://github.com/hfp/libxs/tree/main/samples/ozaki#ozaki-scheme-low-precision-gemm).

## References

**[1]** [CP2K](https://cp2k.org/) — Open Source Molecular Dynamics. LIBXSTREAM provides the OpenCL accelerator backend for CP2K's [DBCSR](https://dbcsr.cp2k.org/) sparse matrix library.

## License

[BSD 3-Clause](LICENSE.md)
