# OpenCL Backend

[`include/libxstream_opencl.h`](https://github.com/hfp/libxstream/blob/main/include/libxstream_opencl.h) is the internal OpenCL layer that powers every public `libxstream_*` function. It owns the OpenCL platform/device/context lifecycle, memory management, kernel compilation, and error handling. Sample code and other LIBXSTREAM extensions (e.g., LIBSMM, Ozaki) include this header to access the OpenCL runtime directly.

## Compile-Time Configuration

The header is guarded by `__OPENCL` (set automatically when `__OFFLOAD_OPENCL` is defined). Key compile-time knobs:

| Macro | Default | Description |
|---|---|---|
| `LIBXSTREAM_MAXALIGN` | 2 MB | Maximum alignment for device allocations |
| `LIBXSTREAM_BUFFERSIZE` | 8 KB | Internal scratch-buffer size |
| `LIBXSTREAM_MAXSTRLEN` | 48 | Maximum string length for names |
| `LIBXSTREAM_MAXNDEVS` | 64 | Maximum number of OpenCL devices |
| `LIBXSTREAM_MAXNITEMS` | 1024 | Per-thread maximum item count |
| `LIBXSTREAM_DELIMS` | `",;"` | CSV delimiter characters |
| `LIBXSTREAM_USM` | auto | Unified Shared Memory level (0 = off, 1 = OpenCL 2.0, 2 = Intel USM) |

## Data Types

### `libxstream_opencl_config_t`

The central singleton (`libxstream_opencl_config`) populated by `libxstream_init`. It holds:

* **Device table** — ordered array of discovered `cl_device_id` entries.
* **Active device** (`libxstream_opencl_device_t`) — context, default stream, error slot, OpenCL standard level, workgroup limits, memory caps, vendor flags, and optional USM function pointers.
* **Resource pools** — lock objects, streams, events, memory-pointer registrations, and a host-memory pool (`libxs_malloc_pool_t`).
* **Runtime switches** — verbosity, async mode, debug/dump level, profiling, execution hints, and workaround level.
* **Histograms** — optional transfer-time histograms for H2D, D2H, and D2D copies.

### `libxstream_opencl_stream_t` / `libxstream_event_t`

Thin wrappers around `cl_command_queue` and `cl_event` respectively. Streams additionally carry a thread-ID and optional priority.

### `libxstream_opencl_info_memptr_t`

Associates a `cl_mem` buffer object with its host-side pointer, used to translate between SVM/USM pointers and buffer-based memory.

### `libxstream_opencl_atomic_fp_t`

Enumerates floating-point atomics support: none, 32-bit, or 64-bit.

## Error Handling Macros

| Macro | Description |
|---|---|
| `CL_CHECK(RESULT, CALL)` | Execute an OpenCL call; on failure record the error code and human-readable name |
| `CL_ERROR_REPORT(NAME)` | Print the last error to stderr (if verbosity is enabled) |
| `CL_RETURN(RESULT, NAME)` | Return from function, reporting the error if non-zero |

## Key Functions

### Device and Context

| Function | Description |
|---|---|
| `libxstream_opencl_set_active_device` | Internal device activation (lock-aware) |
| `libxstream_opencl_create_context` | Create an OpenCL context for a given device |
| `libxstream_opencl_device_name` | Return device name, platform name, and UID |
| `libxstream_opencl_device_level` | Query OpenCL version and device type |
| `libxstream_opencl_device_vendor` | Confirm a device's vendor string |
| `libxstream_opencl_device_ext` | Check for required OpenCL extensions |
| `libxstream_opencl_device_uid` | Capture or compute a unique device identifier |
| `libxstream_opencl_info_devmem` | Query free/total/local device memory |

### Memory

| Function | Description |
|---|---|
| `libxstream_opencl_info_devptr` | Look up a device-pointer registration (read-only) |
| `libxstream_opencl_info_devptr_modify` | Look up a device-pointer registration (writable) |
| `libxstream_opencl_info_hostptr` | Look up a host-pointer registration |
| `libxstream_opencl_memset` | Fill device memory with an arbitrary byte pattern |
| `libxstream_opencl_use_cmem` | Whether OpenCL constant-memory hints apply |
| `libxstream_opencl_set_kernel_ptr` | Set a pointer kernel argument (USM-aware) |

### Kernel Build

| Function | Description |
|---|---|
| `libxstream_opencl_program` | Compile an OpenCL program from source, file, or binary |
| `libxstream_opencl_kernel_query` | Extract a named kernel from a compiled program |
| `libxstream_opencl_kernel` | Convenience: build + extract + release in one call |
| `libxstream_opencl_kernel_flags` | Assemble combined build flags from params, options, and extras |
| `libxstream_opencl_defines` | Merge user defines with internal definitions |
| `libxstream_opencl_flags_atomics` | Generate compiler flags for FP-atomic extensions |

### Streams, Events, and Timing

| Function | Description |
|---|---|
| `libxstream_opencl_stream` | Find an existing stream for a thread-ID |
| `libxstream_opencl_stream_default` | Return the device's default (internal) stream |
| `libxstream_opencl_device_synchronize` | Per-thread device synchronization |
| `libxstream_opencl_duration` | Measure elapsed seconds from a `cl_event` |

### Error Utilities

| Function | Description |
|---|---|
| `libxstream_opencl_strerror` | Map a `cl_int` error code to a string |
| `libxstream_opencl_error_consume` | Clear and return the last recorded error |

## See Also

* LIBXSTREAM API (`include/libxstream.h`) — public API built on top of this layer
* [DBCSR ACC Interface](libxstream_dbcsr.md) — the DBCSR compatibility shim
