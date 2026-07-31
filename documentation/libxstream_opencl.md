# OpenCL Backend

[`libxstream/libxstream_opencl.h`](https://github.com/hfp/libxstream/blob/main/libxstream/libxstream_opencl.h) is the internal OpenCL layer that powers every public `libxstream_*` function. It owns the OpenCL platform/device/context lifecycle, memory management, kernel compilation, and error handling. Sample code and other LIBXSTREAM extensions (e.g., LIBSMM, Ozaki) include this header to access the OpenCL runtime directly.

## Compile-Time Configuration

The header is guarded by `__OPENCL` (set automatically when `__OFFLOAD_OPENCL` is defined). Key compile-time knobs:

| Macro | Default | Description |
|---|---|---|
| `LIBXSTREAM_MAXALIGN` | 2 MB | Maximum alignment for device allocations |
| `LIBXSTREAM_BUFFERSIZE` | 8 KB | Internal scratch-buffer size |
| `LIBXSTREAM_MAXSTRLEN` | 48 | Maximum string length for names |
| `LIBXSTREAM_MAXNDEVS` | 64 | Maximum number of OpenCL devices |
| `LIBXSTREAM_MAXNITEMS` | 1024 | Per-thread maximum item count |
| `LIBXSTREAM_MAXNKERNELS` | 32 | Maximum number of distinct kernels that can be profiled |
| `LIBXSTREAM_PROFILE_TICKS` | 10 | Device-timer ticks a sample must span to be recorded |
| `LIBXSTREAM_USM` | SVM coarse-grain | Runtime Unified Shared Memory level (unset = OpenCL 2.0 SVM coarse-grain, same as 2; 0 = off, 1 = Intel USM, 2 = OpenCL 2.0 SVM coarse-grain, 3 = OpenCL 2.0 SVM reported caps) |
| `LIBXSTREAM_SUBBUFFER` | 0 | Sub-buffers for offset kernel-arguments (0 = off, 1 = on); unused where USM is active |

Levels 1 and 3 are opt-in and never reached by the default, even though both are faster in a microbenchmark. Level 1 (`cl_intel_unified_shared_memory`) is the only path with a genuinely asynchronous transfer -- `clEnqueueMemcpyINTEL` measured 45.3 GB/s against 9.1 GB/s for a 128 MB H2D on a GPU Max 1550, where every SVM path instead ends in a host `memcpy` that runs single-threaded inside the enqueue and cannot overlap a kernel. It stays opt-in regardless, because the default must behave predictably across drivers rather than peak on one; request it explicitly where it is known good. A warning is emitted at verbosity 2+ if an explicit level 1 cannot load the extensions, since the only symptom is a slower run.

Level 3 is excluded from the default for a different reason. It adopts whatever `CL_DEVICE_SVM_CAPABILITIES` reports, and that varies by kernel driver: on i915 a GPU Max 1550 offers only coarse-grain buffers, whereas Xe also reports fine-grain, including fine-grain *system* allocations. That is a substantially broader contract than coarse-grain buffers and not something to acquire implicitly from a capability bit, so levels below 3 mask the capabilities down to `CL_DEVICE_SVM_COARSE_GRAIN_BUFFER`.

Nothing is lost by that: the grain is not a performance lever. Coarse-grain adds a `clEnqueueSVMMap`/`clEnqueueSVMUnmap` pair that fine-grain omits, but that is bookkeeping rather than data movement and the `memcpy` both paths end in dominates. Measured on a Xeon Platinum 8480+ (the device here reporting both grains), 64 MB H2D: 15.2 GB/s forced coarse against 15.1 GB/s with fine-grain.

## Data Types

### `libxstream_opencl_config_t`

The central singleton (`libxstream_opencl_config`) populated by `libxstream_init`. It holds:

* **Device table** — ordered array of discovered `cl_device_id` entries.
* **Active device** (`libxstream_opencl_device_t`) — context, default stream, error slot, OpenCL standard level, workgroup limits, memory caps, vendor flags, and optional USM function pointers.
* **Resource pools** — lock objects, streams, events, memory-pointer registrations, and a host-memory pool (`libxs_malloc_pool_t`).
* **Runtime switches** — verbosity, async mode, debug/dump level, profiling, execution hints, and workaround level.
* **Histograms** — optional per-kernel duration histograms, and transfer-time histograms for H2D, D2H, D2D, and zero-fill operations.

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
| `libxstream_opencl_use_cmem_size` | Whether OpenCL constant-memory hints apply |
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
| `libxstream_opencl_launch` | Launch a kernel (drop-in for `clEnqueueNDRangeKernel`, profiling-aware) |
| `libxstream_opencl_duration` | Measure elapsed seconds from a `cl_event` |

### Error Utilities

| Function | Description |
|---|---|
| `libxstream_opencl_strerror` | Map a `cl_int` error code to a string |
| `libxstream_opencl_error_consume` | Clear and return the last recorded error |

## Profiling

Two independent facilities, reported at exit on `stderr` and silent unless requested. They are separate because they measure different quantities in different units: a kernel has no byte count and hence no transfer rate, so mixing the rows would invite reading one as the other. Both may be set at once, which is the only case in which both appear.

| Variable | Reports |
|---|---|
| `LIBXSTREAM_PROFILE` | Per-kernel durations (microseconds), one row per distinct kernel |
| `LIBXSTREAM_PROFILE_MEM` | Transfer rates (GB/s) for H2D, D2H, D2D, and zero-fill |

A positive value sets the histogram resolution; a negative value additionally traces every individual sample as it is recorded. Setting either variable enables `CL_QUEUE_PROFILING_ENABLE` on all streams, so profiling is not meant for production runs.

Kernels are identified by `CL_KERNEL_FUNCTION_NAME`, read once per distinct `cl_kernel` handle. Call sites therefore pass no identifier: replacing `clEnqueueNDRangeKernel` with `libxstream_opencl_launch` is sufficient to make a kernel appear in the report.

Samples whose duration spans fewer than `LIBXSTREAM_PROFILE_TICKS` ticks of the device timer (`CL_DEVICE_PROFILING_TIMER_RESOLUTION`) are counted as discarded rather than recorded, because a rate derived from one or two ticks is quantization noise. The report states the timer resolution and the resulting floor only when samples were actually dropped, and reports `no samples recorded` when a requested profile collected nothing at all — silence there would be indistinguishable from a run that performed no work.

Every value a histogram carries is averaged per sample, never accumulated, so that a bucket's amount and its duration refer to the same single transfer and their ratio is a rate. Accumulating the duration instead divides a per-sample amount by a bucket total, which understates the rate by roughly the number of samples that share the bucket — and equally sized transfers all share one, so the error is largest exactly where the report is most useful.

The headline rate on a transfer row is the histogram's **mode**, not its median. Transfer sizes are commonly multi-modal — a whole operand alongside per-panel blocks, or a zero-fill covering both a small exponent array and a large slice plane — and a median can fall between the clusters and thus describe no observed transfer, whereas the mode always names a bucket that samples landed in. Where a workload issues transfers of one size the two agree, so nothing is lost where the median was already right. Kernel rows report the median duration.

## See Also

* LIBXSTREAM API (`libxstream/libxstream.h`) — public API built on top of this layer
* [DBCSR ACC Interface](libxstream_dbcsr.md) — the DBCSR compatibility shim
