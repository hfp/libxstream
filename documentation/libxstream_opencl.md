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
| `LIBXSTREAM_USM` | 1 | USM support built in: 0 = none, 1 = Intel USM and OpenCL 2.0 SVM, 2 = SVM only (default with `__OFFLOAD_UNIFIED_MEMORY`); 0 without OpenCL 2.0 |

## Runtime Configuration

Environment variables are read when the library initializes, except those concerning a device or a kernel build, which are read when a device is activated or a kernel is built. Devices are numbered in the order `LIBXSTREAM_VERBOSE` lists them.

| Variable | Description |
|---|---|
| `LIBXSTREAM_VERBOSE` | 0: silent, 1: errors and the activated device, 2: information (device list, build logs, warnings), 3: more; negative: everything; default: 0 |
| `LIBXSTREAM_DEVICE` | Device to activate, by number; setting it exposes only that device (`libxstream_device_count` yields 1); default: 0 |
| `LIBXSTREAM_DEVTYPE` | Kind of device: `gpu`, `cpu`, or `acc` (also `other`); default: all |
| `LIBXSTREAM_DEVIDS` | Devices to keep, by number (separated by comma, semicolon, colon, or space); default: all |
| `LIBXSTREAM_VENDOR` | Devices to keep, by vendor name (case-insensitive substring, e.g., `intel`); default: all |
| `LIBXSTREAM_DEVSPLIT` | Split each device into equal sub-devices: the number of sub-devices, or negative for one per compute unit; 1 does not split but sets `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` (see `LIBXSTREAM_WA`); default: 0 (no split) |
| `LIBXSTREAM_DEVMATCH` | Device whose tuned kernel parameters are taken: a device name or ID (e.g., `[0x0bd5]`) to use another device's parameters, 0 to accept parameters of any device; default: the active device |
| `LIBXSTREAM_PIN` | Route for host memory, see [CUDA Host Registration](#cuda-host-registration); default: resolved per device |
| `LIBXSTREAM_STAGE` | Staging window per thread in MB, for host memory staged through a runtime-owned buffer (`LIBXSTREAM_PIN=1`); default: 32 |
| `LIBXSTREAM_STAGE_NT` | Threads of the staging copy; default: 32 |
| `LIBXSTREAM_STAGE_GRAIN` | Block of the staging copy in KB, handed out round-robin; default: 32 |
| `LIBXSTREAM_USM` | Unified Shared Memory level: 0 = off, 1 = Intel USM, 2 = OpenCL 2.0 SVM coarse-grain, 3 = OpenCL 2.0 SVM reported caps. Unset means 1 on an Intel discrete GPU, 0 on an Intel integrated GPU (which also transfers synchronously), off on NVIDIA, and 2 otherwise. Intel MPI cannot exchange Intel USM host memory without `I_MPI_OFFLOAD=1`: on a system with Intel GPUs only it is set unless given. If `I_MPI_OFFLOAD=0` is given, unset means 2, while an explicit 1 takes effect as requested (and fails). Where it could not be set, an explicit 1 takes host memory off Intel USM |
| `LIBXSTREAM_SUBBUFFER` | Sub-buffers for offset kernel-arguments (0 = off, 1 = on); unused where USM is active; default: 0 |
| `LIBXSTREAM_CACHE` | 0: no kernel cache, i.e., the vendor's compiled-kernel cache is off (`CUDA_CACHE_DISABLE=1` and `NEO_CACHE_PERSISTENT=0`, unless set); default: the vendor's cache |

Level 3 is opt-in and never reached by the default; level 1 is the default on an Intel discrete GPU only, and opt-in elsewhere. Level 1 (`cl_intel_unified_shared_memory`) is the only level with asynchronous host-device transfers. Levels below 3 use coarse-grain SVM buffers only, whereas level 3 adopts whatever `CL_DEVICE_SVM_CAPABILITIES` reports (possibly fine-grain system allocations). At verbosity 2+, a warning says when an explicit level 1 cannot load the extensions, since the only symptom is a slower run.

On NVIDIA the default leaves USM off and no device memory pool is created. An explicit level is honoured there and correct, but slower: leave it unset on that vendor.

### Tuning and Diagnostics

| Variable | Description |
|---|---|
| `LIBXSTREAM_ASYNC` | Asynchronous operations (bitmask): 1 = H2D, 2 = D2H, 4 = D2D, 8 = memset/zero; a cleared bit makes that operation synchronous; default: 15 |
| `LIBXSTREAM_XHINTS` | Execution hints (bitmask): 1 = Intel thread-local queues, 2 = Intel queue families, 4 = host memory as `CL_MEM_USE_HOST_PTR` buffers (NVIDIA, or any vendor when the variable is set), 8 = Intel allocation flags on discrete GPUs, 16 = write-invalidate mapping of host buffers, 32 = query event status before waiting, 64 = order Intel GPUs by PCI bus, 128 = streams do not keep their last event; default: 79 (1+2+4+8+64) |
| `LIBXSTREAM_WA` | Workarounds (bitmask); default: 30, or 31 with `LIBXSTREAM_DEVSPLIT=1`. Unless already set: 1 sets `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, 2 `EnableRecoverablePageFaults=0`, 4 `DirectSubmissionOverrideBlitterSupport=0`, 8 `UCX_RCACHE_ENABLE=0`. 16 = no USM and synchronous transfers on an Intel integrated GPU, 32 = stream sync waits on a marker instead of `clFinish`, 64 = event sync finishes the event's queue. `NEOReadDebugKeys=1` is set unless given, and `NEOReadDebugKeys=0` suppresses bits 1-8 |
| `LIBXSTREAM_NLOCKS` | Lock domains, 1 to 4 (main, memory, stream, event); default: 1 |
| `LIBXSTREAM_NCCS` | Intel compute command streamers per device (`ZEX_NUMBER_OF_CCS`, unless set): 1 or 2, other non-zero: 4; default: 0 (not set) |
| `LIBXSTREAM_INTEL` | Overrides the detected Intel level: 0 = not Intel, 1 = GPU, 2 = GPU with XMX |
| `LIBXSTREAM_NV` | Overrides the detected NVIDIA level: 0 = not NVIDIA, 1 = generic, 2 = SM 7.5+, 3 = SM 8.0+, 4 = SM 9.0+ |
| `LIBXSTREAM_BIGGRF` | 1: Intel kernels are built with `-cl-intel-256-GRF-per-thread`; default: 0 |
| `LIBXSTREAM_ATOMICS` | FP atomics of kernels built with `libxstream_opencl_flags_atomics`: non-zero number = force native, `cmpxchg` or `xchg` = emulation, 0 = unsynchronized updates (not thread-safe); default: detected |
| `LIBXSTREAM_BARRIER` | 0: such kernels are built without a work-group barrier definition; default: defined |
| `LIBXSTREAM_DUMP` | 1: write each kernel built from source as `<name>.cl`, 2 (or negative): also the program binary as `<name>.dump`, 3: the written source is preprocessed (see `LIBXSTREAM_CPP`); default: 0, or `IGC_ShaderDumpEnable` |
| `LIBXSTREAM_DEBUG` | Non-zero: kernels are built without `-DNDEBUG` (and with `-gline-tables-only` on an Intel GPU), build flags are printed at verbosity 1+, and `DisableScratchPages=1` is set unless given; default: `LIBXSTREAM_DUMP` |
| `LIBXSTREAM_CPP` | 1: preprocess written kernel sources, 0: never; default: with `LIBXSTREAM_DUMP=3` and `libxstream_opencl_dump` |
| `LIBXSTREAM_CPPBIN` | Preprocessor; default: `/usr/bin/cpp` |
| `LIBXSTREAM_CPPFLAGS` | Extra preprocessor flags, e.g., `-E -x cl` for a clang driver |
| `LIBXSTREAM_PRIORITY` | Read, but without effect: stream priorities are compiled out; default: 3 |

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
| `libxstream_opencl_device_match` | Select the entry of a list of device names a device matches |
| `libxstream_opencl_info_devmem` | Query free/total/local device memory |
| `libxstream_opencl_device_id` | Index of a device, and optionally its global index |
| `libxstream_opencl_device_name_cleanup` | Strip UID brackets, decorators, and memory sizes from a device name |
| `libxstream_opencl_device_pageable` | Whether the active device reads pageable host memory (1, 0, or -1 if unknown) |

### Memory

| Function | Description |
|---|---|
| `libxstream_opencl_info_devptr` | Look up a device-pointer registration (read-only) |
| `libxstream_opencl_info_devptr_modify` | Look up a device-pointer registration (writable) |
| `libxstream_opencl_info_devptr_lock` | As `libxstream_opencl_info_devptr`, with a caller-given lock |
| `libxstream_opencl_info_hostptr` | Look up a host-pointer registration |
| `libxstream_opencl_memset` | Fill device memory with an arbitrary byte pattern |
| `libxstream_opencl_use_cmem_size` | Whether OpenCL constant-memory hints apply |
| `libxstream_opencl_use_cmem` | As `libxstream_opencl_use_cmem_size` without a size |
| `libxstream_opencl_set_kernel_ptr` | Set a pointer kernel argument (USM-aware) |

#### CUDA Host Registration

Where the application links the CUDA runtime, host memory from `libxstream_mem_host_allocate` is registered with it (`cudaHostRegister`); otherwise, a transfer CUDA issues from such memory is pageable and slower. Nothing needs to be enabled, and there is no build or link dependency on CUDA. USM/SVM memory is registered per pool block, and blocks from plain `malloc` are not registered.

`LIBXSTREAM_PIN` selects the route for the library's own allocations and for ranges the caller declared (see below):

| Value | Route |
|---|---|
| unset | Recommended: resolved once a device is activated; own allocations are registered, declared ranges are staged where the vendor runtime reports no pageable access to host memory and registered otherwise |
| 0 | Nothing is registered or staged (pageable transfers) |
| 1 | Own allocations are registered, declared ranges are staged through a runtime-owned buffer (`LIBXSTREAM_STAGE*`) |
| 2 | Own allocations and declared ranges are registered |
| 3 | As 2, and loads the CUDA runtime if the application did not link it, e.g., for a drop-in BLAS replacement linking only `libOpenCL` |

`LIBXSTREAM_VERBOSE=2` reports how many allocations were offered and how many the CUDA runtime accepted. Memory it refused stays pageable, which shows as a slow kernel rather than as a slow copy.

A range the application allocated itself is declared with `libxstream_mem_host_pin` and withdrawn with `libxstream_mem_host_unpin`. Memory from `libxstream_mem_host_allocate` must not be declared, since it is page-locked already. Declaring neither initializes the library nor activates a device; the route applies to all declared ranges once a device is activated. Up to `LIBXSTREAM_MAXNPINS` ranges (32 by default) can be held at a time; a declaration beyond that is rejected. Failure is not fatal: the range still transfers, only slower.

### Kernel Build

| Function | Description |
|---|---|
| `libxstream_opencl_program` | Compile an OpenCL program from source, file, or binary |
| `libxstream_opencl_kernel_query` | Extract a named kernel from a compiled program |
| `libxstream_opencl_kernel` | Convenience: build + extract + release in one call |
| `libxstream_opencl_kernel_flags` | Assemble combined build flags from params, options, and extras |
| `libxstream_opencl_defines` | Merge user defines with internal definitions |
| `libxstream_opencl_flags_atomics` | Generate compiler flags for FP-atomic extensions |
| `libxstream_opencl_dump` | Write a kernel instance `<name>.cl` with build parameters applied, without a device |
| `libxstream_opencl_program_binary` | Retrieve a built program's binary (NUL-terminated) |
| `libxstream_opencl_retarget_ptx` | Rewrite NVIDIA PTX for the accelerated target (`sm_NNa`), optionally capping registers |

### Streams, Events, and Timing

| Function | Description |
|---|---|
| `libxstream_opencl_stream` | Find an existing stream for a thread-ID |
| `libxstream_opencl_stream_default` | Return the device's default (internal) stream |
| `libxstream_opencl_device_synchronize` | Per-thread device synchronization |
| `libxstream_opencl_launch` | Launch a kernel (drop-in for `clEnqueueNDRangeKernel`, profiling-aware) |
| `libxstream_opencl_launch_work` | As `libxstream_opencl_launch`, with FLOP and byte counts so the profile reports rates |
| `libxstream_opencl_interval` | Begin and end of an event on the device clock (nanoseconds) |
| `libxstream_stream_priority_range` | Least and greatest stream priority (-1 if unsupported) |
| `libxstream_opencl_duration` | Measure elapsed seconds from a `cl_event` |

### Error Utilities

| Function | Description |
|---|---|
| `libxstream_opencl_strerror` | Map a `cl_int` error code to a string |
| `libxstream_opencl_error_consume` | Clear and return the last recorded error |

## Profiling

Two independent facilities, reported at exit on `stderr` and silent unless requested. Both may be set at once.

| Variable | Reports |
|---|---|
| `LIBXSTREAM_PROFILE` | Per-kernel durations (microseconds), one row per distinct kernel |
| `LIBXSTREAM_PROFILE_MEM` | Transfer rates (GB/s) for H2D, D2H, D2D, and zero-fill |

A positive value sets the histogram resolution; a negative value additionally traces every sample as it is recorded, with its interval as `ns=<begin>-<end>` in device-clock nanoseconds. Setting either variable enables `CL_QUEUE_PROFILING_ENABLE` on all streams, so profiling is not meant for production runs.

Kernels are identified by `CL_KERNEL_FUNCTION_NAME`: launching through `libxstream_opencl_launch` instead of `clEnqueueNDRangeKernel` makes a kernel appear in the report.

Samples spanning fewer than `LIBXSTREAM_PROFILE_TICKS` ticks of the device timer (`CL_DEVICE_PROFILING_TIMER_RESOLUTION`) are counted as discarded rather than recorded. The report states the timer resolution and the resulting floor only when samples were dropped, and says `no samples recorded` when a requested profile collected nothing. A kernel whose every sample was discarded still gets a row that says so.

The rate on a transfer row is the histogram's mode (the most populated bucket); a kernel row reports the median duration.

### Overlap

A row gains one field, `inflight>=`, when its samples overlapped in time:

```
PROF ACC/OpenCL: ID=<id> ZERO=<rate> GB/s inflight>=<ratio>
PROF ACC/OpenCL: ID=<id> device inflight>=<ratio> (3 kernels, 3 transfer kinds)
```

It is a lower bound on the average number of these operations running at once (summed durations over the time their intervals cover). A value above 1 is overlap; the field is absent if no overlap shows in two decimals. Intervals arriving too late to be merged are added whole, and their count is reported.

A `device` row combines every kernel and every transfer, states how many of each contributed, and thereby shows overlap between kernels or between a kernel and a transfer.

## OpenCL-MPS

Several processes can share one context per device through a server process; see [OpenCL-MPS](libxstream_mps.md).

## See Also

* LIBXSTREAM API (`libxstream/libxstream.h`) — public API built on top of this layer
* [DBCSR ACC Interface](libxstream_dbcsr.md) — the DBCSR compatibility shim
