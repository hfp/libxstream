# DBCSR ACC Interface

[`libxstream/libxstream_dbcsr.h`](https://github.com/hfp/libxstream/blob/main/libxstream/libxstream_dbcsr.h) implements the [DBCSR ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/) on top of LIBXSTREAM's OpenCL runtime — the accelerator backend contract defined by the [DBCSR](https://dbcsr.cp2k.org/) sparse-matrix library used in [CP2K](https://cp2k.org/). DBCSR configured with `USE_ACCEL=opencl` uses it. DBCSR finds LIBXSTREAM with `find_package(libxstream)` (e.g., via `CMAKE_PREFIX_PATH`), links `libxstream::libxstream`, and builds LIBSMM from `samples/smm` into its own target with `libxstream_add_smm`.

## Purpose

DBCSR expects every accelerator backend to expose a flat C API with the `c_dbcsr_acc_*` prefix covering six domains:

| Domain | Functions | Description |
|---|---|---|
| **Initialization** | `c_dbcsr_acc_init`, `c_dbcsr_acc_finalize` | Library setup and teardown |
| **Errors** | `c_dbcsr_acc_clear_errors` | Clears the last recorded error |
| **Devices** | `c_dbcsr_acc_get_ndevices`, `c_dbcsr_acc_set_active_device`, `c_dbcsr_acc_device_synchronize` | Device enumeration, selection, and synchronization |
| **Streams** | `c_dbcsr_acc_stream_create`, `c_dbcsr_acc_stream_destroy`, `c_dbcsr_acc_stream_sync`, `c_dbcsr_acc_stream_wait_event`, `c_dbcsr_acc_stream_priority_range` | Asynchronous command queues |
| **Events** | `c_dbcsr_acc_event_create`, `c_dbcsr_acc_event_destroy`, `c_dbcsr_acc_event_record`, `c_dbcsr_acc_event_query`, `c_dbcsr_acc_event_synchronize` | Fine-grained synchronization primitives |
| **Memory** | `c_dbcsr_acc_dev_mem_allocate`, `c_dbcsr_acc_dev_mem_deallocate`, `c_dbcsr_acc_dev_mem_set_ptr`, `c_dbcsr_acc_host_mem_allocate`, `c_dbcsr_acc_host_mem_deallocate`, `c_dbcsr_acc_memcpy_h2d`, `c_dbcsr_acc_memcpy_d2h`, `c_dbcsr_acc_memcpy_d2d`, `c_dbcsr_acc_memset_zero`, `c_dbcsr_acc_dev_mem_info` | Device/host allocation, transfers (H2D, D2H, D2D), and memory queries |

Every function delegates to the corresponding `libxstream_*` routine (e.g., `c_dbcsr_acc_stream_create` calls `libxstream_stream_create`). `c_dbcsr_acc_event_query` reports through `c_dbcsr_acc_bool_t` (an `int`), the header's only type.

`c_dbcsr_acc_clear_errors` acts only in OpenCL builds (`__OFFLOAD_OPENCL`). `c_dbcsr_acc_stream_create` ignores `priority` unless LIBXSTREAM is built with `LIBXSTREAM_STREAM_PRIORITIES`, which is off by default.

## Profiling

The header also declares `c_dbcsr_timeset` and `c_dbcsr_timestop`, which are DBCSR's Fortran-side timer callbacks and are provided by DBCSR at link time. Every ACC function except `c_dbcsr_acc_clear_errors` is bracketed by these calls, so that individual backend operations appear in DBCSR's timing report, if two conditions hold:

- The ACC functions are compiled with `LIBXSTREAM_PROFILE_DBCSR`, which `__DBCSR_ACC` defines. A LIBXSTREAM library built on its own does not define it; this optional debug aid is enabled with `make ECFLAGS=-DLIBXSTREAM_PROFILE_DBCSR=8`, and such a library then only links together with DBCSR.
- `LIBXSTREAM_PROFILE` is non-zero at runtime, which also enables LIBXSTREAM's own per-kernel profile.

## LIBSMM

The second half of the ACC interface, `samples/smm/acc_libsmm.h`, is not part of the library. `libxstream_add_smm` compiles it into DBCSR from `samples/smm` (see the SMM sample for its environment variables and tuned parameters):

| Function | Description |
|---|---|
| `libsmm_acc_init`, `libsmm_acc_finalize` | Setup and teardown (repeated calls are allowed) |
| `libsmm_acc_is_thread_safe` | True if built with OpenMP |
| `libsmm_acc_transpose` | Batched in-place transpose of the matrices an offset stack names |
| `libsmm_acc_process` | Batched SMM on a homogeneous stack of `float` or `double` with M·N ≤ `max_kernel_dim`²; other stacks return -1 |
| `c_calculate_norms` | Not implemented (returns -1) |

## Utility Macros

| Macro | Description |
|---|---|
| `DBCSR_STRINGIFY(SYMBOL)` | Stringifies a preprocessor token |
| `DBCSR_CONCATENATE(A, B)` | Concatenates two preprocessor tokens |
| `DBCSR_MARK_USED(x)` | Silences unused-variable warnings |

## See Also

- LIBXSTREAM API (`libxstream/libxstream.h`) — the underlying OpenCL backend API
- [DBCSR ACC specification](https://github.com/cp2k/dbcsr/blob/develop/src/acc/) — upstream interface definition
