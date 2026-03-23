# DBCSR ACC Interface

[`include/libxstream_dbcsr.h`](https://github.com/hfp/libxstream/blob/main/include/libxstream_dbcsr.h) implements the [DBCSR ACC interface](https://github.com/cp2k/dbcsr/blob/develop/src/acc/) — the accelerator backend contract defined by the [DBCSR](https://dbcsr.cp2k.org/) sparse-matrix library used in [CP2K](https://cp2k.org/). By providing this interface on top of LIBXSTREAM's OpenCL runtime, the library acts as a **drop-in OpenCL backend** for DBCSR: no changes to DBCSR or CP2K source code are required.

## Purpose

DBCSR expects every accelerator backend to expose a flat C API with the `c_dbcsr_acc_*` prefix covering five domains:

| Domain | Functions | Description |
|---|---|---|
| **Initialization** | `c_dbcsr_acc_init`, `c_dbcsr_acc_finalize` | Library setup and teardown |
| **Devices** | `c_dbcsr_acc_get_ndevices`, `c_dbcsr_acc_set_active_device`, `c_dbcsr_acc_device_synchronize` | Device enumeration, selection, and synchronization |
| **Streams** | `c_dbcsr_acc_stream_create`, `c_dbcsr_acc_stream_destroy`, `c_dbcsr_acc_stream_sync`, `c_dbcsr_acc_stream_wait_event`, `c_dbcsr_acc_stream_priority_range` | Asynchronous command queues |
| **Events** | `c_dbcsr_acc_event_create`, `c_dbcsr_acc_event_destroy`, `c_dbcsr_acc_event_record`, `c_dbcsr_acc_event_query`, `c_dbcsr_acc_event_synchronize` | Fine-grained synchronization primitives |
| **Memory** | `c_dbcsr_acc_dev_mem_allocate`, `c_dbcsr_acc_dev_mem_deallocate`, `c_dbcsr_acc_dev_mem_set_ptr`, `c_dbcsr_acc_host_mem_allocate`, `c_dbcsr_acc_host_mem_deallocate`, `c_dbcsr_acc_memcpy_h2d`, `c_dbcsr_acc_memcpy_d2h`, `c_dbcsr_acc_memcpy_d2d`, `c_dbcsr_acc_memset_zero`, `c_dbcsr_acc_dev_mem_info` | Device/host allocation, transfers (H2D, D2H, D2D), and memory queries |

Every function delegates to the corresponding `libxstream_*` routine (e.g., `c_dbcsr_acc_stream_create` calls `libxstream_stream_create`), translating between DBCSR's opaque `void*` handles and LIBXSTREAM's typed `libxstream_stream_t` / `libxstream_event_t` pointers.

## Profiling

The header also declares `c_dbcsr_timeset` and `c_dbcsr_timestop`, which are DBCSR's Fortran-side timer callbacks. When profiling is enabled (`LIBXSTREAM_PROFILE_DBCSR`), every ACC function is bracketed by these calls so that individual backend operations appear in DBCSR's timing report.

## Utility Macros

| Macro | Description |
|---|---|
| `DBCSR_STRINGIFY(SYMBOL)` | Stringifies a preprocessor token |
| `DBCSR_CONCATENATE(A, B)` | Concatenates two preprocessor tokens |
| `DBCSR_MARK_USED(x)` | Silences unused-variable warnings |

## See Also

* LIBXSTREAM API (`include/libxstream.h`) — the underlying OpenCL backend API
* [DBCSR ACC specification](https://github.com/cp2k/dbcsr/blob/develop/src/acc/) — upstream interface definition
