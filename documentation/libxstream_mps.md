# OpenCL-MPS

[`src/libxstream_mps.c`](https://github.com/hfp/libxstream/blob/main/src/libxstream_mps.c) and [`src/libxstream_mpsd.c`](https://github.com/hfp/libxstream/blob/main/src/libxstream_mpsd.c) implement OpenCL-MPS, the multi-process service of LIBXSTREAM; its functions are declared in [`libxstream/libxstream_opencl.h`](https://github.com/hfp/libxstream/blob/main/libxstream/libxstream_opencl.h). Several processes sharing one GPU are time-sliced by the device, one context at a time. OpenCL-MPS gives all processes of a node one context per device: each process forwards its OpenCL calls to a server process (`libxstream_mpsd`, installed into `bin/`), which exits once no process was attached for a while. MPI is not involved; one server runs per node and job.

## Build and Installation

OpenCL-MPS is available on Linux if `CL/cl_icd.h` is found at build time, and requires the shared library (`make STATIC=0`, or CMake with `BUILD_SHARED_LIBS=ON`). The build places the server `libxstream_mpsd` and the ICD `libxstream_icd.so` beside the shared library. Installed, the server is in `bin/` and the ICD, versioned like the library, in `lib/libxstream/`; the library locates both relative to its own directory.

## Usage

There are two ways to opt in:

```bash
libxstream_mpsd &              # a running server is the opt-in
mpirun ...
```

```bash
export LIBXSTREAM_MPS=1        # or: the first process starts the server
mpirun ...
```

LIBXSTREAM then points the OpenCL loader to `libxstream_icd.so` (installed into `lib/libxstream/`) when it is loaded, before any OpenCL call, which hides the installed ICDs. An OpenCL application that does not use LIBXSTREAM is served the same way, unchanged:

```bash
LIBXSTREAM_MPS=1 LD_PRELOAD=libxstream.so ./app
```

This relies on the library's constructor, which a C89 build (`PEDANTIC=2`) does not have. Setting both `OCL_ICD_VENDORS` (ocl-icd) and `OCL_ICD_FILENAMES` (Khronos loader) to `libxstream_icd.so` works as well. If the OpenCL driver is only found through these variables (a driver outside the default location), forwarding stays off: give their value to the server instead (`LIBXSTREAM_MPS_ICD`), and unset both.

Host buffers are shared with the server only if they come from OpenCL, i.e., with `LIBXSTREAM_XHINTS=75` (bit 4 cleared); other host memory is copied through a socket.

## Limitations

The forwarding platform reports the vendor's platform and devices, with ` (LIBXSTREAM-MPS)` appended to the platform name. Of several platforms, the server serves one: that of the device type `LIBXSTREAM_DEVTYPE` asks for, otherwise GPUs first. Profiling (`LIBXSTREAM_PROFILE*`), event callbacks, SVM, sub-devices, and programs from IL are not supported. A context spanning several devices is shared only with processes using the same set of devices.

## Environment Variables

| Variable | Description |
|---|---|
| `LIBXSTREAM_MPS` | 1: forward, and start the server if none runs; 0: never forward; default: forward if a server runs |
| `LIBXSTREAM_MPS_TIMEOUT` | Seconds the server stays after the last process detached; negative: until killed; default: 2 |
| `LIBXSTREAM_MPS_KEY` | Names the server to attach to; default: the job ID (`SLURM_JOB_ID`, `PBS_JOBID`, `LSB_JOBID`) |
| `LIBXSTREAM_MPS_SERVER` | Server executable; default: `libxstream_mpsd` in the installed `bin/` (beside the library in a build tree), else found by `PATH` |
| `LIBXSTREAM_MPS_ICD` | Vendor ICDs the server loads, as for `OCL_ICD_FILENAMES`; default: the installed ICDs |
| `LIBXSTREAM_MPS_ARENA` | Host memory in GB a process may share with the server (reserved, not committed); default: 256 |
| `LIBXSTREAM_MPS_CONTEXTS` | Contexts per device the server spreads the processes over (round robin); default: 2 with four or more ranks per node, otherwise 1 (1 if the loader variables are set by hand) |
| `LIBXSTREAM_MPS_ISSUERS` | Commands the server passes to the driver at once, over all processes; default: 0 (unbounded) |

## Functions

| Function | Description |
|---|---|
| `libxstream_mps_server` | Serve the node (what `libxstream_mpsd` runs) |
| `libxstream_mps_icd_function`, `libxstream_mps_icd_platforms` | ICD entry points exported by `libxstream_icd.so` |

## See Also

- [OpenCL Backend](libxstream_opencl.md) — device selection (`LIBXSTREAM_DEVTYPE`) and the other runtime settings
