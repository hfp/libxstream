# DBM — OpenCL

OpenCL backend of CP2K's DBM (Distributed Block-sparse Matrices). DBM
processes batches of small matrix multiplications (tasks):

```C
typedef struct {
  int m;
  int n;
  int k;
  int offset_a;
  int offset_b;
  int offset_c;
} dbm_task_t;
```

Each task is characterized by its M, N, and K and by offsets into compact
arrays holding the A, B, and C matrices of an entire batch. A universal kernel
covers a general range of shapes, and homogeneous batches are specialized at
runtime. The entry point is `dbm_multiply_opencl_launch_kernel` in
[dbm_opencl.h](dbm_opencl.h).

## Build

```bash
cd samples/dbm
make
```

Produces `dbm_bench.x`, which is the benchmark driver of the
[SMM sample](../smm/README.md) linked with the DBM backend. The SMM objects are
built by the SMM sample, hence its make variables (`WITH_GPU`, `ELEM_TYPE`)
apply. Requires an OpenCL runtime, BLAS, and LIBXS built from a sibling
directory.

## Test

```bash
make test
```

Runs `dbm_bench.x` with `DBM_MULTIPLY_SMM=-1`, which routes the homogeneous
LIBSMM batches into the DBM kernel and validates the result. The arguments of
`dbm_bench.x` are those of `acc_bench.x`.

## CP2K

CP2K compiles `dbm_opencl.c` and generates `dbm_kernels.h` from
`kernels/dbm_multiply.cl`: CMake finds this directory as `LIBXSTREAM_DBM_DIR`
(`find_package(libxstream)`), and the Makefile in CP2K's `src/dbm` finds it
under `LIBXSTREAMROOT`. To build CP2K's DBM miniapp against this directory:

```bash
make miniapp [CP2K_ROOT=/path/to/cp2k]
```

`CP2K_ROOT` defaults to a `cp2k` directory next to LIBXSTREAM, and
`dbm_miniapp.x` is built in `$CP2K_ROOT/src/dbm`. Unlike `make test`, the
miniapp also exercises heterogeneous batches.

## Environment Variables

| Variable            | Default | Description                                                        |
|---------------------|---------|--------------------------------------------------------------------|
| DBM_MULTIPLY_SMM    | 0       | Positive: homogeneous batches up to this size use LIBSMM (1 = 64)  |
|                     |         | Negative: LIBSMM uses the DBM kernel for its homogeneous batches   |
| DBM_MULTIPLY_KERNEL | -       | Path to a `.cl` file that replaces the embedded kernel             |
| DBM_MULTIPLY_FP     | 0       | 1: compute in single precision (data remains double precision)     |
| DBM_MULTIPLY_BN     | auto    | Column tile (1–32), 8 or 2 on NVIDIA                               |
| DBM_MULTIPLY_BK     | 0       | Caps the K-block (1, 4, 8, or 16 derived from the batch's max K)   |
| DBM_MULTIPLY_SM     | 0       | Non-zero: accumulate in shared local memory                        |
| DBM_MULTIPLY_WG     | auto    | Work-group size                                                    |
| DBM_MULTIPLY_LU     | 0       | Unrolling: -2 full, -1 no hints, 0 inner, 1 outer-dehint           |
| DBM_MULTIPLY_RO     | -1      | Read-only operands: negative = auto, 0 = global, 1 = constant      |
| DBM_MULTIPLY_XF     | -1      | Intel 256-GRF mode: negative = device default, 0 = off, 1 = on     |
| DBM_MULTIPLY_NZ     | 0       | Non-zero: skip zero contributions when accumulating into C         |
| DBM_MULTIPLY_LIN    | 0       | Non-zero: swap the access pattern of A and B                       |
| DBM_MULTIPLY_SGB    | 1       | Sub-group broadcast on GPUs (0 = off)                              |
| DBM_MULTIPLY_BLK    | 1       | Sub-group block reads on Intel GPUs (0 = off)                      |

`LIBXSTREAM_VERBOSE=2` (or higher) prints the kernel configuration and the
compilation of each specialization, and values above 2 trace every launch.
