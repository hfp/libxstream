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
[dbm_opencl.h](https://github.com/hfp/libxstream/blob/main/samples/dbm/dbm_opencl.h).

## Build

```bash
cd samples/dbm
make
```

Produces `dbm_bench.x`, which is the benchmark driver of the
[SMM sample](https://github.com/hfp/libxstream/blob/main/samples/smm/README.md)
linked with the DBM backend. The SMM objects are built by the SMM sample,
hence its make variables (`WITH_GPU`, `ELEM_TYPE`) apply. Requires an OpenCL runtime, BLAS, and LIBXS built from a sibling
directory.

## Test

```bash
make test
```

Runs `dbm_bench.x` with `DBM_MULTIPLY_SMM=-1`, which routes the homogeneous
LIBSMM batches into the DBM kernel and validates the result. The test fails if
the DBM kernel did not run. The arguments of `dbm_bench.x` are those of
`acc_bench.x` and can be given as `make test TESTARG="5 3000 13x5x7"`.

## CP2K

CP2K builds the backend into its own targets without naming its files: CMake
calls `libxstream_add_dbm(<target>)` after `find_package(libxstream)`, and the
Makefile in CP2K's `src/dbm` includes `dbm.mk` from this directory. Both
generate `dbm_kernels.h` into the consumer's build tree. To build CP2K's DBM
miniapp against this directory:

```bash
make miniapp CP2K_ROOT=/path/to/cp2k
```

`CP2K_ROOT` is optional and defaults to a `cp2k` directory next to
LIBXSTREAM, and `dbm_miniapp.x` is built in `$CP2K_ROOT/src/dbm`. Unlike
`make test`, the miniapp also exercises heterogeneous batches. `SMM=1`
links LIBSMM from the SMM sample, which `DBM_MULTIPLY_SMM` needs.

## Environment Variables

| Variable            | Default | Description                                                        |
|---------------------|---------|--------------------------------------------------------------------|
| DBM_MULTIPLY_SMM    | 0       | Positive: homogeneous batches with M, N, and K up to this value use LIBSMM (1 = 64, at most 127). Negative: LIBSMM (DBCSR) uses the DBM kernel for homogeneous batches (backward usage, experimental) |
| DBM_MULTIPLY_SORT   | 0       | Order of tasks in a batch: 0 as generated, 1 by shape              |
| DBM_MULTIPLY_KERNEL | -       | Path to a `.cl` file that replaces the embedded kernel; the embedded kernel is used if the file cannot be opened |
| DBM_MULTIPLY_FP     | 0       | 1: compute in single precision (data remains double precision)     |
| DBM_MULTIPLY_BN     | auto    | Column tile (1–32): 8 (2 on NVIDIA), or that times twice SM if SM is non-zero |
| DBM_MULTIPLY_BK     | 0       | Caps the K-block of mixed shapes (1, 4, 8, or 16 derived from the batch's max K); homogeneous batches use their exact K |
| DBM_MULTIPLY_SM     | 0       | Non-zero: accumulate in shared local memory                        |
| DBM_MULTIPLY_WG     | auto    | Work-group size                                                    |
| DBM_MULTIPLY_LU     | 0       | Unrolling: -2 full, -1 no hints, 0 inner, 1 outer-dehint           |
| DBM_MULTIPLY_RO     | -1      | Read-only operands: negative = auto, 0 = global, 1 = constant      |
| DBM_MULTIPLY_XF     | -1      | Intel 256-GRF mode: negative = device default, 0 = off, 1 = on     |
| DBM_MULTIPLY_NZ     | 0       | Non-zero: skip zero contributions when accumulating into C         |
| DBM_MULTIPLY_LIN    | 0       | Non-zero: swap the access pattern of A and B                       |
| DBM_MULTIPLY_SGB    | auto    | Sub-group broadcast: on by default on Intel GPUs (0 = off), opt-in on NVIDIA (1) |
| DBM_MULTIPLY_BLK    | auto    | Row per lane (1, needs LIN=0) for mixed shapes, on Intel GPUs also for homogeneous batches; 2 adds homogeneous batches elsewhere, 0 = off |
| DBM_MULTIPLY_FUSE   | auto    | Tasks per work-group fusing those sharing A: 16 with BLK, else 1   |

`LIBXSTREAM_VERBOSE=2` (or higher) prints the kernel configuration, the
compilation of each specialization, and a summary at exit; values above 2
or negative values also trace every launch.
