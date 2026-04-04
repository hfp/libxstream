# Small Matrix Multiplication (SMM) - OpenCL

This sample implements the ACC LIBSMM interface for batched small matrix multiplications (SMM) on OpenCL devices. It originates from [DBCSR](https://github.com/cp2k/dbcsr)'s OpenCL backend and has been adapted to run on top of LIBXSTREAM. The implementation includes GPU-accelerated kernels for batched matrix multiplication and transpose, a benchmark driver (`acc_bench`), and a complete auto-tuning framework.

## Build

```bash
cd samples/smm
make [WITH_GPU=<device>] [ELEM_TYPE=<type>]
```

Produces `acc_bench.x`. Requires an OpenCL runtime, BLAS (linked via `BLAS=2`), and LIBXS built from a sibling directory.

| Make variable | Default | Description |
|---------------|---------|-------------|
| `ELEM_TYPE` | `double` | Element precision: `double` or `float`. |
| `WITH_GPU` | auto | Device name for selecting tuned parameters, e.g., `PVC`, `A100`, `V100`. Falls back to all CSV files in `params/`. |

## Benchmark Driver

```
./acc_bench.x [nrepeat [batchsize [M [N [K [nc [na [nb]]]]]]]]
```

The kernel shape can also be given as `MxNxK`:

```
./acc_bench.x 5 30000 13x5x7
```

Alternatively, the first argument can be a file containing one set of parameters per line (for batch testing multiple kernels).

| Argument | Default | Description |
|----------|---------|-------------|
| `nrepeat` | 66 (or 3 with `CHECK`) | Number of timed repetitions |
| `batchsize` | 30000 | Number of matrix products per batch (the "stack size") |
| `M` | 23 | Rows of A and C |
| `N` | M | Columns of B and C |
| `K` | M | Columns of A / rows of B |
| `nc` | batchsize/16 | Number of unique C-matrices |
| `na` | 10*nc | Number of unique A-matrices |
| `nb` | 10*nc | Number of unique B-matrices |

The batchsize argument also accepts memory-budget notation with K/M/G suffixes (e.g., `256M`), in which case the stack size is derived from the memory budget and matrix dimensions.

### Environment Variables (Benchmark)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECK` | -1 (enabled) | Accuracy validation: negative = auto-threshold, 0 = disabled, positive = custom threshold. |
| `CHECK_H2D` | - | Minimum H2D bandwidth (GB/s); fail if below. |
| `CHECK_DEV` | - | Minimum device GFLOPS/s; fail if below. |
| `CHECK_HST` | - | Minimum host GFLOPS/s; fail if below. |
| `DEVICE` | 0 | Device index (with multiple ranks, device = rank % ndevices). |
| `NREPEAT_H2D` | 1 | Number of H2D copy repetitions for bandwidth measurement. |
| `NREPEAT_SMM` | 1 | Number of SMM kernel launches per timed iteration (for profiling). |
| `BATCHSIZE_SMM` | - | Override batchsize and optionally nrepeat (`batchsize,nrepeat`). |

### Compile-Time Knobs (Benchmark)

| Macro | Default | Description |
|-------|---------|-------------|
| `ELEM_TYPE` | `double` | Element type (`double` or `float`). |
| `BATCHSIZE` | 30000 | Default batch/stack size. |
| `BATCHGRAIN` | 100 | Granularity for randomized batch sizes. |
| `NREPEAT` | 3 | Default repetitions (with `CHECK`). |
| `XREPEAT` | 66 | Default repetitions (without `CHECK`). |
| `WARMUP` | 2 | Number of untimed warmup iterations. |
| `MAX_KERNEL_DIM` | 80 | Maximum supported matrix dimension. |
| `ALIGNMENT` | `LIBXS_ALIGNMENT` | Buffer alignment in bytes. |

## LIBSMM Kernel Parameters

The LIBSMM library generates OpenCL kernels at runtime for any requested (M, N, K) shape within `MAX_KERNEL_DIM`. Performance can be improved by supplying tuned parameters, either embedded at build time or loaded at runtime.

### Environment Variables (LIBSMM)

**Transpose kernel:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCL_LIBSMM_TRANS_BUILDOPTS` | - | Extra OpenCL build options for transpose kernels. |
| `OPENCL_LIBSMM_TRANS_INPLACE` | 0 | Non-zero: in-place transpose (no local memory). |
| `OPENCL_LIBSMM_TRANS_BM` | auto | Block size in M-direction (0 < BM <= M). |

**Multiply kernel:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCL_LIBSMM_SMM_BUILDOPTS` | - | Extra OpenCL build options for multiply kernels. |
| `OPENCL_LIBSMM_SMM_PARAMS` | auto | Tuned parameters: `0` = disable, or path to CSV file. |
| `OPENCL_LIBSMM_SMM_BS` | auto | Intra-kernel mini-batchsize for amortizing atomic updates. |
| `OPENCL_LIBSMM_SMM_BM` | auto | Block size in M-direction (0 < BM <= M). |
| `OPENCL_LIBSMM_SMM_BN` | auto | Block size in N-direction (0 < BN <= N). |
| `OPENCL_LIBSMM_SMM_AP` | auto | Access pattern for the parameter/stack array. |
| `OPENCL_LIBSMM_SMM_AA` | auto | Access pattern for the A-matrix array. |
| `OPENCL_LIBSMM_SMM_AB` | auto | Access pattern for the B-matrix array. |
| `OPENCL_LIBSMM_SMM_AC` | auto | Access pattern for the C-matrix array. |

The full list of tunable parameters and their accepted values is available via `tune_multiply.py --help`.

**NOTE**: Some parameters produce distinct code paths (e.g., `OPENCL_LIBSMM_SMM_BS=1` vs. `OPENCL_LIBSMM_SMM_BS=2`), so the parameter space is non-smooth.

## Tuned Parameters

Pre-tuned parameter sets for various GPUs are stored as CSV files in the `params/` directory:

| File | Device |
|------|--------|
| `tune_multiply_PVC.csv` | Intel Data Center GPU Max (PVC) |
| `tune_multiply_BMG.csv` | Intel Battlemage (BMG) |
| `tune_multiply_A100.csv` | NVIDIA A100 |
| `tune_multiply_H100.csv` | NVIDIA H100 |
| `tune_multiply_GH200.csv` | NVIDIA GH200 |
| `tune_multiply_V100.csv` | NVIDIA V100 |
| `tune_multiply_P100.csv` | NVIDIA P100 |
| `tune_multiply_Mi250.csv` | AMD MI250 |

Parameters are selected by matching the device ID at runtime. When no match is found, the best-matching set is used as fallback. Tuned parameters for single and double precision can coexist in the same CSV and are distinguished automatically.

### Loading Parameters

Parameters are embedded into the binary at build time (via `tool_opencl.sh`). They can also be overridden at runtime:

```bash
# Use embedded parameters (default)
./acc_bench.x 5 30000 13 5 7

# Disable tuned parameters (use defaults)
OPENCL_LIBSMM_SMM_PARAMS=0 ./acc_bench.x 5 30000 13 5 7

# Load parameters from a specific CSV file
OPENCL_LIBSMM_SMM_PARAMS=params/tune_multiply_PVC.csv ./acc_bench.x 5 30000 13 5 7
```

To rebuild with a specific device's parameters embedded:

```bash
make realclean
make WITH_GPU=PVC
```

## Auto Tuning

The auto-tuning framework uses [OpenTuner](http://opentuner.org/) to explore the parameter space for each (M, N, K) kernel. The tuner communicates with the benchmark driver solely through environment variables.

### Setup

```bash
cd samples/smm
pip install -r requirements.txt
```

### Tuning a Single Kernel

```bash
./tune_multiply.py 13x5x7
```

The script accepts `--stop-after=N` to limit tuning to N seconds. Without a limit, OpenTuner decides when to stop. Tuning can be interrupted with Ctrl-C and results are still written.

Output is a JSON file encoding the benchmark, precision, kernel shape, problem size, and achieved GFLOPS/s, e.g., `tune_multiply-float-12x12x12-s15-60gflops.json`.

**NOTE**: Delete the `opentuner.db` directory before tuning a different kernel to avoid cross-contamination (`tune_multiply.sh` does this automatically).

If an environment variable for a parameter is already set (e.g., `OPENCL_LIBSMM_SMM_BM`), that parameter is held fixed and excluded from tuning, allowing directed exploration of the remaining parameter space.

### Tuning Multiple Kernels

The `tune_multiply.sh` wrapper script handles multiple kernels via triplet specifications:

```bash
./tune_multiply.sh -t 300 -j 8 -i 1  4 10 15, 6 7 8, 23
```

Triplets are comma-separated groups of (M, N, K)-extents. Each group is expanded as a Cartesian product, groups are concatenated, and duplicates are removed. The example above expands to 55 kernels.

| Option | Description |
|--------|-------------|
| `-t <seconds>` | Time limit per kernel (default: 160s for updates). |
| `-j <parts>` | Total number of partitions for distributed tuning. |
| `-i <index>` | Partition index to process (1-based). |
| `-c` / `--continue` | Continue after Ctrl-C instead of stopping all kernels. |
| `-u` / `--update` | Re-tune existing JSON files (parse M,N,K from filenames). |
| `-p <path>` | Directory containing JSON files to update. |
| `-a <level>` | Tuning level (default: 1 for updates). |

The script tunes 1266 default kernels. With `-t 300 -j 8`, each partition takes approximately 13 hours.

### Bulk Tuning with MPI

Multiple devices can be used in parallel by launching `tune_multiply.sh` under MPI (one process per device):

```bash
MAXTIME=200 NPARTS=8 UPDATE=1 JSONDIR=params/pvc mpirun \
  ./tune_multiply.sh -i 1 : \
  ./tune_multiply.sh -i 2 : \
  ./tune_multiply.sh -i 3 : \
  ./tune_multiply.sh -i 4 : \
  ./tune_multiply.sh -i 5 : \
  ./tune_multiply.sh -i 6 : \
  ./tune_multiply.sh -i 7 : \
  ./tune_multiply.sh -i 8 \
>out.log 2>&1
```

### Managing Tuned Parameters

JSON files are automatically merged into a CSV file by `tune_multiply.py -m`. The device name in JSON files can be updated to match the current driver version with `tune_multiply.py -u`. Different problem sizes are collapsed to a maximum, so tuning for a non-default problem size requires care to avoid dominance by larger results.

To retune an existing parameter set:

```bash
make realclean
make WITH_GPU=P100                          # embed current parameters
smm/tune_multiply.sh -p smm/params/p100 -u # retune from embedded baseline
cp tune_multiply.csv smm/params/tune_multiply_P100.csv
```

**NOTE**: Keep the GPU driver state persistent during tuning to avoid repeated initialization overhead (e.g., `nvidia-smi -pm ENABLED` on headless Nvidia systems).

## Files

| File | Description |
|------|-------------|
| `acc_bench.c` | Benchmark driver: batched SMM with H2D, device, and host timing. |
| `acc_bench.h` | Stack initialization and matrix setup helpers. |
| `acc_libsmm.h` | ACC LIBSMM interface definition (transpose + multiply). |
| `opencl_libsmm.c` | OpenCL LIBSMM implementation: kernel generation, parameter handling, dispatch. |
| `opencl_libsmm.h` | LIBSMM internal types: kernel configs, tunables, parameter I/O. |
| `kernels/multiply.cl` | OpenCL C multiply kernel. |
| `kernels/transpose.cl` | OpenCL C transpose kernel. |
| `tune_multiply.py` | OpenTuner-based auto-tuning script for multiply kernels. |
| `tune_multiply.sh` | Wrapper: triplet expansion, partitioning, bulk/MPI tuning. |
| `acc_triplets.sh` | Triplet generation helper. |
| `params/*.csv` | Pre-tuned parameter sets per device. |
