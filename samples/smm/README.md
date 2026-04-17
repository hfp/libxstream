# Small Matrix Multiplication (SMM) -- OpenCL

Batched small matrix multiplications (SMM) on OpenCL devices via the
ACC LIBSMM interface. Originated from the DBCSR OpenCL backend,
adapted to run on top of LIBXSTREAM. Includes GPU-accelerated kernels,
a benchmark driver, and an auto-tuning framework.

## Build

```bash
cd samples/smm
make [WITH_GPU=<device>] [ELEM_TYPE=<type>]
```

Produces `acc_bench.x`. Requires an OpenCL runtime, BLAS (`BLAS=2`),
and LIBXS built from a sibling directory.

| Make variable | Default  | Description                                                      |
|---------------|----------|------------------------------------------------------------------|
| ELEM_TYPE     | `double` | Element precision: `double` or `float`                           |
| WITH_GPU      | auto     | Device for tuned parameters (PVC, A100, ...). Fallback: all CSVs |

## Benchmark Driver

```bash
./acc_bench.x [nrepeat [batchsize [M [N [K [nc [na [nb]]]]]]]]
```

The kernel shape can also be given as `MxNxK`:

```bash
./acc_bench.x 5 30000 13x5x7
```

The first argument can be a file with one set of parameters per line.
The batchsize argument accepts K/M/G suffixes for memory-budget mode.

| Argument  | Default      | Description                            |
|-----------|--------------|----------------------------------------|
| nrepeat   | 66 (3+CHECK) | Number of timed repetitions            |
| batchsize | 30000        | Matrix products per batch (stack size) |
| M         | 23           | Rows of A and C                        |
| N         | M            | Columns of B and C                     |
| K         | M            | Columns of A / rows of B               |
| nc        | batchsize/16 | Number of unique C-matrices            |
| na        | 10*nc        | Number of unique A-matrices            |
| nb        | 10*nc        | Number of unique B-matrices            |

### Environment Variables (Benchmark)

| Variable      | Default | Description                                               |
|---------------|---------|-----------------------------------------------------------|
| CHECK         | -1      | Accuracy: negative=auto-threshold, 0=off, positive=custom |
| CHECK_H2D     | -       | Minimum H2D bandwidth (GB/s); fail if below               |
| CHECK_DEV     | -       | Minimum device GFLOPS/s; fail if below                    |
| CHECK_HST     | -       | Minimum host GFLOPS/s; fail if below                      |
| DEVICE        | 0       | Device index (multi-rank: device = rank % ndevices)       |
| NREPEAT_H2D   | 1       | H2D copy repetitions for bandwidth measurement            |
| NREPEAT_SMM   | 1       | SMM kernel launches per timed iteration (for profiling)   |
| BATCHSIZE_SMM | -       | Override batchsize (and optionally nrepeat: `bs,nrep`)    |

## LIBSMM Kernel Parameters

OpenCL kernels are generated at runtime for any (M, N, K) shape
within MAX_KERNEL_DIM. Tuned parameters can be embedded at build time
or loaded at runtime for better performance.

### Environment Variables (LIBSMM)

Transpose kernel:

| Variable                      | Default | Description                             |
|-------------------------------|---------|-----------------------------------------|
| OPENCL_LIBSMM_TRANS_BUILDOPTS | -       | Extra OpenCL build options              |
| OPENCL_LIBSMM_TRANS_INPLACE   | 0       | Non-zero: in-place transpose (no LDS)   |
| OPENCL_LIBSMM_TRANS_BM        | auto    | Block size in M-direction (0 < BM <= M) |

Multiply kernel:

| Variable                    | Default | Description                              |
|-----------------------------|---------|------------------------------------------|
| OPENCL_LIBSMM_SMM_BUILDOPTS | -       | Extra OpenCL build options               |
| OPENCL_LIBSMM_SMM_PARAMS    | auto    | Tuned parameters: 0=disable, or CSV path |
| OPENCL_LIBSMM_SMM_BS        | auto    | Intra-kernel mini-batchsize              |
| OPENCL_LIBSMM_SMM_BM        | auto    | Block size in M-direction (0 < BM <= M)  |
| OPENCL_LIBSMM_SMM_BN        | auto    | Block size in N-direction (0 < BN <= N)  |
| OPENCL_LIBSMM_SMM_AP        | auto    | Access pattern for parameter/stack array |
| OPENCL_LIBSMM_SMM_AA        | auto    | Access pattern for A-matrix array        |
| OPENCL_LIBSMM_SMM_AB        | auto    | Access pattern for B-matrix array        |
| OPENCL_LIBSMM_SMM_AC        | auto    | Access pattern for C-matrix array        |

The full list of tunable parameters is available via
`tune_multiply.py --help`. Some parameters produce distinct code paths
(e.g., SMM_BS=1 vs SMM_BS=2), so the parameter space is non-smooth.

## Tuned Parameters

Pre-tuned parameter sets in `params/`:

| File                    | Device                          |
|-------------------------|---------------------------------|
| tune_multiply_PVC.csv   | Intel Data Center GPU Max (PVC) |
| tune_multiply_BMG.csv   | Intel Battlemage (BMG)          |
| tune_multiply_A100.csv  | NVIDIA A100                     |
| tune_multiply_H100.csv  | NVIDIA H100                     |
| tune_multiply_GH200.csv | NVIDIA GH200                    |
| tune_multiply_V100.csv  | NVIDIA V100                     |
| tune_multiply_P100.csv  | NVIDIA P100                     |
| tune_multiply_Mi250.csv | AMD MI250                       |

Parameters are matched by device ID at runtime with best-match
fallback. Single and double precision coexist in the same CSV.

### Loading Parameters

```bash
# Use embedded parameters (default)
./acc_bench.x 5 30000 13 5 7

# Disable tuned parameters
OPENCL_LIBSMM_SMM_PARAMS=0 ./acc_bench.x 5 30000 13 5 7

# Load from a specific CSV file
OPENCL_LIBSMM_SMM_PARAMS=params/tune_multiply_PVC.csv ./acc_bench.x 5 30000 13 5 7
```

Rebuild with a specific device's parameters embedded:

```bash
make realclean
make WITH_GPU=PVC
```

## Auto Tuning

Uses OpenTuner to explore the parameter space per (M, N, K) kernel.
The tuner communicates with the benchmark driver through environment
variables.

### Setup

```bash
cd samples/smm
pip install -r requirements.txt
```

### Tuning a Single Kernel

```bash
./tune_multiply.py 13x5x7
```

Use `--stop-after=N` to limit to N seconds. Tuning can be interrupted
with Ctrl-C and results are still written. Delete the `opentuner.db`
directory before tuning a different kernel (`tune_multiply.sh` does
this automatically).

### Tuning Multiple Kernels

```bash
./tune_multiply.sh -t 300 -j 8 -i 1  4 10 15, 6 7 8, 23
```

Triplets are comma-separated groups expanded as a Cartesian product.
The example above expands to 55 kernels.

| Option          | Description                                       |
|-----------------|---------------------------------------------------|
| -t seconds      | Time limit per kernel (default: 160s for updates) |
| -j parts        | Total partitions for distributed tuning           |
| -i index        | Partition index to process (1-based)              |
| -c / --continue | Continue after Ctrl-C instead of stopping         |
| -u / --update   | Re-tune existing JSON files                       |
| -p path         | Directory containing JSON files to update         |
| -a level        | Tuning level (default: 1 for updates)             |

### Bulk Tuning with MPI

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

JSON files are merged into CSV by `tune_multiply.py -m`. Device names
in JSON can be updated with `tune_multiply.py -u`. To retune:

```bash
make realclean
make WITH_GPU=P100
smm/tune_multiply.sh -p smm/params/p100 -u
cp tune_multiply.csv smm/params/tune_multiply_P100.csv
```

Keep GPU driver state persistent during tuning (e.g.,
`nvidia-smi -pm ENABLED` on headless NVIDIA systems).
