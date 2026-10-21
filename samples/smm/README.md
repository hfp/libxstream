# Small Matrix Multiplication (SMM) — OpenCL

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
and LIBXS in a sibling directory (header-only unless a LIBXS library
is built).

| Make variable | Default  | Description                                                |
|---------------|----------|------------------------------------------------------------|
| ELEM_TYPE     | `double` | Element precision: `double` or `float`                     |
| WITH_GPU      | `GPUVER` | Device for tuned parameters (PVC, A100, ...), see below    |
| SYM           | 0        | Non-zero also keeps comments in the embedded kernel source |

`make test` builds and runs `acc_bench.x` with its defaults.

The build embeds `params/tune_multiply_<WITH_GPU>.csv`. Without such a
file, it embeds `tune_multiply.csv` from this directory if present
(the default output of a merge), and otherwise every `params/*.csv`
unless `WITH_GPU` was given on the command line.

A CMake project (e.g., DBCSR) builds LIBSMM into its own target with
`libxstream_add_smm(<target>)` after `find_package(libxstream)`, which also
generates `smm_kernels.h` into the target's build tree.

A Makefile project (e.g., CP2K's DBM miniapp) includes `smm.mk` instead,
optionally setting `LIBXSTREAM_SMM_GENDIR` (directory of the generated
`smm_kernels.h`, default: current directory) and `LIBXSTREAM_SMM_GENARG`
(e.g., `-d` to keep comments). It then adds `LIBXSTREAM_SMM_IFLAGS` to
its compiler flags, `LIBXSTREAM_SMM_OBJS` to its objects, and builds them
from `$(LIBXSTREAM_SMM_DIR)/%.c` with `$(LIBXSTREAM_SMM_HEADERS)` as
prerequisites.

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
A negative batchsize, e.g., -30000, draws a random multiple of 100 up to
its magnitude (also via `BATCHSIZE_SMM`).

| Argument  | Default         | Description                            |
|-----------|-----------------|----------------------------------------|
| nrepeat   | 3 (CHECK=0: 66) | Number of timed repetitions            |
| batchsize | 30000           | Matrix products per batch (stack size) |
| M         | 23              | Rows of A and C                        |
| N         | M               | Columns of B and C                     |
| K         | M               | Columns of A / rows of B               |
| nc        | batchsize/16    | Number of unique C-matrices            |
| na        | 10*nc           | Number of unique A-matrices            |
| nb        | 10*nc           | Number of unique B-matrices            |

### Environment Variables (Benchmark)

| Variable      | Default | Description                                                |
|---------------|---------|------------------------------------------------------------|
| CHECK         | 1       | Tolerance: CHECK × 1E-3 (float: 2E-3), sign ignored, 0=off |
| CHECK_H2D     | -       | Minimum H2D bandwidth (GB/s); fail if below                |
| CHECK_DEV     | -       | Minimum device GFLOPS/s; fail if below                     |
| CHECK_HST     | -       | Minimum host GFLOPS/s; fail if below                       |
| DEVICE        | 0       | Device index (multi-rank: device = rank % ndevices)        |
| NREPEAT_H2D   | 1       | H2D copy repetitions for bandwidth measurement             |
| NREPEAT_SMM   | 1       | Repeats the batch inside each SMM kernel (for profiling)   |
| BATCHSIZE_SMM | -       | Override batchsize (and optionally nrepeat: `bs,nrep`)     |
| UNIFIED       | 0       | 1: kernels read the stack from host memory if possible     |

## LIBSMM Kernel Parameters

OpenCL kernels are generated at runtime for any (M, N, K) shape with
M·N ≤ MAX_KERNEL_DIM² (the caller's limit; `acc_bench.x` uses 80).
Tuned parameters can be embedded at build time or loaded at runtime for
better performance. `LIBXSTREAM_DEVMATCH` (which device's parameters are
taken) and `LIBXSTREAM_VERBOSE` (INFO output about loaded parameters and
generated kernels) are described on the LIBXSTREAM OpenCL page.

### Environment Variables (LIBSMM)

Transpose kernel:

| Variable                      | Default | Description                             |
|-------------------------------|---------|-----------------------------------------|
| OPENCL_LIBSMM_TRANS_BUILDOPTS | -       | Extra OpenCL build options              |
| OPENCL_LIBSMM_TRANS_INPLACE   | 0       | Non-zero: in-place if square (no LDS)   |
| OPENCL_LIBSMM_TRANS_BM        | auto    | Block size in M-direction (0 < BM <= M) |
| OPENCL_LIBSMM_TRANS_BS        | 8       | Matrices per work-group                 |
| OPENCL_LIBSMM_TRANS_PARAMS    | -       | 0: ignore TRANS_BM and TRANS_BS         |
| OPENCL_LIBSMM_TRANS_PREDICT   | -       | Positive: ignore TRANS_BM and TRANS_BS  |

Multiply kernel:

| Variable                       | Default  | Description                                                          |
|--------------------------------|----------|----------------------------------------------------------------------|
| OPENCL_LIBSMM_SMM_BUILDOPTS    | -        | Extra OpenCL build options                                           |
| OPENCL_LIBSMM_SMM_KERNEL       | -        | OpenCL source file replacing the embedded multiply kernel            |
| OPENCL_LIBSMM_SMM_PARAMS       | auto     | CSV path, a parameter row, or 0 for plain defaults (see below)       |
| OPENCL_LIBSMM_SMM_PREDICT      | 0        | Model: -1=off (default if PARAMS=0), 0=untuned shapes, 1=all shapes  |
| OPENCL_LIBSMM_SMM_PREDICT_CONF | 0.5      | Confidence a predicted parameter needs; below, its default is used   |
| OPENCL_LIBSMM_SMM_DEVID        | -        | Device ID whose parameters are used, e.g., `0x0bd5`                  |
| OPENCL_LIBSMM_SMM_S            | 64       | BS > 1 needs a batch of at least BS×S; negative: always BS=1         |
| OPENCL_LIBSMM_SMM_BS           | auto     | Intra-kernel mini-batchsize                                          |
| OPENCL_LIBSMM_SMM_BM           | auto     | Block size in M-direction (0 < BM <= M), ignored if LU > 1           |
| OPENCL_LIBSMM_SMM_BN           | auto     | Block size in N-direction (0 < BN <= N)                              |
| OPENCL_LIBSMM_SMM_BK           | auto     | Block size BK (0 < BK <= M)                                          |
| OPENCL_LIBSMM_SMM_WS           | max(M,N) | Minimum work-group size (WS <= M·N)                                  |
| OPENCL_LIBSMM_SMM_WG           | auto     | Work-group size: 0=exact, non-zero=sub-group multiple, 2=power of 2  |
| OPENCL_LIBSMM_SMM_LU           | auto     | Unrolling: -2=full, -1=no hints, 0=inner, 1=outer, >1=LU blocks of M |
| OPENCL_LIBSMM_SMM_NZ           | 0        | 1: skip atomic updates of zero values                                |
| OPENCL_LIBSMM_SMM_AL           | 0        | Ignored in this build                                                |
| OPENCL_LIBSMM_SMM_TB           | 0        | 1: load B (AB=2) only when it changes within a mini-batch            |
| OPENCL_LIBSMM_SMM_TC           | 1        | 1: write C back only when it changes                                 |
| OPENCL_LIBSMM_SMM_AP           | 0        | Parameter stack: 0=global, 1=shared                                  |
| OPENCL_LIBSMM_SMM_AA           | auto     | A-matrix: 0=global, 1=shared, 2=register                             |
| OPENCL_LIBSMM_SMM_AB           | auto     | B-matrix: 0=global, 1=shared, 2=register                             |
| OPENCL_LIBSMM_SMM_AC           | 0        | C-matrix: 0=register, 1=shared                                       |
| OPENCL_LIBSMM_SMM_XF           | 0        | 1: 256 GRF per thread on Intel GPUs (see below)                      |

Defaults apply where no tuned or predicted parameter exists.
`OPENCL_LIBSMM_SMM_XF` defaults to 1 if `OPENCL_LIBSMM_SMM_BUILDOPTS`
contains `-cl-intel-256-GRF-per-thread`. A value of
`OPENCL_LIBSMM_SMM_PREDICT` above 1 also selects a LIBXS prediction mode.

`OPENCL_LIBSMM_SMM_PARAMS=0` runs with plain defaults: no tuned table, no
prediction (unless `OPENCL_LIBSMM_SMM_PREDICT` is set), and the two-letter
variables above (`OPENCL_LIBSMM_SMM_BS` to `_XF`) are ignored, as they are
with a positive `OPENCL_LIBSMM_SMM_PREDICT`. In the default build, whose
tuned parameters come from the models, `OPENCL_LIBSMM_SMM_PREDICT=-1` alone
already drops all tuned parameters.

`OPENCL_LIBSMM_SMM_PARAMS` given a CSV file applies its parameters to any
device (device matching is off). Instead of a file, it also takes one
parameter row: the CSV columns without DEVICE, S, and GFLOPS, e.g.,
`3;23;23;23;8;23;1;1;23;0;-1;0;0;0;1;0;2;0;0` (TYPEID;M;N;K;BS;...;AC).

The full list of tunable parameters is available via
`tune_multiply.py --help`.

## Tuned Parameters

Pre-tuned parameter sets in `params/`:

| File                       | Device                          |
|----------------------------|---------------------------------|
| tune_multiply_PVC.csv      | Intel Data Center GPU Max (PVC) |
| tune_multiply_BMG.csv      | Intel Battlemage (BMG)          |
| tune_multiply_A100.csv     | NVIDIA A100                     |
| tune_multiply_H100-PCI.csv | NVIDIA H100 PCIe                |
| tune_multiply_H100-SXM.csv | NVIDIA H100 SXM                 |
| tune_multiply_GH200.csv    | NVIDIA GH200                    |
| tune_multiply_V100.csv     | NVIDIA V100                     |
| tune_multiply_P100.csv     | NVIDIA P100                     |
| tune_multiply_Mi250.csv    | AMD MI250                       |

Parameters are matched by device ID at runtime with best-match
fallback. The CSV format takes single and double precision in the same
file, and the shipped files hold double precision only.

Each CSV file comes with a prediction model of the same name (`.bin`),
built from its double-precision entries; predictions apply to double
precision only, other types use the defaults. If every CSV file of a build has a model, the models are
embedded instead of the CSV rows. `smm_params.sh` rebuilds the model
after each merge, using `predict_params.x` from LIBXS (`samples/predict`,
built on demand) found via `LIBXSROOT`, a sibling `libxs`, or
`$HOME/libxs`. Without LIBXS, the merge reports the model as stale. To
rebuild a model by hand:

```bash
predict_params.x hknn params/tune_multiply_PVC.csv params/tune_multiply_PVC.bin
```

### Loading Parameters

```bash
# Use embedded parameters (default)
./acc_bench.x 5 30000 13 5 7

# Plain defaults
OPENCL_LIBSMM_SMM_PARAMS=0 ./acc_bench.x 5 30000 13 5 7

# Load from a specific CSV file
OPENCL_LIBSMM_SMM_PARAMS=params/tune_multiply_PVC.csv ./acc_bench.x 5 30000 13 5 7
```

Rebuild with a specific device's parameters embedded:

```bash
make realclean
make WITH_GPU=PVC
```

Show which embedded parameter set a device name selects:

```bash
make match
./acc_match.x "NVIDIA H200"
```

Names given after the device are compared instead of the embedded ones:

```bash
./acc_match.x "NVIDIA H200" "NVIDIA H100 PCIe" "NVIDIA GH200 480GB"
```

## Auto Tuning

Uses OpenTuner to explore the parameter space per (M, N, K) kernel.

### Setup

```bash
cd samples/smm
pip install -r requirements.txt
```

Build `acc_bench.x` before tuning. Keep GPU clocks and driver state as
stable as practical during a run.

### tune_multiply.py

Use `tune_multiply.py` when you want direct control over one tuning
run or over JSON maintenance.

Tune one kernel and write JSON results in the current directory:

```bash
./tune_multiply.py 13x5x7
```

Limit the search time, choose the JSON directory, and set the benchmark
batch size:

```bash
mkdir -p params/local
./tune_multiply.py 13x5x7 --stop-after=300 -p params/local -s 30000
```

Tune several explicit kernels from a file, one `MxNxK` per line. A `#`
starts a comment, so a line can be annotated or taken out of the list:

```bash
printf '%s\n' 13x5x7 23x23x23 32x32x32 > kernels.txt
./tune_multiply.py kernels.txt --stop-after=180 -p params/local
```

Merge JSON files into a CSV file (without a shape, `-p` or `--prefer`
alone merges as well):

```bash
./tune_multiply.py -m -p params/local -o tune_multiply_local.csv
```

`-m` alone (or a merge without a shape) merges the precision that
`acc_bench.x` was built for and runs it to find out, which needs a device.
`-m2` (double), `-m1` (single), and `-m0` (all) merge without one.

A JSON file is named after its kernel and a hash of its parameters, for
example `tune_multiply-double-23x23x23-1a2b3c4d.json`, with `-s<N>` after
the shape if the batch size is not the default 30000 (`N` is the base-2
logarithm of the batch size, rounded up). Keep one directory per
device: a same-named file from another device is reported and left
alone.

Update the device name stored in every JSON file of a directory, either
as given or, without a name, as the device reports it (which requires
running on the target machine):

```bash
./tune_multiply.py -u "Some Device [0x1234]" -p params/local
./tune_multiply.py -u -p params/local
```

Write the JSON files a CSV file was merged from (the inverse of `-m`),
which needs neither a device nor the benchmark:

```bash
./tune_multiply.py -M params/tune_multiply_PVC.csv -p tmp/PVC
```

A file that exists already is left unchanged.

Check existing JSONs without re-tuning:

```bash
./tune_multiply.py -c -p params/local
```

When two JSON files describe the same kernel on the same device,
`--prefer` decides which one wins the CSV: `fast` (default) keeps the
higher GFLOPS/s, and `new` keeps the more recent measurement (e.g., after
a driver change). A merge lists the losing files (`Remove N (use -d)`),
and `-d` removes them:

```bash
./tune_multiply.py -p params/local --prefer new
./tune_multiply.py -p params/local --prefer new -d
```

Useful options:

| Option              | Meaning                                      |
|---------------------|----------------------------------------------|
| --stop-after N      | Stop the search after N seconds              |
| -p path             | JSON directory; without a shape, merge it    |
| -s size             | Benchmark batch size, also called stack size |
| -a level            | Tuning level: 0=all ... 4=least tunables     |
| -m [mode]           | Merge JSON files into a CSV file (see above) |
| -M file             | Write the JSON files of a CSV file (see -p)  |
| -o file             | CSV output file                              |
| -u [device]         | Update JSON device names (see above)         |
| -c [epsilon]        | Validate JSON entries                        |
| --prefer fast\|new  | Duplicate winner; without a shape, merge     |
| -d                  | Delete losing duplicates during merge        |
| -x                  | Write GFLOPS as 0 into the CSV file          |
| -e char             | CSV separator (default: `;`)                 |
| -r count            | Repetitions per benchmark run (0: default)   |
| -mb size            | Largest mini-batch size to try (default: 64) |
| --timeout seconds   | Limit per benchmark run (0: none)            |
| -q                  | Skip configurations with AA=1 or AB=1        |
| -v                  | Print the settings of each run               |

Setting `OPENCL_LIBSMM_SMM_<X>` (e.g., `OPENCL_LIBSMM_SMM_BS=8`) fixes
that parameter during tuning. A value of `tune`, `enabled`, or `on`
keeps it tunable unless the tuning level fixes it. `OPENCL_LIBSMM_SMM_WS`
is fixed unless set to `tune`.

If a shared LIBXS library loads (`LIBXS_LIB` names the file or its
directory, otherwise `LIBXSROOT/lib`, a sibling `libxs/lib`, or the
system path), the OpenTuner technique `LibxsBanditA` becomes available
(`--technique=LibxsBanditA`). `LIBXS_SURROGATE_KAPPA` (1.0),
`LIBXS_SURROGATE_CANDIDATES` (512), `LIBXS_SURROGATE_WARMUP` (8), and
`LIBXS_SURROGATE_NOVELTY` (1.0) adjust it.

The tuner can run under MPI. It detects local MPI rank variables and
uses them to select `LIBXSTREAM_DEVICE`, so ranks on the same node can
use different devices. This is most useful when each rank is tuning a
different kernel.

```bash
mpirun \
  ./tune_multiply.py 13x5x7 --stop-after=300 -p params/local : \
  ./tune_multiply.py 23x23x23 --stop-after=300 -p params/local
```

Interrupted tuning writes the best result seen so far. Unless
`--database` names one, the OpenTuner database is kept per process in
the system's temporary directory (`opentuner/<pid>.db`, under MPI
`opentuner<device>/<pid>.db`), so sessions do not share it.

### tune_multiply.sh

Use `tune_multiply.sh` when you want to expand triplet specifications,
retune a directory, or split a larger tuning job into parts.

Tune a compact Cartesian-product specification:

```bash
./tune_multiply.sh -t 300 4 10 15, 6 7 8, 23
```

Triplets are comma-separated groups. The command above expands to all
`MxNxK` combinations with `M` in `4 10 15`, `N` in `6 7 8`, and `K=23`.

Tune an explicit list instead:

```bash
./tune_multiply.sh -t 300 1x1x1 2x2x2 13x5x7 23x23x23
```

Split the same work into four parts and run the second part:

```bash
./tune_multiply.sh -t 300 -j 4 -i 2 4 10 15, 6 7 8, 23
```

Parts also split a long job into sessions on a single device, where
each session repeats the same command with the next part number:

```bash
PART=1  # next session: 2, then 3 and 4
./tune_multiply.sh -t 300 -j 4 -i ${PART} -p params/local 4 10 15, 6 7 8, 23
```

Keep the options that shape the list unchanged between sessions (`-r`,
`-m`, `-n`, `-b`, `-f`, `-k`). With `-u`, do not add new shapes to the
JSON directory between sessions.

Retune every JSON file found in a directory:

```bash
./tune_multiply.sh -u -p params/local -t 300
```

#### Retuning by age

`--plan` lists the kernels of a JSON directory oldest first and writes
that list as a file of triplets, or to standard output if no file is
given. A kernel is dated by its youngest JSON file:

```bash
./tune_multiply.sh --plan retune.txt -p params/local -n 40
./tune_multiply.sh -f retune.txt -p params/local -t 300
```

The first command takes the 40 kernels that were tuned least recently
(`-b` takes the most recent ones instead, and `-r`, `-m`, and `-n` limit
the plan as they limit any other list). The second command works the
plan. Keep the plan file between sessions rather than generating it
again.

Each kernel that tuned successfully is marked `#done` in the plan and is
skipped when the plan is used again, so an interrupted session resumes
where it stopped and a kernel that failed stays in the plan. Only a
generated plan is marked (a handwritten list is left alone), and only in
a single session (`-j 1`). Delete the marks to tune the plan again:

```bash
sed "s/^#done //" retune.txt > retune.tmp && mv retune.tmp retune.txt
```

A plan lists every kernel of the directory. `-n` with `--plan` limits
the plan, and `-n` with `-f` limits a single session, so repeating the
same command works down the plan:

```bash
./tune_multiply.sh --plan retune.txt -p params/local
./tune_multiply.sh -f retune.txt -p params/local -n 2 -t 300
```

Once no unmarked line is left, generate the next plan:

```bash
grep -qv "^#" retune.txt || ./tune_multiply.sh --plan retune.txt -p params/local
./tune_multiply.sh -f retune.txt -p params/local -n 2 -t 300
```

Limit the generated work before splitting it:

```bash
./tune_multiply.sh -t 180 -r 4 32 -m 64 -n 100 \
  4 8 16 32, 4 8 16 32, 4 8 16 32
```

Useful options:

| Option        | Meaning                                            |
|---------------|----------------------------------------------------|
| -t seconds    | Time limit per kernel (default: 160)               |
| -p path       | Directory for JSON files                           |
| -s size       | Benchmark batch size, also called stack size       |
| -a level      | Tuning level: 0=all ... 4=least tunables           |
| -u            | Retune JSON files found under `-p`                 |
| --plan [file] | Write kernels of `-p` by age, oldest first         |
| -d            | Ask the merge step to delete losing JSONs          |
| --prefer new  | Prefer newest duplicate (default: fastest)         |
| -c            | Continue with the next kernel after an error       |
| -b            | Reverse triplets, before `-n` and before parts     |
| -j parts      | Total number of tuning parts                       |
| -i index      | Part to run, using 1-based numbering               |
| -r low high   | Keep kernels with low**3 < M*N*K <= high**3        |
| -m extent     | Keep kernels with M, N, and K no larger than this  |
| -n count      | Keep only the first count kernels (see `-b`)       |
| -f file       | Read MxNxK list from a file, resuming a plan       |
| -k id         | Use a predefined triplet set                       |
| -w seconds    | Delay before tuning starts (default: 12, parts: 0) |
| -h            | Print usage                                        |

With `-c` and without `-t`, there is no time limit per kernel. Option
variables that are already set in the environment (e.g., `PART`,
`NPARTS`, `JSONDIR`, `MAXTIME`, `CONTINUE`) act as if given.

Options the wrapper does not know are passed to `tune_multiply.py`.
Such an option must carry its value as `--opt=value`, since a separate
value would be read as the start of the triplet specification:

```bash
./tune_multiply.sh -t 300 --check=0 23x23x23
```

The tuning level defaults to `-1`, which fixes the same tunables as
level 2. Levels 3 and 4 fix successively more of them.

### Tuning from a File

A text file with one `MxNxK` per line can drive the tuning session.
Lines starting with `#` are comments, and inline comments after `#`
are stripped. Whitespace within an entry is ignored:

```bash
./tune_multiply.sh -t 300 -f retune_shapes.txt -p params/local
```

Under MPI, the file entries are partitioned across ranks as usual:

```bash
mpirun -np 8 ./tune_multiply.sh -t 300 -f retune_shapes.txt -p params/local
```

### Bulk Tuning with MPI

Under MPI, the wrapper defaults `-j` to the MPI world size and `-i` to
`rank + 1`. It also forwards a normalized local rank to
`tune_multiply.py`, so the Python tuner can select a different device
per local rank.

For most MPI launchers, this is enough:

```bash
mpirun -np 8 ./tune_multiply.sh -t 300 -p params/local \
  4 10 15, 6 7 8, 23
```

You can still spell out parts explicitly when the launcher or scheduler
needs it:

```bash
mpirun \
  ./tune_multiply.sh -t 300 -j 4 -i 1 -p params/local \
    4 10 15, 6 7 8, 23 : \
  ./tune_multiply.sh -t 300 -j 4 -i 2 -p params/local \
    4 10 15, 6 7 8, 23 : \
  ./tune_multiply.sh -t 300 -j 4 -i 3 -p params/local \
    4 10 15, 6 7 8, 23 : \
  ./tune_multiply.sh -t 300 -j 4 -i 4 -p params/local \
    4 10 15, 6 7 8, 23 \
>out.log 2>&1
```

To retune an existing JSON directory across eight MPI ranks:

```bash
mpirun -np 8 ./tune_multiply.sh -u -p params/local -t 300
```

### Managing Tuned Parameters

JSON files are the working format (`params/*/` is not tracked), and CSV
files are the deployable format. `smm_params.sh` converts between the
two for every device at once, pairing a directory `params/<device>-0x<id>` with the CSV file
`params/tune_multiply_<device>.csv` (double precision only):

```bash
./smm_params.sh            # merge every device
./smm_params.sh "H100*"    # merge matching devices only
./smm_params.sh -e         # the reverse: write JSONs to tmp/
```

The device id in the directory name must match the id of the merged
parameters and that of the CSV file being replaced. Otherwise the merge
reports both ids and writes nothing. A device without a CSV file needs
`-f`.

A typical retune flow is:

```bash
make realclean
make WITH_GPU=P100
./tune_multiply.sh --plan retune.txt -p params/P100-0x2f6c
./tune_multiply.sh -f retune.txt -p params/P100-0x2f6c -t 300
./smm_params.sh P100
```

The last step also rebuilds the prediction model (see Tuned Parameters).

Keep GPU driver state persistent during tuning (e.g.,
`nvidia-smi -pm ENABLED` on headless NVIDIA systems).

### Starting a New Device

A new device can be seeded with the parameters of a similar one (e.g.,
the previous generation of the same vendor): `-e` expands its CSV file
into JSON files:

```bash
./smm_params.sh -e P100
mv tmp/P100-0x2f6c params/NEW-0x0000
```

`-e` writes below `tmp/` only. The expanded files still name the device
they came from. On the target machine, update the device name:

```bash
./tune_multiply.py -u -p params/NEW-0x0000
./smm_params.sh -f NEW
```

The merge reports the id it found. Rename the directory accordingly:

```bash
mv params/NEW-0x0000 params/NEW-0x1234
./smm_params.sh -f NEW
```

From there, retune with the normal flow.
