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

Show which embedded parameter set a device name selects:

```bash
make match
./acc_match.x "NVIDIA H200"
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

A JSON file is named after its kernel and a hash of its parameters, for
example `tune_multiply-double-23x23x23-1a2b3c4d.json`, with `-s<N>` after
the shape if the batch size is not the default 30000. Tuning that finds
the same parameters again keeps one file with the higher GFLOPS/s. A
merge renames files of earlier forms and keeps their modification time,
which `--prefer new` relies on. Keep one
directory per device: a same-named file from another device is reported
and left alone.

Update the device name stored in every JSON file of a directory. Given a
name, the tuner writes it as-is and runs nowhere in particular. Given no
name, it asks the device how it identifies itself, which requires
running on the target machine:

```bash
./tune_multiply.py -u "Some Device [0x1234]" -p params/local
./tune_multiply.py -u -p params/local
```

Write the JSON files a CSV file was merged from (the inverse of `-m`),
which needs neither a device nor the benchmark:

```bash
./tune_multiply.py -M params/tune_multiply_PVC.csv -p tmp/PVC
```

The names are reproduced as a tuning session would have written them,
because a JSON file is named after a hash of its parameters. A file that
exists is kept rather than rewritten, so an existing directory does not
lose the file dates that `--plan` orders by.

Check existing JSONs without re-tuning:

```bash
./tune_multiply.py -c -p params/local
```

When two JSON files describe the same kernel on the same device,
`--prefer` decides which one wins the CSV: `fast` (default) keeps the
higher GFLOPS/s, and `new` keeps the more recent measurement, which is
what a driver or hardware change calls for. A merge lists the losing
files, and `-d` removes them:

```bash
./tune_multiply.py -p params/local --prefer new
./tune_multiply.py -p params/local --prefer new -d
```

The first command prints `Remove N (use -d)` followed by the files the
second one deletes, so a merge is also the preview.

Neither direction is inherently safer. Deleting the older file
discards a configuration that was faster and may be faster again,
while deleting the newer one discards the evidence of a regression.

Useful options:

| Option              | Meaning                                      |
|---------------------|----------------------------------------------|
| --stop-after N      | Stop the search after N seconds              |
| -p path             | JSON directory; without a shape, merge it    |
| -s size             | Benchmark batch size, also called stack size |
| -a level            | Tuning level: 0=all ... 4=least tunables     |
| -m                  | Merge JSON files into a CSV file             |
| -M file             | Write the JSON files of a CSV file (see -p)  |
| -o file             | CSV output file                              |
| -u [device]         | Update JSON device names (see above)         |
| -c [epsilon]        | Validate JSON entries                        |
| --prefer fast\|new  | Duplicate winner; without a shape, merge     |
| -d                  | Delete losing duplicates during merge        |

The tuner can run under MPI. It detects local MPI rank variables and
uses them to select `LIBXSTREAM_DEVICE`, so ranks on the same node can
use different devices. This is most useful when each rank is tuning a
different kernel.

```bash
mpirun \
  ./tune_multiply.py 13x5x7 --stop-after=300 -p params/local : \
  ./tune_multiply.py 23x23x23 --stop-after=300 -p params/local
```

Interrupted tuning writes the best result seen so far. If you tune a
different kernel with `tune_multiply.py` directly, remove any old
OpenTuner database first:

```bash
rm -rf opentuner.db
```

The wrapper below does this cleanup automatically.

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

Parts also split a long job into sessions on a single device. The
split points depend only on the triplet list and `-j`, so each session
repeats the same command and only the part number changes:

```bash
PART=1  # next session: 2, then 3 and 4
./tune_multiply.sh -t 300 -j 4 -i ${PART} -p params/local 4 10 15, 6 7 8, 23
```

Keep the options that shape the list unchanged between sessions (`-r`,
`-m`, `-n`, `-b`, `-f`, `-k`), since any change moves the split points.
With `-u`, the list comes from the JSON directory, which stays the same
as long as no new shapes are added to it between sessions.

Retune every JSON file found in a directory:

```bash
./tune_multiply.sh -u -p params/local -t 300
```

#### Retuning by age

`--plan` lists the kernels of a JSON directory oldest first and writes
that list as a file of triplets, or to standard output if no file is
given. A kernel is dated by its youngest JSON file, so a kernel counts
as old only once all of its files are:

```bash
./tune_multiply.sh --plan retune.txt -p params/local -n 40
./tune_multiply.sh -f retune.txt -p params/local -t 300
```

The first command takes the 40 kernels that were tuned least recently
(`-b` takes the most recent ones instead, and `-r`, `-m`, and `-n` limit
the plan as they limit any other list). The second command works the
plan. Tuning does not necessarily rewrite a JSON file: finding the same
parameters again with a lower GFLOPS/s keeps the file and its date, so
an order taken from the directory a second time can repeat the kernels
it just tuned. The plan avoids this by holding the order still, which is
also what makes a plan the list to keep between sessions of a split job.

A plan records its own progress: each kernel that tuned successfully is
marked `#done` and is skipped when the plan is used again, so an
interrupted session resumes where it stopped and a kernel that failed
stays in the plan. Marking rewrites the plan, which is why it is limited
to the generated file (a handwritten list is left alone) and to a single
session (`-j 1`). Delete the marks to tune the plan again:

```bash
sed "s/^#done //" retune.txt > retune.tmp && mv retune.tmp retune.txt
```

There is no minimum age: a plan lists every kernel of the directory, and
`-n` decides how much of it to take. Taken with `--plan`, it fixes a
smaller campaign, and taken with `-f`, it sizes a single session, so the
same command repeated works down the plan by the same budget each time:

```bash
./tune_multiply.sh --plan retune.txt -p params/local
./tune_multiply.sh -f retune.txt -p params/local -n 2 -t 300
```

A complete plan has no unmarked line left, which is the point to
generate the next one:

```bash
grep -qv "^#" retune.txt || ./tune_multiply.sh --plan retune.txt -p params/local
./tune_multiply.sh -f retune.txt -p params/local -n 2 -t 300
```

A date tells when a kernel was last improved, not when it was last
tuned, and `#done` is the only record of an attempt. A kernel that
retuning never improves therefore keeps its date and heads the next
generated plan again.

Limit the generated work before splitting it:

```bash
./tune_multiply.sh -t 180 -r 4 32 -m 64 -n 100 \
  4 8 16 32, 4 8 16 32, 4 8 16 32
```

Useful options:

| Option          | Meaning                                           |
|-----------------|---------------------------------------------------|
| -t seconds      | Time limit per kernel                             |
| -p path         | Directory for JSON files                          |
| -s size         | Benchmark batch size, also called stack size      |
| -a level        | Tuning level: 0=all ... 4=least tunables          |
| -u              | Retune JSON files found under `-p`                |
| --plan [file]   | Write kernels of `-p` by age, oldest first        |
| -d              | Ask the merge step to delete losing JSONs         |
| --prefer new    | Prefer newest duplicate (default: fastest)        |
| -c              | Continue with the next kernel after an error      |
| -b              | Reverse triplets, before `-n` and before parts    |
| -j parts        | Total number of tuning parts                      |
| -i index        | Part to run, using 1-based numbering              |
| -r low high     | Keep kernels with low**3 < M*N*K <= high**3       |
| -m extent       | Keep kernels with M, N, and K no larger than this |
| -n count        | Keep only the first count kernels (see `-b`)      |
| -f file         | Read MxNxK list from a file, resuming a plan      |
| -k id           | Use a predefined triplet set                      |

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

JSON files are the working format and stay on the machine that tuned
them, since `params/*/` is not tracked. CSV files are the deployable
format. `smm_params.sh` converts between the two for every device at
once, pairing a directory `params/<device>-0x<id>` with the CSV file
`params/tune_multiply_<device>.csv`:

```bash
./smm_params.sh            # merge every device
./smm_params.sh "H100*"    # merge matching devices only
./smm_params.sh -e         # the reverse: write JSONs to tmp/
```

The device id in the directory name is what tells two variants of the
same GPU apart. A merge compares it against the id of the parameters it
merged and against the CSV file it is about to replace, and reports both
ids rather than overwriting the wrong file. A device without a CSV file
needs `-f`, which is how a new device gets its first one.

A typical retune flow is:

```bash
make realclean
make WITH_GPU=P100
./tune_multiply.sh --plan retune.txt -p params/P100-0x2f6c
./tune_multiply.sh -f retune.txt -p params/P100-0x2f6c -t 300
./smm_params.sh P100
```

Keep GPU driver state persistent during tuning (e.g.,
`nvidia-smi -pm ENABLED` on headless NVIDIA systems).

### Starting a New Device

Tuning from scratch starts from the benchmark's own defaults. Starting
from parameters that already work on a similar GPU is usually better, so
`-e` expands a CSV file back into JSON files and any published device
can seed a new one. Pick the closest relative, e.g. the previous
generation of the same vendor:

```bash
./smm_params.sh -e P100
mv tmp/P100-0x2f6c params/NEW-0x0000
```

`-e` writes below `tmp/` and never into a corpus, so it cannot reset the
file dates of parameters that are already tuned. The expanded files
still name the device they came from. On the target machine, let the
tuner ask the device instead:

```bash
./tune_multiply.py -u -p params/NEW-0x0000
./smm_params.sh -f NEW
```

The merge reports the id it actually found, which is the id to put into
the directory name. Rename accordingly and the pair is consistent:

```bash
mv params/NEW-0x0000 params/NEW-0x1234
./smm_params.sh -f NEW
```

From there, retuning is the normal flow: the seeded parameters are what
the search starts from, and `--plan` keeps a record of what was retuned.
