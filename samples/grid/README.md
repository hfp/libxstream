# Grid

CP2K's grid library collocates products of Gaussians onto real-space grids and
integrates potentials against them. This directory builds CP2K's grid miniapp
and unit test from a CP2K source tree, outside of that tree, with every backend
the build supports: reference, CPU, DGEMM, and CUDA if `nvcc` is found.

## Build

```bash
cd samples/grid
make
```

`CP2K_ROOT` defaults to a `cp2k` directory next to LIBXSTREAM; without a CP2K
tree, nothing is built. Make variables:

| Variable  | Default        | Description                                  |
|-----------|----------------|----------------------------------------------|
| CP2K_ROOT | `../../../cp2k`| CP2K source tree                             |
| BLAS_LIBS | `-lblas`       | BLAS for the DGEMM backend                   |
| CUDA      | auto           | 0: no CUDA backend even if `nvcc` is found   |
| CUDA_ARCH | `sm_70`        | Architecture given to `nvcc`                 |

## Test

```bash
make test
```

Runs CP2K's unit test, records each of CP2K's sample tasks as task list
(collocate and integrate) with the reference backend, and replays every task
list with the backends in `TESTBACKENDS` (default: `ref cpu`, and `gpu` with
CUDA). `TESTARG` is the number of repetitions of each replay (default: 3).

## Task Lists

A task list holds one call of CP2K's grid library: the task list itself, the
arguments of the call, and its results. CP2K writes task lists when the input
says so, e.g., the first collocate and the first integrate call of each rank:

```text
&GLOBAL
  &GRID
    DUMP_TASK_LISTS 1
  &END GRID
&END GLOBAL
```

The miniapp replays a task list with the selected backend, compares the results
with the recorded ones, and reports the time of creating the task list apart
from the time of the repeated calls:

```bash
./grid_miniapp.x --backend cpu 10 grid_collocate_0_0.tasklist
```

The backend is `auto`, `ref`, `cpu`, `dgemm`, or `gpu`, and `--cutoff` applies
the spherical cutoff of the DGEMM and GPU backends as CP2K's input does by
default. The results do not depend on the number of repetitions. The miniapp
also takes CP2K's `.task` files (a single task, `--batch` repeats it), and
`--dump <calls>` records task lists from such a run.
