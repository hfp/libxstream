# Grid

CP2K's grid library collocates products of Gaussians onto real-space grids and
integrates potentials against them. This directory builds CP2K's grid miniapp
and unit test from a CP2K source tree, outside of that tree, with every backend
the build supports: reference, CPU, DGEMM, and as GPU backend either CUDA if
`nvcc` is found or LIBXSTREAM's OpenCL backend (`grid_opencl.c`).

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
| OPENCL    | auto           | 0: no OpenCL backend (needs LIBXS)           |

## Test

```bash
make test
```

Runs CP2K's unit test, records each of CP2K's sample tasks as task list
(collocate and integrate) with the reference backend, and replays every task
list with the backends in `TESTBACKENDS` (default: `ref cpu`, and `gpu` with
CUDA or OpenCL). `TESTARG` is the number of repetitions of each replay (default: 3).

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

## OpenCL Backend

`grid_opencl.c` implements CP2K's GPU backend with OpenCL, which an OpenCL build
of CP2K uses unless `__NO_OFFLOAD_GRID` is defined. With CMake, `-DCP2K_USE_ACCEL=OPENCL` calls
`libxstream_add_grid(<target> <grid>)` (unless `CP2K_ENABLE_GRID_GPU=OFF`), which
generates `grid_kernels.h` from the headers in `<grid>/common` into the
consumer's build tree. A task contributes to the same points as with CP2K's CPU
backend, or with `GRID_CENTRED=1` to the points within its radius plus one grid
spacing about the center of the product (fewer, more accurate). With `GRID_WINDOW=1`, the
contribution fades out over that margin, so that points do not enter or leave a task's sphere with a step. Forces and
virial include the terms of the window. Each work-group owns a tile of points and sums the tile's
tasks in a fixed order, hence the result is reproducible without atomics. The
task lists of the tiles are held packed (`libxs_vpack` at level 1, about a fifth
of 32-bit indices) and decoded by the work-groups as they walk them
(`opencl/libxstream_pack.h`).
Integration uses the same model: persistent engines take tasks from a queue and
give each task lanes by its size; a task's lanes reduce its rows in a fixed order, and each block's part is owned by one work-item, hence
no atomics either. Distributed grids and tasks with a border mask use CP2K's CPU
backend.

| Environment | Default | Description                                         |
|-------------|---------|-----------------------------------------------------|
| GRID_TILE   | `8,4,4` | Points of a tile (x,y,z) owned by a work-group      |
| GRID_ENGINE | auto    | Lanes of an engine (power of two), default: sub-group size |
| GRID_ROWS   | `8`     | Rows per lane before a task gets more lanes         |
| GRID_SLOTS  | `8`     | Engines per compute unit                            |
| GRID_CENTRED | `0`    | 1: points within radius + h of the product center, else CP2K's rules |
| GRID_WINDOW | `0`     | 1: weight falling smoothly (C2) from 1 at the radius to 0 at radius + h, implies GRID_CENTRED |
| GRID_SG     | auto    | Sub-group size of the main kernels (0: the compiler's choice), default: 16 where offered |
| GRID_FP32   | `-1`    | Margin (bits) for single precision in the tails of collocation (16 is safe), <0: off |
