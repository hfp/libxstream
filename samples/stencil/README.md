# Stencil

Finite-difference stencil computation for seismic wave propagation
(RTM, FWI) on GPUs, with an 8th-order (radius-4) operator by default.
Three kernel paths are available:

- **FP32** (default): the stencil in FP32 arithmetic, on any OpenCL
  device. Supports XYZ and ZYX memory layouts, PML absorbing boundaries,
  and the compact operator methods.
- **BF16-DPAS** (STENCIL_BF16=1): the stencil as BF16 matrix products on
  Intel matrix units (DPAS/XMX), accumulated in FP32.
- **INT8** (STENCIL_INT8=1): the stencil as INT8 matrix products (DPAS on
  Intel GPUs, dp4a on NVIDIA).

## Build

```bash
make
```

Requires LIBXS in a sibling directory of LIBXSTREAM (`../../../libxs`
from here) and an OpenCL runtime. The FP32 path (default) runs on any
OpenCL device. The BF16 and INT8 paths need DPAS and
cl_intel_subgroup_2d_block_io on Intel GPUs, the INT8 path also runs on
NVIDIA (dp4a), and other devices fall back to scalar code
(`STENCIL_BF16=2`, `STENCIL_INT8=2`), whose performance is not
representative.

Build-time defines go into ECFLAGS, e.g.,
`make ECFLAGS="-DSTENCIL_PML_W=27"` (DFLAGS and CFLAGS are assembled by
the build and a command line assignment would replace them):

- `STENCIL_PML_W`: absorbing layer width in cells (default: 20). A
  caller supplying its own eta sets its own damping width here; a larger
  value is safe, a smaller one leaves damped cells unattenuated.

### Host build

```bash
make OCL=0
```

Compiles the FP32 kernel (kernels/stencil_fp32.cl) with the C compiler
and runs it on the CPU instead of the GPU. Needs neither OpenCL nor the
LIBXSTREAM library. The kernel is specialized at build time, so the
work-group shape and optionally the grid extent are pinned by CPUDEF:

```bash
make OCL=0 CPUDEF="-DWG_X=128 -DWG_Y=8"
```

| Define (CPUDEF)                    | Meaning                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `-DWG_X=`, `-DWG_Y=`               | work-group shape (default: full fast axis with capacity 1024, and 8)    |
| `-DSTENCIL_NX=`, `-DSTENCIL_NY=`, `-DSTENCIL_NZ=` | pin the grid (all three or none); another grid is refused |
| `-DSTENCIL_LAYOUT=`                | layout built in (0=XYZ, 1=blocked, 2=ZYX; default: 0)                   |
| `-DSTENCIL_PML=1`                  | PML built in (default: 0)                                               |
| `-DSTENCIL_CPU_COMPACT=0`          | drop the compact radii (8 instead of 32 instances)                      |
| `-DSTENCIL_PADDED=`                | force the boundary treatment (0=clamp, 1=halo)                          |
| `-DSTENCIL_CPU_LANES=0`            | 1x1 work-groups instead of lane loops (default: 1); with `-DLIBXSTREAM_CPU_TEAM=1`, an OpenMP team per work-group |

`STENCIL_BF16`, `STENCIL_INT8`, and `STENCIL_BLOCKED` are refused with
OCL=0. Bind the threads, which is what `make OCL=0 test` does:

```bash
OMP_PROC_BIND=spread OMP_PLACES=cores ./stencil.x -n 512
```

The host build carries one kernel instance per combination of these
environment variables and selects between them at run time:

| Variable       | Meaning                                              |
|----------------|------------------------------------------------------|
| STENCIL_BF16S  | wavefield storage limbs (1 = single BF16, 2 = two)   |
| STENCIL_FP16S  | IEEE FP16 wavefield storage (0/1)                    |
| STENCIL_HALO   | halo padding size per axis                           |
| STENCIL_METHOD | operator method (0–3)                                |
| STENCIL_LAYOUT | accepted when it names the layout built in           |
| STENCIL_PML    | accepted when it matches the build                   |

The default host build carries 32 instances. Drop the compact radii for
a build of eight:

```bash
make OCL=0 CPUDEF="-DSTENCIL_CPU_COMPACT=0"
```

Asking for a combination the build does not carry is an error naming the
flag to rebuild without, never a silent substitution. `STENCIL_PADDED`
forces the boundary treatment and then compiles only that half.

The benchmark driver reaches the host build through `--exe`, which makes
the CPU kernel sweepable like any device variant:

```bash
make OCL=0 BLDDIR=/tmp/b/obj OUTDIR=/tmp/b
./stencil.py --exe /tmp/b/stencil.x --kernel bf16s --sizes 256,512
```

FP16S is slow in the host build. BF16S=2 (`--kernel bf16s-split`) stores
two limbs, i.e., as many bytes as FP32.

## Usage

```text
./stencil.x [options]

  -n <N>         grid dimension (NxNxN, default 256)
  -nx/ny/nz <N>  individual grid dimensions
  -t <steps>     time steps (default 100)
  -d <dims>      operator terms: 3=isotropic, 9=TTI (default 3, 9 needs STENCIL_BF16=1)
  -m <method>    operator method 0–3 (overrides STENCIL_METHOD)
  -i <init>      initial wavefield: rand | zero | gauss (default rand)
  -s <spacing>   grid spacing in meters (default 10.0), also --spacing
  -v <model>     velocity: const | grad | layered | <file.bin> (default grad)
  -vmin <vel>    minimum velocity m/s (default 1500)
  -vmax <vel>    maximum velocity m/s (default 4500)
  -w <steps>     warmup steps excluded from timing (default 5)
  -h, --help     show the options and exit

  -seg-salt      preset: SEG/EAGE Salt (676x676x210, h=20m)
  -overthrust    preset: SEG/EAGE Overthrust (801x801x187, h=25m)
```

A preset sets the grid, the spacing, the velocity range, and a synthetic
velocity model; `-v` given after the preset replaces the model.

Performance is reported as GPoints/s (grid points updated per second):

```text
GPoints/s = (nx x ny x nz x nsteps) / (time x 1e9)
```

`stencil.py` runs a sweep of `stencil.x` and writes CSV (optionally a
plot); `./stencil.py --help` lists all options:

| Option                      | Meaning                                                                         |
|-----------------------------|---------------------------------------------------------------------------------|
| `--sizes`                   | grid sizes, comma lists, or ranges start:stop[:step] (default: 800)             |
| `--kernel`                  | fp32 (default), fp32-split, bf16, int8, int8-split, bf16s, bf16s-split, fp16s   |
| `--exe`                     | path to stencil.x (default: ./stencil.x)                                        |
| `--output`, `--input`       | CSV to write (default: stencil-<kernel>.csv), or CSV to read instead of running |
| `--plot`                    | PNG output path                                                                 |
| `--layout`                  | sets STENCIL_LAYOUT (0, 1, 2)                                                   |
| `--pml`                     | sets STENCIL_PML=1                                                              |
| `--check`                   | sets STENCIL_CHECK=1                                                            |
| `--ppw`, `--fit`, `--radius-fit` | set STENCIL_PPW, STENCIL_FIT, STENCIL_RADIUS_FIT                           |

### TTI mode

With -d 9, the kernel adds the cross-derivative terms of tilted
transverse isotropy. This requires the BF16-DPAS path
(`STENCIL_BF16=1`, not with the ZYX layout); otherwise `-d 9` is refused
with an error. The CPU check (`STENCIL_CHECK`) covers the isotropic terms
only.

## Velocity Models

The driver loads an external velocity model as a flat binary file
(float32), one velocity volume without anisotropy parameters. The two
3D SEG/EAGE models below are publicly available.

### SEG/EAGE Salt Body (3D)

- Grid: 676 x 676 x 210 (n1=210 depth, n2=n3=676 lateral)
- Spacing: 20 m
- Range: 1500–4482 m/s
- Size: ~385 MB (float32)
- Reference: Aminzadeh et al. (1997)

Downloads:

- Original (20 m): <https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/fvp>
- Resampled (40 m, 115x338x338, for cheaper runs): <https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/BENCH_27PT/v.bin>

### SEG/EAGE Overthrust (3D)

- Grid: 801 x 801 x 187 (n1=187 depth, n2=n3=801 lateral)
- Spacing: 25 m
- Range: 2179–6000 m/s
- Size: ~450 MB (float32)
- Reference: Aminzadeh, Brac, and Kunz (1997)

Downloads:

- Original (25 m): <https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp>
- Resampled (50 m, 94x401x401): <https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp.r50>

Both models are hosted by the WIND project at Universite Cote d'Azur /
Geoazur: <https://www.geoazur.fr/WIND/bin/view/Main/Data/>

### Usage with this benchmark

The binary format is flat float32, read without transposition into
`-nx` x `-ny` x `-nz` with x fastest. The files above store depth (n1)
fastest, hence depth becomes the x-axis. Velocities are in m/s; the
driver squares them internally. Byte order is native (little-endian on
x86). Download the velocity binary and run:

```bash
wget https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/fvp
./stencil.x -seg-salt -nx 210 -ny 676 -nz 676 -v fvp -t 1000

wget https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp
./stencil.x -overthrust -nx 187 -ny 801 -nz 801 -v fvp -t 1000
```

## Operator Methods

Methods (selected via STENCIL_METHOD environment variable) replace the
radius-4 operator by a smaller radius r:

| Method | Name        | K | r | Description                                  |
|--------|-------------|---|---|----------------------------------------------|
| 0      | direct      | 1 | 4 | standard high-order (default)                |
| 1      | compact-r1  | 4 | 1 | compact tridiagonal runtime path             |
| 2      | compact-r2  | 2 | 2 | compact pentadiagonal runtime path           |
| 3      | compact-fit | 2 | 3 | dispersion-fitted coefficients (r: `STENCIL_RADIUS_FIT`) |

The compact-fit coefficients target a number of points per wavelength
(`STENCIL_PPW`, default 8). The compact methods apply to the isotropic
terms; TTI cross-terms use the direct operator.

## Environment Variables

| Variable              | Meaning                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------|
| STENCIL_BF16          | BF16-DPAS kernel (1=native BF16, 2=FP32 arithmetic without DPAS, the fallback on devices without DPAS) |
| STENCIL_BF16S         | BF16 wavefield storage limbs (0=off, 1=single 2-byte limb halves traffic, 2=two limbs FP32-equivalent) |
| STENCIL_FP16S         | IEEE FP16 wavefield storage (0/1); single 2-byte value per point, halves traffic (10-bit mantissa, FP32 kernel) |
| STENCIL_INT8          | INT8 kernel (1=DPAS on Intel or dp4a on NVIDIA, 2=scalar integer code, the fallback on other devices) |
| STENCIL_NDIGITS_A     | operator digits of the BF16 and INT8 paths (1–3, default: 1)                                  |
| STENCIL_I8_OP         | INT8 operator: `implicit` (compact, default) or `banded` (radius-4 FD weights)                |
| STENCIL_NDIGITS_X     | INT8 wavefield digits (1–3, default: 2 with the implicit operator, else 3)                    |
| STENCIL_R_GATHER      | INT8 gather radius (1–4, default: 2 with the implicit operator, else 4)                       |
| STENCIL_METHOD        | operator method (0–3, default 0)                                                              |
| STENCIL_PPW           | compact-fit points per wavelength (default: 8)                                                |
| STENCIL_FIT           | compact-fit method (0=uniform L2, 1=Ricker-weighted, 2=minimax; default: 2)                   |
| STENCIL_RADIUS_FIT    | compact-fit radius (2 or 3, default: 3)                                                       |
| STENCIL_STRIPS_PER_WG | adjacent N-strips handled by one work-group (default: 2, INT8: 1; must divide 64)             |
| STENCIL_SG            | subgroup size override (default: device preferred)                                            |
| STENCIL_GRF256        | force 256-GRF mode (0/1, default: 0)                                                          |
| STENCIL_TRIM          | drop least-significant digit products (BF16 and INT8 paths, default: 1)                       |
| STENCIL_LU            | loop unroll strategy (-1=none, 0=inner, 1=outer; default: 0)                                  |
| STENCIL_LAYOUT        | memory layout (0=XYZ, 1=blocked, 2=ZYX; default: 0); with ZYX, `STENCIL_BF16` falls back to the FP32 kernel |
| STENCIL_BLOCKED       | 1 selects layout 1 (blocked) when STENCIL_LAYOUT is not set                                   |
| STENCIL_HALO          | halo/padding size per axis (default: 0)                                                       |
| STENCIL_PML           | enable PML absorbing boundary (0/1, default: 0)                                               |
| STENCIL_PML_ETA       | PML damping peak (default: 0.0199)                                                            |
| STENCIL_FP32_WG_X, STENCIL_FP32_WG_Y | FP32 direct-kernel work-group shape (default: 32x8); powers of two with X ≤ 64, Y ≤ 32, and X·Y = 256, else the default |
| STENCIL_FP32_SBLOCK   | FP32 direct-kernel slow-axis SLM microblock (1 or 2, default: 2 except ZYX+PML)               |
| STENCIL_FP32_BLOCK_IO | enable FP32 Intel 2D block reads for padded 32x8 cases (0/1, default: 0); needs `STENCIL_FP32_SBLOCK=1` |
| STENCIL_CHECK         | unset: print output statistics; 1: also compare against a CPU reference; 0: neither           |
| STENCIL_HINT          | debug: non-zero allocates device buffers with the atomics hint (default: 0)                   |
| STENCIL_TRACE         | debug: if set, trace the driver phases to stderr                                              |
| STENCIL_DUMP_OP       | debug: if set, print the reconstruction error of the BF16 operator                            |

## Integration

The stencil API (stencil_opencl.h) accepts device pointers for all
buffer arguments (wavefield, output, velocity). No host-device
transfers happen inside the dispatch path.

```C
stencil_context_t ctx;
stencil_init(&ctx, verbosity, method);  /* method < 0: STENCIL_METHOD */
ctx.nterms = nterms;                    /* 3 (default) or 9 */
stencil_configure(&ctx, nx, ny, nz);
stencil_precompute_operators(&ctx, fd_weights, STENCIL_RADIUS);
stencil_seed_exp_buf(&ctx, p_host, nx, ny, nz);  /* INT8 path only */
for (t = 0; t < nsteps; ++t) {          /* p_old receives the next step */
  stencil_apply_laplacian(&ctx, p_cur, p_old, vel, dt * dt, h, nterms);
  tmp = p_cur; p_cur = p_old; p_old = tmp;
}
libxstream_stream_sync(ctx.stream);
stencil_finalize(&ctx);
```

`fd_weights` holds the 2·STENCIL_RADIUS+1 second-derivative weights for
the grid spacing h (center at index STENCIL_RADIUS), and `vel` holds the
squared velocity. Launches are asynchronous on `ctx.stream`. A caller
with its own absorbing boundary (STENCIL_PML=1) sets the device buffers
`ctx.eta` and `ctx.phi` between `stencil_init` and `stencil_configure`.

Buffers are expected in the configured layout and storage format. These
helpers convert from an XYZ float32 array on the host:

| Helper                       | Converts to                                                       |
|------------------------------|-------------------------------------------------------------------|
| `stencil_pack_blocked`       | blocked layout (size: `stencil_blocked_size`)                     |
| `stencil_pack_bf16s`         | BF16S/FP16S storage (ndigits: 0 = FP16, 1 or 2 = BF16 limbs)      |
| `stencil_pack_bf16s_blocked` | BF16S/FP16S storage in the blocked layout                         |
| `stencil_pack_bf16s_zyx`     | BF16S/FP16S storage in the ZYX layout with halo                   |
| `stencil_unpack_bf16s`       | back from BF16S/FP16S storage to float32                          |

### Initialization and USM control

LIBXSTREAM provides libxstream_init_config for explicit control over
the memory model before initialization:

```C
libxstream_init_config_t cfg;
libxstream_init_config_default(&cfg);  /* all fields = -1 (default) */
cfg.usm = 1;      /* force Intel USM extensions */
cfg.device = 0;   /* select device index */
libxstream_init_config(&cfg);
```

USM levels (cfg.usm):

| Level | Meaning                                                                                                   |
|-------|-----------------------------------------------------------------------------------------------------------|
| -1    | env/default (LIBXSTREAM_USM): level 1 on an Intel discrete GPU unless Intel MPI runs without offload support, disabled on Intel GPUs with unified memory, else as level 2 |
| 0     | disable USM, force clCreateBuffer path                                                                    |
| 1     | Intel USM extensions (clDeviceMemAllocINTEL)                                                              |
| 2     | OpenCL 2.0 SVM coarse-grain only                                                                          |
| 3     | OpenCL 2.0 SVM with device-reported capabilities                                                          |

When USM is active (level 1), device pointers from any source
(SYCL, Level Zero, raw clDeviceMemAllocINTEL) can be passed to
the stencil API without conversion.

### SYCL interoperability

SYCL USM device allocations (sycl::malloc_device) can be passed
directly to stencil_apply_laplacian on Intel GPUs. Requirements:

- Initialize LIBXSTREAM with cfg.usm = 1, or set the environment
  variable LIBXSTREAM_USM=1.
- The SYCL queue and the libxstream OpenCL context must target the
  same physical device (shared Level Zero context).
- Synchronization: use sycl::queue::ext_oneapi_submit_barrier() before
  calling stencil_apply_laplacian, and libxstream_stream_sync after, to
  order work correctly.
- No format conversion needed: float32 USM buffers are used as-is
  (velocity is expected as v^2).

### External OpenCL consumers

Any OpenCL application can call the API by passing cl_mem-backed
device pointers obtained via libxstream_mem_allocate, or USM
pointers from clDeviceMemAllocINTEL directly.

## File Layout

```text
samples/stencil/
  Makefile             build rules
  README.md            this file
  stencil.c            host driver (benchmark, model loading)
  stencil.py           Python benchmark harness (CSV/plot output)
  stencil_opencl.c     OpenCL context, kernel dispatch, exp_buf management
  stencil_opencl.h     public API, compile-time parameters
  stencil_kernels.h    generated at build time from .cl sources
  stencil_pack.c       layout and storage conversion helpers
  stencil_pml.c/.h     absorbing boundary setup
  stencil_weights.c/.h operator weights of the methods
  stencil_cpu.c        host build (OCL=0): kernel instances and launcher
  stencil_cpu_set.h    host build: list of kernel instances
  stencil_cpu_instance.h  host build: one kernel instance
  stencil_hostmem.h    host build: stand-in for the LIBXSTREAM memory API
  kernels/
    stencil_common.cl  parameters, gather macros, layout indexing
    stencil_fp32.cl    FP32 path: stencil_apply_direct (default)
    stencil_bf16.cl    BF16 path: stencil_apply, stencil_apply_tti
    stencil_int8.cl    INT8 path: stencil_apply_int8

libxstream/opencl/
  libxstream_common.h  IEEE utilities, BF16 conversion, EXP2I, unroll macros
  libxstream_ozaki.h   Dekker split, BF16 DPAS macros
  libxstream_cpu_begin.h, libxstream_cpu_end.h  host translation of a kernel (OCL=0)
```
