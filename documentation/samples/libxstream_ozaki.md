# Ozaki Scheme — OpenCL

High-precision GEMM on OpenCL devices via mantissa slicing (Scheme 1)
or Chinese Remainder Theorem (Scheme 2). Both schemes decompose FP
matrices into int8/u8 tiles and use the DPAS/XMX matrix engines (Intel)
or dp4a, `mma.sync` and warp-group MMA (NVIDIA) when available; Scheme 2
can also carry its residues in bf16 (`OZAKI_BF16`). This is an OpenCL
adaptation of the CPU-based Ozaki sample in
[LIBXS](https://github.com/hfp/libxs/blob/main/samples/ozaki/README.md).

## Build

```bash
cd samples/ozaki
make [DBG=1]
```

Requires an OpenCL runtime and headers, a LIBXS checkout next to this
repository (`../libxs`, which provides `Makefile.inc`), and a host BLAS
for the reference GEMM (linked via `BLAS=2`). `make test` runs
`./ozaki.x 64 64 64`. Compile-time defines go into ECFLAGS, e.g.
`make ECFLAGS="-DOZAKI_CUBLAS_WORKSPACE=0"`.

## Run

```bash
./ozaki.x [M [N [K [transa [transb [alpha [beta [lda [ldb [ldc]]]]]]]]]]
```

All arguments are positional and optional:

| Pos. | Argument | Default | Description                |
|------|----------|---------|----------------------------|
| 1    | M        | 257     | Rows of C and op(A)        |
| 2    | N        | M       | Columns of C and op(B)     |
| 3    | K        | M       | Inner dimension            |
| 4    | transa   | 0       | 0=N, 1=T for A             |
| 5    | transb   | 0       | 0=N, 1=T for B             |
| 6    | alpha    | 1       | Scalar multiplier for A\*B |
| 7    | beta     | 1       | Scalar multiplier for C    |
| 8    | lda      | auto    | Leading dimension of A     |
| 9    | ldb      | auto    | Leading dimension of B     |
| 10   | ldc      | M       | Leading dimension of C     |

```bash
./ozaki.x 256                      # adaptive scheme (default)
OZAKI=2 ./ozaki.x 4096             # Scheme 2
OZAKI_CACHE=3 ./ozaki.x 4096       # adaptive, with preprocessing cache
```

## Environment Variables

### Scheme Selection

| Variable    | Default | Description                                                                                              |
|-------------|---------|----------------------------------------------------------------------------------------------------------|
| OZAKI       | 3       | 1=mantissa slicing, 2=CRT, 3=adaptive                                                                    |
| OZAKI_FP    | 64      | 64=fp64 (double), 32=fp32 (float)                                                                        |
| OZAKI_N     | (auto)  | Slices (Sch.1: fp64=8, fp32=4) or moduli (Sch.2: 2–20, see below). Under OZAKI=3 it sets the moduli only |
| OZAKI_XOVER | 27      | Sch.1/2 crossover weight used by OZAKI=3 (fp64, non-NVIDIA)                                              |

The default modulus count is the smallest that carries the exact
product for K up to `OZAKI_MAXK`: 16 (fp64) and 9 (fp32) at the default
`OZAKI_MAXK`, more for a larger one.

OZAKI=3 picks a scheme per call from that call alone, so mixed matrix
sizes in one process do not influence each other. On NVIDIA it selects
Scheme 2, and in fp32 on other devices Scheme 1.

### Accuracy

| Variable       | Default | Description                                                                                                             |
|----------------|---------|-------------------------------------------------------------------------------------------------------------------------|
| OZAKI_FLAGS    | 3       | Sch.1 bitmask: 1=Triangular, 2=Symmetrize, 0=full S^2. No Sch.2 (OZAKI=2 defaults to 0)                                 |
| OZAKI_TRIM     | 0       | Levels to trim (0=default). ~7 bits (Sch.1), ~4 bits (Sch.2); negative buys precision back                              |
| OZAKI_MAXK     | 32768   | Largest K one preprocessing pass covers; sizes the moduli and the K-grouping. 0 or negative=default                     |
| OZAKI_FRACCRT  | (auto)  | Sch.2: 0=Garner, 1=flat fractional (not exact), 2=fractional CRT. Auto: 0 if unfused, on NV MMA or without fp64, else 2 |
| OZAKI_TZDETECT | 0       | Sch.2: report the largest `OZAKI_TRIM` that stays exact for the data (reporting only)                                   |

Under OZAKI=3, a positive `OZAKI_TRIM` applies to Scheme 1 only; a
negative one applies to both schemes. There is no unbounded
`OZAKI_MAXK`.

Only `OZAKI_FLAGS` 0 and 3 compute the full product; 1 and 2 are for
symmetric operands only. The LIBXS interceptor accepts only 0 and 3.

`OZAKI_FRACCRT` 0 and 2 are exact; 1 is magnitude-bounded. Without fp64
only Garner (0) is available.

`OZAKI_TZDETECT=1` prints per call how many moduli the data needs and
the matching `OZAKI_TRIM`; it adds a device-to-host readback per call.

### Scheme 2 (CRT)

| Variable             | Default | Description                                                                                                    |
|----------------------|---------|----------------------------------------------------------------------------------------------------------------|
| OZAKI_HIER           | (auto)  | Hierarchical (two-level) CRT reconstruction (default on)                                                       |
| OZAKI_HIER_L2        | (auto)  | Level-2 combine: 1=tree merge, 0=Garner. Auto: tree merge for at most 2 groups (more fall back to Garner)      |
| OZAKI_L1_GARNER      | 0       | 1=Garner at level 1 instead of the explicit CRT (not with OZAKI_FRACCRT=2)                                     |
| OZAKI_EXTRACT_FLAT   | 0       | fp64: 1=extract the residues per modulus instead of per group                                                  |
| OZAKI_UNFUSE         | (auto)  | Reconstruct in a 2nd kernel (`gemm_crt_reduce`). On for GPUs                                                   |
| OZAKI_GROUPS         | (auto)  | K-grouping factor (1=off). Consecutive K panels share the reconstruction                                       |
| OZAKI_SYMRES         | (auto)  | Symmetric residues (magnitude m/2 at most). On with bf16, or `OZAKI_MAXK` past 32768                           |
| OZAKI_PB             | 1       | CRT modulus batching factor                                                                                    |
| OZAKI_BF16           | 0       | Carry the residues in bf16 rather than int8 (see below)                                                        |
| OZAKI_COMPLEX_3M     | (auto)  | Complex: 1=3M on residues, 0=4M; auto: 3M from _MIN on. Needs OZAKI_UNFUSE, not with OZAKI_BF16/OZAKI_TZDETECT |
| OZAKI_COMPLEX_3M_MIN | (auto)  | Cube root of M*N*K from which auto takes 3M (1024 coherent, 4096 discrete)                                     |
| OZAKI_SWIZZLE        | 0       | Work-group rasterization width (0=launch order)                                                                |
| OZAKI_NOBOUNDS       | 0       | Drop output range checks (whole-tile shapes only)                                                              |
| OZAKI_ALPHA_ONE      | 0       | Specialize the GEMM for alpha=1                                                                                |
| OZAKI_FIRST          | 0       | Specialize the GEMM for C=0+AB (first accumulation)                                                            |
| OZAKI_SKIP_GARNER    | 0       | Fused epilogue only (OZAKI_UNFUSE=0): skip reconstruction (timing only, result is not a GEMM)                  |

`OZAKI_UNFUSE` gives the same results as the fused epilogue
(`OZAKI_UNFUSE=0`). It needs `OZAKI_HIER`, `OZAKI_PB=1` and no
K-grouping, and one byte of scratch per modulus and output element,
twice the size of C in fp64 at 16 moduli. In the `LIBXSTREAM_PROFILE`
report the two kernel rows add up to the total; the FLOP rate is
attributed to the GEMM row.

`OZAKI_GROUPS` disables the unfused epilogue and is slower for it
(reported under `OZAKI_VERBOSE`); set it only when the per-pass buffers
do not fit. Left unset, it is derived only where an exact result needs it: an
`OZAKI_MAXK` past what an integer accumulator holds exactly (32768
unsigned, 131072 symmetric). `OZAKI_SYMRES` turns on by itself past
32768, so the grouping is derived only past 131072 (or past 32768 with
`OZAKI_SYMRES=0`). The bf16 carrier never needs it. `OZAKI_GROUPS=1`
declines the derived grouping and warns where that leaves K unreduced;
`OZAKI_GROUPS=0` means unset.

Flat reconstruction takes `OZAKI_HIER=0 OZAKI_FRACCRT=0` with
`OZAKI=2`; otherwise `OZAKI_HIER=0` warns and the reconstruction stays
hierarchical.

`OZAKI_BF16=1` runs Scheme 2 on a floating-point matrix engine, for a
device whose bf16 matrix throughput is much higher than its integer
throughput. Moduli, accuracy and results are unchanged (bit-identical to
int8). It needs warp-group MMA with the default A and B layouts, or
warp-level MMA (`mma.sync`), where it selects `OZAKI_BKMAJOR` by itself;
otherwise it disables itself with a message.

The complex variables take effect through the LIBXS interceptor
(ZGEMM/CGEMM); `ozaki.x` issues real GEMMs only.

### NVIDIA

| Variable               | Default | Description                                                                                              |
|------------------------|---------|----------------------------------------------------------------------------------------------------------|
| OZAKI_WGMMA            | (auto)  | Sch.2: warp-group MMA. On where reachable (see below)                                                    |
| OZAKI_WGMMA_N          | 256     | Warp-group tile width, 64, 128 or 256                                                                    |
| OZAKI_WGMMA_M          | 128     | Warp-group tile rows: 64 = one warp group, 128 = two, 256 = four (with OZAKI_WGMMA_N of 128 or less)     |
| OZAKI_WGMMA_KU         | 0       | Staging depth per tile, overriding the derived one (0=derived; still capped to the shared-memory budget) |
| OZAKI_WGMMA_DEFER      | (auto)  | Deferred wait: 1=on, 0=off. Auto: on for the 128x256 tile                                                |
| OZAKI_WGMMA_STAGES     | (auto)  | Staging buffers with the deferred wait, 2 to 4. Auto: 4, 2 with OZAKI_BF16                               |
| OZAKI_WGMMA_RESIDE     | (auto)  | Sch.2: tiles per SM below which 128x128 runs two per SM (0=off)                                          |
| OZAKI_WGMMA_RESIDE_MKT | 2.5E8   | Sch.2: bound on M*K*tiles per SM for 128x128 (0=no bound)                                                |
| OZAKI_MAXNREG          | 0       | Sch.2 wgmma: registers per thread of the GEMM (0=derived)                                                |
| OZAKI_ABLOCK           | 1       | wgmma: A in the fragment layout (0=plain layout; OZAKI_BF16 needs 1)                                     |
| OZAKI_BVNNI            | 1       | Sch.2: pre-interleave B so each operand is one aligned uint. Not with wgmma or OZAKI_BKMAJOR             |
| OZAKI_BKMAJOR          | 0       | Sch.2 without wgmma: transpose B to [N][K] instead (see below)                                           |

Warp-group MMA (`wgmma`) is the default wherever initialization finds it
usable; otherwise `mma.sync` is used and `OZAKI_VERBOSE=1` says so.
`OZAKI_WGMMA=0` selects `mma.sync`. It needs `OZAKI_HIER`, no
K-grouping, `OZAKI_PB=1` and `OZAKI_FRACCRT` other than 1, and it fixes
the tile to `OZAKI_WGMMA_M`x`OZAKI_WGMMA_N` regardless of
`OZAKI_TM`/`OZAKI_TN`. The first JIT per shape takes about twice as
long; a failed patch refuses the kernel, see `OZAKI_VERBOSE=1`.

Under warp-group MMA, `OZAKI_KU` sets the base staging depth (8 at the
default width, 16 with `OZAKI_WGMMA_N` of 64 or 128). The depth per
tile is derived from it within 128 KB of shared memory, so lower it if
a device refuses that.

Below `OZAKI_WGMMA_RESIDE` tiles per SM the tile narrows to 128x128 and
two work-groups share an SM. The default is 8 on a part with a coherent
host link and 2 otherwise; `OZAKI_WGMMA_RESIDE=0` keeps the wide tile.
The wide tile is also kept once M\*K\*(tiles per SM) exceeds
`OZAKI_WGMMA_RESIDE_MKT`, and `OZAKI_WGMMA_RESIDE_MKT=0` lifts that
bound.

`OZAKI_BKMAJOR=1` applies to the `mma.sync` and dp4a paths, not to
warp-group MMA, and is exact like the default layout.

`LIBXSTREAM_DUMP=2` (or a negative value) also writes the patched PTX of
a warp-group kernel as `<name>.ptx`.

### Hardware Control

| Variable         | Default | Description                                                                                                          |
|------------------|---------|----------------------------------------------------------------------------------------------------------------------|
| OZAKI_TM         | (auto)  | Output tile M (BM), with OZAKI_TN. Not under wgmma                                                                   |
| OZAKI_TN         | (auto)  | Output tile N (BN), with OZAKI_TM. Not under wgmma                                                                   |
| OZAKI_RTM        | (auto)  | Register tiling M (power of two). Auto: 2 (Intel, NV MMA), 4 (Intel 256-GRF without bit 2 of OZAKI_FLAGS), 1 (other) |
| OZAKI_RTN        | (auto)  | Register tiling N. Auto: 8 (NV MMA Sch.2), 2 (Intel, NVIDIA), 1 (other)                                              |
| OZAKI_SB         | 1       | Sch.1: slice-block width for the pair loop; must divide the slice count                                              |
| OZAKI_WGS_MAX    | (auto)  | Work-group size ceiling for tile selection (0=hardware bound). Auto: 4 sub-groups on non-Intel GPUs, else 0          |
| OZAKI_TILE_SAT   | (auto)  | Tile selection keeps at least compute units / OZAKI_TILE_SAT work-groups. Auto: 16, 1 on non-Intel GPUs              |
| OZAKI_KU         | 2       | K-loop unroll factor. Auto: 2, 4 (NVIDIA dp4a, Sch.2), 1 (mma.sync, not adjustable); wgmma: see NVIDIA               |
| OZAKI_RC         | 8       | DPAS repeat count (8 or 4)                                                                                           |
| OZAKI_BIGGRF     | (auto)  | Override 256-GRF detection (0=off, 1=on). HIER defaults to 128; LIBXSTREAM_BIGGRF applies if this is unset           |
| OZAKI_LU         | 0       | Loop-unroll hints: -2=all loops, -1=none, 0=inner loops, 1=inner loops and outer loops not unrolled                  |
| OZAKI_PREFETCH   | 0       | Sch.1: enable prefetching                                                                                            |
| OZAKI_SCALAR_ACC | 0       | Sch.1: force scalar accumulation                                                                                     |
| OZAKI_DEFINES    | (unset) | Extra build options (e.g. -DNAME=value) for the Scheme-1 and Scheme-2 kernel programs                                |
| OZAKI_WG         | 0       | Read but has no effect                                                                                               |
| OZAKI_SG         | (auto)  | Has no effect: the sub-group size is 16, or 32 with NV MMA                                                           |

`OZAKI_TM` and `OZAKI_TN` take effect only together and not under
warp-group MMA (reported at `OZAKI_VERBOSE=1`); otherwise the output
tile is chosen per call from M and N, keeping enough work-groups to fill
the compute units (`OZAKI_TILE_SAT`).

On Intel GPUs the register tiling is chosen per call as well:
`OZAKI_VERBOSE=3` prints the range (e.g. `rtm=2..4`) and the `JIT` line
the value used (`rt=`). Setting `OZAKI_RTM` or `OZAKI_RTN` disables this
for both; the one not set keeps its base value.

Kernels are compiled on first use per tile, register tiling and
bounds-checking variant (Scheme 1 also per cutoff, Scheme 2 per modulus
count) and kept for later calls.

### Memory and Caching

| Variable     | Default | Description                                                     |
|--------------|---------|-----------------------------------------------------------------|
| OZAKI_CACHE  | 0       | Preprocessing cache bitmask: 1=A, 2=B, 3=both (or any negative) |
| OZAKI_ARENA  | (auto)  | Device scratch arena in MB (0=off). Auto: on if no pool         |
| OZAKI_NPANEL | 1       | Sch.2: N-panel width (0=auto, 1=disable). See below             |

`OZAKI_CACHE` assumes that the content at a cached pointer does not
change between calls; enable it when A or B stays constant. Only
`OZAKI_CACHE=0` disables it: a negative value is another spelling of 3.
Scheme-1 calls on cached operands also skip the occupancy readback.

`OZAKI_ARENA` keeps the per-call device scratch (operand planes, residue
planes, device copy of C) in one allocation for the life of the context.
Unset, it is on wherever LIBXSTREAM has no device memory pool.
`OZAKI_ARENA=0` restores per-call allocation. A positive value caps the
arena in MB and also enables it where a pool exists (e.g.
`LIBXSTREAM_USM` on NVIDIA); a call needing more than the cap falls back
per buffer, and a first call that exceeds it gets no arena at all.
`ozaki_scratch_size` reports the requirement. Planes kept by
`OZAKI_CACHE` are never carved from the arena. A caller that owns device
memory can supply it with `ozaki_scratch_size`/`ozaki_scratch_set` (the
benchmark does so under `OZAKI_SCRATCH=1`); the LIBXS interceptor relies
on the default.

`OZAKI_NPANEL=0` (Scheme 2) splits N into panels whose upload,
preprocessing, GEMM and download overlap; a value above 1 sets the panel
width (rounded up to the tile). Panelling is skipped with too few tiles
per panel, `N` not above the tile width, K-grouping, device-resident
operands, or a B cache (`OZAKI_CACHE=2` or 3), which takes precedence.
With `transb`, B is uploaded whole. The panel slots are `OZAKI_NSLOTS`
(compile-time, 2 by default).

### Benchmark

| Variable      | Default | Description                                                                        |
|---------------|---------|------------------------------------------------------------------------------------|
| NREPEAT       | 1       | Number of benchmark repetitions                                                    |
| OZAKI_VERBOSE | 0       | 0=silent, non-zero=notes and warnings, 3+=all, negative=all                        |
| GRADE         | 0       | Componentwise grade of every result (non-zero=on)                                  |
| EVIL          | 0       | Per-column exponent span of 2n binades (n>0 only)                                  |
| TAME          | 0       | Keep n mantissa bits of A and B, clear the rest                                    |
| XREF          | 0       | Compensated reference on n rows (negative=all)                                     |
| OZAKI_SCRATCH | 0       | Non-zero=supply caller-owned device scratch (ozaki_scratch_size/ozaki_scratch_set) |
| OZAKI_SWITCH  | (unset) | Switch to n moduli after the warm-up call (2 up to the initial count)              |

`XREF=n` grades the Ozaki, the host BLAS and the cuBLAS result against
a reference computed by `libxs_dot2_gemm` (as if in twice the double
precision) on n rows spread over C, and prints one line
`XREF: rows=n ozaki_linf_rel=.. blas_linf_rel=.. cublas_linf_rel=..`
with `_l2_rel` and, under `GRADE`, `_grade` per result. With `GRADE`
and a cuBLAS result, a `vs-cuBLAS DIFF` line also grades the Ozaki
result against cuBLAS.

Additional variables for accuracy monitoring and complex GEMM dispatch
(`OZAKI_THRESHOLD`, `OZAKI_STAT`, `OZAKI_EPS`, `OZAKI_RSQ`, `OZAKI_EXIT`,
`OZAKI_COMPLEX`) are handled by the
[LIBXS Ozaki sample](https://github.com/hfp/libxs/blob/main/samples/ozaki/README.md),
which owns the GEMM interceptor. There, `OZAKI`, `OZAKI_N`,
`OZAKI_FLAGS`, `OZAKI_TRIM`, `OZAKI_MAXK`, `OZAKI_GROUPS`, `OZAKI_TM`,
`OZAKI_TN` and `OZAKI_VERBOSE` are read by LIBXS and passed on, with the
restrictions its README states; the kernel and memory variables above
are read by LIBXSTREAM itself in either case.

In `ozaki.x`, the driver reads `OZAKI`, `OZAKI_FP`, `OZAKI_FLAGS`,
`OZAKI_TRIM`, `OZAKI_GROUPS` and `OZAKI_VERBOSE` and passes them to
`ozaki_init`, which reads the other kernel and memory variables.

Kernel timings come from LIBXSTREAM: `LIBXSTREAM_PROFILE=1` reports one
row per kernel, and `LIBXSTREAM_PROFILE_MEM=1` reports transfer rates.

### cuBLAS Reference

When the OpenCL headers are taken from a CUDA installation, the sample
additionally links cuBLAS and reports a device-side reference GEMM
(`cuBLAS GEMM`, and `cuBLAS DIFF` against the host BLAS). Build with
`make CUBLAS=0` to opt out. `CUBLAS=1` (the default) also forces
`PEDANTIC=0`. A failing cuBLAS run is not fatal.

By default the sample requests FP64 emulation without the fallback to
native FP64 (CUDA 13.0u2 and later, compute capability 8.0 and later;
fp32 emulation needs compute capability 10.0). The variables below are
populated before cuBLAS is entered; any of them already set wins:

| Variable                                       | Sample default                        |
|------------------------------------------------|---------------------------------------|
| CUBLAS_EMULATE_DOUBLE_PRECISION                | 1                                     |
| CUBLAS_EMULATE_SINGLE_PRECISION                | 1 (`OZAKI_FP=32`)                     |
| CUBLAS_EMULATION_STRATEGY                      | eager                                 |
| CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK   | 0                                     |
| CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT | from `OZAKI_CUBLAS_BITS` (unset if 0) |

| Variable          | Default | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| OZAKI_CUBLAS_BITS | 0       | Mantissa bits: 0=default, <0=match the OZAKI_N slices    |
| OZAKI_CUBLAS_XPTR | 0       | 1=pass LIBXSTREAM device pointers to cuBLAS (experiment) |

A non-zero `OZAKI_CUBLAS_BITS` fixes the number of int8 slices
(`slices = ceil((bits + 1) / 8)`) and is passed as
`CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT` unless that is already
set. A negative value takes the component count of this sample, which
is a slice count for OZAKI=1 and a modulus count otherwise.
`OZAKI_CUBLAS_XPTR=1` is expected to fail: LIBXSTREAM device pointers
are not addresses of the CUDA context.

Emulation needs a workspace, which the sample supplies (compile-time
`OZAKI_CUBLAS_WORKSPACE` in bytes, 2 GB by default, 0 to disable; see
Build); a workspace that is too small silently disables emulation.
