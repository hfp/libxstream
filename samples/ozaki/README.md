# Ozaki Scheme -- OpenCL

High-precision GEMM on OpenCL devices via mantissa slicing (Scheme 1)
or Chinese Remainder Theorem (Scheme 2). Both schemes decompose FP
matrices into int8/u8 tiles and use DPAS/XMX matrix engines when
available. This is an OpenCL adaptation of the CPU-based Ozaki sample
in [LIBXS](https://github.com/hfp/libxs).

## Build

```bash
cd samples/ozaki
make [DBG=1]
```

Requires an OpenCL runtime and headers. BLAS is linked via `BLAS=2`
for the reference GEMM.

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

## Environment Variables

### Scheme Selection

| Variable | Default | Description                                                       |
|----------|---------|-------------------------------------------------------------------|
| OZAKI    | 2       | 1=mantissa slicing, 2=CRT (default), 3=adaptive, 0=bypass BLAS    |
| OZAKI_FP | 64      | 64=fp64 (double), 32=fp32 (float)                                 |
| OZAKI_N  | (auto)  | Slices (Sch.1: fp64=8, fp32=4) or primes (Sch.2: fp64=16, fp32=9) |

OZAKI=3 (adaptive) starts with Scheme 1 on the first call to learn
the effective cutoff from preprocessing occupancy data. Subsequent
calls compare the Scheme-1 pair count against the Scheme-2 prime
count and pick the cheaper path. The cutoff is cached alongside the
preprocessed buffers and reused on cache hits without any device-to-
host readback.

### Accuracy

| Variable     | Default | Description                                                          |
|--------------|---------|----------------------------------------------------------------------|
| OZAKI_FLAGS  | 3       | Sch.1 bitmask: 1=Triangular, 2=Symmetrize, 0=full S^2. No Sch.2      |
| OZAKI_TRIM   | 0       | Precision levels to trim (0=exact). ~7 bits (Sch.1), ~4 bits (Sch.2) |
| OZAKI_I8     | 0       | Sch.2: use signed i8 residues (moduli<=128) instead of u8            |
| OZAKI_GROUPS | 0       | Sch.2: K-grouping factor, consecutive K panels share reconstr.       |

### Hardware Control

| Variable         | Default | Description                                                      |
|------------------|---------|------------------------------------------------------------------|
| OZAKI_TM         | (auto)  | Output tile M (BM). Overrides size-aware selection                |
| OZAKI_TN         | (auto)  | Output tile N (BN). Overrides size-aware selection                |
| OZAKI_RTM        | (auto)  | Register tiling M (power of two). Auto: 2 (HIER), 4 (256-GRF)    |
| OZAKI_RTN        | (auto)  | Register tiling N (power of two). Auto: 2 (Intel GPU), 1 (other) |
| OZAKI_WG         | 0       | Work-group size hint (0=no hint)                                 |
| OZAKI_SG         | (auto)  | Sub-group size (forced to 16 with XMX)                           |
| OZAKI_BIGGRF     | (auto)  | Override 256-GRF detection (0=off, 1=on). HIER defaults to 128   |
| OZAKI_KU         | 2       | K-loop unroll factor                                             |
| OZAKI_RC         | 8       | DPAS repeat count (8 or 4)                                       |
| OZAKI_PB         | 1       | Sch.2: CRT prime batching factor                                 |
| OZAKI_HIER       | (auto)  | Sch.2: hierarchical CRT (default on). Two-level Garner reconstr. |
| OZAKI_PREFETCH   | 0       | Sch.1: enable prefetching                                        |
| OZAKI_SCALAR_ACC | 0       | Sch.1: force scalar accumulation                                 |

### Memory and Caching

| Variable      | Default | Description                                                         |
|---------------|---------|---------------------------------------------------------------------|
| OZAKI_DEVPOOL | 0       | Device memory pool via USM/SVM (eliminates per-call alloc overhead) |
| OZAKI_CACHE   | 0       | Preprocessing cache bitmask: 1=A, 2=B, 3=both. Skips on match       |
| OZAKI_NPANEL  | 1       | Sch.2: N-panel width (0=auto, 1=disable). See Panel Pipeline        |

The preprocessing cache also stores the last effective cutoff from
Scheme 1 occupancy detection. On cache hits the cutoff is reused
without device-to-host readback, eliminating the sync bubble.

### Benchmark

| Variable      | Default | Description                                       |
|---------------|---------|---------------------------------------------------|
| NREPEAT       | 1       | Number of benchmark repetitions                   |
| OZAKI_VERBOSE | 0       | 0=silent, 1=errors, 2=warnings, 3+=all. Neg.=all  |

Additional variables for accuracy monitoring and complex GEMM dispatch
(OZAKI_THRESHOLD, OZAKI_STAT, OZAKI_EPS, OZAKI_RSQ, OZAKI_EXIT,
OZAKI_COMPLEX) are handled by the LIBXS Ozaki sample
([LIBXS](https://github.com/hfp/libxs)), which owns the GEMM
interceptor. See its README for details.

Kernel timings come from LIBXSTREAM rather than from a sample-specific
facility: `LIBXSTREAM_PROFILE=1` reports one row per kernel
(preprocess A, preprocess B, the fused GEMM, and so on), and
`LIBXSTREAM_PROFILE_MEM=1` reports transfer rates. Neither needs a
phase to be selected up front -- every kernel is recorded and the
interesting rows are read from the report.

### cuBLAS Reference

When the OpenCL headers are taken from a CUDA installation, the sample
additionally links cuBLAS and reports a device-side reference GEMM
(`cuBLAS GEMM` and `cuBLAS DIFF`). Build with `make CUBLAS=0` to opt
out. The host BLAS stays the accuracy reference for both the Ozaki and
the cuBLAS result, and a failing cuBLAS run is not fatal.

By default the sample requests FP64 emulation without the built-in
fallback to native FP64 (CUDA 13.0u2 and later, compute capability 8.0
and later). The variables below are populated before cuBLAS is
entered, hence any of them that is already set wins:

| Variable                                        | Sample default |
|-------------------------------------------------|----------------|
| CUBLAS_EMULATE_DOUBLE_PRECISION                 | 1              |
| CUBLAS_EMULATION_STRATEGY                       | eager          |
| CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK    | 0              |

Setting `CUBLAS_EMULATE_DOUBLE_PRECISION=0` yields the native device
GEMM, which is the baseline to compare against. For `OZAKI_FP=32` the
counterpart is `CUBLAS_EMULATE_SINGLE_PRECISION`, which requires
compute capability 10.0 and later.

| Variable          | Default | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| OZAKI_CUBLAS_BITS | 0       | Mantissa bits: 0=default, <0=match the OZAKI_N slices    |
| OZAKI_CUBLAS_PIN  | 0       | 1=register the host buffers with CUDA (pinned transfers) |
| OZAKI_CUBLAS_XPTR | 0       | 1=pass LIBXSTREAM device pointers to cuBLAS (experiment) |

A non-zero `OZAKI_CUBLAS_BITS` fixes the number of int8 slices
(`slices = ceil((bits + 1) / 8)`) instead of letting cuBLAS pick the
precision per call. A negative value matches the component count of
this sample, which is a slice count for OZAKI=1 but a prime count for
OZAKI=2 -- only the former is comparable.

`OZAKI_CUBLAS_XPTR=1` is an experiment and expected to fail: the
device-side pointers of LIBXSTREAM are not addresses of the CUDA
context that cuBLAS runs in.

Like the Ozaki timing, the cuBLAS timing covers the transfers, i.e. A,
B and C are uploaded and C is read back per iteration; CUDA events
report the device-side split (`gemm`, `h2d`, `d2h`) separately. With
`OZAKI_CUBLAS_XPTR=1` the transfers are not on the CUDA timeline and
read as zero.

Two properties to keep in mind when reading the numbers. The mode
printed alongside the timing is the one that was requested: whether a
call was emulated cannot be queried, and `nsys profile ./ozaki.x` is
the way to confirm it. And emulation needs a workspace, which the
sample supplies (`-DOZAKI_CUBLAS_WORKSPACE=<bytes>`, 2 GB by default,
0 to disable) because a workspace that is too small does not fail the
call but silently falls back to non-emulated kernels.

## Output Tile Selection

The output tile per work-group (BM x BN) is chosen per call from M and
N, because the best tile depends strongly on problem size: the total
work-item count is invariant to the tile, so the tile only controls how
many work-groups the problem splits into and how far M and N are padded
to a tile multiple. A large tile maximizes operand reuse but produces
too few work-groups to fill the device at small sizes; on a GPU Max
1550 the fixed 128x256 tile costs up to 1.7x at M=N=256..640 while
being optimal from ~1024 upward.

Selection maximizes arithmetic intensity `BM*BN/(BM+BN)` per unit of
padded work, subject to two constraints: the work-group size bound
`SG * (BM/(XMX_M*RTM)) * (BN/(XMX_N*RTN)) <= max_wgs`, and enough
work-groups to saturate the compute units (OZAKI_TILE_SAT). The
intensity term is maximal for square tiles, which keeps the split
balanced rather than degenerate. Set OZAKI_TM/OZAKI_TN to bypass
selection and pin a tile.

## Panel Pipeline (Scheme 2)

Only ~30% of a Scheme-2 call is the GEMM kernel; the rest is uploading
A/B, preprocessing them into CRT residues, and downloading C. Scheme 2
therefore splits **N** into panels and pipelines them: while panel *j*
runs its GEMM, panel *j+1* uploads and preprocesses its B columns and
panel *j-1* downloads its C block. A is uploaded and preprocessed once
as a prologue.

N is the right axis because each panel then owns a disjoint column block
of B and C. C is read and written exactly once no matter how many panels
there are, and panel GEMMs are mutually independent. Splitting K instead
would re-read and re-write all of C per group -- measured at several ms
per pass at n=4096, which is most of what the pipeline is trying to hide
-- and would serialize the panel GEMMs through that accumulation.

Panel B-slices are double-buffered across `OZAKI_NSLOTS` slots (2 by
default; 3 and 4 measured slower). A panel reusing a slot waits only on
the GEMM that last read it, which is what decouples consecutive panels.
Because a panel is a slice of B rather than all of B, panelling also
lowers peak memory relative to the unpanelled path.

Panelling is skipped when it cannot pay: too few tiles per panel to
saturate the device, `N <= tn`, K-grouping active, device-resident
operands, or `OZAKI_NPANEL=1`. Requesting a B cache (`OZAKI_CACHE=2` or
`3`) takes precedence -- a cached slice buffer must hold all of B, so
caching and panelling are alternative ways to avoid the same work
(reuse across calls versus overlap within one). With `transb` a panel is
a strided row block, so B is uploaded whole and only the preprocessing
and download overlap.

Disabled by default because the automatic width does not account for
shape: measured on PVC (fp64) it costs up to 12% on square shapes and
gains up to 7% where `N` is much larger than `M`. Set `OZAKI_NPANEL=0`
to enable it where `N` is the dominant dimension.

## Kernel Registry

Both schemes compile fused GEMM kernels on demand via a JIT registry
keyed by the output tile plus the bounds-checking variant; Scheme 1
additionally keys on the compile-time cutoff (OZAKI_CUTOFF), which lets
the compiler eliminate dead slice-pair iterations and reduce register
pressure. The first call with a given key triggers JIT compilation
(~100 ms); subsequent calls hit the registry cache. A workload with a
single matrix shape produces one specialization per scheme.

## Example

```bash
./ozaki.x 256
```

Scheme 2 on a large matrix:

```bash
OZAKI=2 ./ozaki.x 4096
```

Adaptive scheme selection with caching:

```bash
OZAKI=3 OZAKI_CACHE=3 ./ozaki.x 4096
```

## Quick Tuning Guide

Scheme 2 (CRT, OZAKI=2, default): fixed cost of P integer GEMMs plus
hierarchical Garner reconstruction. Predictable performance regardless
of data distribution. Use OZAKI_GROUPS for K-grouping at large sizes.
The hierarchical CRT (OZAKI_HIER, on by default) halves private
residue arrays and enables GRF128 for doubled thread occupancy.

Scheme 1 (mantissa slicing, OZAKI=1): up to S\*(S+1)/2 integer GEMMs,
but adaptive cutoff can reduce this substantially for narrow exponent
spans. Use OZAKI_TRIM to trade accuracy for speed.

Adaptive (OZAKI=3): automatically picks the cheaper scheme per call
based on preprocessing occupancy. Best with OZAKI_CACHE=3 to avoid
repeated occupancy readbacks.

Enable OZAKI_CACHE=3 when A or B stays constant across calls.
Enable OZAKI_DEVPOOL=1 for repeated calls with similar sizes.
