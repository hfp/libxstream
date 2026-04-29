# Ozaki Scheme -- OpenCL

High-precision GEMM on OpenCL devices via mantissa slicing (Scheme 1)
or Chinese Remainder Theorem (Scheme 2). Both schemes decompose FP
matrices into int8/u8 tiles and use DPAS/XMX matrix engines when
available. This is an OpenCL adaptation of the CPU-based Ozaki sample
in [LIBXS](https://github.com/hfp/libxs).

## Build

```bash
cd samples/ozaki
make [GNU=1] [DBG=1]
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

The preprocessing cache also stores the last effective cutoff from
Scheme 1 occupancy detection. On cache hits the cutoff is reused
without device-to-host readback, eliminating the sync bubble.

### Benchmark

| Variable      | Default | Description                                       |
|---------------|---------|---------------------------------------------------|
| NREPEAT       | 1       | Number of benchmark repetitions                   |
| OZAKI_VERBOSE | 0       | 0=silent, 1=errors, 2=warnings, 3+=all. Neg.=all  |

Additional variables for profiling, accuracy monitoring, and complex
GEMM dispatch (OZAKI_PROFILE, OZAKI_THRESHOLD, OZAKI_STAT, OZAKI_EPS,
OZAKI_RSQ, OZAKI_EXIT, OZAKI_COMPLEX) are handled by the LIBXS Ozaki
sample ([LIBXS](https://github.com/hfp/libxs)), which owns the GEMM
interceptor. See its README for details.

## Kernel Registry

Scheme 1 fused GEMM kernels are compiled on demand via a JIT registry.
The compile-time cutoff (OZAKI_CUTOFF) is baked into each kernel
specialization, allowing the compiler to eliminate dead slice-pair
iterations and reduce register pressure. The first call with a given
cutoff value triggers JIT compilation (~100 ms); subsequent calls hit
the registry cache. Typical workloads produce 2-3 specializations
(full cutoff, reduced cutoff, each with/without bounds checking).

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
