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

| Pos. | Argument | Default | Description               |
|------|----------|---------|---------------------------|
| 1    | M        | 257     | Rows of C and op(A)       |
| 2    | N        | M       | Columns of C and op(B)    |
| 3    | K        | M       | Inner dimension           |
| 4    | transa   | 0       | 0=N, 1=T for A            |
| 5    | transb   | 0       | 0=N, 1=T for B            |
| 6    | alpha    | 1       | Scalar multiplier for A*B |
| 7    | beta     | 1       | Scalar multiplier for C   |
| 8    | lda      | auto    | Leading dimension of A    |
| 9    | ldb      | auto    | Leading dimension of B    |
| 10   | ldc      | M       | Leading dimension of C    |

## Environment Variables

### Scheme Selection

| Variable | Default | Description                                                               |
|----------|---------|---------------------------------------------------------------------------|
| OZAKI    | 1       | 1=mantissa slicing, 2=CRT, 0=bypass (call BLAS directly)                  |
| OZAKI_FP | 64      | 64=fp64 (double), 32=fp32 (float)                                         |
| OZAKI_N  | (auto)  | Slices (Sch.1: fp64=8, fp32=4) or primes (Sch.2: fp64=16, fp32=9; max 20) |

### Accuracy

| Variable     | Default | Description                                                                |
|--------------|---------|----------------------------------------------------------------------------|
| OZAKI_FLAGS  | 3       | Sch.1 bitmask: 1=Triangular, 2=Symmetrize, 0=full S^2. Ignored for Sch.2   |
| OZAKI_TRIM   | 0       | Precision levels to trim (0=exact). ~7 bits/level (Sch.1), ~4 bits (Sch.2) |
| OZAKI_I8     | 0       | Sch.2: use signed i8 residues (moduli<=128) instead of u8                  |
| OZAKI_GROUPS | 0       | Sch.2: K-grouping factor, consecutive K panels share one reconstruction    |

### Hardware Control

| Variable         | Default | Description                                                         |
|------------------|---------|---------------------------------------------------------------------|
| OZAKI_RTM        | (auto)  | Register tiling M (power of two). Auto: 4/2/1 for 256/128-GRF/other |
| OZAKI_RTN        | (auto)  | Register tiling N (power of two). Auto: 2 (Intel GPU), 1 (other)    |
| OZAKI_WG         | 0       | Work-group size hint (0=no hint)                                    |
| OZAKI_SG         | (auto)  | Sub-group size (forced to 16 with XMX)                              |
| OZAKI_BIGGRF     | (auto)  | Override 256-GRF detection (0=off, 1=on)                            |
| OZAKI_KU         | 2       | K-loop unroll factor                                                |
| OZAKI_RC         | 8       | DPAS repeat count (8 or 4)                                          |
| OZAKI_PB         | 1       | Sch.2: CRT prime batching factor                                    |
| OZAKI_PREFETCH   | 0       | Sch.1: enable prefetching                                           |
| OZAKI_BOUNDS     | 0       | Sch.1: force bounds checking (auto for non-aligned sizes)           |
| OZAKI_SCALAR_ACC | 0       | Sch.1: force scalar accumulation                                    |

### Memory and Caching

| Variable      | Default | Description                                                              |
|---------------|---------|--------------------------------------------------------------------------|
| OZAKI_DEVPOOL | 0       | Device memory pool via USM/SVM (eliminates per-call allocation overhead) |
| OZAKI_CACHE   | 0       | Preprocessing cache bitmask: 1=A, 2=B, 3=both. Skips on matching pointer |

### TinyTC Kernels

| Variable     | Default | Description                                                               |
|--------------|---------|---------------------------------------------------------------------------|
| OZAKI_TINYTC | 0       | Load TinyTC SPIR-V from .clx path. 0=use embedded or OpenCL C. Sch.1 only |

### Benchmark

| Variable      | Default | Description                                               |
|---------------|---------|-----------------------------------------------------------|
| NREPEAT       | 1       | Number of benchmark repetitions                           |
| OZAKI_VERBOSE | 0       | 0=silent, 1=errors, 2=warnings, 3+=all info. Negative=all |

Additional variables for profiling, accuracy monitoring, and complex
GEMM dispatch (OZAKI_PROFILE, OZAKI_THRESHOLD, OZAKI_STAT, OZAKI_EPS,
OZAKI_RSQ, OZAKI_EXIT, OZAKI_COMPLEX) are handled by the LIBXS Ozaki
sample ([LIBXS](https://github.com/hfp/libxs)), which owns the GEMM
interceptor. See its README for details.

## Example

```bash
./ozaki.x 256
```

Scheme 2 with K-grouping on a large matrix:

```bash
OZAKI=2 OZAKI_GROUPS=4 ./ozaki.x 4096
```

## Quick Tuning Guide

Scheme 1 (mantissa slicing, OZAKI=1): suited for smaller matrices.
Use OZAKI_TRIM to trade accuracy for speed. Default 8 slices (fp64)
yield S*(S+1)/2 = 36 dot products with triangular+symmetrize.

Scheme 2 (CRT, OZAKI=2): more efficient for large matrices. Use
OZAKI_GROUPS=4 to amortize Garner reconstruction across K panels
(~3x speedup on PVC at sizes >=2048).

Enable OZAKI_CACHE=3 when A or B stays constant across calls.
Enable OZAKI_DEVPOOL=1 for repeated calls with similar sizes.
