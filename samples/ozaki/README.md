# Ozaki Scheme 1 — OpenCL

This sample demonstrates high-precision GEMM emulation via **mantissa slicing**
(Ozaki Scheme 1) fully offloaded to an OpenCL device. It is an OpenCL adaptation
of the CPU-based Ozaki sample in [LIBXS](https://github.com/hfp/libxs).

## Algorithm

Each IEEE-754 mantissa is decomposed into 7-bit int8 slices. The GEMM
`C = alpha * A * B + beta * C` is then computed as a sum of `S*(S+1)/2`
(with triangular + symmetrize) pairwise int8 dot products over `BM×BN×BK`
tiles, matching the granularity of GPU matrix engines.

Three OpenCL kernels implement the pipeline:
1. **preprocess_a** — extract sign/exponent/mantissa from A, find per-row
   max exponent, align and slice into int8 panels
2. **preprocess_b** — same for B (per-column)
3. **dotprod** — iterate slice pairs, accumulate scaled int8 dot products
   into double/float C tiles

When Intel XMX hardware is detected (`cl_intel_subgroup_matrix_multiply_accumulate`
and `cl_intel_subgroup_2d_block_io`), the dotprod kernel uses **DPAS** (Data
Processing Accelerator Systolic) instructions with 2D block I/O for data
movement (SG=16, 8×32 A reads, 32×16 VNNI-transformed B reads). Otherwise a
scalar fallback is used.

A double-buffered K-batch pipeline overlaps preprocessing of batch N+1 with the
dot-product computation of batch N across three concurrent streams.

## Build

```bash
cd samples/ozaki
make [GNU=1] [DBG=1] [PEDANTIC=2]
```

Requires an OpenCL runtime and headers (e.g., `opencl-c-headers`, `ocl-icd-dev`,
or Intel oneAPI). BLAS is linked via `BLAS=2` (2=parallelized) for the reference GEMM.

## Run

```bash
./ozaki.x [M [N [K [transa [transb [alpha [beta [lda [ldb [ldc]]]]]]]]]]
```

All arguments are positional and optional (defaults shown):

| Position | Argument  | Default   | Description                             |
|----------|-----------|-----------|-----------------------------------------|
| 1        | `M`       | 257       | Number of rows of C and op(A)           |
| 2        | `N`       | M         | Number of columns of C and op(B)        |
| 3        | `K`       | M         | Inner dimension                         |
| 4        | `transa`  | 0         | 0 = 'N', 1 = 'T' for A                 |
| 5        | `transb`  | 0         | 0 = 'N', 1 = 'T' for B                 |
| 6        | `alpha`   | 1         | Scalar multiplier for A*B               |
| 7        | `beta`    | 1         | Scalar multiplier for C                 |
| 8        | `lda`     | auto      | Leading dimension of A                  |
| 9        | `ldb`     | auto      | Leading dimension of B                  |
| 10       | `ldc`     | M         | Leading dimension of C                  |

## Environment Variables

| Variable         | Default | Description                                    |
|------------------|---------|------------------------------------------------|
| `GEMM_OZFLAGS`   | 3       | Scheme 1 bitmask: Triangular (1), Symmetrize (2). 0 = full S² square. |
| `GEMM_OZTRIM`    | 0       | Diagonal trim: drop T least significant diagonals (~7 bits each). |
| `GEMM_OZN`       | 8       | Number of int8 slices per mantissa.            |
| `OZAKI_VERBOSE`  | 0       | Verbosity level (1 = info, 2+ = debug).        |
| `OZAKI_XMX`      | auto    | Override XMX detection (0 = force off, 1 = on).|
| `OZAKI_WG`       | 0       | Work-group size hint (0 = no hint).            |
| `OZAKI_SG`       | auto    | Sub-group size (forced to 16 when XMX active). |
| `OZAKI_CONSTANT` | 0       | 1 = use `constant` address space for read-only buffers. |

The Ozaki context auto-selects XMX-friendly defaults when hardware support is
detected: `BK=32`, `BM=16`, `BN=16`, `SG=16`, `nslices=8`, `batch_k=4`.

## Example

```
$ ./ozaki.x 256
Ozaki Scheme 1 OpenCL benchmark
NN M=256 N=256 K=256 lda=256 ldb=256 ldc=256 alpha=1 beta=1
Device: Intel(R) Data Center GPU Max 1550 (GPU)
Ozaki GEMM: 12.345 ms
BLAS  GEMM: 1.234 ms
GEMM: linf=0.000000 linf_rel=0.000000 l2_rel=0.000000 eps=0.000000 rsq=1.000000
```

## Limitations

- Only Scheme 1 (mantissa slicing) is implemented; Scheme 2 (CRT) is not
  included.
- Complex GEMM (3M method) is not yet supported.
