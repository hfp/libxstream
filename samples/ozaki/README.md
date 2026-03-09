# Ozaki Scheme — OpenCL

This sample demonstrates high-precision GEMM emulation via **mantissa slicing**
(Ozaki Scheme 1) and **Chinese Remainder Theorem** (Ozaki Scheme 2), both
fully offloaded to an OpenCL device. It is an OpenCL adaptation of the
CPU-based Ozaki sample in [LIBXS](https://github.com/hfp/libxs).

## Algorithm

### Scheme 1 — Mantissa Slicing

Two kernel variants implement the mantissa decomposition:

- **int8** (`ozaki1_int8.cl`, default, `OZAKI=1`) — Each IEEE-754 mantissa is
  decomposed into 7-bit signed int8 slices with a shared per-row/per-column
  exponent. The DPAS path uses `i8_i8_matrix_mad_k32` (BK=32).

- **bf16** (`ozaki1_bf16.cl`, `OZAKI=3`) — Dekker splitting: each element
  is successively rounded to bf16 and the residual re-split.  Each bf16 slice
  carries its own sign and exponent, eliminating the exponent panels and
  exponent-reconstruction step during accumulation.  The DPAS path uses
  `bf16_bf16_matrix_mad_k16` (BK=16).

The GEMM `C = alpha * A * B + beta * C` is then computed as a sum of
`S*(S+1)/2` (with triangular + symmetrize) pairwise dot products over
`BM×BN×BK` tiles, matching the granularity of GPU matrix engines.

### Scheme 2 — Chinese Remainder Theorem (CRT)

- **int8** (`ozaki2_int8.cl`, `OZAKI=2`) — Each IEEE-754 mantissa, shifted by
  a per-row/per-column max exponent, is reduced modulo each of up to 18
  pairwise coprime moduli (≤ 128, fitting int8).  One int8 dot product is
  performed per modulus channel (NPRIMES independent products instead of
  S*(S+1)/2 pairwise slice products).  The results are reconstructed via
  **Garner's algorithm** and evaluated with **Horner's method** in a
  mixed-radix representation.  The product of 17 moduli exceeds 2^111,
  providing sufficient range for fp64.

  XMX / DPAS acceleration uses a **fused** `dotprod` kernel that computes
  DPAS int8 matmuls per modulus channel, reduces to unsigned residues in
  registers, and immediately runs Garner reconstruction and Horner
  evaluation — no intermediate global buffer is needed.  A scalar
  fallback follows the same fused structure.  The triangular and
  symmetrize optimisations do not apply (each modulus channel is
  independent).

  **K-grouping** (`OZAKI_TRIM`): When `OZAKI_TRIM` > 0, KGROUP = 2^OZAKI_TRIM
  consecutive K sub-panels share a common max exponent and their DPAS dot
  products are accumulated before a single Garner reconstruction per group.
  This amortises the (expensive) Garner/Horner phase across KGROUP panels.
  KGROUP > 1 automatically uses 18 primes (instead of 17) to provide
  sufficient CRT range for accumulated dot products.  With `OZAKI_TRIM=2`
  (KGROUP=4) the Garner cost is cut by 4× and overall speedups of ~3× have
  been measured on PVC at large matrix sizes.

### Pipeline

Both schemes share the same three-kernel pipeline:
1. **preprocess_a** — decompose rows of A into int8/bf16 slices (Scheme 1)
   or modular residues (Scheme 2)
2. **preprocess_b** — decompose columns of B into int8/bf16 slices or residues
3. **dotprod** — iterate slice pairs or modulus channels, accumulate into C

Scheme 2 fuses Garner reconstruction and Horner evaluation directly into
the `dotprod` kernel (both XMX and scalar paths), eliminating the need for
an intermediate residue buffer or a separate postprocess phase.

Shared IEEE-754 field extraction is factored into `ozaki_common.cl`
(`ieee_decompose`), included by both `ozaki1_int8.cl` and `ozaki2_int8.cl`.
Scheme 2 additionally factors Garner reconstruction and Horner evaluation
into file-local inline helpers (`oz2_garner_reconstruct`,
`oz2_horner_accumulate`) shared between the XMX and scalar `dotprod` paths.

When Intel XMX hardware is detected (`cl_intel_subgroup_matrix_multiply_accumulate`
and `cl_intel_subgroup_2d_block_io`), the Scheme 1 dotprod kernel uses **DPAS**
(Data Processing Accelerator Systolic) instructions with 2D block I/O for data
movement (SG=16, 8×32 A reads, 32×16 VNNI-transformed B reads). Otherwise a
scalar fallback is used.

A double-buffered K-batch pipeline overlaps preprocessing of batch N+1 with the
dot-product computation of batch N across three concurrent streams.

When Intel USM or OpenCL SVM is available, `ozaki_init` creates a device-memory
pool (backed by `libxs_malloc_xpool` with `LIBXS_MALLOC_NATIVE`).  Buffers
allocated during `ozaki_gemm` are returned to the pool instead of being freed,
eliminating per-call allocation overhead for repeated GEMM calls.  With the pool
active, `ozaki_gemm` returns without synchronizing — the caller is responsible
for syncing the stream.  On the rare grow path (larger problem size), the
wrapped deallocator syncs all streams before reallocating.  The pool falls back
to direct `libxstream_memdev_allocate` transparently if USM/SVM is not supported.

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
| `OZAKI`          | 1       | Kernel variant: 1 = int8 mantissa slices, 2 = int8 CRT, 3 = bf16 Dekker slices. |
| `OZAKI_FLAGS`    | 3       | Scheme 1 bitmask: Triangular (1), Symmetrize (2). 0 = full S² square. Ignored for Scheme 2. |
| `OZAKI_TRIM`     | 0       | Scheme 1: diagonal trim (drop T least significant diagonals). Scheme 2: K-grouping exponent — KGROUP = 2^TRIM consecutive K sub-panels share one exponent and one Garner reconstruction (0 = no grouping, 1 = pairs, 2 = quads). |
| `OZAKI_N`        | 8/17    | Scheme 1: number of slices per element. Scheme 2: number of CRT primes (default 17, max 18; automatically raised to 18 when KGROUP > 1). |
| `OZAKI_VERBOSE`  | 0       | Verbosity level: 0 = silent, 1 = errors only, 2 = errors + warnings, 3+ = all info. Negative values also enable all output. |
| `OZAKI_XMX`      | auto    | Override XMX detection (0 = force off, 1 = on).|
| `OZAKI_WG`       | 0       | Work-group size hint (0 = no hint).            |
| `OZAKI_SG`       | auto    | Sub-group size (forced to 16 when XMX active). |
| `OZAKI_GRF256`   | 0       | 1 = request 256 GRF per thread (Intel XMX only). |
| `OZAKI_CONSTANT` | 0       | 1 = use `constant` address space for read-only buffers. |

The Ozaki context auto-selects XMX-friendly defaults when hardware support is
detected.  For int8 Scheme 1 (default): `BK=32`, `BM=16`, `BN=16`.  For bf16
(`OZAKI=3`): `BK=16`, `BM=16`, `BN=16`.  For CRT (`OZAKI=2`): `nprimes=17`
(18 when KGROUP > 1), XMX uses `BK=32` with fused in-register Garner/Horner.
Common defaults: `SG=16`, `batch_k=4`.

## Example

```
$ ./ozaki.x 256
OpenCL benchmark for Ozaki's methods
GEMM: NN M=256 N=256 K=256 lda=256 ldb=256 ldc=256 alpha=1 beta=1
Ozaki GEMM: 12.345 ms
BLAS  GEMM: 1.234 ms
DIFF: linf=0.000000 linf_rel=0.000000 l2_rel=0.000000 eps=0.000000 rsq=1.000000
```

## Limitations

- Complex GEMM (3M method) is not yet supported.
