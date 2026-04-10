# Ozaki Scheme - OpenCL

This sample demonstrates high-precision GEMM emulation via **mantissa slicing**
(Ozaki Scheme 1) and **Chinese Remainder Theorem** (Ozaki Scheme 2), both
fully offloaded to an OpenCL device. It is an OpenCL adaptation of the
CPU-based Ozaki sample in [LIBXS](https://github.com/hfp/libxs).

## Algorithm

### Scheme 1 - Mantissa Slicing

Two kernel variants implement the mantissa decomposition:

- **int8** (`ozaki1_int8.cl`, default, `OZAKI=1`) - Each IEEE-754 mantissa is
  decomposed into 7-bit signed int8 slices with a shared per-row/per-column
  exponent. The DPAS path uses `i8_i8_matrix_mad_k32` (BK=32).

The GEMM `C = alpha * A * B + beta * C` is then computed as a sum of
`S*(S+1)/2` (with triangular + symmetrize) pairwise dot products over
`BMxBNxBK` tiles, matching the granularity of GPU matrix engines.

### Scheme 2 - Chinese Remainder Theorem (CRT)

- **u8 default** (`ozaki2_int8.cl`, `OZAKI=2`) - Each IEEE-754 mantissa, shifted
  by a per-row/per-column max exponent, is reduced modulo each of up to 20
  pairwise coprime moduli.  One dot product is performed per modulus channel
  (NPRIMES independent products instead of S*(S+1)/2 pairwise slice products).
  The results are reconstructed via **Garner's algorithm** and evaluated with
  **Horner's method** in a mixed-radix representation.

  Residues are unsigned uint8 with moduli up to 256 (prime powers 256=2^8,
  243=3^5, 169=13^2 alongside large primes).  Sign is encoded via the modular
  additive inverse (p - r).  The DPAS path uses `u8_u8_matrix_mad_k32`;
  a signed i8 equivalent is available via `OZAKI_I8=1`.

  XMX / DPAS acceleration uses a **fused** `dotprod` kernel that computes
  DPAS matmuls per modulus channel, reduces to unsigned residues in registers,
  and immediately runs Garner reconstruction and Horner evaluation - no
  intermediate global buffer is needed.  A scalar fallback follows the same
  fused structure.  The triangular and symmetrize optimisations do not apply
  (each modulus channel is independent).

  **K-grouping** (`OZAKI_GROUPS`): When `OZAKI_GROUPS` > 1, that many
  consecutive K sub-panels share a common max exponent and their DPAS dot
  products are accumulated before a single Garner reconstruction per group.
  This amortises the (expensive) Garner/Horner phase across OZAKI_GROUPS panels.
  With `OZAKI_GROUPS=4` the Garner cost is cut by 4x and overall speedups of ~3x
  have been measured on PVC at large matrix sizes.

### Pipeline

Both schemes share the same three-kernel pipeline:

1. **preprocess_a** - decompose rows of A into int8 slices (Scheme 1)
   or modular residues (Scheme 2)
2. **preprocess_b** - decompose columns of B into int8 slices or residues
3. **dotprod** - iterate slice pairs or modulus channels, accumulate into C

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
movement (SG=16, 8x32 A reads, 32x16 VNNI-transformed B reads). Otherwise a
scalar fallback is used.

A double-buffered K-batch pipeline overlaps preprocessing of batch N+1 with the
dot-product computation of batch N across three concurrent streams.

### TinyTC Kernels

Optionally, Scheme 1 can use **TinyTC**-generated SPIR-V kernels for the
dotprod phase. TinyTC is a tensor compiler that generates optimized GPU code.
Precompiled `.clx` files in the `kernels/` directory can be loaded at runtime
(via `OZAKI_TINYTC` env var) or embedded at build time (with
`-DOZAKI_TINYTC_EMBED`). The filenames encode block sizes and configuration:
`ozaki1_{f64|f32}_{BM}x{BN}_n{N}t{TRIM}_{tri|sq}.clx`.

### Performance Features

**Device Memory Pool**: When Intel USM or OpenCL SVM is available, `ozaki_init`
creates a device-memory pool (backed by `libxs_malloc_xpool` with
`LIBXS_MALLOC_NATIVE`). Buffers allocated during `ozaki_gemm` are returned to
the pool instead of being freed, eliminating per-call allocation overhead for
repeated GEMM calls. With the pool active, `ozaki_gemm` returns without
synchronizing - the caller is responsible for syncing the stream. On the rare
grow path (larger problem size), the wrapped deallocator syncs all streams
before reallocating. The pool falls back to direct `libxstream_mem_allocate`
transparently if USM/SVM is not supported.

**Preprocessing Cache**: When `OZAKI_CACHE` is enabled (bitmask: 1=A, 2=B,
3=both), preprocessed slices and exponents are cached per input matrix pointer.
If the same matrix is passed to `ozaki_gemm` with matching dimensions and
transpose flag, preprocessing is skipped entirely. This is particularly
effective when A or B remains constant across multiple calls (e.g., batched
inference with a fixed weight matrix).

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

| Variable         | Default | Description                                                                                                                                                   |
|------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OZAKI`          | 1       | Kernel variant for real GEMM: 1 = int8 mantissa slices, 2 = int8 CRT.                                                                                        |
| `OZAKI_3M`       | *auto*  | Complex GEMM dispatch (ZGEMM/CGEMM): 0 = original BLAS, 1 = CPU 3M, 2 = GPU 3M (falls back to CPU). Default: 0 if `OZAKI=0`, else 2. Controlled by [LIBXS](https://github.com/hfp/libxs). |
| `OZAKI_MAXK`     | 32768   | Max K elements per preprocessing pass. 0 = no grouping (full K in one pass). Trades exponent-scope precision vs FP accumulation rounding. Applies to both CPU and GPU. |
| `OZAKI_FLAGS`    | 3       | Scheme 1 bitmask: Triangular (1), Symmetrize (2). 0 = full S^2 square. Ignored for Scheme 2.                                                                  |
| `OZAKI_TRIM`     | 0       | Precision levels to trim. Scheme 1: drops diagonals from slice-pair iteration (~7 product bits/level). Scheme 2: truncates mantissa before CRT (~2 input bits/level, ~4 product bits/level). The level semantics are calibrated so the same trim value gives comparable accuracy across schemes. |
| `OZAKI_N`        | 8/16    | Number of decomposition components. Scheme 1: number of slices per element (fp64: default 8, fp32: default 4). Scheme 2: number of CRT primes (fp64: default 16, fp32: default 9; max 20). |
| `OZAKI_I8`       | 0       | Scheme 2 only: use signed i8 residues (moduli <= 128) instead of the default unsigned u8. |
| `OZAKI_GROUPS`   | 0       | Scheme 2 only: K-grouping factor - that many consecutive K sub-panels share one exponent and one Garner reconstruction (0/1 = no grouping, 4 = quads).      |
| `OZAKI_CACHE`    | 0       | Preprocessing cache bitmask: 1 = cache A, 2 = cache B, 3 = cache both. Skips preprocessing when same matrix pointer is reused with matching dims/transpose.  |
| `OZAKI_TINYTC`   | 0       | Load TinyTC SPIR-V kernel from .clx file path (e.g., `kernels/ozaki1_f64_256x128_n8t3_tri.clx`). 0 = use embedded or OpenCL C kernels.                      |
| `OZAKI_VERBOSE`  | 0       | Verbosity level: 0 = silent, 1 = errors only, 2 = errors + warnings, 3+ = all info. Negative values also enable all output.                                 |
| `OZAKI_XMX`      | auto    | Override XMX detection (0 = force off, 1 = on).                                                                                                              |
| `OZAKI_WG`       | 0       | Work-group size hint (0 = no hint).                                                                                                                          |
| `OZAKI_SG`       | auto    | Sub-group size (forced to 16 when XMX active).                                                                                                               |
| `OZAKI_BIGGRF`   | auto    | Override 256 GRF per thread (0 = force off, 1 = on). Auto-enabled for Intel GPUs.                                                                            |
| `OZAKI_FP`       | 64      | Floating-point precision: 64 = fp64 (double), 32 = fp32 (float).                                                                                            |
| `OZAKI_RTM`      | auto    | Register tiling factor for M dimension (power of two). Auto: 4 (256-GRF Intel GPU), 2 (128-GRF), 1 (other).                                                 |
| `OZAKI_RTN`      | auto    | Register tiling factor for N dimension (power of two). Auto: 2 (Intel GPU), 1 (other).                                                                      |
| `OZAKI_KU`       | 2       | K-loop unroll factor (compiled into kernel).                                                                                                                 |
| `OZAKI_RC`       | 8       | DPAS repeat count: 8 (default) or 4 (split).                                                                                                                |
| `OZAKI_PB`       | 1       | Scheme 2 only: CRT prime batching factor (compiled into kernel).                                                                                             |
| `OZAKI_PREFETCH` | 0       | 1 = enable prefetching in Scheme 1 dotprod kernel.                                                                                                           |
| `OZAKI_BOUNDS`   | 0       | 1 = force bounds checking in Scheme 1 dotprod kernel (automatic for non-tile-aligned sizes).                                                                 |
| `OZAKI_SCALAR_ACC` | 0     | 1 = force scalar accumulation in Scheme 1 dotprod kernel.                                                                                                    |
| `NREPEAT`        | 1       | Number of times to repeat the benchmark (for timing measurements).                                                                                           |

Additional environment variables (`OZAKI_PROFILE`, `OZAKI_THRESHOLD`, `OZAKI_STAT`,
`OZAKI_EPS`, `OZAKI_RSQ`, `OZAKI_EXIT`) are handled by the
[LIBXS Ozaki sample](https://github.com/hfp/libxs) which owns the GEMM interceptor,
histogram-based profiling, and accuracy monitoring. See the LIBXS Ozaki README for
their documentation.

The Ozaki context auto-selects XMX-friendly defaults when hardware support is
detected.  For int8 Scheme 1 (default): `BK=32`, `BM=16`, `BN=16`.  For CRT
Scheme 2: XMX uses `BK=32` with fused in-register Garner/Horner.  Common
defaults: `SG=16`.  Scheme 2 auto-default for fp64 is 16 primes (u8) or 19
primes (i8 with `OZAKI_I8=1`).

## Example

```bash
$ ./ozaki.x 256
OpenCL benchmark for Ozaki's methods
GEMM: NN M=256 N=256 K=256 lda=256 ldb=256 ldc=256 alpha=1 beta=1
Ozaki GEMM: 12.345 ms
BLAS  GEMM: 1.234 ms
DIFF: linf=0.000000 linf_rel=0.000000 l2_rel=0.000000 eps=0.000000 rsq=1.000000
```

With Scheme 2 CRT and K-grouping:

```bash
$ OZAKI=2 OZAKI_GROUPS=4 ./ozaki.x 4096
OpenCL benchmark for Ozaki's methods
GEMM: NN M=4096 N=4096 K=4096 lda=4096 ldb=4096 ldc=4096 alpha=1 beta=1
Device: Intel(R) Data Center GPU Max 1550 (GPU)
Ozaki GEMM: 234.567 ms
BLAS  GEMM: 12.345 ms
DIFF: linf=0.000000 linf_rel=0.000000 l2_rel=0.000000 eps=0.000000 rsq=1.000000
```

## Files and Directory Structure

| File/Directory          | Description                                                                        |
|------------------------|------------------------------------------------------------------------------------|
| `ozaki_bench.c`        | Main benchmark driver (initializes context, runs GEMM, compares with BLAS)        |
| `ozaki_gemm.c`         | GEMM implementation (preprocessing pipeline, dotprod launch, buffer management)   |
| `ozaki_opencl.c`       | Context initialization (device selection, kernel compilation, parameter tuning)   |
| `ozaki_gemm3m.c`       | Complex GEMM via 3M (Karatsuba) method - all intermediates stay on device          |
| `ozaki_opencl.h`       | Public API and context structure definitions                                       |
| `ozaki_kernels.h`      | Auto-generated header embedding OpenCL C kernel sources                            |
| `ozaki_tinytc.h`       | Auto-generated header embedding specialized TinyTC SPIR-V kernels                  |
| `kernels/ozaki1_int8.cl` | Scheme 1 OpenCL C kernels (preprocess + XMX/scalar dotprod)                     |
| `kernels/ozaki2_int8.cl` | Scheme 2 OpenCL C kernels (preprocess + fused CRT dotprod)                      |
| `kernels/gemm3m.cl`     | Complex GEMM 3M helper kernels (deinterleave, matadd, finalize)                 |
| `kernels/ozaki_common.cl` | Shared IEEE-754 field extraction helpers                                        |
| `kernels/ozaki1.tinytc`  | TinyTC kernel definition for generic Scheme 1 dotprod                            |
| `kernels/ozaki1_prod_*.tinytc` | Specialized TinyTC definitions (triangular/square)                         |
| `kernels/*.clx`        | Precompiled TinyTC SPIR-V binaries (embedded at build time or loaded at runtime) |

## Performance Tuning

### Scheme Selection

- **Scheme 1 (int8 mantissa slicing)**: Better suited for smaller matrices and when
  high accuracy is needed with moderate slice counts (default: 8 slices for fp64).
  Triangular + symmetrize optimizations reduce slice-pair count from S^2 to S(S+1)/2.
  Use `OZAKI_TRIM` to drop least significant diagonals for speed at the cost of accuracy.

- **Scheme 2 (CRT)**: More efficient for large matrices, especially with K-grouping.
  Default 18 primes provide excellent accuracy. Use `OZAKI_GROUPS=4` on large GEMMs
  to amortize Garner reconstruction cost across K panels - speedups of 3x have been
  observed on Intel Data Center GPU Max (PVC) at sizes >=2048.

### Cache and Memory

- Enable `OZAKI_CACHE=3` when repeatedly calling GEMM with the same A and B matrices
  (e.g., batched inference with fixed weights). Preprocessing is skipped entirely on
  cache hits, yielding 2-5x speedup for the overall GEMM.

- The device memory pool (requires USM/SVM) eliminates allocation overhead across
  repeated calls (`export OZAKI_DEVPOOL=1`). If your workload involves many
  different matrix sizes, consider keeping it disabled.

### TinyTC vs OpenCL C

- TinyTC kernels are specialized at compile time per (ndecomp, trim, scheme) and
  offer 10-30% better performance than the generic OpenCL C kernels for Scheme 1.
  Use `OZAKI_TINYTC=kernels/ozaki1_f64_256x128_n8t3_tri.clx` to load a specific
  variant, or embed them at build time with `-DOZAKI_TINYTC_EMBED` (requires
  `.clx` files in `kernels/`).

- The OpenCL C kernels are always available as fallback and support runtime
  parameterization. TinyTC support is Scheme 1 only.

### Profiling

Profiling is controlled by the [LIBXS Ozaki sample](https://github.com/hfp/libxs)
via the `OZAKI_PROFILE` environment variable. Both GPU and CPU paths push into the
same histogram, and at exit a summary line reports effective GFLOPS/s together with
the implied INT8-TOPS/s (derived from the scheme's decomposition multiplier).
See the LIBXS Ozaki README for details on profile modes and output format.

## Limitations

- Complex GEMM (3M method) is implemented (`ozaki_gemm3m` in `ozaki_gemm3m.c`) but not called from this sample's benchmark driver. It is exercised by the [LIBXS Ozaki sample](https://github.com/hfp/libxs), whose interceptor-based driver dispatches ZGEMM/CGEMM through the GPU 3M path when LIBXSTREAM is available. The `OZAKI_3M` environment variable (0 = original BLAS, 1 = CPU 3M, 2 = GPU 3M) controls the complex GEMM dispatch; see the LIBXS Ozaki README for details.
- TinyTC kernels are Scheme 1 only (Scheme 2 uses OpenCL C kernels).
