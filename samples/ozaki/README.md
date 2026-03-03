# Ozaki Scheme 1 — OpenCL

This sample demonstrates high-precision GEMM emulation via **mantissa slicing**
(Ozaki Scheme 1) fully offloaded to an OpenCL device. It is an OpenCL adaptation
of the CPU-based Ozaki sample in [LIBXS](https://github.com/hfp/libxs).

## Algorithm

Each IEEE-754 mantissa is decomposed into 7-bit int8 slices. The GEMM
`C = alpha * A * B + beta * C` is then computed as a sum of `S*(S+1)/2`
(with triangular + symmetrize) pairwise int8 dot products over `BM×BN×BK`
tiles (default 16×16×16), matching the granularity of GPU matrix engines
(Tensor Cores, XMX, AMD Matrix Cores).

Three OpenCL kernels implement the pipeline:
1. **preprocess_a** — extract sign/exponent/mantissa from A, find per-row
   max exponent, align and slice into int8 panels
2. **preprocess_b** — same for B (per-column)
3. **dotprod** — iterate slice pairs, accumulate scaled int8 dot products
   into double/float C tiles

## Build

```bash
cd samples/ozaki
make [DBG=1]
```

Requires an OpenCL runtime and headers (e.g., `opencl-c-headers`, `ocl-icd-dev`,
or Intel oneAPI).

## Run

```bash
./ozaki_bench M N K [nslices]
```

Environment variables:
- `OZAKI_FLAGS` — bitmask: 1=triangular, 2=symmetrize (default 3)
- `OZAKI_TRIM` — number of least-significant diagonals to skip (default 0)

## Example

```
$ ./ozaki_bench 256 256 256
Ozaki Scheme 1 OpenCL benchmark
M=256 N=256 K=256 nslices=8 flags=3 trim=0
Device: Intel(R) Data Center GPU Max 1550
Ozaki GEMM: 12.345 ms
Ref   GEMM: 45.678 ms
Max absolute diff: 1.234567e-12
Max relative diff: 5.678901e-14
```

## Limitations

- This is an initial proof-of-concept; the dot-product kernel uses scalar int8
  multiply-accumulate. A production version would target hardware matrix
  instructions via `cl_intel_subgroup_matrix_multiply_accumulate` or equivalent.
- Only Scheme 1 (mantissa slicing) is implemented; Scheme 2 (CRT) is not
  included.
- Complex GEMM (3M method) is not yet supported.
