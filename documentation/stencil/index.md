# Stencils as Small Matrix Multiplications

## RTM, TTI, long-leg stencils, and BF16-DPAS

LIBXSTREAM stencil sample

30-minute technical overview

---

## Talk Goal

- Show what the stencil sample implements today.
- Explain why seismic stencils have long legs.
- Map stencil families to small matrix multiplications.
- Show how BF16-DPAS and Ozaki splitting preserve FP32-like accuracy.
- Motivate compact staged operators as an alternative to explicit long legs.

---

## 30 Minute Route

| Time      | Topic                                   |
|-----------|-----------------------------------------|
| 0-5 min   | Application: RTM and wave propagation   |
| 5-10 min  | Stencil math and long legs              |
| 10-18 min | Mapping stencils to GEMM/DPAS           |
| 18-24 min | TTI and compact staged paths            |
| 24-30 min | Implementation, performance, discussion |

---

## Why Seismic Stencils?

Reverse Time Migration (RTM) and Full Waveform Inversion (FWI)  
solve wave equations over 3D grids.

$$p_{next} = 2\,p_{now} - p_{prev} + \Delta t^2\, v^2\, \mathcal{L}(p_{now})$$

- $p$: pressure wavefield
- $v$: velocity model
- $\mathcal{L}$: spatial differential operator

The hot loop is repeated stencil evaluation over (many) grid points.

---

## Implemented in the Sample

The current sample is a GPU stencil benchmark and integration example.

| Mode                          | CLI            | Implemented path                             |
|-------------------------------|----------------|----------------------------------------------|
| Isotropic RTM-style Laplacian | `-d 3`         | fused 3-axis BF16-DPAS apply                 |
| TTI-style anisotropic terms   | `-d 9`         | pure terms plus cross-derivative DPAS phases |
| Direct high-order stencil     | `-m 0`         | radius-4 per axis                            |
| Compact staged variants       | `-m 1`, `-m 2` | radius-1/radius-2 compact runtime paths      |
| Staged fitting hook           | `-m 3`         | placeholder for fitted compact coefficients  |

---

## Block View

The sample updates one `32 x 32 x 32` output cube per block.

```text
BLK       = 32
RADIUS    = 4       direct 8th-order FD
K_BASE    = BLK + 2 RADIUS = 40
K_PAD     = align16(K_BASE) = 48
XMX tile  = 8 x 16
```

For one axis, each block becomes a matrix multiplication.

```text
D: 32 x 48          operator rows and haloed line
P: 48 x 1024        gathered wavefield lines
Y: 32 x 1024        output block contribution
```

---

## Direct Long-Leg Stencil

An 8th-order second derivative has a radius-4 stencil.

$$u''(i) \approx c_0\,u(i) + \sum_{k=1}^{4} c_k\bigl[u(i-k) + u(i+k)\bigr]$$

Each output point reads nine positions along one axis.  
In 3D isotropic mode this is applied along $x$, $y$, and $z$.

---

## Long Legs as a Matrix

The 1D stencil is a banded operator matrix.

$$D = \begin{bmatrix}
c_4 & c_3 & c_2 & c_1 & c_0 & c_1 & c_2 & c_3 & c_4 & 0 & \cdots \\\\
0 & c_4 & c_3 & c_2 & c_1 & c_0 & c_1 & c_2 & c_3 & c_4 & \cdots \\\\
& & & & \ddots & & & & & &
\end{bmatrix}$$

$$Y = D\,P$$

The sample stores $D$ as a dense BF16 surface because DPAS wants regular tiles.  
The zeros are structural convenience.

---

## Three Isotropic GEMMs

The isotropic Laplacian separates into three 1D operators.

$$\mathcal{L}(p) = D_x\,p + D_y\,p + D_z\,p$$

For each axis, the kernel gathers a `K_PAD x XMX_N` panel into SLM
and applies the same DPAS micro-kernel.

```text
for dim in x, y, z:
    gather haloed lines into SLM
    split wavefield into BF16 digits
    accumulate D_dim * P_dim into FP32
```

---

## Low Precision Compute

Use matrix compute units but without giving up accuracy.

Ozaki/Dekker-style splitting represents FP32 values as BF16 digit sums.

$$A \approx A_0 + A_1 \qquad (p = 2)$$

$$X \approx X_0 + X_1 + X_2 \qquad (q = 3)$$

$$A\,X \approx \sum_i \sum_j A_i\,X_j$$

Each $A_i X_j$ product is BF16 × BF16 DPAS with FP32 accumulation.

---

## DPAS Work Count

For each 1D operator:

$$p \times q = 2 \times 3 = 6 \text{ DPAS products}$$

| Operator family   | Matrix work per output block        |
|-------------------|-------------------------------------|
| Isotropic, direct | 3 axes × 6 products                 |
| TTI pure terms    | 3 axes × 6 products                 |
| TTI cross terms   | two DPAS phases per cross term      |
| Compact staged    | same DPAS primitive, smaller radius |

The shape is always small and regular: `8 x 16` DPAS tiles over the K dimension.

---

## Hardware Mapping

The kernel uses Intel GPU matrix and block I/O features.

```text
A-side: 8 rows x 16 K values   operator D
B-side: 16 K x 16 columns      wavefield panel
C:      8 rows x 16 columns    FP32 accumulator
```

- `BF16_LOAD_A`: 2D block read of operator rows
- `B` panel: VNNI-transformed local BF16 data
- `BF16_DPAS_ONE`: one 16-wide K step
- SLM holds the gathered and split wavefield panel

---

## TTI: Why It Is Different

Tilted Transverse Isotropy introduces mixed derivatives.

$$\mathcal{L}_\text{TTI}(p) = \text{pure terms} + \text{cross terms}$$

$$\text{cross term: } D_i\bigl(c_{ij} \cdot D_j\,p\bigr)$$

This is not just a wider 1D stencil. It is a composition of two
directional derivatives with a pointwise anisotropy field in between.

---

## TTI as Two GEMM Phases

The sample implements each cross term as a two-phase DPAS pipeline.

$$T = D_j\,P \qquad \text{(first GEMM)}$$
$$T = c_{ij} \cdot T \qquad \text{(pointwise scaling)}$$
$$Y \mathrel{+}= D_i\,T \qquad \text{(second GEMM)}$$

Implementation details:

- `x_slm`: gathered wavefield digits
- `t_slm`: BF16 re-split intermediate
- `stencil_apply_tti`: cross-derivative kernel
- Pure terms still use the isotropic `stencil_apply` path

---

## Stencil Kinds and GEMM Shapes

| Stencil kind        | Math                       | GEMM form                          |
|---------------------|----------------------------|------------------------------------|
| 1D direct FD        | $D_i\,P$                   | $32 \times 48$ by $48 \times 1024$ |
| 3D isotropic        | $D_x P + D_y P + D_z P$    | three independent 1D GEMMs         |
| VTI-like pure terms | scaled pure axes           | three GEMMs plus coefficients      |
| TTI cross term      | $D_i(c_{ij}\,D_j\,P)$      | GEMM, scale, GEMM                  |
| Compact staged      | repeated compact evolution | smaller-radius $D_r\,P$ per step   |

The key design choice is to make the stencil look like many dense,  
small, predictable GEMMs.

---

## Long-Leg Motivation

High-order FD stencils use long spatial legs to reduce dispersion error.

| Benefit | Cost on large 3D grids |
|---------|------------------------|
| better wave propagation accuracy | wider block halos |
| fewer time-step artifacts | more distant memory accesses |
| familiar RTM/TTI formulation | more L2/TLB pressure |

Can time evolution provide the effective reach while each update  
touches only a compact neighborhood?
<span style="opacity: 0.3;">Yes, approximately.</span>

Note: Repeated compact updates compose into a wider domain of dependence.
The hard part is fitting the compact coefficients so the composed symbol
matches the long-leg stencil's dispersion behavior.

---

## Compact Staged Idea

Instead of applying the long-leg radius-4 operator directly,  
use compact operators over time.

```text
direct:       one radius-4 operator

staged-r1:    compact radius-1 operator, K=4
staged-r2:    compact radius-2 operator, K=2
staged-fit:   future fitted compact coefficients
```

Current implementation uses compact runtime paths for isotropic mode.
The longer effective behavior is intended to arise from repeated time updates,
not from loading the long halo every time.

---

## What Is Implemented Today

| Method     | CLI    | Radius used by kernel | Status                   |
|------------|--------|-----------------------|--------------------------|
| Direct     | `-m 0` | `r=4`                 | baseline high-order path |
| Staged r1  | `-m 1` | `r=1`                 | compact isotropic path   |
| Staged r2  | `-m 2` | `r=2`                 | compact isotropic path   |
| Staged fit | `-m 3` | `r=1`                 | placeholder coefficients |
| TTI        | `-d 9` | direct two-phase      | cross terms implemented  |

Important caveat: fitted dispersion coefficients are not implemented yet.

---

## Kernel Structure

```text
host:
  choose method, radius, strip grouping
  precompute BF16 operator surfaces
  JIT OpenCL with method-specific constants

kernel stencil_apply:
  for dim in active pure terms:
    gather compact/direct halo into SLM
    split wavefield into BF16 digits
    DPAS accumulate into FP32
  write leapfrog update

kernel stencil_apply_tti:
  GEMM -> scale -> re-split -> GEMM
```

---

## Runtime Controls

```text
./stencil.x -n 800 -d 3 -m 1

-d 3    isotropic pure terms
-d 9    TTI-style pure plus cross terms
-m 0    direct radius-4
-m 1    staged-r1
-m 2    staged-r2
```

Useful environment controls:

```text
STENCIL_STRIPS_PER_WG=2   default, best measured grouping
STENCIL_TRIM=N            accuracy/performance tradeoff
STENCIL_GRF256=1          tested slower on target system
```

---

## Target-System Snapshot

Measured on TODO, `./stencil.x -n 800 -d 3`.

TODO

`TRIM` drops least-significant digit products, so it is
a controlled accuracy tradeoff, not the default.

---

## Demo Script

Build on the target system:

```bash
git clone https://github.com/hfp/libxs.git
git clone https://github.com/hfp/libxstream.git
cd libxstream/samples/stencil
echo "Make OpenCL runtime available"
make GNU=1
```

Run the useful comparison (`stencil.py`):

```bash
./stencil.x -m 0 -n 800 -d 3
./stencil.x -m 1 -n 800 -d 3
./stencil.x -m 2 -n 800 -d 3
STENCIL_TRIM=3 ./stencil.x -m 1 -n 800 -d 3
```

---

## Where to Go Next

1. Add correctness and norm comparisons against direct/reference paths.
2. Implement real `staged-fit` coefficient generation for dispersion targets.
3. Extend compact staging to TTI cross terms, or explain why not.
4. Study accuracy/performance of `STENCIL_TRIM` on realistic wavefields.
5. Compare Salt/Overthrust model runs, not only synthetic gradient velocity.

---

## Takeaway

Seismic stencils as dense small matrix multiplications.

- BF16-DPAS with FP32 accumulation via Ozaki splitting
- Compact staged paths: long-leg reach, short-leg cost
- TTI: pure terms + GEMM-scale-GEMM cross terms
- RTM isotropic: three directional GEMMs

The hook: expressing stencil structure so matrix engines can execute it.

---

## OpenCL?

OpenCL is interoperable with respective vendor model, e.g., SYCL.

- Intel: Driver and SYCL already deliver OpenCL, otherwise
  - Install opencl-c-headers, ocl-icd-libopencl1, ocl-icd-opencl-dev
  - Install https://github.com/intel/compute-runtime
- Nvidia: Driver and CUDA already deliver OpenCL

---

## LIBXSTREAM

Minimal compiler requirements (C90), e.g., GNU\* Compiler.

- API to ease buffer management and carrying kernel code
- Based on LIBXS, can make use of powerful primitives
  - For example, predicting tuning parameters

Leverage runtime code generation to specialize kernels (JIT).
