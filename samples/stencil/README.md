# Stencil

Finite-difference stencil computation for seismic wave propagation
(RTM, FWI) on GPUs.  Three kernel paths are available:

- **FP32** (default): dedicated scalar kernel with SLM blocking
- **BF16-DPAS** (STENCIL_BF16=1): Dekker-split BF16 via hardware
  matrix units (DPAS/XMX)
- **INT8-DPAS** (STENCIL_INT8=1): Ozaki-1 INT8 with carried-forward
  exponent

The DPAS paths achieve single-precision accuracy from low-precision
tensor operations.

## Method

The 3D Laplacian is decomposed into three 1D operators applied along
each axis (dimension splitting).  Each 1D operator D is a BLK x BLK
banded Toeplitz matrix (bandwidth 2R+1) representing the FD stencil.

Parameters (compile-time):

    BLK       = 32    block side length (32^3 output cube)
    RADIUS    = 4     half-order (8th-order FD, 9-point stencil)

### FP32 path (default)

The dedicated FP32 kernel uses SLM tiling with a sliding window along
the slow axis.  Work-group size 32x8, cooperative SLM fill, no DPAS
dependency.  Supports XYZ and ZYX memory layouts, PML absorbing
boundaries, and compact/dispersion-fitted operator methods.

### BF16 path (STENCIL_BF16=1)

The wavefield block P and operator D are Dekker-split into BF16 digits.
The matrix product D x P is computed as a sum of BF16 x BF16 DPAS
calls that accumulate into FP32.

    NDIGITS_A = 2     BF16 digits for operator (11-bit range)
    NDIGITS_X = 3     BF16 digits for wavefield

Products per dimension: NDIGITS_A x NDIGITS_X = 6 DPAS calls.
Isotropic total: 3 x 6 = 18 DPAS calls per output block.

### INT8 path (STENCIL_INT8=1)

Ozaki-1 slicing: the FD operator fits in a single 7-bit signed digit
(NSLICES_A=1).  The wavefield is sliced into 1-3 digits at runtime
depending on local dynamic range.

    NSLICES_A = 1     INT8 digits for operator
    NSLICES_X = 1-3   INT8 digits for wavefield (runtime adaptive)
    K_PAD_I8  = 64    K dimension (k=32 DPAS alignment)

Products per dimension: 1 x nslices_eff = 1-3 DPAS calls.
Isotropic total: 3 x (1-3) = 3-9 DPAS calls per output block.

A per-block carried-forward exponent (exp_buf) tracks the dynamic
range across time steps.  Double-buffered output-based tracking avoids
race conditions between neighboring blocks.

## 2D Block I/O

The kernel exploits Intel 2D block load instructions for the A-side
(operator) and sub-group block reads for the B-side (wavefield in SLM):

BF16 path:

    A: intel_sub_group_2d_block_read_16b_8r16x1c
    B: intel_sub_group_2d_block_read_transform_16b_16r16x1c (VNNI)

INT8 path:

    A: intel_sub_group_2d_block_read_8b_8r32x1c (k=32)
    B: intel_sub_group_block_read8 from row-major SLM

The operator D is stored dense (banded zeros included).  The structural
zeros cost no memory bandwidth (D fits in L1 after first access) and no
effective compute (kernel is memory-bound on wavefield reads).

The INT8 kernel gathers the wavefield directly into SLM during the fill
loop (no separate preprocess kernel).

## Build

    make GNU=1 DBG=1 PEDANTIC=2

Requires LIBXS (sibling directory ../../../libxs) and an OpenCL 2.0
runtime with cl_intel_subgroup_2d_block_io and DPAS support.

## Usage

    ./stencil.x [options]

    -n <N>         grid dimension (NxNxN, default 256)
    -nx/ny/nz <N>  individual grid dimensions
    -t <steps>     time steps (default 100)
    -d <dims>      operator terms: 3=isotropic, 9=TTI (default 3)
    -h <spacing>   grid spacing in meters (default 10.0)
    -v <model>     velocity: const | grad | layered | <file.bin>
    -vmin <vel>    minimum velocity m/s (default 1500)
    -vmax <vel>    maximum velocity m/s (default 4500)
    -f <freq>      source frequency Hz (default 25)
    -w <steps>     warmup steps excluded from timing (default 5)

    -seg-salt      preset: SEG/EAGE Salt (676x676x210, h=20m)
    -overthrust    preset: SEG/EAGE Overthrust (801x801x187, h=25m)

Performance is reported as GPoints/s (grid points updated per second).

## Velocity Models

The driver supports loading external velocity models as flat binary
files (float32, x-fastest order).  Several standard models are
publicly available for benchmarking.

### SEG/EAGE Salt Body (3D)

    Grid:    676 x 676 x 210   (n1=210 depth, n2=n3=676 lateral)
    Spacing: 20 m
    Range:   1500 - 4482 m/s
    Size:    ~385 MB (float32)
    Notes:   Salt body with sharp velocity contrasts.
             Standard RTM benchmark (Aminzadeh et al., 1997).

  Original (20 m):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/fvp

  Resampled (40 m, 115x338x338, for cheaper runs):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/BENCH_27PT/v.bin

  Reference wavefield (frequency-domain CBS solution):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/BENCH_27PT/cbs_salt_real.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/BENCH_27PT/cbs_salt_imag.bin

### SEG/EAGE Overthrust (3D)

    Grid:    801 x 801 x 187   (n1=187 depth, n2=n3=801 lateral)
    Spacing: 25 m
    Range:   2179 - 6000 m/s
    Size:    ~450 MB (float32)
    Notes:   Layered geology with thrust faults.  Smoother than Salt.
             Reference: Aminzadeh, Brac, and Kunz (1997).

  Original (25 m):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp

  Resampled (50 m, 94x401x401):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp.r50

  2D section (slice i3=455, 187x801):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/overthrust2D

  Reference wavefield (CBS, 50 m grid):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/BENCH_27PT/v.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/BENCH_27PT/cbs_overthrust_real.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/BENCH_27PT/cbs_overthrust_imag.bin

### Marmousi (2D)

    Grid:    2301 x 751         (n1=751 depth, n2=2301 lateral)
    Spacing: 4 m
    Notes:   Classic 2D migration benchmark (Versteeg, 1994).

  Velocity (vp.bin):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi/vp.bin

  Density (rho.bin):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi/rho.bin

### BP 2004 Velocity (2D, isotropic)

    Grid:    1911 x 12601
    Spacing: 6.25 m
    Notes:   Challenging sub-salt imaging (Billette and Brandsberg-Dahl, 2004).
             Available from SEG open data.

  SEG wiki page (registration may be required):
    https://wiki.seg.org/wiki/2004_BP_Velocity_Estimation_Benchmark_Model

### Valhall 2D (VTI -- anisotropic)

    Grid:    641 x 209           (n1=209 depth, n2=641 lateral)
    Spacing: 25 m
    Type:    Acoustic VTI (vertical transverse isotropy)
    Fields:  Vp, epsilon, delta, eta, Vnmo, density
    Notes:   Representative of North Sea environments.
             Exercises anisotropic operator (pure terms with
             modified coefficients; cross-terms vanish for VTI).

  Download (Geoazur WIND):
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Valhall2D/vp_true.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Valhall2D/epsilon_true.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Valhall2D/delta_true.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Valhall2D/eta_true.bin
    https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Valhall2D/rho_true.bin

### BP 2007 TTI (2D -- tilted anisotropy)

    Grid:    ~1911 x 12480
    Spacing: 6.25 m
    Type:    Acoustic TTI (tilted transverse isotropy)
    Fields:  Vp, epsilon, delta, theta (tilt angle)
    Notes:   Canonical TTI benchmark (Shafiq et al., 2007).
             Exercises full cross-derivative kernel with non-zero
             tilt angles.  This is the standard reference for TTI
             stencil performance.

  SEG wiki page (registration required):
    https://wiki.seg.org/wiki/2007_BP_Anisotropic_Velocity_Benchmark_Model

### Synthetic 3D TTI (from Overthrust)

No public 3D TTI model exists.  The standard practice in GPU stencil
papers is to overlay synthetic Thomsen parameters on the 3D Overthrust
velocity model:

    Vp:      from Overthrust (801x801x187, 25 m)
    epsilon: constant 0.1 - 0.2 (or linear gradient)
    delta:   constant 0.05 - 0.1
    theta:   smooth tilt (e.g., 20-45 degrees, linearly varying with depth)
    phi:     azimuthal angle (0 or smooth variation)

This gives a realistic velocity structure with controlled anisotropy
parameters for benchmarking the TTI kernel (-d 9 mode).

### Data hosting (Geoazur WIND project)

All models above (except BP 2004) are hosted by the WIND project at
Universite Cote d'Azur / Geoazur.  Index page:

    https://www.geoazur.fr/WIND/bin/view/Main/Data/

### Usage with this benchmark

Download the velocity binary and run:

    wget https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Salt/fvp
    ./stencil.x -seg-salt -v fvp -t 1000

    wget https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Overthrust/fvp
    ./stencil.x -overthrust -v fvp -t 1000

The binary format is flat float32 with n1 (depth) as the fastest
dimension.  Velocities are in m/s; the driver squares them internally.
Byte order is native (little-endian on x86).

## Performance Metric

The standard metric for stencil codes is GPoints/s:

    GPoints/s = (nx x ny x nz x nsteps) / (time x 1e9)

This measures throughput independent of the internal algorithm (direct
FD, GEMM-based, FFT, etc.) and is directly comparable across:

    - Devito (Imperial College) -- symbolic PDE -> optimized C/OpenCL
    - minimod (TotalEnergies) -- open-source GPU stencil mini-app
    - Published results on PVC, A770, H100, MI300X

For reference, a memory-bandwidth-bound 8th-order stencil on PVC
(HBM bandwidth ~3.2 TB/s) achieves approximately 200-400 GPoints/s
depending on grid size and occupancy.

## TTI (Anisotropic) Mode

With -d 9, the kernel computes the full tilted transverse isotropy
operator with cross-derivative terms.  Each cross-term uses SLM to
buffer the intermediate result between two GEMM phases:

    T  = D_j x P         (first GEMM)
    T  = c_ij . T        (point-wise anisotropy scaling)
    Y += D_i x T         (second GEMM)

SLM budget per cross-term: 2 x K_PAD x XMX_N x sizeof(ushort),
rounded up to the DPAS K block size.
Total for 3 cross-terms with barrier: fits within 128 KB (Xe2/BMG).

## Operator Paths

The standard direct kernel applies one wide-reach operator (radius-4,
9-point) per dimension.  The compact variants are independent runtime
paths: they compile the same Ozaki-split BF16 x BF16 DPAS primitive
with a smaller gather radius and rely on time evolution to build longer
effective reach.

Methods (selected via STENCIL_METHOD environment variable):

  0 = direct      K=1, r=4  standard high-order (default)
  1 = compact-r1  K=4, r=1  compact tridiagonal runtime path
  2 = compact-r2  K=2, r=2  compact pentadiagonal runtime path
  3 = compact-fit K=4, r=1  placeholder for dispersion-fitted coefficients

The "compact-fit" mode uses the free degrees of freedom in the K-factor
coefficient product to match or exceed the dispersion quality of the
standard 8th-order stencil at target frequencies, while retaining the
memory savings of the cascade.  Coefficients are precomputed once at
initialization for the grid's frequency content.

The first compact path is implemented for the isotropic apply kernel in
methods 1-3.  TTI cross-terms still use the direct two-phase DPAS path.

Environment variables controlling kernel specialization:

    STENCIL_BF16     BF16-DPAS kernel (1=native BF16, 2=FP32-split via BF16)
    STENCIL_BF16S    BF16 split wavefield storage format (0/1)
    STENCIL_INT8     INT8-DPAS kernel (1=native INT8, 2=FP32-split via INT8)
    STENCIL_METHOD   operator method (0-3, default 0)
    STENCIL_STRIPS_PER_WG
         adjacent N-strips handled by one work-group (default: 2)
    STENCIL_SG       subgroup size override (default: device preferred)
    STENCIL_GRF256   force 256-GRF mode (0/1, default: auto)
    STENCIL_TRIM     drop least-significant digit products (BF16 path)
    STENCIL_LU       loop unroll strategy (-1=none, 0=inner, 1=outer)
    STENCIL_LAYOUT   memory layout (0=XYZ, 1=blocked, 2=ZYX)
    STENCIL_HALO     halo/padding size per axis
    STENCIL_PML      enable PML absorbing boundary (0/1)
    STENCIL_FP32_WG_X, STENCIL_FP32_WG_Y
       FP32 direct-kernel work-group shape (default: 32x8)
    STENCIL_FP32_BLOCK_IO
       enable FP32 Intel 2D block reads for padded 32x8 cases (0/1, default: 0)

Specialized kernels are compiled on first use and cached in a
thread-safe registry keyed by the method, compact-step parameters,
  strip grouping, subgroup size, packed specialization flags, FP32
  work-group shape, grid shape, and term count.
Subsequent launches with the same parameters are zero-cost.

Future extension: per-block adaptive method selection (different K in
regions with different velocity/frequency content).  This requires
per-block dispatch logic and is not yet implemented.

## Integration

The stencil API (stencil_opencl.h) accepts device pointers for all
buffer arguments (wavefield, output, velocity).  No host-device
transfers happen inside the dispatch path -- data stays on-device
across the full time-stepping loop.

### Initialization and USM control

LIBXSTREAM provides libxstream_init_config for explicit control over
the memory model before initialization:

    libxstream_init_config_t cfg;
    libxstream_init_config_default(&cfg);  /* all fields = -1 (default) */
    cfg.usm = 1;      /* force Intel USM extensions */
    cfg.device = 0;   /* select device index */
    libxstream_init_config(&cfg);

USM levels (cfg.usm):

    -1  env/default (LIBXSTREAM_USM, or SVM coarse-grain fallback)
     0  disable USM, force clCreateBuffer path
     1  Intel USM extensions (clDeviceMemAllocINTEL)
     2  OpenCL 2.0 SVM coarse-grain only
     3  OpenCL 2.0 SVM with device-reported capabilities

When USM is active (level 1), device pointers from any source
(SYCL, Level Zero, raw clDeviceMemAllocINTEL) can be passed to
the stencil API without conversion.

### SYCL interoperability

SYCL USM device allocations (sycl::malloc_device) can be passed
directly to stencil_apply_laplacian on Intel GPUs.  The underlying
mechanism is clSetKernelArgMemPointerINTEL, which accepts the same
raw pointers that SYCL produces via the Level Zero unified runtime.

Requirements:

    - Initialize LIBXSTREAM with cfg.usm = 1, or set the
      environment variable LIBXSTREAM_USM=1.
    - The SYCL queue and the libxstream OpenCL context must target
      the same physical device (shared Level Zero context).
    - Synchronization: use sycl::queue::ext_oneapi_submit_barrier()
      before calling stencil_apply_laplacian, and
      libxstream_stream_sync after, to order work correctly.
    - No format conversion needed: float32 USM buffers are used
      as-is (velocity is expected as v^2).

### External OpenCL consumers

Any OpenCL application can call the API by passing cl_mem-backed
device pointers obtained via libxstream_mem_allocate, or USM
pointers from clDeviceMemAllocINTEL directly.  The dispatch layer
auto-detects USM availability and selects the appropriate kernel
argument-setting path.

### Scalar proxy (non-DPAS devices)

The kernels include a scalar fallback that runs on any OpenCL 1.2+
device without DPAS/XMX hardware.  This enables functional testing
and correctness verification on iGPUs, CPUs, and other devices.
Performance is not representative -- the proxy exists purely for
development and validation.

## File Layout

    samples/stencil/
      Makefile             build rules
      README.md            this file
      stencil.c            host driver (benchmark, model loading)
      stencil.py           Python benchmark harness (CSV/plot output)
      stencil_opencl.c     OpenCL context, kernel dispatch, exp_buf management
      stencil_opencl.h     public API, compile-time parameters
      stencil_kernels.h    generated at build time from .cl sources
      kernels/
        stencil_common.cl  parameters, gather macros, layout indexing
        stencil_fp32.cl    FP32 path: stencil_apply_direct (default)
        stencil_bf16.cl    BF16 path: stencil_apply, stencil_apply_tti
        stencil_int8.cl    INT8 path: stencil_apply_int8

    libxstream/opencl/
      libxstream_common.h  IEEE utilities, BF16 conversion, EXP2I, unroll macros
      libxstream_ozaki.h   Dekker split, BF16 DPAS macros
