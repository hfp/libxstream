# Stencil BF16-DPAS

Finite-difference stencil computation using Dekker-split BF16 and
hardware matrix units (DPAS/XMX) to achieve single-precision accuracy
from low-precision tensor operations.

Target application: seismic wave propagation (RTM, FWI) on GPUs.

## Method

The 3D Laplacian is decomposed into three 1D operators applied along
each axis (dimension splitting).  Each 1D operator D is a BLK x BLK
banded Toeplitz matrix (bandwidth 2R+1) representing the FD stencil.

The wavefield block P (BLK x BLK^2) and operator D are Dekker-split
into BF16 digits.  The matrix product D x P is then computed as a sum
of BF16 x BF16 DPAS calls that accumulate into FP32.

Parameters (compile-time):

    BLK       = 32    block side length (32^3 output cube)
    RADIUS    = 4     half-order (8th-order FD, 9-point stencil)
    NDIGITS_A = 2     BF16 digits for operator (11-bit range)
    NDIGITS_X = 3     BF16 digits for wavefield

Products per dimension: NDIGITS_A x NDIGITS_X = 6 DPAS calls.
Isotropic total: 3 x 6 = 18 DPAS calls per output block.
TTI total:       9 x 6 = 54 DPAS calls per output block.

## 2D Block I/O

The kernel exploits Intel 2D block load instructions for both the
A-side (operator) and B-side (wavefield):

    A: intel_sub_group_2d_block_read_16b_8r16x1c
    B: intel_sub_group_2d_block_read_transform_16b_16r16x1c (VNNI)

The operator D is stored dense (banded zeros included).  The structural
zeros cost no memory bandwidth (D fits in L1 after first access at
4 KB) and no effective compute (kernel is memory-bound on wavefield
reads at 192 KB per dimension per digit).

A preprocess kernel gathers wavefield data along strided dimensions
(y, z) into K-contiguous BF16 surfaces that satisfy the 2D block I/O
surface constraints (width >= 64 bytes, pitch 16-byte aligned).

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

SLM budget per cross-term: 2 x K_PAD x XMX_N x sizeof(ushort) = 2 KB.
Total for 3 cross-terms with barrier: fits within 128 KB (Xe2/BMG).

## Operator Cascade (Densification)

The standard kernel applies one wide-reach operator (radius-4, 9-point)
per dimension.  The cascade variants factor this into K sub-steps each
with a smaller radius r, such that K*r >= R_target.  Sub-step results
stay in registers (no memory round-trip), trading extra DPAS calls for
reduced halo size and fully dense operator matrices.

Methods (selected via STENCIL_METHOD environment variable):

    0 = sparse    K=1, r=4  standard high-order (default)
    1 = dense     K=4, r=1  pure cascade, tridiagonal, minimal halo
    2 = hybrid    K=2, r=2  balanced (pentadiagonal, half halo)
    3 = best      coefficients dispersion-optimized (static)

The "best" mode uses the free degrees of freedom in the K-factor
coefficient product to match or exceed the dispersion quality of the
standard 8th-order stencil at target frequencies, while retaining the
memory savings of the cascade.  Coefficients are precomputed once at
initialization for the grid's frequency content.

Environment variables controlling kernel specialization:

    STENCIL_METHOD   operator method (0-3, default 0)
    STENCIL_BK       K-unroll block size (default: K_PAD)
    STENCIL_SG       subgroup size override (default: device preferred)
    STENCIL_GRF256   force 256-GRF mode (0/1, default: auto)

Specialized kernels are compiled on first use and cached in a
thread-safe registry keyed by the (method, k_steps, r_per_step, sg)
tuple.  Subsequent launches with the same parameters are zero-cost.

Future extension: per-block adaptive method selection (different K in
regions with different velocity/frequency content).  This requires
per-block dispatch logic and is not yet implemented.

## File Layout

    samples/stencil/
      Makefile             build rules
      README.md            this file
      stencil.c            host driver (benchmark, model loading)
      stencil_opencl.c     OpenCL context, kernel dispatch
      stencil_opencl.h     public API
      stencil_kernels.h    generated at build time from .cl sources
      kernels/
        stencil_common.cl  parameters, includes libxstream_ozaki.h
        stencil_bf16.cl    preprocess_x, stencil_apply, stencil_apply_tti

    libxstream/opencl/
      libxstream_common.h  IEEE utilities, BF16 conversion, EXP2I
      libxstream_ozaki.h   Dekker split, BF16 DPAS macros
