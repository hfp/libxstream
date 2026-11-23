# SYRK — OpenCL

Symmetric rank-k and rank-2k updates of [LIBXS](https://github.com/hfp/libxs/blob/main/documentation/libxs_syrk.md) with the panels on an OpenCL device. LIBXS keeps the algorithm, local or distributed over MPI, and calls this sample's device (`libxs_syrk_device_t`) for every panel of a plan. The device also offers the weighted form C := alpha\*op(A)\*diag(d)\*op(A)^T + beta\*C, and keeps A on the device across calls if asked to (`syrk_opencl_resident`).

## Build

```bash
cd samples/syrk
make
```

Requires an OpenCL runtime, LIBXS in a sibling directory, and a host BLAS for the reference.

## Run

```bash
./syrk.x [n [k [nrepeat [trans [upper [syr2k [weighted [single]]]]]]]]
```

Defaults: n=1000, k=n, 3 timed repetitions, then the lower triangle of SYRK in double precision without weights (each flag is 0 or 1). The result is compared with LIBXS on the host. The sample prints the relative error, and the best time and rate of LIBXS on the host and on the device, with A copied per call or resident. A mismatch fails the run.

```bash
./syrk.x 1088 54272 2         # a shape of RI-RPA (CP2K's contract_S_to_Q)
./syrk.x 1088 54272 2 0 0 0 1 # weighted, as with the frequencies of RPA
```

## Use

```C
syrk_opencl_t* ctx;
libxs_syrk_device_t device;
syrk_opencl_create(&ctx);            /* on the active device */
syrk_opencl_device(ctx, &device);
shape.device = &device;              /* libxs_syrk_shape_t, also with shape.dist */
libxs_syrk_dispatch(&plan, &shape);
syrk_opencl_weights(ctx, d);         /* optional, per call */
libxs_syrk(&plan, alpha, a, NULL, beta, c);
```

The kernel (`kernels/syrk.cl`) computes a panel in blocks of 64x64 (16x16 work-items with 4x4 elements each) through local memory, and the host merges the panel's triangle into C. A panel the device cannot compute (e.g., no fp64) is declined, and LIBXS computes it on the host.
