# SYRK — OpenCL

Symmetric rank-k and rank-2k updates of [LIBXS](https://github.com/hfp/libxs/blob/main/documentation/libxs_syrk.md) with the panels on an OpenCL device: a benchmark and test of LIBXSTREAM's [SYRK device](../../documentation/libxstream_opencl.md#syrk-device) (`libxstream_syrk.h`). LIBXS keeps the algorithm, local or distributed over MPI, and calls the device (`libxs_syrk_device_t`) for every panel of a plan.

## Build

```bash
cd samples/syrk
make
```

Requires an OpenCL runtime, LIBXS in a sibling directory, and a host BLAS for the reference. `make MPI=1` builds with the MPI wrapper of the C compiler (`MPICC`, default `mpicc`): every rank then holds its share of k, C is block-cyclic on a near-square grid of the ranks (blocks of 64), and the reference is LIBXS' distributed plan on the host.

```bash
make MPI=1
mpiexec -n 4 ./syrk.x 1088 54272 2
```

## Run

```bash
./syrk.x [n [k [nrepeat [trans [upper [syr2k [weighted [single]]]]]]]]
```

Defaults: n=1000, k=n, 3 timed repetitions, then the lower triangle of SYRK in double precision without weights (each flag is 0 or 1). The result is compared with LIBXS on the host. The sample prints the relative error, and the best time and rate of LIBXS on the host and on the device, with A copied per call or resident. A mismatch fails the run.

```bash
./syrk.x 1088 54272 2         # a shape of RI-RPA (CP2K's contract_S_to_Q)
./syrk.x 1088 54272 2 0 0 0 1 # weighted, as with the frequencies of RPA
```

The sample prints what computed the panels: the device BLAS by its plugin's name (e.g., `mkl`), `kernel`, or `host` for panels the device declined (`LIBXSTREAM_SYRK_BLAS=0` takes the kernel).
