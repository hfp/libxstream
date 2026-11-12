# Mixes — OpenCL

Three kernels, each a known mixture of instructions: a STREAM triad
(`mixes_stream`), a GEMM tiled in local memory (`mixes_gemm`), and FP atomics
(`mixes_atomics`, with the expression `libxstream_opencl_flags_atomics` selects
for the device). Timed on a device, they say what each mixture costs there.
Counted by a simulator, they are what another kernel's instruction counts are
expressed in ([tool_score.py](https://github.com/hfp/libxstream/blob/main/scripts/tool_score.py)).

## Build

```bash
cd samples/mixes
make
```

Produces `mixes.x`. `ELEM_TYPE=float` builds single precision (default: `double`).
Requires an OpenCL runtime and LIBXS in a sibling directory.

## Run

```bash
./mixes.x [stream-elements [gemm-order [atomic-slots [launches]]]]
```

Defaults: 1048576 elements, order 256 (rounded up to the tile of 16), 1024 slots
(the atomics accumulate the stream elements into them), and 3 timed launches
after one warm-up. Every result is checked; a mismatch fails the run.
