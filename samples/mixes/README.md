# Mixes — OpenCL

Kernels that are each a known mixture of instructions: a STREAM triad
(`mixes_stream`), a GEMM tiled in local memory (`mixes_gemm`), FP atomics
(`mixes_atomics`, with the expression `libxstream_opencl_flags_atomics` selects
for the device), small blocks gathered by index and multiplied in local memory
(`mixes_gather`), and an empty kernel over as many work-items as the stream
(`mixes_empty`), which is what dispatching them costs. Timed on a device, they
say what each mixture costs there. Counted by a simulator, they are what another
kernel's instruction counts are expressed in ([tool_score.py](https://github.com/hfp/libxstream/blob/main/scripts/tool_score.py)).

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
after one warm-up. The gather covers as many 16x16 blocks as the stream has
elements. Every result is checked; a mismatch fails the run.
