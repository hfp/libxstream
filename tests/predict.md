# Predict Model Fixtures (Format Version 1)

Reference `libxs_predict_save` output in the **v1.0.0** serialization format,
kept so that `libxs_predict_load` can be tested against a released format it
must keep reading. A round-trip test cannot cover this: it only ever exercises
the current writer.

`libxs_predict.h` documents that files written by any released version load, so
each of these must continue to load, and re-saving one must produce a file the
current reader accepts.

| file | model | exercises |
| :--- | :--- | :--- |
| `predict_v1_flat.bin` | flat, `auto` | baseline flat container |
| `predict_v1_flat_c09.bin` | flat, `compress` | compressed clusters |
| `predict_v1_flat_interp.bin` | flat, `interp` | interpolated outputs |
| `predict_v1_flat_rf.bin` | flat, `rf` | random-forest block (100 trees x 16 outputs) |
| `predict_v1_hknn.bin` | hierarchical kNN | per-output clusters, one group per output |
| `predict_v1_hknn_c09.bin` | hierarchical kNN, `compress` | the above, compressed |

All six are 3 inputs (M,N,K) and 16 outputs, built from
`tune_multiply_V100.csv` (157 entries).

## Provenance

Generated with LIBXS at tag `1.0.0` (2026-06-26), which predates every later
format change, so these cannot be produced by the current tree:

```sh
git -C libxs worktree add ../libxs-1.0.0 1.0.0
make -C ../libxs-1.0.0
make -C ../libxs-1.0.0/samples/predict predict_params.x
../libxs-1.0.0/samples/predict/predict_params.x <mode> \
  samples/smm/params/tune_multiply_V100.csv tests/predict_v1_<name>.bin
```

Unreleased or development versions may not be covered by backward compatibility.

## Adding fixtures for a new format version

At each release that changes the layout, add a `predict_v<N>_*.bin` set
generated from that release's tag, and keep the older sets. The version in the
name is what makes a fixture interpretable years later.
