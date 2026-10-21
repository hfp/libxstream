# Tuned Parameters

Each CSV file contains auto-tuned kernel parameters for a specific
GPU, e.g., `tune_multiply_PVC.csv` targets Intel Data Center GPU Max.
The format takes single and double precision in the same file, and the
shipped files hold double precision only.

Each CSV file comes with a prediction model of the same name (`.bin`),
built from its double-precision entries (predictions apply to double
precision only). If every CSV file of a build has a model, the models are
embedded instead of the CSV rows. `smm_params.sh` rebuilds the model
after each merge using LIBXS (`samples/predict`); by hand, e.g.:

```bash
predict_params.x hknn tune_multiply_PVC.csv tune_multiply_PVC.bin
```

At build time, the matching CSV (or its model) is embedded (selected
via `make WITH_GPU=<device>`). At runtime, parameters are selected by
matching the device ID with fallback to the best-matching set. A CSV
can also be loaded explicitly via `OPENCL_LIBSMM_SMM_PARAMS=<path>`,
and then applies to any device.

See the SMM sample README (`samples/smm/README.md`) for details on
auto-tuning, bulk tuning, and parameter management.
