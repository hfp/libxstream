# Tuned Parameters

Each CSV file contains auto-tuned kernel parameters for a specific GPU, e.g., `tune_multiply_PVC.csv` targets Intel Data Center GPU Max. Parameters for single and double precision coexist in the same file.

At build time, the matching CSV is embedded into the binary (selected via `make WITH_GPU=<device>`). At runtime, parameters are selected by matching the device ID with fallback to the best-matching set. A CSV file can also be loaded explicitly via `OPENCL_LIBSMM_SMM_PARAMS=<path>`.

See the [main README](../README.md) for details on auto-tuning, bulk tuning, and parameter management.
