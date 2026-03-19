# Scripts

Helper scripts for building, inspecting, and maintaining LIBXSTREAM.

## tool\_checkabi.sh

Checks the ABI (Application Binary Interface) stability of the LIBXSTREAM shared library. The script uses `nm` to extract exported symbols from `lib/*.so` (or `lib/*.a` as fallback) and verifies that no previously published symbol has been removed or renamed compared to an earlier baseline (`.abi.txt`). Non-conforming symbol names cause an immediate error.

```bash
scripts/tool_checkabi.sh
```

**NOTE**: For full coverage the library must be built with `make STATIC=0 SYM=1` (or `DBG=1`) so that symbol information is present.

## tool\_getenvars.sh

Scans the LIBXSTREAM source tree (`src/*.c`) for calls to `getenv` and prints a sorted, deduplicated list of every environment variable used at runtime, separated into LIBXSTREAM-specific (`LIBXSTREAM_*`) and other variables.

```bash
scripts/tool_getenvars.sh
```

## tool\_opencl.sh

Converts OpenCL kernel files (`.cl`) — and optionally CSV parameter files — into a C header whose string literals can be compiled straight into the host binary. Include guards are stripped by default, and `#include` directives inside kernels are recursively resolved (inline).

```bash
scripts/tool_opencl.sh [options] infile.cl [infile2.cl ..] [infile.csv ..] outfile.h
```

| Flag | Description |
|---|---|
| `-k`, `--keep` | Keep include guards (normally stripped) |
| `-b N`, `--banner N` | Copy the first N lines of the first `.cl` file as a banner |
| `-p DIR`, `--params DIR` | Directory containing CSV parameter files |
| `-c`, `-d`, `--debug`, `--comments` | Preserve comments in the generated source |
| `-v`, `--verbose` | Echo the command line |

## tool\_version.sh

Derives the project version from Git tags and revision count. It can emit a plain version string, individual version components, or a complete C header with `#define` guards.

```bash
# full version string (branch-tag-patch)
scripts/tool_version.sh

# generate a C version header with a given symbol prefix
scripts/tool_version.sh LIBXSTREAM -1

# single component: 1=major, 2=minor, 3=update, 4+=patch
scripts/tool_version.sh LIBXSTREAM 2
```

| Argument | Description |
|---|---|
| `PREFIX` | Symbol prefix used in the generated header (uppercased) |
| `CMPNT` | `0` (default) — version string; negative — C header; `1`‒`3` — major/minor/update; `>3` — patch count |
| `SHIFT` | Added to the patch count (default: `0`) |
