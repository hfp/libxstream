# Scripts

Helper scripts for building, inspecting, and maintaining LIBXSTREAM.

## tool\_checkabi.sh

Checks the ABI (Application Binary Interface) stability of the LIBXSTREAM shared library. The script uses `nm` to extract exported symbols from `lib/*.so` (or `lib/*.a` as fallback) and verifies that no previously published symbol has been removed or renamed compared to an earlier baseline (`.abi.txt`). Non-conforming symbol names cause an immediate error.

```bash
scripts/tool_checkabi.sh
```

**NOTE**: For full coverage the library must be built with `make STATIC=0 SYM=1` (or `DBG=1`) so that symbol information is present.

## tool\_checkenvars.sh

Scans the sources and public headers (`*.c` and `*.h` under `src/` and `libxstream/`) for calls to `getenv`. With `--list` it prints a sorted, deduplicated list of every environment variable used at runtime, separated into LIBXSTREAM-specific (`LIBXSTREAM_*`) and other variables. Without an argument it reports the `LIBXSTREAM_*` variables that no Markdown file documents, which is what the pre-commit hook of the same name runs; `--all` reports the deferred ones as well, i.e., those listed in `scripts/tool_checkenvars.todo`.

```bash
scripts/tool_checkenvars.sh --list
scripts/tool_checkenvars.sh
```

## tool\_checkstruct.py

Checks the structural rules of the contribution policy that span more than one line (e.g., a single function exit, blank lines). Run by the pre-commit hooks.

```bash
scripts/tool_checkstruct.py FILE...   # report violations, exit 1 if any
scripts/tool_checkstruct.py --list    # name the checks
```

## tool\_clangformat.sh

Runs `clang-format` with all arguments forwarded, preferring the newest `clang-format-N` found. `make reformat` uses it.

```bash
scripts/tool_clangformat.sh -style=file:.clang-format -i src/libxstream.c
```

## tool\_normalize.sh

Front-end for the rules in `.pre-commit-config.yaml` (requires `pre-commit`).

```bash
scripts/tool_normalize.sh --install   # install the Git hook (once per clone)
scripts/tool_normalize.sh [DIR]       # check and fix the tree, or just DIR
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
| `-I DIR`, `--include DIR` | Search path for angle-bracket `#include <...>` directives inside kernels (repeatable) |
| `-c`, `-d`, `--debug`, `--comments` | Preserve comments in the generated source |
| `-v`, `--verbose` | Echo the command line |
| `-h`, `--help` | Print usage |

## tool\_pkgversion.sh

Synchronizes the packaging metadata (RPM spec, Debian changelog) with `version.txt`, changing files only if the version differs. The Makefile runs it when it generates the version header.

```bash
scripts/tool_pkgversion.sh
```

## tool\_source.sh

Prints the header-only header (`libxstream_source.h`) that includes every library source file; an optional argument sets the directory the includes refer to (default: `../src`). `make header-only` uses it.

```bash
scripts/tool_source.sh >libxstream_source.h
```

## tool\_version.sh

Derives the project version from `version.txt` (`MAJOR.MINOR.PATCH`) and, as patch count, the number of non-merge commits since the nearest `MAJOR.MINOR.PATCH` tag (or all commits if there is none). It can emit a plain version string, individual version components, or a complete C header with `#define` guards.

```bash
# full version string (branch-version-patch)
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
