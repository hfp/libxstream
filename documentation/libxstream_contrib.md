# Contributing

This is the contribution policy for LIBXS and LIBXSTREAM. Both projects follow
the same policy, so the text is project-neutral: it is maintained in LIBXS
(`documentation/libxs_contrib.md`) and copied into dependent projects by `make
policies`, the same way `Makefile.inc` is shared. Please edit it in LIBXS — the
copy is overwritten whenever the original changes.

The code base is small, long-lived, and read far more often than it is written,
so uniformity is what keeps diffs reviewable. When a rule below does not answer
your case, the strongest rule is: **match the surrounding code**.

Not every rule is derived from first principles. Some are taste, some are habit,
and some are plain superstition. They are the policy regardless: a uniform code
base is worth more than an individually optimal choice, and rules that are
merely arbitrary are still cheap to follow.

## Character Encoding and Whitespace

| Files                                                          | Encoding        |
| -------------------------------------------------------------- | --------------- |
| Source, headers, kernels, Makefiles, scripts, YAML, plain text | US-ASCII only   |
| Markdown (`*.md`)                                              | any UTF-8       |

No Unicode in code or build files — no typographic quotes, no em dashes, no
non-breaking spaces. Such characters survive editors, diffs, and generators
poorly, and neither the compiler nor `sed` is required to handle them.

Markdown has no such restriction: a spaced em dash (—) instead of ` -- `, an en
dash for numeric ranges (1–4), and mathematical notation (α, ≤, ⌈n/2⌉, ×, ∑) are
all welcome wherever they read better than an ASCII transliteration.

Beyond encoding:

- **No tabs**, except where a Makefile requires them.
- LF line endings. CRLF is rejected.
- No trailing whitespace.
- No whitespace before `#` in a preprocessor directive.
- **No French spacing** in either sense: no space before `,`, `;`, `:`, `!`, or
  `?`, and a single space after a sentence-ending period. This holds for code,
  comments, commit messages, and documentation alike.
- A script carrying a shebang is executable in the index (mode 755); every
  other file is 644.

These are enforced mechanically rather than by review. `.pre-commit-config.yaml`
combines the standard [pre-commit](https://pre-commit.com/) hooks (trailing
whitespace, line endings, byte-order marks, shebang and exec-bit consistency,
YAML syntax) with the project-specific rules (US-ASCII, tabs, C++ comments,
whitespace before `#`, `exit()` in library code, `sed -i` in scripts). Install
the Git hook once per clone; the same configuration runs in continuous
integration, so a violation fails the pull request:

```bash
scripts/tool_normalize.sh --install   # once per clone
scripts/tool_normalize.sh            # check and fix the whole tree
scripts/tool_normalize.sh src        # or one directory
```

`.editorconfig` keeps most of it from happening in the first place. Data files
(`*.csv`, `*.tsv`, `*.dat`) and generated sources are exempt — tabs and line
endings are part of their format, and their producer owns them. Of the rules
above only the two spacing conventions are checked partially: a space before a
comma or semicolon is caught in C sources, the rest is on the author.

## C Source File Structure

A translation unit is strictly grouped, in this order:

1. Includes
2. Macros
3. Types (typedefs, struct definitions)
4. Translation-unit variables (file-scope statics and globals)
5. Functions

No interleaving. A macro used only by the last function still belongs in the
macro section. This makes the shape of every file predictable: what it depends
on, what it configures, what state it holds, and what it does — in that order.

Every file carries the SPDX license header (BSD-3-Clause) verbatim as found in
existing files.

## C Dialect

**C89 (ANSI C) in general.** The pre-submit configuration (`PEDANTIC=2`)
compiles with `-std=c89`, so a C99-only construct breaks the build for everyone
else even when it compiles for you:

- Declarations at the beginning of a block, before any statement. No
  declarations mixed into the middle of a block, and no declaration in a `for`
  initializer.
- `/* ... */` comments only.
- No variable-length arrays, compound literals, designated initializers,
  `long long`, `restrict`, or `//`-style line continuation tricks.
- Newer facilities are reached through the macros and typedefs the public
  headers already provide, not by raising the dialect locally.

Where a C99 (or later) construct is genuinely required, it is guarded and
confined, in the same style as the existing guards.

## Functions

- **A function has a single exit.** No early `return`, no multiple return
  paths, no `goto`.
- Use a `result` variable, gate subsequent work on `EXIT_SUCCESS == result`, and
  return `result` at the single exit point.
- Constants go on the left-hand side of a comparison (`EXIT_SUCCESS == result`,
  `NULL != ptr`), which turns an accidental assignment into a compile error.
- Two blank lines between function definitions.
- At most one blank line inside a function body.

```c
int example(const void* input, void** output) {
  int result = EXIT_SUCCESS;
  if (NULL == input || NULL == output) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    result = prepare(input);
  }
  if (EXIT_SUCCESS == result) {
    result = finish(input, output);
  }
  return result;
}
```

## Comments

- **A comment adds value on top of the code.** Never document what the code
  obviously does — the code already says that. Document what it cannot say: why
  this way, what breaks otherwise, which assumption is being relied on.
- Prefer no comment at all. Then prefer one line.
- **Single-line comments by default.** A multi-line comment has to earn its
  size: it is reserved for what is absolutely necessary, i.e., a risk or a trap
  worth spelling out — typically something that has already been gotten wrong
  once and would be gotten wrong again without the warning. Everything else is
  one line or nothing.
- API documentation in public headers is the other place a multi-line block
  belongs.
- `/* ... */` only. **C++ comments (`//`) are rejected** in `.c` and `.h`.
- **No decorative or banner-style comment blocks.** No `/*==== Section ====*/`,
  no boxes, no ASCII rules. The license header is the sole exception.
- Never write ` -- ` inside a comment (the ASCII rule keeps the em dash out, and
  a double hyphen reads as a typo); rephrase instead.

## Blank Lines

- Never three or more consecutive blank lines, anywhere.
- Exactly two blank lines separate function definitions.
- At most one blank line separates logical blocks inside a function.

## Formatting

A `.clang-format` file is present (LLVM-based, 2-space indent, 96-column limit,
never tabs), and `scripts/tool_clangformat.sh` selects the newest available
version of the tool.

**Do not bulk-reformat as part of a change.** Recent clang-format versions
reflow entire files, which buries a small change in hundreds of unrelated lines
and makes review impossible. This is why clang-format is deliberately absent
from the hook set. Format new and edited code by hand to match the file around
it; the formatter is a maintenance tool, run deliberately and committed on its
own.

Do not mix reformatting, renaming, and behavioural change in one commit.

## Library Code

- Library code does not terminate the process: no direct `exit(...)` in `src/`.
  Return a status and let the caller decide. The macro that wraps the one
  unavoidable case is the single exception.
- Environment variables carry the project's own prefix (`LIBXS_*` or
  `LIBXSTREAM_*`). `scripts/tool_getenvars.sh` lists what the source reads.
- **Header-only mode must keep working.** The amalgamated header
  (`*_source.h`, or the corresponding `-D*_SOURCE`) has to be includable from
  multiple translation units, so a new file-scope symbol in `src/*.c` needs
  internal linkage or the established macro treatment.
- Fix the cause, not the symptom. When a sample, test, or dependent project runs
  into a limitation of the library, change the library rather than working
  around it at the call site.

## Fortran

Fortran sources are **fixed-form** (`.f`). Free-form (`.f90`) is not used;
please do not propose converting them.

## Scripts

- POSIX-portable shell. `sed -i` is rejected: it is not portable (macOS).
- Shell scripts pass `shellcheck`, which the hooks run.
- Python is formatted with `black -l79` and passes `flake8`; `mypy` covers the
  tooling under `scripts/` and `.theme/`, not sample code.
- Where a hook carries an exclusion, it names the open findings it defers. An
  exclusion is a backlog item, not a permission.

## Documentation

Documentation is terse and written for the person *using* the code, not for the
person who wrote it.

- **Every `README.md` is user documentation**: what the thing does, how to enable
  it, how to run it, which environment variables and build knobs exist. Nothing
  more.
- Insights stay out. Design rationale, derivations, and performance analysis do
  not belong in a README — the single exception being a surprising usability
  implication the user has to know about (a knob that silently changes accuracy,
  a mode that only works on one vendor). If it does not change how someone uses
  the code, it is not user documentation.
- A short *why* belongs in the commit message.
- Document every new environment variable. A variable that `tool_getenvars.sh`
  reports but the documentation does not mention is a defect.
- A new page under `documentation/` needs a `nav` entry in `mkdocs.yml`.

Markdown may use any UTF-8, but the PDF is produced through LaTeX, which cannot
render an arbitrary glyph. `PDF_UTF8_SED` in `Makefile.inc` transliterates a
known set (Greek letters, arrows, comparison and set operators, ceiling and
floor brackets). If `make documentation` fails on a character, add it there
rather than removing it from the text.

Parts of `documentation/` are **generated**, and editing the output is lost
work: the landing page comes from `README.md`, the development page from
`scripts/README.md`, one page per sample from `samples/*/README.md`, one page
per test from `tests/*.c`, this page from LIBXS, and the PDFs from all of the
above. Version and amalgamated headers are generated too. Edit the source, then:

```bash
make documentation   # PDFs
make mkdocs          # serve the site with live reload
make mkslides        # serve a slide deck (SLIDES=<topic>)
```

The Makefile is authoritative about what is generated from what.

## Build Systems

**GNU Make is primary.** The `Makefile` and the shared `Makefile.inc` define the
defaults, the knobs, and the behaviour; they are the reference. **CMake is
secondary**: it has to produce the same artifacts, but it does not get to define
anything.

- A new source file, build knob, or changed default lands in the Makefile first,
  then in `CMakeLists.txt`. `CMakeLists.txt` mirrors the Makefile's defaults and
  says so where it matters — when a default moves, move both.
- Every step is taken to keep the two interchangeable *from the consumer's side*.
  In particular, `make` writes the files a CMake consumer expects — the package
  configuration under `lib/cmake/<project>/` alongside the pkg-config `.pc`
  files — so a tree built and installed with GNU Make is usable via
  `find_package()` without CMake ever having run. GNU Make pretending to be
  CMake is a feature, not a workaround.
- Continuous integration builds and installs both ways and then consumes the
  result via `find_package()` and pkg-config. A change that only works in one of
  the two is incomplete.

## Building and Testing

Build without `DBG=1` by default. Debug builds are `-O0`, which makes any
runtime-bearing sample prohibitively slow — never measure performance with one.

```bash
make -j $(nproc)                     # default build
make -j $(nproc) test                # test suite
make -j $(nproc) DBG=1 PEDANTIC=2    # correctness check before submitting
```

`PEDANTIC=2` enables strict warnings and `ANALYZE=1` runs the compiler's static
analyzer; where a project ships `scripts/tool_analyze.sh`, that runs cppcheck on
top. A change is expected to
be warning-free under `DBG=1 PEDANTIC=2`, because that is what continuous
integration builds: GCC, Intel oneAPI, and macOS, covering release and strict
debug configurations as well as a header-only build compiled as C++. See
`.github/workflows/` for the exact matrix.

## ABI and Versioning

The version derives from Git tags and `version.txt`; the shared library carries
an SOVERSION. `scripts/tool_checkabi.sh` compares exported symbols against the
recorded baseline. Do not remove or rename a published symbol — add a new one
instead. Run the checker on a build that has symbol information:

```bash
make STATIC=0 SYM=1
scripts/tool_checkabi.sh
```

A symbol name outside the project's namespace is an error, not a warning.

## Commits and Pull Requests

- One concern per commit. Keep unrelated formatting out of it.
- Subject line: short, capitalized, no trailing period, e.g. `Improved device
  memory allocation`. An area prefix is fine: `CMake: updated to test kernels`.
- Explain *why* in the body when the subject cannot carry it.
- Contributions are accepted under the project's BSD-3-Clause license.

Before submitting:

```bash
scripts/tool_normalize.sh --install       # once per clone, then automatic
scripts/tool_normalize.sh                 # whitespace, encoding, lint
make -j $(nproc) DBG=1 PEDANTIC=2 test    # strict build and tests
scripts/tool_checkabi.sh                  # only if public symbols changed
```

## Workflow with Assistants

If an AI assistant is used, the same policies apply, plus two more:

- Discuss design options and trade-offs before implementing.
- Present options as inline text, not as interactive choice widgets.

`CLAUDE.md` at the repository root deliberately holds no policy of its own: it
points here and adds only the working agreements that are specific to an
assistant. One policy, one place — a second copy drifts, and the drift is
invisible.

`CLAUDE.md`, `.editorconfig`, `.pre-commit-config.yaml`, the lint workflow, and
`scripts/tool_normalize.sh` travel the same one-directional route as this page:
maintained in LIBXS, copied by `make policies`. In a dependent project they are
generated files — edit them in LIBXS.
