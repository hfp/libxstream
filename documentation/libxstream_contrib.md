# Contributing

This is the contribution policy for LIBXS and LIBXSTREAM. Both projects follow
the same policy, so the text is project-neutral: it is maintained in LIBXS
(`documentation/libxs_contrib.md`) and copied into dependent projects by `make
documentation`, the same way `Makefile.inc` is shared. Please edit it in LIBXS — the
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
whitespace before `#`, `exit()` in library code, `sed -i` in scripts, `goto`,
a declaration in a `for` initializer, banner comments, ` -- ` in a comment,
three blank lines, and column 73 in fixed-form Fortran). Install the Git hook
once per clone; the same configuration runs in continuous integration, so a
violation fails the pull request:

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

The rules that span more than one line are checked by
`scripts/tool_checkstruct.py`: a single function exit, blank lines inside and
between functions, stacked single-line comments, where an opening brace goes,
whether a multi-line `if` or loop is braced, and a constant on the left-hand
side. It works on the source text, with comments, literals, and
preprocessor directives masked out, so a `return` in `#if` and another in
`#else` count as one exit. A parser is not used on purpose: the declarations
carry `LIBXS_API` and friends, which a preprocessor-less parser turns into an
error node every few lines, and a preprocessed compiler dump no longer says
which file a construct came from. `scripts/tool_checkenvars.sh` compares the
prefixed variables the source reads against what the documentation mentions.

**The policy holds everywhere; the hooks enforce most of it on library code and
public headers.** A sample or a test that bails out of `main()` early is not the
defect a multi-exit library function is, so `single-exit`, `function-gap` and
the rest are scoped to `src`, the public headers and the kernels. Three of them
are not, because they mislead whoever reads the code wherever it sits: a comment
that has come loose from what it documents, two closing braces sharing a column
while their blocks are nested, and a second blank line inside a function body.
Those run over the whole tree.

Both keep their open findings in a to-do file beside them,
`scripts/tool_checkstruct.todo` and `scripts/tool_checkenvars.todo`, rather
than in a file-level exclusion. Three properties follow, and each is the point:

- The list is **per rule and per file**, so a file listed for one rule is still
  checked by the others.
- The list is **per project and not propagated**. The two scripts are policy
  files that `make documentation` copies from LIBXS; a count lowered in a copy
  would be reverted, so the state cannot live inside them.
- The list is a **to-do, not a permission**. Each entry carries what it defers,
  and going the other way fails as well: one finding more than listed is a
  regression, one less means the entry outlived its findings, and an entry
  naming a file that no longer exists is reported too. An exclusion that
  outlives its cause is how a list starts lying.

The list maintains itself in the one direction that is safe. A fix followed by
`tool_normalize.sh` lowers the count, and drops the entry when nothing is left;
an entry whose file is gone drops too. The run still fails and says what it
changed, exactly as the whitespace hooks do, so the smaller list is reviewed and
committed rather than applied silently. Upwards never happens: a count that
grew, or a new undocumented variable, is a regression and stays an error.
`tool_checkstruct.py --counts` prints the table from scratch if a list needs
rebuilding wholesale.

## C Source File Structure

A translation unit is strictly grouped, in this order:

1. Includes
2. Macros
3. Types (typedefs, struct definitions)
4. Translation-unit variables (file-scope statics and globals)
5. Prototypes, where forward declarations are needed
6. Functions

No interleaving. A macro used only by the last function still belongs in the
macro section, and a type or a table used only by the last function belongs in
its own. This makes the shape of every file predictable: what it depends on,
what it configures, what state it holds, and what it does — in that order.
Grouping is also what keeps the sections reviewable: a macro parked next to its
first use hides how many of them the file already has.

Two exceptions, and no others:

- A translation unit split into implementation fragments includes those
  fragments where their code belongs, not at the top. Such an include is a piece
  of the implementation rather than a dependency, and hoisting it would reorder
  the definitions it contains.
- A type may sit immediately above the one entry point it parameterizes, when
  that is what makes the interface readable. This covers an argument or callback
  type in a public header, not an internal type shared by several functions.

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
  `restrict`, or `//`-style line continuation tricks.
- Newer facilities are reached through the macros and typedefs the public
  headers already provide, not by raising the dialect locally.

Where a C99 (or later) construct is genuinely required, it is guarded and
confined, in the same style as the existing guards.

**OpenCL kernels (`*.cl`) follow every rule above except the dialect.** They
are compiled as OpenCL C, which is C99, so a declaration in a `for` initializer
is correct there and the hooks exempt kernels from that one rule. Everything
else holds unchanged: US-ASCII, no tabs, `/* ... */` comments only, a single
function exit, two blank lines between functions, capitalized macros, and the
SPDX header. A kernel is source, not data.

**A 64-bit integer is `uint64_t` or `int64_t`.** They are exactly 64 bits wide
rather than merely at least that wide, they add no name of ours to the API, and
they cost the user nothing: `libxs_macros.h` already includes `<stdint.h>` and
`<inttypes.h>`, so every consumer of the public headers has them, and the API
already returns `uint64_t` from `libxs_hilbert` and `libxs_morton`. `long long`
is kept only where something outside the project spells it — the Intel
intrinsics take `long long*` (`_mm256_i64gather_epi64`), the `__atomic_*_8`
builtins take `long long`, and a value printed with `%llu` is cast to `unsigned
long long`, which is portable without composing the format string out of
`PRIu64`. Those are the reasons `-Wno-long-long` is set for a pedantic build,
and they are the only ones: new code does not reach for `long long` to hold a
number of its own.

## Functions

- **A function has a single exit.** No early `return`, no multiple return
  paths, no `goto`.
- **No trailing underscore on a parameter or a local**, in a definition or a
  declaration. That mark is reserved for a variable a *macro* declares, so that
  such a name cannot collide with one the caller already has in scope; a
  function wearing it anywhere takes the mark away from the one thing it is
  for. Passing an underscored name *to* a function is a different matter and
  fine: inside a macro body that is exactly what the macro's own local is for.
- Use a `result` variable, gate subsequent work on `EXIT_SUCCESS == result`, and
  return `result` at the single exit point.
- Constants go on the left-hand side of a comparison (`EXIT_SUCCESS == result`,
  `NULL != ptr`), which turns an accidental assignment into a compile error.
- Two blank lines between function definitions.
- At most one blank line inside a function body.

```c
int example(const void* input, void** output)
{
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

## Macros

**A macro is capitalized, and so are its parameters**: `LIBXS_ALIGN(POINTER,
ALIGNMENT)`, not `LIBXS_ALIGN(pointer, alignment)`. The parameters are the part
that is easy to forget, and they are the part that matters at the point of use:
a capitalized argument is what tells the reader that the expression may be
evaluated more than once. **A parameter carries no trailing underscore**: that
marks a variable the macro declares itself, and the two must stay apart, since
the whole point of the underscore is to say "this name is mine, not yours".

**A variable a macro declares is the other way round**: lowercase, with a
trailing underscore, and ideally prefixed by the macro's own name.

```c
#define MACRO() { int macro_i_ = 0; }
```

The capitalization tells the reader which names come from the call site and
which the macro invented; the trailing underscore and the prefix are what keep
the invented one from colliding with a variable the caller already has in
scope. A macro that declares plain `i`, `s`, or `p` is a trap for whoever
expands it next to their own `i`. The members of an aggregate the macro
declares are not locals — in `union { float v; float a[8]; } u_` it is `u_`
that carries the underscore — and a `*_DECL(A)` macro that declares a variable
named by its own parameter keeps the caller's spelling.

Two kinds of macro name are lowercase on purpose, and both are exempt:

- A trailing lowercase segment is a token pasted onto the name — a type
  (`LIBXS_TYPECHAR_double`), a lock kind (`LIBXS_LOCK_ACQUIRE_spin`), an
  address space. It has to match the spelling of what it names.
- A macro standing in for something that is not ours keeps that spelling: a
  compiler builtin (`__builtin_nan`), a language keyword the OpenCL-on-CPU
  path defines away (`kernel`, `restrict`), a foreign API (`offloadSuccess`),
  or the lowercase alias a paste target needs (`libxs_crc32_b8`).

## Comments

- **A comment adds value on top of the code.** Never document what the code
  obviously does — the code already says that. Document what it cannot say: why
  this way, what breaks otherwise, which assumption is being relied on.
- Prefer no comment at all. Then prefer one line.
- **Never stack comments.** Two comments with no code between them are either
  one comment — make it one line, or a block if it earns the size — or they
  describe different subjects, and then the code each one describes belongs
  between them. **A blank line between them does not separate them**: it is
  still two comments and no code. This covers blocks as much as single lines;
  the license header, being the file's leading comment, is exempt.
- **Single-line comments by default.** A multi-line comment has to earn its
  size: it is reserved for what is absolutely necessary, i.e., a risk or a trap
  worth spelling out — typically something that has already been gotten wrong
  once and would be gotten wrong again without the warning. Everything else is
  one line or nothing.
- API documentation in public headers is the other place a multi-line block
  belongs.
- A multi-line block opens on its own line and continues with ` * `.
- `/* ... */` only. **C++ comments (`//`) are rejected** in `.c` and `.h`.
- **No decorative or banner-style comment blocks.** No `/*==== Section ====*/`,
  no boxes, no ASCII rules. The license header is the sole exception.
- Never write ` -- ` inside a comment (the ASCII rule keeps the em dash out, and
  a double hyphen reads as a typo); rephrase instead.
- These rules govern the comments a change **writes**. Do not restyle existing
  comments along the way: one that already earned its size keeps it, and
  reflowing a file's comments buries the actual change exactly as a bulk
  reformat does.

```c
/* one line is the default, and most comments need no more */

/**
 * A block earns its size by naming a trap: what was already gotten wrong once,
 * and would be gotten wrong again without the warning.
 */
```

## Blank Lines

- Never three or more consecutive blank lines, anywhere.
- Exactly two blank lines separate function definitions in a `.c` or `.cl`
  file. **A header may use one**, and the files that carry small
  `LIBXS_API_INLINE` definitions are uniform about it: `libxs_gemm.h` and
  `libxs_token.h` use one throughout, `libxs_math.h` for sixteen of its
  twenty-one. One or two, but not a mixture within a file, which is the
  surrounding code a change there has to match.
- At most one blank line separates logical blocks inside a function. Two is what
  separates the functions themselves, so two never appear inside one.

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

**Where the opening brace goes.** It stays on the line of the construct it
belongs to — `if (0 < n) {`, `for (...) {`, `} else {` — with two exceptions,
and both are checked:

- **A function body opens on its own line in an implementation unit**, a `.c` or
  a `.cl` file: 1971 definitions do it and 78 do not.
- **A brace whose parentheses were broken across lines opens on its own line**,
  because the brace is then what tells the reader the condition has ended:

```c
if (0 == first &&
    0 != second)
{
  ...
}
```

**In a header the brace may trail the signature**, and it usually should. What
rules a header is a readable API: the declarations are read as a list, and an
inline definition sits in that list. It is also outside the ABI and meant to
stay a small tool, so a body long enough to want the brace on a line of its own
needs a reason — `libxs_gemm.h` has one. Both forms are accepted there, which is
the difference from an implementation unit; a broken parameter list still takes
the brace onto its own line.

A bare block has no construct line to sit on, so nothing is required of it:
`{ int scope_ = n;` and a brace alone on the line are both in use and both fine.
A struct, a union and an initializer are left alone as well.

**Whatever an `if`, an `else` or a loop controls stays on the keyword's line, or
it is braced.** Once the construct reaches a second line the braces are what say
where it ends, and that includes the case where only the condition wrapped:

```c
while (npos < (int)ctx->text_size
  && 0 != isspace(ctx->text[npos]))
{
  ++npos;
}
```

Each keyword is judged on its own, so an `if` with a braced body and a one-line
`else` is two decisions rather than one. An `else if` is the inner `if` and is
judged there.

Two more places keep the brace on its own line, and the check knows both. One is
a construct whose line is followed by a preprocessor directive: the line above
the brace is then `#endif`, and moving the brace up would carry it into the
branch. The other is a header that is included inside a function body, where the
file's own control flow sits at brace depth zero and is not a definition.

```c
#if defined(LIBXS_CPUID_ARM_MODEL_FALLBACK)
  if (NULL != info)
#endif
  {
    ...
  }
```

A directive in that position also excuses the braces themselves: the keyword and
what it controls then belong to different configurations, and one pair of braces
cannot serve both.

Indentation is not reformatted by a hook, but one thing about it is checked:
**two closing braces in a row must step left.** Sharing a column means they
close blocks that are nested, so a level is missing from the indentation even
though the braces balance and the compiler is content. The check is local and
makes no assumption about the indent unit, and a preprocessor directive between
the two resets it, because which brace belongs to which block then depends on
the configuration. `} else {` is not a closer for this purpose: it reopens, so
the next closer legitimately shares its column.

## Library Code

- Library code does not terminate the process: no direct `exit(...)` in `src/`.
  Return a status and let the caller decide. The macro that wraps the one
  unavoidable case is the single exception.
- Environment variables carry the project's own prefix (`LIBXS_*` or
  `LIBXSTREAM_*`). `scripts/tool_checkenvars.sh --list` lists what the source
  reads, ours and foreign.
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

- Statements occupy columns 7 to 72. Columns 73 and beyond are ignored, so text
  reaching column 73 is **silently truncated** rather than diagnosed.
- A continued line carries `&` in column 6. Sources additionally place a
  trailing `&` in column 73, so the same file also reads as free-form; that
  marker is the only thing allowed at column 73.
- `.fi` files are included into fixed-form sources and follow the same rules.

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
- Document every new environment variable. A variable that
  `tool_checkenvars.sh` reports but the documentation does not mention is a
  defect, and the hook of the same name says so.
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
