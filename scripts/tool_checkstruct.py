#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Maintained in LIBXS and copied into dependent projects by "make documentation".
# Edit it in LIBXS: a change made in a copy is overwritten.
"""Check the structural rules of the contribution policy.

These rules span more than one line, so a pattern-based hook cannot express
them. A parser cannot either: the declarations carry LIBXS_API and friends,
so tree-sitter reports an error node every few lines, and a preprocessed
GCC dump no longer says which file a construct came from. What is left is
the source text itself, which is what the rules are about anyway.

The checks are lexical. Comments, literals, and preprocessor directives are
blanked out first, then braces are counted: a "{" at depth zero preceded by
")" opens a function body. Exits are counted per configuration rather than
per file, so a return in #if and another in #else are one exit, not two.

  tool_checkstruct.py FILE...   report violations, exit 1 if any
  tool_checkstruct.py --list    name the checks and exit
"""

import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

# Matched at a known line start, so anchoring is positional, not by "^".
CONDITIONAL = re.compile(r"[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b")
RETURN = re.compile(r"\breturn\b")
COMMENT = re.compile(r"^([ \t]*)/\*.*\*/[ \t]*$")
# Checked on the masked text, so a comment mentioning "== NULL" is not a hit.
CONSTANT = re.compile(
    r"[\w)\]][ \t]*[!=]=[ \t]*(NULL|EXIT_SUCCESS|EXIT_FAILURE)\b"
)
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\(([^)]*)\))?")
BLOCK = re.compile(r"/\*.*?\*/")
# All-caps, with at most one trailing lowercase segment: that segment is a
# token pasted onto the name (a type, a keyword, an address space, a lock
# kind), which is why it is not capitalized.
MACRO = re.compile(r"^[A-Z][A-Z0-9_]*(_[a-z0-9_]+)?$|^_[A-Z0-9_]+$")
LITERAL = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
# A declaration inside a macro body. The multi-word types come first, or
# "unsigned long long x" reads as the type "long" declaring a variable "long".
BASE = (
    r"(?:unsigned\s+|signed\s+)?(?:long\s+long|long\s+double|long\s+int"
    r"|short\s+int|long|short|int|char|float|double|size_t|ptrdiff_t"
    r"|uchar|ushort|uint|ulong|half|bool|u?int(?:8|16|32|64)_t"
    r"|__m\d+\w*|\w+_t)"
)
LOCAL = re.compile(
    r"(?:const\s+|volatile\s+|static\s+|register\s+)*"
    + BASE
    + r"(?:\s*\*+)?\s+(\w+)\s*(?=[;,=\[)])"
)
# The members of an aggregate are not locals: in "union { float v; float a[8];
# } u_ " the name that has to carry the underscore is u_, not v or a.
MEMBERS = re.compile(r"\b(?:struct|union|enum)\b[^{;]*\{[^{}]*\}")
KEYWORDS = (
    "long",
    "short",
    "int",
    "char",
    "float",
    "double",
    "unsigned",
    "signed",
    "const",
    "void",
)
Finding = Tuple[str, int, str]
# Suppression is per rule and per file, not per file: a file that is listed
# for one rule is still checked by the others.
#
# EXEMPT is permanent. These files define names that are not ours, so the
# names keep the spelling they have elsewhere: compiler builtins and language
# keywords, a CP2K name, the lowercase aliases LIBXS_CRC32(N) pastes onto, and
# the comment style of the DBCSR-derived header.
EXEMPT = (
    ("macro-name", "libxs/libxs_macros.h"),
    ("macro-name", "src/libxs_crc32.h"),
    ("macro-name", "libxstream/libxstream_cp2k.h"),
    ("macro-name", "libxstream/opencl/libxstream_cpu_begin.h"),
    ("macro-name", "libxstream/opencl/libxstream_cpu_end.h"),
    ("macro-parameter", "libxstream/libxstream_dbcsr.h"),
    ("stacked-comments", "libxstream/libxstream_dbcsr.h"),
)
# The backlog is not here: it is per-project state, and this script is a
# policy file that "make documentation" copies into dependent projects, which
# would revert a count lowered in the copy. It lives next to this script, in
# tool_checkstruct.todo, which is NOT propagated.
TODO = "tool_checkstruct.todo"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = (
    "single-exit",
    "function-gap",
    "blank-in-function",
    "stacked-comments",
    "constant-left",
    "macro-name",
    "macro-parameter",
    "macro-local",
    "closer-nesting",
)


def mask(
    text: str, directives: bool = False
) -> Tuple[str, List[Tuple[int, str]]]:
    """Blank comments, literals, and directives; keep offsets and newlines.

    Returns the masked text plus the conditional directives as
    (offset, keyword), which the exit count needs and the mask removed.
    Directive text is blanked because a macro body carries braces and
    returns that would derail the brace count; pass directives=True where
    the check is line-local and wants to see into a macro definition.
    """
    out = list(text)
    conds: List[Tuple[int, str]] = []
    quote = ""
    state = "code"
    start = 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if "code" == state:
            if "/" == c and "*" == nxt:
                state, start = "block", i
                i += 2
                continue
            if "/" == c and "/" == nxt:
                state, start = "line", i
                i += 2
                continue
            if c in "\"'":
                state, start, quote = "literal", i, c
                i += 1
                continue
            if "#" == c and not text[:i].rsplit("\n", 1)[-1].strip():
                match = CONDITIONAL.match(text, text.rfind("\n", 0, i) + 1)
                if match:
                    conds.append((i, match.group(1)))
                state, start = "directive", i
                i += 1
                continue
        elif "block" == state:
            if "*" == c and "/" == nxt:
                blank(out, start, i + 2)
                state = "code"
                i += 2
                continue
        elif "line" == state:
            if "\n" == c:
                blank(out, start, i)
                state = "code"
        elif "literal" == state:
            if "\\" == c:
                i += 2
                continue
            if quote == c:
                blank(out, start, i + 1)
                state = "code"
        elif "directive" == state:
            if "\n" == c and "\\" != text[i - 1]:
                if not directives:
                    blank(out, start, i)
                state = "code"
        i += 1
    if "code" != state:
        blank(out, start, n)
    return "".join(out), conds


def blank(out: List[str], start: int, end: int) -> None:
    """Overwrite a span with spaces, leaving the line structure intact."""
    for k in range(start, end):
        if "\n" != out[k]:
            out[k] = " "


def exits(body: str, conds: Sequence[Tuple[int, str]], base: int) -> int:
    """Count the exits of the worst-case preprocessor configuration.

    A region contributes its own returns plus, per nested conditional, the
    maximum over that conditional's branches.
    """
    stack: List[List[int]] = [[0]]
    pos = 0
    for offset, keyword in conds:
        here = offset - base
        if here < pos or len(body) <= here:
            continue
        stack[-1][-1] += len(RETURN.findall(body[pos:here]))
        pos = here
        if keyword in ("if", "ifdef", "ifndef"):
            stack.append([0])
        elif keyword in ("elif", "else"):
            if 1 < len(stack):
                stack[-1].append(0)
        elif 1 < len(stack):
            branches = stack.pop()
            stack[-1][-1] += max(branches)
    stack[-1][-1] += len(RETURN.findall(body[pos:]))
    while 1 < len(stack):
        stack[-2][-1] += max(stack.pop())
    return max(stack[0])


def scopes(masked: str) -> List[Tuple[int, int, bool]]:
    """Locate every brace-depth-zero region as (open, close, is_function)."""
    found: List[Tuple[int, int, bool]] = []
    stack: List[Tuple[int, bool]] = []
    depth = 0
    for i, c in enumerate(masked):
        if "{" == c:
            if 0 == depth:
                stack.append((i, masked[:i].rstrip().endswith(")")))
            depth += 1
        elif "}" == c:
            depth -= 1
            if 0 == depth and stack:
                opened, isfunc = stack.pop()
                found.append((opened, i, isfunc))
            if 0 > depth:
                depth = 0
    return found


def runs(lines: Sequence[str]) -> int:
    """Return the longest run of blank lines."""
    longest = run = 0
    for line in lines:
        run = run + 1 if not line.strip() else 0
        longest = max(longest, run)
    return longest


def check(path: str, enabled: Sequence[str]) -> List[Finding]:
    """Report the violations of one file as (rule, line, detail)."""
    report: List[Finding] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as error:
        return [("unreadable", 0, str(error))]
    masked, conds = mask(text)
    lines = text.split("\n")
    regions = scopes(masked)
    for opened, closed, isfunc in regions:
        if not isfunc:
            continue
        first = masked.count("\n", 0, opened) + 1
        last = masked.count("\n", 0, closed) + 1
        if "single-exit" in enabled:
            count = exits(masked[opened:closed], conds, opened)
            if 1 < count:
                report.append(("single-exit", first, "%i exits" % count))
        if "blank-in-function" in enabled:
            if 1 < runs(lines[first:last]):
                report.append(
                    ("blank-in-function", first, "more than one blank line")
                )
        if "function-gap" in enabled:
            gap, at = 0, last
            while at < len(lines) and not lines[at].strip():
                gap += 1
                at += 1
            if at < len(lines) and not lines[at].lstrip().startswith("#"):
                # A header carries small inline definitions and the files that
                # hold them are uniform about one blank line, which is the
                # surrounding code a change there has to match.
                want = (1, 2) if path.endswith(".h") else (2,)
                if gap not in want:
                    report.append(
                        (
                            "function-gap",
                            last,
                            "%i blank lines, expected %s"
                            % (gap, " or ".join(str(w) for w in want)),
                        )
                    )
    if "stacked-comments" in enabled:
        report += stacked(text, lines)
    if "closer-nesting" in enabled:
        report += nesting(text, masked)
    if "constant-left" in enabled:
        visible, _ = mask(text, True)
        for number, line in enumerate(visible.split("\n"), 1):
            if CONSTANT.search(line):
                report.append(
                    ("constant-left", number, "put the constant on the left")
                )
    if [name for name in enabled if name.startswith("macro-")]:
        report += macros(lines, enabled)
    return report


def stacked(text: str, lines: Sequence[str]) -> List[Finding]:
    """Report comments that follow one another with no code between them.

    Blank lines do not separate them: two comments with only whitespace in
    between are one comment split in two, or they describe different things
    and the code each describes belongs between them. The license header is
    the file's leading comment and is exempt by construction.
    """
    report: List[Finding] = []
    visible, _ = mask(text, True)
    first = 1
    for number, line in enumerate(visible.split("\n"), 1):
        if line.strip():
            first = number
            break
    previous = None
    at, size = 0, len(text)
    while at < size:
        opened = text.find("/*", at)
        if 0 > opened:
            break
        closed = text.find("*/", opened + 2)
        if 0 > closed:
            break
        start = text.count("\n", 0, opened) + 1
        end = text.count("\n", 0, closed) + 1
        head = text.rfind("\n", 0, opened) + 1
        tail = text.find("\n", closed)
        tail = size if 0 > tail else tail
        alone = (
            not text[head:opened].strip()
            and not text[closed + 2 : tail].strip()
        )
        if alone and previous is not None and start > first:
            between = lines[previous : start - 1]
            if not [line for line in between if line.strip()]:
                report.append(
                    (
                        "stacked-comments",
                        start,
                        "follows the comment at line %i" % previous,
                    )
                )
        previous = end if alone else None
        at = closed + 2
    return report


def nesting(text: str, masked: str) -> List[Finding]:
    """Report a closing brace that does not step left from the one above it.

    Two closers on the same column close blocks that are nested, so one of
    the two levels is missing from the indentation. The line shape is taken
    from the source and the braces from the masked text, or a line whose
    only code is the "};" of an initializer looks like a closer. A directive
    resets the comparison: which brace belongs to which block then depends
    on the configuration.
    """
    report: List[Finding] = []
    previous = None
    for number, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("#"):
            previous = None
            continue
        strip = line.strip()
        if not strip:
            continue
        indent = len(line) - len(line.lstrip())
        if strip.startswith("}") and previous is not None:
            if indent >= previous[1]:
                report.append(
                    (
                        "closer-nesting",
                        number,
                        "shares column %i with the closer at line %i"
                        % (indent, previous[0]),
                    )
                )
        alone = strip.startswith("}") and "{" not in strip
        previous = (number, indent) if alone else None
    return report


def body(lines: Sequence[str], number: int, rest: str) -> Tuple[str, int]:
    """Collect a macro body, following its backslash continuations."""
    collected = [rest]
    while collected[-1].rstrip().endswith("\\") and number < len(lines):
        collected.append(LITERAL.sub('""', BLOCK.sub("", lines[number])))
        number += 1
    return "\n".join(collected), number


def macros(lines: Sequence[str], enabled: Sequence[str]) -> List[Finding]:
    """Report the macros whose names or locals break the naming rules.

    A comment is stripped per line rather than by mask(), which leaves a
    directive intact and would otherwise offer "/*note*/" as a parameter.
    """
    report: List[Finding] = []
    number = 0
    while number < len(lines):
        line = LITERAL.sub('""', BLOCK.sub("", lines[number]))
        number += 1
        match = DEFINE.match(line)
        if not match:
            continue
        where = number
        name, params = match.group(1), match.group(3)
        text, number = body(lines, number, line[match.end() :])
        if "macro-local" in enabled:
            given = [p.strip() for p in (params or "").split(",") if p.strip()]
            for local in LOCAL.findall(MEMBERS.sub(" ", text)):
                # A DECL macro names the variable after its own parameter,
                # so the caller owns that name and its capitalization.
                if local in given or local in KEYWORDS:
                    continue
                if re.search(r"[A-Z]", local) or not local.endswith("_"):
                    report.append(
                        (
                            "macro-local",
                            where,
                            "%s wants a lowercase name and a trailing"
                            " underscore" % local,
                        )
                    )
        if "macro-name" in enabled and not MACRO.match(name):
            report.append(
                ("macro-name", where, "%s is not capitalized" % name)
            )
        if "macro-parameter" in enabled and params is not None:
            for param in [p.strip() for p in params.split(",") if p.strip()]:
                if "..." != param and re.search(r"[a-z]", param):
                    report.append(
                        (
                            "macro-parameter",
                            where,
                            "%s is not capitalized" % param,
                        )
                    )
    return report


def backlog() -> Dict[Tuple[str, str], int]:
    """Read the per-project backlog that sits next to this script.

    A line is "<rule> <count> <path>". No file means no backlog, so every
    finding is reported: a project that has not started a list gets the
    rules in full.
    """
    listed: Dict[Tuple[str, str], int] = {}
    try:
        with open(os.path.join(ROOT, "scripts", TODO), encoding="utf-8") as f:
            for line in f:
                field = line.split("#", 1)[0].split()
                if 3 == len(field) and field[1].isdigit():
                    listed[(field[0], field[2])] = int(field[1])
    except OSError:
        pass
    return listed


def suppress(
    path: str,
    enabled: Sequence[str],
    report: Sequence[Finding],
    listed: Dict[Tuple[str, str], int],
) -> Tuple[List[str], List[str]]:
    """Apply the suppression lists, returning (violations, complaints).

    A violation is a finding the lists do not cover. A complaint is about a
    list entry itself: a count that has been overrun, or one that is now too
    high because the backlog shrank. The second kind is what keeps the list
    from going stale.
    """
    violations: List[str] = []
    complaints: List[str] = []
    seen: Dict[str, List[Finding]] = {}
    for rule, line, detail in report:
        seen.setdefault(rule, []).append((rule, line, detail))
    for rule, found in sorted(seen.items()):
        if (rule, path) in EXEMPT:
            continue
        allowed = listed.get((rule, path))
        if allowed is None:
            violations += [
                "%s:%i: %s: %s" % (path, line, rule, detail)
                for rule, line, detail in found
            ]
        elif allowed < len(found):
            violations += [
                "%s:%i: %s: %s" % (path, line, rule, detail)
                for rule, line, detail in found
            ]
            complaints.append(
                "%s: %s: %i findings, the list allows %i"
                % (path, rule, len(found), allowed)
            )
    return violations, complaints


def lower(
    listed: Dict[Tuple[str, str], int],
    observed: Dict[Tuple[str, str], int],
    examined: Sequence[str],
    enabled: Sequence[str],
) -> List[str]:
    """Bring the to-do file down to what is left, and say what changed.

    Only downwards: a count that shrank is rewritten and an entry that ran
    out is deleted, because lowering the list is always correct. A count
    that grew is never touched here, or a regression would legalize itself.
    An entry whose file is gone goes too.
    """
    said: List[str] = []
    change: Dict[Tuple[str, str], int] = {}
    for (rule, path), allowed in sorted(listed.items()):
        if not os.path.exists(os.path.join(ROOT, path)):
            change[(rule, path)] = 0
            said.append(
                "%s: %s: file is gone, dropped from %s" % (path, rule, TODO)
            )
        elif path in examined and rule in enabled:
            count = observed.get((rule, path), 0)
            if count < allowed:
                change[(rule, path)] = count
                said.append(
                    "%s: %s: down to %i, %s in %s"
                    % (
                        path,
                        rule,
                        count,
                        "dropped" if 0 == count else "lowered",
                        TODO,
                    )
                )
    if change:
        name = os.path.join(ROOT, "scripts", TODO)
        with open(name, encoding="utf-8") as handle:
            lines = handle.readlines()
        out = []
        for line in lines:
            field = line.split("#", 1)[0].split()
            if 3 == len(field) and field[1].isdigit():
                left = change.get((field[0], field[2]))
                if left is not None:
                    if 0 == left:
                        continue
                    line = "%-17s %3i  %s\n" % (field[0], left, field[2])
            out.append(line)
        with open(name, "w", encoding="utf-8") as handle:
            handle.writelines(out)
    return said


def main(argv: Sequence[str]) -> int:
    """Check every named file, or list the checks, or print the counts."""
    result = 0
    if "--list" in argv:
        print("\n".join(CHECKS))
    else:
        skip: Dict[str, bool] = {}
        only: List[str] = []
        names = []
        counts = "--counts" in argv
        for argument in argv[1:]:
            if argument.startswith("--no-"):
                skip[argument[5:]] = True
            elif argument.startswith("--only="):
                only += argument[7:].split(",")
            elif not argument.startswith("--"):
                names.append(argument)
        enabled = [
            check
            for check in CHECKS
            if check not in skip and (not only or check in only)
        ]
        listed = backlog()
        observed: Dict[Tuple[str, str], int] = {}
        for path in names:
            report = check(path, enabled)
            for rule, _, _ in report:
                key = (rule, path)
                observed[key] = observed.get(key, 0) + 1
            if counts:
                continue
            violations, complaints = suppress(path, enabled, report, listed)
            if violations or complaints:
                print("\n".join(violations + complaints))
                result = 1
        if not counts:
            for said in lower(listed, observed, names, enabled):
                print(said)
                result = 1
        if counts:
            # Ready for tool_checkstruct.todo, minus what EXEMPT covers.
            for (rule, path), count in sorted(observed.items()):
                if (rule, path) not in EXEMPT:
                    print("%-17s %3i  %s" % (rule, count, path))
            for key in sorted(listed):
                if key not in observed and key[1] in names:
                    print("# clean now, drop: %s %s" % key)
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv))
