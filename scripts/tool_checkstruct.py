#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Maintained in LIBXS and copied into dependent projects by "make policies".
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
CHECKS = (
    "single-exit",
    "function-gap",
    "blank-in-function",
    "stacked-comments",
    "constant-left",
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


def check(path: str, enabled: Sequence[str]) -> List[str]:
    """Report the violations of one file as "path:line: check: detail"."""
    report: List[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as error:
        return ["%s: cannot read (%s)" % (path, error)]
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
                report.append(
                    "%s:%i: single-exit: %i exits" % (path, first, count)
                )
        if "blank-in-function" in enabled:
            if 1 < runs(lines[first:last]):
                report.append(
                    "%s:%i: blank-in-function: more than one blank line"
                    % (path, first)
                )
        if "function-gap" in enabled:
            gap, at = 0, last
            while at < len(lines) and not lines[at].strip():
                gap += 1
                at += 1
            if at < len(lines) and not lines[at].lstrip().startswith("#"):
                if 2 != gap:
                    report.append(
                        "%s:%i: function-gap: %i blank lines, expected 2"
                        % (path, last, gap)
                    )
    if "stacked-comments" in enabled:
        previous = ""
        for number, line in enumerate(lines, 1):
            match = COMMENT.match(line)
            if match and previous == match.group(1):
                report.append(
                    "%s:%i: stacked-comments: merge or separate them"
                    % (path, number)
                )
            previous = match.group(1) if match else "\n"
    if "constant-left" in enabled:
        visible, _ = mask(text, True)
        for number, line in enumerate(visible.split("\n"), 1):
            if CONSTANT.search(line):
                report.append(
                    "%s:%i: constant-left: put the constant on the left"
                    % (path, number)
                )
    return report


def main(argv: Sequence[str]) -> int:
    """Check every named file, or list the checks."""
    result = 0
    if "--list" in argv:
        print("\n".join(CHECKS))
    else:
        skip: Dict[str, bool] = {}
        names = []
        for argument in argv[1:]:
            if argument.startswith("--no-"):
                skip[argument[5:]] = True
            else:
                names.append(argument)
        enabled = [check for check in CHECKS if check not in skip]
        for path in names:
            report = check(path, enabled)
            if report:
                print("\n".join(report))
                result = 1
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv))
