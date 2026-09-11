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
# An identifier wearing the trailing underscore that belongs to a macro local.
TRAILING = re.compile(r"\b(\w*[A-Za-z0-9]_)\b")
# A block with a controlling construct, which is what the brace may trail.
CONTROL = re.compile(r"\b(if|for|while|switch|else|do)\s*$")
DIRECTIVE = re.compile(r"[ \t]*#")
# The same keywords where they start a statement, to find what they control.
CONTROLLED = re.compile(r"\b(if|for|while|switch|else|do)\b")
ELSEIF = re.compile(r"if\b")
# The include guard, whose "#define" is the file's own name rather than a
# member of the macro section.
GUARD = re.compile(
    r"^[ \t]*#[ \t]*ifndef[ \t]+(\w+)[ \t]*\n[ \t]*#[ \t]*define[ \t]+\1\b",
    re.M,
)
KEYWORD = re.compile(r"[ \t]*#[ \t]*(\w+)")
NAMED = re.compile(r"(\w+)[ \t]*\(")
# The declarator that trails the closing brace of a type definition, as in
# "} libxs_gemm_shape_t;": the construct ends at the semicolon, not at the
# brace. Kept to one line, or the next construct would be swallowed too.
DECLARATOR = re.compile(r"[ \t*,\[\]\w]*;")
# The sections of the policy, in their order, as the report words them.
SECTIONS = (
    "an include",
    "a macro",
    "a type",
    "a translation-unit variable",
    "a prototype",
    "a function",
)
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
    "function-parameter",
    "function-local",
    "brace-placement",
    "multiline-block",
    "section-order",
)


def header(path: str) -> bool:
    """True for a header, which is any ".h*" file: .h, .hpp, .hxx, .h.in."""
    return ".h" in os.path.basename(path)


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


def runs(lines: Sequence[str], allowed: int = 1) -> int:
    """Return the offset of the first blank line beyond a run of "allowed"."""
    result, run = -1, 0
    for at, line in enumerate(lines):
        run = run + 1 if not line.strip() else 0
        if allowed < run and 0 > result:
            result = at
    return result


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
            # One blank line separates the blocks of a body; two are what
            # separate the functions themselves, so they cannot be inside one.
            blank = runs(lines[first:last])
            if 0 <= blank:
                report.append(
                    (
                        "blank-in-function",
                        first + blank + 1,
                        "a second blank line inside a function body",
                    )
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
                want = (1, 2) if header(path) else (2,)
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
    if "function-parameter" in enabled:
        report += parameters(masked)
    if "brace-placement" in enabled:
        report += braces(path, text, masked)
    if "multiline-block" in enabled:
        report += blocks(text, masked)
    if "section-order" in enabled:
        report += sections(text)
    if "function-local" in enabled:
        for opened, closed, isfunc in regions:
            if not isfunc:
                continue
            # Blanked rather than removed, to keep the offsets and with
            # them the line the declaration is actually on.
            inside = MEMBERS.sub(
                lambda m: " " * len(m.group(0)), masked[opened:closed]
            )
            for match in LOCAL.finditer(inside):
                # The mask removed the directives, so a macro body cannot
                # reach here: this is the function's own declaration.
                local = match.group(1)
                if local.endswith("_") and local not in KEYWORDS:
                    report.append(
                        (
                            "function-local",
                            masked.count("\n", 0, opened + match.start()) + 1,
                            "%s carries the underscore of a macro local"
                            % local,
                        )
                    )
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


def braces(path: str, text: str, masked: str) -> List[Finding]:
    """Report an opening brace that is not where its kind of block wants it.

    A function body opens on its own line, and so does a block whose
    parentheses were broken across lines, because the brace is what tells the
    reader the condition has ended. Everything else keeps the brace on the
    line of the construct it belongs to. A bare block has no such line and is
    left alone, as are a struct, a union and an initializer.

    A header holds inline definitions, and there the brace may trail the
    signature: the definition is part of a declaration list, so keeping it on
    one line is what the surrounding declarations do.
    """
    report: List[Finding] = []
    # Comments blanked but the code kept: a comment after the brace does not
    # make the line occupied, and a string cannot reach the cases checked.
    visible, _ = mask(text, True)
    raw = visible.split("\n")
    depth = 0
    for at, char in enumerate(masked):
        if "}" == char:
            depth -= 1
            continue
        if "{" != char:
            continue
        line = masked.count("\n", 0, at) + 1
        column = at - (masked.rfind("\n", 0, at) + 1)
        source = raw[line - 1]
        alone = (
            not source[:column].strip() and not source[column + 1 :].strip()
        )
        before = masked[:at].rstrip()
        paren = before.endswith(")")
        # A directive between the construct and the brace: the line above is
        # "#endif", and moving the brace up would take it into the branch.
        above = masked.count("\n", 0, max(len(before) - 1, 0)) + 1
        gap = any(DIRECTIVE.match(entry) for entry in raw[above : line - 1])
        split, head = False, ""
        if paren:
            nest, back = 0, len(before) - 1
            while 0 <= back:
                if ")" == before[back]:
                    nest += 1
                elif "(" == before[back]:
                    nest -= 1
                    if 0 == nest:
                        break
                back -= 1
            if 0 <= back:
                split = "\n" in before[back:]
                head = before[:back].rstrip()
            else:
                # No opener for this parenthesis, so the construct is composed
                # across preprocessor branches: an OpenCL kernel whose
                # parameter list ends, and whose body opens, once per
                # configuration. Neither the paren nor the brace count says
                # anything there, so the brace is not judged.
                paren = False
        # An included body fragment carries its control flow at depth zero, so
        # a control keyword ahead of the parentheses rules out a definition.
        if 0 == depth and paren and not CONTROL.search(head):
            if not alone and not header(path):
                report.append(
                    (
                        "brace-placement",
                        line,
                        "a function body opens on its own line",
                    )
                )
            elif not alone and split:
                report.append(
                    (
                        "brace-placement",
                        line,
                        "the parentheses span lines, so the brace opens on its own",
                    )
                )
        elif split:
            if not alone:
                report.append(
                    (
                        "brace-placement",
                        line,
                        "the parentheses span lines, so the brace opens on its own",
                    )
                )
        elif (paren or CONTROL.search(before)) and alone and not gap:
            report.append(
                (
                    "brace-placement",
                    line,
                    "the brace belongs on the line above",
                )
            )
        depth += 1
    return report


def blocks(text: str, masked: str) -> List[Finding]:
    """Report a control statement that spans lines without being a block.

    Whatever belongs to an "if", an "else" or a loop stays on the keyword's
    line, or it is braced: once the construct occupies a second line, the
    braces are what say where it ends. Each keyword is judged on its own, so
    an "if" with a braced body and a one-line "else" is two decisions, not
    one. An "else if" chain is the inner "if" and is judged there.
    """
    report: List[Finding] = []
    visible, _ = mask(text, True)
    raw = visible.split("\n")
    for match in CONTROLLED.finditer(masked):
        keyword = match.group(1)
        at = match.end()
        if keyword in ("if", "for", "while", "switch"):
            while at < len(masked) and masked[at].isspace():
                at += 1
            if at >= len(masked) or "(" != masked[at]:
                continue
            nest = 0
            while at < len(masked):
                if "(" == masked[at]:
                    nest += 1
                elif ")" == masked[at]:
                    nest -= 1
                    if 0 == nest:
                        at += 1
                        break
                at += 1
        while at < len(masked) and masked[at].isspace():
            at += 1
        if at >= len(masked) or masked[at] in "{;":
            # A block says where it ends, and an empty statement is the tail
            # of a do-while or a wait loop, which has nothing to brace.
            continue
        if "else" == keyword and ELSEIF.match(masked, at):
            continue
        nest, end = 0, at
        while end < len(masked):
            if "(" == masked[end]:
                nest += 1
            elif ")" == masked[end]:
                nest -= 1
            elif 0 == nest and masked[end] in "{}":
                end = -1
                break
            elif 0 == nest and ";" == masked[end]:
                break
            end += 1
        if end < 0 or end >= len(masked):
            continue
        line = masked.count("\n", 0, match.start()) + 1
        # A directive between the keyword and what it controls: the statement
        # belongs to one configuration, and braces cannot span the two.
        if any(
            DIRECTIVE.match(entry)
            for entry in raw[line : masked.count("\n", 0, at)]
        ):
            continue
        if line != masked.count("\n", 0, end) + 1:
            report.append(
                (
                    "multiline-block",
                    line,
                    '"%s" spans lines, so it wants braces' % keyword,
                )
            )
    return report


def kind(text: str) -> int:
    """Rank one top-level construct by what it defines."""
    flat = " ".join(text.split())
    head = flat.split("{")[0]
    if flat.startswith("typedef"):
        return 3
    if "{" in flat:
        # An initializer carries braces of its own, so what decides is the
        # text ahead of them: an "=" makes the braces a value, and a ")"
        # makes them a body.
        if re.search(r"=[^=]*$", head):
            return 4
        return 6 if head.rstrip().endswith(")") else 3
    if "=" not in flat and NAMED.search(flat) and flat.rstrip().endswith(")"):
        return 5
    return 4


def named(text: str) -> str:
    """Name the entity a declaration declares, or "" where there is none.

    The name is the identifier ahead of the outermost parameter list, which
    is neither the first nor the last in the text: a decoration such as
    LIBXS_INTRINSICS(...) brings its own parentheses ahead of the name, and
    a function parameter brings its own behind it.
    """
    result = ""
    depth = 0
    for match in re.finditer(r"(\w+)?[ \t\n]*([()])", text.split("{")[0]):
        if "(" == match.group(2):
            if 0 == depth and match.group(1):
                result = match.group(1)
            depth += 1
        else:
            depth -= 1
    return result


def constructs(visible: str) -> List[Tuple[int, int, str, int]]:
    """Rank the top-level constructs as (offset, rank, text, conditional).

    The rank is the position in the section order, from 1 for an include to
    6 for a function definition, and 0 for what has no section of its own,
    which is every directive that is neither an include nor a define. The
    conditional depth rides along because an include or a define inside an
    "#if" is a feature test rather than a member of a section.
    """
    found: List[Tuple[int, int, str, int]] = []
    depth, cond, start = 0, 0, -1
    i, n = 0, len(visible)
    while i < n:
        char = visible[i]
        if (
            "#" == char
            and 0 == depth
            and not visible[:i].rsplit("\n", 1)[-1].strip()
        ):
            end = visible.find("\n", i)
            while -1 != end and "\\" == visible[end - 1 : end]:
                end = visible.find("\n", end + 1)
            end = n if -1 == end else end
            match = KEYWORD.match(visible, i)
            name = match.group(1) if match else ""
            if name in ("if", "ifdef", "ifndef"):
                cond += 1
            elif "endif" == name:
                cond = max(cond - 1, 0)
            rank = {"include": 1, "define": 2}.get(name, 0)
            if 0 != rank:
                found.append((i, rank, visible[i:end], cond))
            i, start = end, -1
            continue
        if 0 > start:
            if char in " \t\n":
                i += 1
                continue
            start = i
        if "{" == char:
            depth += 1
        elif "}" == char:
            depth -= 1
            if 0 >= depth:
                depth = 0
                match = DECLARATOR.match(visible, i + 1)
                stop = match.end() if match else i + 1
                entry = visible[start:stop]
                found.append((start, kind(entry), entry, cond))
                i, start = stop, -1
                continue
        elif ";" == char and 0 == depth:
            entry = visible[start:i]
            found.append((start, kind(entry), entry, cond))
            start = -1
        i += 1
    return found


def sections(text: str) -> List[Finding]:
    """Report a construct that reopens a section the file already left.

    The sections are strictly ordered, so the rank of the constructs never
    decreases: where it does, something was parked next to its first use
    instead of grouped with its kind. The report names both sections, which
    is the direction of the fix, and one backward step is reported once
    rather than once per construct: after a misplaced typedef, every macro
    below it is out of order too, and that is one defect, not a hundred.

    Three things are ranked out of the ordering, because in each the text
    that looks like a section member is not one: the include guard, whose
    "#define" is the file's own name; an include or a define inside an
    "#if", which is a feature test and belongs where the test is; and a
    prototype immediately above the definition it repeats, which is how a
    static definition answers -Wmissing-prototypes.
    """
    report: List[Finding] = []
    visible, _ = mask(text, True)
    match = GUARD.search(visible)
    guard = match.group(1) if match else ""
    found = constructs(visible)
    top, again = 0, 0
    for at, (offset, rank, entry, cond) in enumerate(found):
        if 0 == rank:
            continue
        if guard and 2 == rank:
            if re.match(r"[ \t]*#[ \t]*define[ \t]+%s\b" % guard, entry):
                continue
        if 0 < cond and rank in (1, 2):
            continue
        if 5 == rank and at + 1 < len(found):
            below = found[at + 1]
            name = named(entry)
            if name and 6 == below[1] and name == named(below[2]):
                continue
        if rank < top:
            if again != rank:
                report.append(
                    (
                        "section-order",
                        visible.count("\n", 0, offset) + 1,
                        "%s after %s"
                        % (SECTIONS[rank - 1], SECTIONS[top - 1]),
                    )
                )
            again = rank
        else:
            again = 0
        top = max(top, rank)
    return report


def parameters(masked: str) -> List[Finding]:
    """Report a function parameter that carries a macro local's underscore.

    A parameter list at brace depth zero is followed by "{" for a definition
    and ";" for a declaration, which is what tells it apart from a call. A
    macro body cannot appear here: the mask has removed the directives.
    """
    report: List[Finding] = []
    depth = 0
    at, size = 0, len(masked)
    while at < size:
        c = masked[at]
        if "{" == c:
            depth += 1
        elif "}" == c:
            depth = max(0, depth - 1)
        elif "(" == c and 0 == depth:
            nest, close = 0, at
            while close < size:
                if "(" == masked[close]:
                    nest += 1
                elif ")" == masked[close]:
                    nest -= 1
                    if 0 == nest:
                        break
                close += 1
            after = close + 1
            while after < size and masked[after] in " \t\n":
                after += 1
            if after < size and masked[after] in "{;":
                for name in TRAILING.findall(masked[at + 1 : close]):
                    report.append(
                        (
                            "function-parameter",
                            masked.count("\n", 0, at) + 1,
                            "%s carries the underscore of a macro local"
                            % name,
                        )
                    )
            at = close
        at += 1
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
                if "..." == param:
                    continue
                if re.search(r"[a-z]", param):
                    report.append(
                        (
                            "macro-parameter",
                            where,
                            "%s is not capitalized" % param,
                        )
                    )
                elif param.endswith("_"):
                    # The trailing underscore marks a local the macro
                    # declares; a parameter wearing it hides the difference.
                    report.append(
                        (
                            "macro-parameter",
                            where,
                            "%s carries the underscore of a local" % param,
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
