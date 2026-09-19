#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
"""Capture the SMM build strings and generate the shim flavors from them.

tests/shim_smm_fp64.c and shim_smm_fp32.c hold the kernels' build strings as
the host emitted them, not composed by hand: "capture" runs acc_bench on a
device through a logging preprocessor (LIBXSTREAM_CPPBIN, LIBXSTREAM_DUMP=3)
over the tuner's overrides, and "generate" turns the strings into instances.

  shim_smm.py capture -o defs.txt BENCH64 [BENCH32]
  shim_smm.py generate defs.txt double -o shim_smm_fp64.c
  shim_smm.py generate defs.txt float -o shim_smm_fp32.c

BENCH32 is acc_bench built with ELEM_TYPE=float. The overrides only apply
without OPENCL_LIBSMM_SMM_PARAMS=0 (it suppresses every one of them), and a
batch above one needs OPENCL_LIBSMM_SMM_S=0 as the tuner sets it.
"""
import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile

P = "OPENCL_LIBSMM_SMM_"
# the tuner's parameters, one family at a time, and the transpose's own
CONFIGS = [
    "",
    P + "AA=1 " + P + "AB=1 " + P + "AC=1",
    P + "AA=2 " + P + "AB=2 " + P + "AC=2",
    P + "AA=3 " + P + "AB=3 " + P + "AP=1",
    P + "AA=0 " + P + "AB=0 " + P + "AC=0 " + P + "AP=0",
    P + "BS=1 " + P + "NZ=1",
    P + "BM=2 " + P + "BN=2 " + P + "BK=2",
    P + "BM=4 " + P + "BN=1 " + P + "BK=1 " + P + "AA=3",
    P + "BS=1 " + P + "BM=3 " + P + "BN=2 " + P + "AB=3 " + P + "NZ=1",
    P + "LU=-2",
    P + "LU=1",
    P + "WG=0",
    P + "WG=2",
    P + "S=0 " + P + "BS=4",
    P + "S=0 " + P + "BS=4 " + P + "TB=1 " + P + "TC=1",
    P + "S=0 " + P + "BS=4 " + P + "AP=1 " + P + "AC=1",
    P + "S=0 " + P + "BS=4 " + P + "AA=3 " + P + "AB=3 " + P + "NZ=1",
    P + "S=0 " + P + "BS=4 " + P + "BM=2 " + P + "BN=2 " + P + "BK=2",
    P + "S=0 " + P + "BS=4 " + P + "BM=4 " + P + "BN=1 " + P + "BK=1",
    P + "S=0 " + P + "BS=4 " + P + "AA=0 " + P + "AB=0 " + P + "AC=0",
    P + "S=0 " + P + "BS=4 " + P + "AC=2 " + P + "BM=3 " + P + "BN=2",
    "OPENCL_LIBSMM_TRANS_INPLACE=1",
    "OPENCL_LIBSMM_TRANS_BM=4",
    "OPENCL_LIBSMM_TRANS_BM=8 OPENCL_LIBSMM_TRANS_BS=4",
]
SHAPES = ["23 23 23", "5 7 9", "4 4 4", "32 32 32"]

# the device part a generic OpenCL 1.2 device without sub-groups receives from
# libxstream_opencl_flags_atomics; every instance of a unit shares it
DEVICE = {
    "double": ["TAN 2", "TA long", "CMPXCHG atom_cmpxchg"],
    "float": ["TAN 1", "TA int", "CMPXCHG atomic_cmpxchg"],
}
SHARED = {"GPU", "CONSTANT", "SG", "INTEL"}
DROP = SHARED | {"TAN", "TA", "TA2", "TF", "CMPXCHG"}
DROP |= {"ATOMIC_ADD_GLOBAL(A,B)", "BARRIER(A)"}
LICENSE = """\
/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXSTREAM library.                                *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxstream/                     *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
"""


def capture(args):
    """Run acc_bench over CONFIGS x SHAPES and keep the build strings."""
    cpp = os.environ.get("CPP", "/usr/bin/cpp")
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "cpp.log")
        wrap = os.path.join(tmp, "cpp")
        with open(wrap, "w") as f:
            f.write('#!/bin/sh\nprintf "%%s\\n" "$*" >> "%s"\n' % log)
            f.write('exec "%s" "$@"\n' % cpp)
        os.chmod(wrap, os.stat(wrap).st_mode | stat.S_IXUSR)
        env = dict(os.environ, LIBXSTREAM_CPPBIN=wrap, LIBXSTREAM_DUMP="3")
        env.update(LIBXSTREAM_ATOMICS="cmpxchg", CUDA_CACHE_DISABLE="1")
        env.update(CHECK="1")
        for bench in args.bench:
            for shape in SHAPES:
                for config in CONFIGS:
                    run = dict(env)
                    run.update(kv.split("=", 1) for kv in config.split())
                    cmd = [bench, "1", "64"] + shape.split()
                    res = subprocess.run(cmd, env=run, cwd=tmp, check=False)
                    if 0 != res.returncode:  # the host may refuse a shape
                        print("refused: %s %s" % (shape, config))
        lines = open(log).read().splitlines() if os.path.exists(log) else []
    defs = set()
    for line in lines:
        m = re.search(r"-DLIBXSTREAM_OCLVER_C=\d+ (-D(?:T|CONSTANT)=.*)", line)
        if m:
            defs.add(re.sub(r" /\S*$", "", m.group(1)).strip())
    with open(args.output, "w") as f:
        f.write("\n".join(sorted(defs)) + "\n")
    print("%s: %d build strings" % (args.output, len(defs)))


def parse(line):
    """The -D tokens of a build string, in order; arguments stay attached."""
    return [t.partition("=")[::2] for t in re.findall(r"-D(\S+)", line)]


def generate(args):
    """Write one flavor: the device part once, then an instance per string."""
    elem, tag = args.elem, ("d" if "double" == args.elem else "s")
    flavor = "fp64" if "double" == elem else "fp32"
    smm, trs, seen = [], [], set()
    for line in open(args.defs).read().splitlines():
        d = parse(line)
        dd = dict(d)
        is_trans = "INPLACE" in dd
        want = ("char8" if "double" == elem else "float") if is_trans else elem
        if "FN" not in dd or dd.get("T") != want:
            continue
        key = tuple((k, v) for k, v in d if k not in ("FN", "LU"))
        if key not in seen:  # LU sets unroll hints, pragmas on the host
            seen.add(key)
            (trs if is_trans else smm).append(d)
    out = [LICENSE]
    out.append(
        "/**\n"
        " * The SMM kernels in %s through the host shim (see shim_smm.h),\n"
        " * one instance per build string smm_kernel.c or smm_trans.c\n"
        " * emitted for acc_bench, captured rather than composed: generated\n"
        " * by tests/shim_smm.py, whose header says how to regenerate. The\n"
        " * device part is what libxstream_opencl_flags_atomics emits for an\n"
        " * OpenCL 1.2 device without sub-groups.\n"
        " */\n" % flavor
    )
    out.append("#define SHIM_SMM_ELEM %s" % elem)
    out.append('#include "shim_smm.h"\n')
    out.append(
        "/* a work-group is an OpenMP team, the barriers need its threads */"
    )
    out.append("#if defined(_OPENMP)\n")
    out.append("#define LIBXSTREAM_CPU_TEAM 1")
    out.append("#define GPU 1\n#define CONSTANT global")
    out.append("#define SG 0\n#define INTEL 0")
    out += ["#define " + d for d in DEVICE[elem]]
    out.append(
        "#define ATOMIC_ADD_GLOBAL(A, B) atomic_add_global_cmpxchg(A, B)"
    )
    out.append("#define BARRIER(A) barrier(A)")
    out.append("#include <libxstream/opencl/libxstream_cpu_begin.h>\n")
    names = {}
    for kind, items, header in (
        ("smm", smm, "shim_smm_instance.h"),
        ("trans", trs, "shim_smm_trans_instance.h"),
    ):
        names[kind] = []
        for i, d in enumerate(items):
            name = "shim_%s%s_%d" % (tag, kind, i + 1)
            names[kind].append(name)
            for k, v in d:
                if k not in DROP:
                    v = name if "FN" == k else v
                    out.append("#define %s%s" % (k, (" " + v) if v else ""))
            out.append('#include "%s"\n' % header)
    out.append("#include <libxstream/opencl/libxstream_cpu_end.h>\n")
    for kind, typ in (
        ("smm", "shim_smm_desc_t"),
        ("trans", "shim_trans_desc_t"),
    ):
        out.append("static const %s* const shim_%s_set[] = {" % (typ, kind))
        out.append(",\n".join("  &%s_desc" % n for n in names[kind]))
        out.append("};")
    out.append("\n\nint main(void)\n{")
    out.append(
        "  return shim_smm(shim_smm_set, (int)(sizeof(shim_smm_set) / "
        "sizeof(*shim_smm_set)), shim_trans_set,"
    )
    out.append(
        "    (int)(sizeof(shim_trans_set) / sizeof(*shim_trans_set)), "
        '"%s");' % flavor
    )
    out.append("}\n")
    out.append("#else\n")
    out.append("int main(void)\n{")
    out.append(
        "  /* stated rather than skipped, as a pass would be "
        "indistinguishable */"
    )
    out.append(
        '  printf("shim: smm %s NOT ATTEMPTED '
        '(work-group barriers need OpenMP)\\n");' % flavor
    )
    out.append("  return EXIT_SUCCESS;\n}\n")
    out.append("#endif")
    with open(args.output, "w") as f:
        f.write("\n".join(out) + "\n")
    print("%s: %d multiply, %d transpose" % (args.output, len(smm), len(trs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="run acc_bench, keep build strings")
    cap.add_argument("bench", nargs="+", help="acc_bench.x (fp64, fp32)")
    cap.add_argument("-o", "--output", default="shim_smm.defs")
    gen = sub.add_parser("generate", help="build strings into a flavor")
    gen.add_argument("defs")
    gen.add_argument("elem", choices=["double", "float"])
    gen.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    capture(args) if "capture" == args.command else generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
