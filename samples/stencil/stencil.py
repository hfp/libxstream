#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time


CASES = (
    {"case": "direct", "method": 0, "trim": None},
    {"case": "staged-r1", "method": 1, "trim": None},
    {"case": "staged-r2", "method": 2, "trim": None},
    {"case": "staged-r1-trim3", "method": 1, "trim": 3},
)

FIELDNAMES = (
    "n",
    "dims",
    "case",
    "method_arg",
    "trim",
    "reported_method",
    "k_steps",
    "r_per_step",
    "strips_per_wg",
    "time_s",
    "gpoints_s",
    "ms_per_step",
    "bandwidth_gbs",
    "returncode",
    "command",
)

METHOD_RE = re.compile(r"Method:\s+([^\s]+)\s+\(K=([0-9]+),\s*r=([0-9]+),\s*strips/WG=([0-9]+)\)")
TIME_RE = re.compile(r"Time:\s+([0-9.]+)\s+s")
THROUGHPUT_RE = re.compile(r"Throughput:\s+([0-9.]+)\s+GPoints/s")
PER_STEP_RE = re.compile(r"Per step:\s+([0-9.]+)\s+ms")
BANDWIDTH_RE = re.compile(r"Bandwidth:\s+([0-9.]+)\s+GB/s")


def parse_sizes(values):
    result = []
    if not values:
        return [800]
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                parts = item.split(":")
                if len(parts) not in (2, 3):
                    raise ValueError("invalid size range '{}'".format(item))
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if 3 == len(parts) else 1
                if 0 == step:
                    raise ValueError("invalid zero step in '{}'".format(item))
                current = start
                if 0 < step:
                    while current <= stop:
                        result.append(current)
                        current += step
                else:
                    while current >= stop:
                        result.append(current)
                        current += step
            else:
                result.append(int(item))
    return result


def run_case(args, n, case):
    env = os.environ.copy()
    trim = case["trim"]
    if trim is None:
        env.pop("STENCIL_TRIM", None)
    else:
        env["STENCIL_TRIM"] = str(trim)

    command = [args.exe, "-m", str(case["method"]), "-n", str(n), "-d", str(args.dims)]
    if args.steps is not None:
        command.extend(["-t", str(args.steps)])
    if args.warmup is not None:
        command.extend(["-w", str(args.warmup)])
    if args.extra:
        command.extend(args.extra)

    shell_command = " ".join(command)
    if trim is not None:
        shell_command = "STENCIL_TRIM={} {}".format(trim, shell_command)

    row = {
        "n": n,
        "dims": args.dims,
        "case": case["case"],
        "method_arg": case["method"],
        "trim": "" if trim is None else trim,
        "reported_method": "",
        "k_steps": "",
        "r_per_step": "",
        "strips_per_wg": "",
        "time_s": "",
        "gpoints_s": "",
        "ms_per_step": "",
        "bandwidth_gbs": "",
        "returncode": "",
        "command": shell_command,
    }

    if args.dry_run:
        print(shell_command)
        row["returncode"] = 0
        return row

    if not args.quiet:
        print("=== n={} case={} ===".format(n, case["case"]))
        print(shell_command)
        sys.stdout.flush()

    started = time.time()
    proc = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=args.timeout,
    )
    elapsed = time.time() - started
    output = proc.stdout
    row["returncode"] = proc.returncode

    method_match = METHOD_RE.search(output)
    if method_match:
        row["reported_method"] = method_match.group(1)
        row["k_steps"] = method_match.group(2)
        row["r_per_step"] = method_match.group(3)
        row["strips_per_wg"] = method_match.group(4)

    for regex, field in (
        (TIME_RE, "time_s"),
        (THROUGHPUT_RE, "gpoints_s"),
        (PER_STEP_RE, "ms_per_step"),
        (BANDWIDTH_RE, "bandwidth_gbs"),
    ):
        match = regex.search(output)
        if match:
            row[field] = match.group(1)

    if args.log_dir:
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        log_name = "n{}_{}.log".format(n, case["case"])
        log_path = os.path.join(args.log_dir, log_name)
        with open(log_path, "w") as log_file:
            log_file.write(output)
        row["log"] = log_path

    if not args.quiet:
        if row["gpoints_s"]:
            print("{} GPoints/s, {} ms/step".format(row["gpoints_s"], row["ms_per_step"]))
        else:
            print("no result parsed, returncode={}, elapsed={:.3f}s".format(proc.returncode, elapsed))
        sys.stdout.flush()

    if 0 != proc.returncode and args.stop_on_error:
        sys.stderr.write(output)
        raise RuntimeError("command failed: {}".format(shell_command))

    return row


def write_csv(path, rows):
    extra_fields = []
    for row in rows:
        for key in row:
            if key not in FIELDNAMES and key not in extra_fields:
                extra_fields.append(key)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(FIELDNAMES) + extra_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, rows):
    with open(path, "w") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, sort_keys=True))
            jsonl_file.write("\n")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Run stencil.x benchmark cases and collect CSV results."
    )
    parser.add_argument(
        "sizes",
        nargs="*",
        help="Grid sizes, comma lists, or ranges start:stop[:step] (default: 800).",
    )
    parser.add_argument(
        "--sizes",
        dest="sizes_option",
        action="append",
        help="Grid sizes, comma lists, or ranges start:stop[:step].",
    )
    parser.add_argument("--exe", default="./stencil.x", help="Path to stencil executable.")
    parser.add_argument("--dims", type=int, default=3, help="Stencil term count passed with -d.")
    parser.add_argument("--steps", type=int, help="Time steps passed with -t.")
    parser.add_argument("--warmup", type=int, help="Warmup steps passed with -w.")
    parser.add_argument("--output", default="stencil-results.csv", help="CSV output path.")
    parser.add_argument("--jsonl", help="Optional JSON-lines output path.")
    parser.add_argument("--log-dir", help="Optional directory for raw stencil.x output logs.")
    parser.add_argument("--timeout", type=float, help="Timeout per case in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed cases.")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra arguments appended to each stencil.x invocation after --.",
    )
    args = parser.parse_args(argv)
    args.stop_on_error = not args.keep_going

    try:
        sizes = parse_sizes((args.sizes_option or []) + args.sizes)
    except ValueError as exc:
        parser.error(str(exc))

    rows = []
    for n in sizes:
        for case in CASES:
            rows.append(run_case(args, n, case))

    if not args.dry_run:
        write_csv(args.output, rows)
        if args.jsonl:
            write_jsonl(args.jsonl, rows)
        if not args.quiet:
            print("wrote {} rows to {}".format(len(rows), args.output))
            if args.jsonl:
                print("wrote {} rows to {}".format(len(rows), args.jsonl))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
