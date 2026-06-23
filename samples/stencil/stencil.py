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

METHOD_RE = re.compile(
    r"Method:\s+([^\s]+)\s+\(K=([0-9]+),\s*r=([0-9]+),\s*strips/WG=([0-9]+)\)"
)
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
            print(
                "{} GPoints/s, {} ms/step".format(row["gpoints_s"], row["ms_per_step"])
            )
        else:
            print(
                "no result parsed, returncode={}, elapsed={:.3f}s".format(
                    proc.returncode, elapsed
                )
            )
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


def read_csv(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def row_float(row, field):
    value = row.get(field, "")
    if "" == value:
        return None
    return float(value)


def memory_roof(rows, peak_bandwidth_gbs):
    bytes_per_point = []
    for row in rows:
        gpoints_s = row_float(row, "gpoints_s")
        bandwidth_gbs = row_float(row, "bandwidth_gbs")
        if gpoints_s and bandwidth_gbs:
            bytes_per_point.append(bandwidth_gbs / gpoints_s)
    if not bytes_per_point:
        return None, None
    bytes_avg = sum(bytes_per_point) / len(bytes_per_point)
    return peak_bandwidth_gbs / bytes_avg, bytes_avg


def format_percent(value):
    if 10.0 <= value:
        return "{:.0f}%".format(value)
    return "{:.1f}%".format(value)


def case_label(case, points, roof_gpoints_s):
    if not roof_gpoints_s:
        return case
    percentages = sorted(100.0 * point[1] / roof_gpoints_s for point in points)
    lower = format_percent(percentages[0])
    upper = format_percent(percentages[-1])
    if lower == upper:
        return "{} ({} of roof)".format(case, lower)
    return "{} ({}-{} of roof)".format(case, lower, upper)


def memory_roof_label(peak_bandwidth_gbs, roof_gpoints_s, bytes_per_point):
    return "Memory roof uses {:.0f} GB/s / {:.1f} B/point (~{} GPoints/s)".format(
        peak_bandwidth_gbs, bytes_per_point, int(round(roof_gpoints_s))
    )


def plot_png(path, rows, metric, title, dpi, peak_bandwidth_gbs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "gpoints_s": "Throughput [GPoints/s]",
        "ms_per_step": "Time per step [ms]",
        "bandwidth_gbs": "Effective bandwidth [GB/s]",
        "time_s": "Total time [s]",
    }
    series = {}
    for row in rows:
        if str(row.get("returncode", "0")) not in ("", "0"):
            continue
        n_value = row_float(row, "n")
        value = row_float(row, metric)
        if n_value is None or value is None:
            continue
        case = row.get("case", "unknown")
        series.setdefault(case, []).append((int(n_value), value))

    if not series:
        raise RuntimeError("no plottable rows found for metric '{}'".format(metric))

    roof_gpoints_s, bytes_per_point = (None, None)
    if peak_bandwidth_gbs and "gpoints_s" == metric:
        roof_gpoints_s, bytes_per_point = memory_roof(rows, peak_bandwidth_gbs)

    fig, axis = plt.subplots(figsize=(7.2, 4.4), dpi=dpi)
    for case in sorted(series):
        points = sorted(series[case])
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=case_label(case, points, roof_gpoints_s),
        )

    if roof_gpoints_s and bytes_per_point:
        x_values = sorted(point[0] for points in series.values() for point in points)
        axis.plot(
            [x_values[0], x_values[-1]],
            [roof_gpoints_s, roof_gpoints_s],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label=memory_roof_label(
                peak_bandwidth_gbs, roof_gpoints_s, bytes_per_point
            ),
        )

    axis.set_xlabel("Grid size n for n x n x n")
    axis.set_ylabel(labels[metric])
    axis.grid(True, color="#d9d9d9", linewidth=0.8)
    axis.legend(frameon=False)
    if title:
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def default_plot_path(csv_path):
    root, ext = os.path.splitext(csv_path)
    if root and ".csv" == ext.lower():
        return root + ".png"
    return csv_path + ".png"


def main(argv):
    parser = argparse.ArgumentParser(
        description="Run stencil.x benchmark cases, collect CSV results, and optionally plot PNG output."
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
    parser.add_argument(
        "--exe", default="./stencil.x", help="Path to stencil executable."
    )
    parser.add_argument(
        "--dims", type=int, default=3, help="Stencil term count passed with -d."
    )
    parser.add_argument("--steps", type=int, help="Time steps passed with -t.")
    parser.add_argument("--warmup", type=int, help="Warmup steps passed with -w.")
    parser.add_argument(
        "--input", help="Read an existing CSV instead of running benchmarks."
    )
    parser.add_argument("--output", default="stencil.csv", help="CSV output path.")
    parser.add_argument("--jsonl", help="Optional JSON-lines output path.")
    parser.add_argument("--plot", help="Optional PNG output path.")
    parser.add_argument(
        "--plot-metric",
        choices=("gpoints_s", "ms_per_step", "bandwidth_gbs", "time_s"),
        default="gpoints_s",
        help="CSV metric to plot (default: gpoints_s).",
    )
    parser.add_argument("--plot-title", help="Optional plot title.")
    parser.add_argument("--plot-dpi", type=int, default=180, help="PNG dots per inch.")
    parser.add_argument(
        "--peak-bandwidth-gbs",
        type=float,
        help="Overlay memory roof using this peak bandwidth in GB/s.",
    )
    parser.add_argument(
        "--log-dir", help="Optional directory for raw stencil.x output logs."
    )
    parser.add_argument("--timeout", type=float, help="Timeout per case in seconds.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue after failed cases."
    )
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

    if args.input:
        if not args.plot:
            args.plot = default_plot_path(args.input)
        rows = read_csv(args.input)
    else:
        rows = []
        for n in sizes:
            for case in CASES:
                rows.append(run_case(args, n, case))

    if not args.dry_run:
        if not args.input:
            write_csv(args.output, rows)
        if args.jsonl:
            write_jsonl(args.jsonl, rows)
        if args.plot:
            plot_png(
                args.plot,
                rows,
                args.plot_metric,
                args.plot_title,
                args.plot_dpi,
                args.peak_bandwidth_gbs,
            )
        if not args.quiet:
            if not args.input:
                print("wrote {} rows to {}".format(len(rows), args.output))
            if args.jsonl:
                print("wrote {} rows to {}".format(len(rows), args.jsonl))
            if args.plot:
                print("wrote plot to {}".format(args.plot))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
