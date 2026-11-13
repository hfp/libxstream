#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
"""Express a kernel's instruction counts in named mixes, and predict its time.

A named mix (e.g., "stream", "gemm", "atomics" of samples/mixes) is a kernel
whose counts are known on a target. A kernel k is projected onto the mixes j
counted on the same target and in the same precision:

  c_k ~ sum_j alpha_j c_j,  alpha >= 0 (non-negative least squares)

i.e., one launch of k is alpha_j launches of mix j. Rows are scaled by the
kernel's own counts, hence the residual is relative, and what it leaves out
(coverage) is reported with the categories that miss most. A category that
counts the sum of others (e.g., "flat" with "flat_read" present) is dropped,
and so is any category one side lacks: counters can come and go.

With times of the mixes on a device (measured, published, or from a datasheet,
each row saying which), a kernel's time is predicted in two ways:

  additive   T_k(d) = overlap * sum_j alpha_j t_j(d)
  relative   T_k(d) = T_k(a) * sum_j beta_j t_j(d) / t_j(a)          (arithmetic)
             T_k(d) = T_k(a) * prod_j (t_j(d) / t_j(a)) ** beta_j    (geometric)

where "a" is an anchor device on which k itself was measured, and beta_j is
the share of the anchor's time the mixture attributes to mix j:
beta_j = alpha_j t_j(a) / sum_i alpha_i t_i(a). The arithmetic form is the
ratio of the additive predictions on both devices: exact under the additive
model, whose overlap factor cancels; the geometric form is
its first-order expansion in log space, less sensitive to one outlying ratio.
Both are weighted means: a mix without a time on both devices drops out and
the others renormalize. The few numbers that are fitted (the overlap factor)
are written to a weights file, which says what they were fitted on.

Times compose arithmetically (a launch is the sum of what its parts cost);
ratios compose geometrically: a score over kernels, devices or subscores is a
geometric mean (independent of the reference, and a missing subscore drops a
dimension of the volume the others span). It is reported as is and trimmed
(--trim of each end, in log space), so that a peak or a low cannot decide it.

A category counts only if the kernel and every mix report it (a zero is a
report): a counter that a newer simulator adds, or an older one lacks, drops
out instead of being read as zero.

  tool_score.py project  --counts MIXES.tsv [--markdown]
  tool_score.py predict  --counts MIXES.tsv --times TIMES.csv [--anchor DEVICE]
                         [--beta-from target|anchor] [--mean arithmetic|geometric]
                         [--weights W.json]
                         [--weights-out W.json] [--markdown]
  tool_score.py validate --counts MIXES.tsv [--times TIMES.csv] [--markdown]

MIXES.tsv: target, kernel, category, value per launch (tests/atomics.sh
writes it); categories "work_*" are the kernel's own work (flops, bytes),
which are reported but not projected. TIMES.csv: device, target, kernel,
time, unit (s, ms, us, ns; default s), source. Only times are accepted, never
rates: time is what adds up, hence what an arithmetic mean may average, and
one orientation ("lower is better") is kept throughout. A rate (GB/s, GFLOPS)
is the work over a time; convert it with the work before it comes here. Every
reported number says its orientation. A kernel name may end in "_double" or
"_float", which
selects the mixes of that precision (--precision says what a kernel without
the suffix is). The backend is "native" (numpy and scipy); "predict" is the
interface for LIBXS PREDICT and not implemented yet.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import sys
from collections import defaultdict

try:
    import numpy
    from scipy.optimize import nnls
    from scipy.stats import spearmanr
except ImportError as error:  # numpy and scipy are the only dependencies
    sys.stderr.write(
        "tool_score.py: %s (numpy and scipy are needed)\n" % error
    )
    sys.exit(2)


# "empty" dispatches as many work-groups as the stream with almost no instructions: with "waves" among
# the categories, it is what a kernel's dispatch costs, apart from what its instructions cost
MIXES = ["stream", "gemm", "atomics", "gather", "empty"]
PRECISIONS = ["double", "float"]
# not instruction categories: the launch's geometry and totals
EXCLUDE = ["launches", "wave", "instructions"]
# a category that is the sum of its parts is dropped when a part is present
PARTS = {
    "flat": ["flat_read", "flat_write", "flat_atomic"],
    "vmem": ["vmem_rd", "vmem_wr"],
}
WEIGHTS = {"version": 1, "overlap": 1.0, "exclude": EXCLUDE, "fitted_on": None}


def split_name(kernel, default):
    """Base name and precision of a kernel ("gemm_float" -> "gemm", "float")."""
    base, _, suffix = kernel.rpartition("_")
    if base and suffix in PRECISIONS:
        return base, suffix
    return kernel, default


def read_counts(path):
    """{target: {kernel: {category: value}}} and the meta line of the file."""
    counts, meta = {}, None
    with open(path) as file:
        lines = []
        for line in file:
            if line.startswith("#"):
                meta = meta or line[1:].strip()
            elif line.strip():
                lines.append(line)
    for row in csv.DictReader(lines, delimiter="\t"):
        kernels = counts.setdefault(row["target"], {})
        kernels.setdefault(row["kernel"], {})[row["category"]] = float(
            row["value"]
        )
    return counts, meta


UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def read_times(path):
    """[{device, target, kernel, seconds, source}] of a CSV file, every time in seconds."""
    with open(path) as file:
        rows = [
            row
            for row in csv.DictReader(
                line for line in file if not line.startswith("#")
            )
        ]
    for row in rows:
        unit = (row.get("unit") or "s").strip()
        if unit not in UNITS:  # e.g., "GB/s": a rate, of the other orientation
            sys.stderr.write(
                'tool_score.py: %s/%s has unit "%s": only times (%s) are accepted\n'
                % (row["device"], row["kernel"], unit, ", ".join(UNITS))
            )
            sys.exit(2)
        value = row.get("time", row.get("seconds"))
        row["seconds"] = float(value) * UNITS[unit]
        if not 0 < row["seconds"]:
            sys.stderr.write(
                "tool_score.py: %s/%s: a time must be positive\n"
                % (row["device"], row["kernel"])
            )
            sys.exit(2)
        row["source"] = row.get("source") or "unknown"
    return rows


def categories(kernel, mixes, exclude):
    """Categories the kernel and all mixes have, without totals, work, and sums of present parts."""
    names = set(kernel)
    for mix in mixes:
        names &= set(mix)
    names = {
        name
        for name in names
        if name not in exclude and not name.startswith("work_")
    }
    for whole, parts in PARTS.items():
        if whole in names and any(part in names for part in parts):
            names.discard(whole)
    return sorted(names)


class Native:
    """Projection by non-negative least squares on relatively scaled rows."""

    name = "native"

    def project(self, kernel, mixes, names):
        b = numpy.array([kernel[name] for name in names])
        a = numpy.array([[mix[name] for mix in mixes] for name in names])
        floor = 1e-3 * max(
            b.sum(), 1.0
        )  # a category the kernel barely has does not dominate
        scale = 1.0 / numpy.maximum(b, floor)
        alpha, _ = nnls(a * scale[:, None], b * scale)
        error = (a @ alpha - b) * scale
        norm = numpy.linalg.norm(b * scale)
        residual = float(numpy.linalg.norm(error) / norm) if 0 < norm else 0.0
        worst = sorted(zip(names, error), key=lambda item: -abs(item[1]))[:3]
        return (
            alpha,
            residual,
            [
                (name, float(value))
                for name, value in worst
                if 1e-3 < abs(value)
            ],
        )


class Predict:
    """LIBXS PREDICT (libxs_predict_eval_flat via ctypes, cf. samples/smm/tune_multiply.py): not implemented.

    The interface is the same as Native.project: counts of a kernel and of the
    mixes on one target in, the mixes' weights, a relative residual, and the
    categories that miss most out. The intent is nonlinear interpolation where
    the linear span of the mixes does not reach.
    """

    name = "predict"

    def project(self, kernel, mixes, names):
        raise NotImplementedError(
            "the LIBXS PREDICT backend is an interface only (use --backend native)"
        )


def projections(counts, backend, exclude, default, only=None):
    """[{target, kernel, precision, mixes, alpha, beta, residual, worst, categories, work}] for every kernel."""
    result = []
    for target in sorted(counts):
        kernels = counts[target]
        for kernel in sorted(kernels):
            base, precision = split_name(kernel, default)
            if base in MIXES and only is None:
                continue  # a mix is projected only by the validation
            if only is not None and kernel != only[0]:
                continue
            names = [
                mix for mix in MIXES if "%s_%s" % (mix, precision) in kernels
            ]
            if only is not None:
                names = [mix for mix in names if mix not in only[1]]
            if not names:
                continue
            mixes = [kernels["%s_%s" % (mix, precision)] for mix in names]
            used = categories(kernels[kernel], mixes, exclude)
            if not used:
                continue
            alpha, residual, worst = backend.project(
                kernels[kernel], mixes, used
            )
            share = numpy.array(
                [
                    alpha[i] * mixes[i].get("instructions", 0.0)
                    for i in range(len(names))
                ]
            )
            beta = share / share.sum() if 0 < share.sum() else share
            work = {
                name[5:]: value
                for name, value in kernels[kernel].items()
                if name.startswith("work_")
            }
            result.append(
                {
                    "target": target,
                    "kernel": kernel,
                    "precision": precision,
                    "mixes": names,
                    "alpha": [float(x) for x in alpha],
                    "beta": [float(x) for x in beta],
                    "residual": residual,
                    "worst": worst,
                    "categories": used,
                    "work": work,
                }
            )
    return result


def geomean(ratios, trim=0.0):
    """Geometric mean of ratios, without the fraction trim of the lowest and of the highest (in log space).

    The ratios must share one orientation (all "higher is better", or all
    "lower is better"): inverting all of them inverts the mean, but a mean of
    mixed orientations means nothing.
    """
    logs = numpy.sort(numpy.log(numpy.asarray(ratios, dtype=float)))
    cut = int(trim * len(logs))
    return math.exp(float(numpy.mean(logs[cut : len(logs) - cut])))


def table(rows, header, markdown):
    """Print rows as a markdown table or tab-separated."""
    if markdown:
        print("| %s |" % " | ".join(header))
        print("|%s" % ("---|" * len(header)))
        for row in rows:
            print("| %s |" % " | ".join(str(cell) for cell in row))
    else:
        print("\t".join(header))
        for row in rows:
            print("\t".join(str(cell) for cell in row))


def mixture(projection):
    """'0.71 stream + 0.03 gemm' (launches of each mix per launch of the kernel)."""
    terms = [
        "%.3g %s" % (a, mix)
        for mix, a in zip(projection["mixes"], projection["alpha"])
        if 1e-9 < a
    ]
    return " + ".join(terms) or "none"


def shares(projection):
    """'stream 91%, gemm 9%' (shares of the kernel's instructions)."""
    terms = [
        "%s %.0f%%" % (mix, 100 * b)
        for mix, b in zip(projection["mixes"], projection["beta"])
        if 0.005 <= b
    ]
    return ", ".join(terms) or "-"


def misses(projection):
    """Categories the mixes miss most, signed (negative: the mixes have too few)."""
    return (
        ", ".join(
            "%s %+.0f%%" % (name, 100 * value)
            for name, value in projection["worst"]
        )
        or "-"
    )


def command_project(args, weights):
    counts, _ = read_counts(args.counts)
    rows = []
    for p in projections(
        counts, args.backend, weights["exclude"], args.precision
    ):
        rows.append(
            [
                p["target"],
                p["kernel"],
                mixture(p),
                shares(p),
                "%.1f%%" % (100 * p["residual"]),
                misses(p),
            ]
        )
    table(
        rows,
        [
            "Target",
            "Kernel",
            "Mixture (launches per launch)",
            "Instructions",
            "Residual [lower is better]",
            "Misses most",
        ],
        args.markdown,
    )
    return 0


def times_of(rows):
    """{(device, kernel): (seconds, source)} and {device: target}."""
    times, targets = {}, {}
    for row in rows:
        times[(row["device"], row["kernel"])] = (row["seconds"], row["source"])
        targets[row["device"]] = row["target"]
    return times, targets


def additive(p, device, times, overlap):
    """overlap * sum_j alpha_j t_j(d), or None if a mix with weight has no time on the device."""
    total = 0.0
    for mix, a in zip(p["mixes"], p["alpha"]):
        if 1e-9 < a:
            key = (device, "%s_%s" % (mix, p["precision"]))
            if key not in times:
                return None
            total += a * times[key][0]
    return overlap * total


def relative(p, device, anchor, times, mean="arithmetic"):
    """T_k(a) times the beta-weighted mean of t_j(d) / t_j(a), beta_j the anchor's time shares (mixes timed on both)."""
    anchored = times.get((anchor, p["kernel"]))
    parts = []
    for mix, a in zip(p["mixes"], p["alpha"]):
        name = "%s_%s" % (mix, p["precision"])
        if 1e-9 < a and (device, name) in times and (anchor, name) in times:
            parts.append(
                (
                    a * times[(anchor, name)][0],
                    times[(device, name)][0] / times[(anchor, name)][0],
                )
            )
    total = sum(weight for weight, _ in parts)
    if anchored is None or total <= 0:
        return None
    if "geometric" == mean:
        return anchored[0] * math.exp(
            sum(weight * math.log(ratio) for weight, ratio in parts) / total
        )
    # arithmetic: the ratio of the additive predictions (the overlap factor cancels)
    on_device, on_anchor = additive(p, device, times, 1.0), additive(
        p, anchor, times, 1.0
    )
    if on_device and on_anchor:
        return anchored[0] * on_device / on_anchor
    return anchored[0] * sum(weight * ratio for weight, ratio in parts) / total


def fit_overlap(counts, times, targets, backend, weights, default):
    """Geometric mean of measured over additive time, over every kernel measured where its mixes are timed."""
    ratios, fitted = [], []
    for device, target in sorted(targets.items()):
        for p in projections(
            {target: counts.get(target, {})},
            backend,
            weights["exclude"],
            default,
        ):
            measured = times.get((device, p["kernel"]))
            predicted = additive(p, device, times, 1.0)
            if (
                measured is not None
                and predicted
                and "measured" == measured[1]
            ):
                ratios.append(math.log(measured[0] / predicted))
                fitted.append("%s/%s" % (device, p["kernel"]))
    return (math.exp(sum(ratios) / len(ratios)) if ratios else None), fitted


def command_predict(args, weights):
    counts, meta = read_counts(args.counts)
    times, targets = times_of(read_times(args.times))
    if args.weights_out:
        overlap, fitted = fit_overlap(
            counts, times, targets, args.backend, weights, args.precision
        )
        if overlap is not None:
            digest = hashlib.sha256()
            for path in (args.counts, args.times):
                with open(path, "rb") as file:
                    digest.update(file.read())
            weights = dict(
                weights,
                overlap=overlap,
                fitted_on={
                    "kernels": fitted,
                    "counts": meta,
                    "inputs_sha256": digest.hexdigest(),
                    "date": datetime.date.today().isoformat(),
                    "backend": args.backend.name,
                },
            )
        with open(args.weights_out, "w") as file:
            json.dump(weights, file, indent=2, sort_keys=True)
            file.write("\n")
    rows = []
    for device, target in sorted(targets.items()):
        for p in projections(
            {target: counts.get(target, {})},
            args.backend,
            weights["exclude"],
            args.precision,
        ):
            if args.beta_from == "anchor" and args.anchor in targets:
                anchored = projections(
                    {
                        targets[args.anchor]: counts.get(
                            targets[args.anchor], {}
                        )
                    },
                    args.backend,
                    weights["exclude"],
                    args.precision,
                    (p["kernel"], []),
                )
                if anchored:
                    p = dict(
                        p,
                        mixes=anchored[0]["mixes"],
                        alpha=anchored[0]["alpha"],
                    )
            sum_ = additive(p, device, times, weights["overlap"])
            rel = (
                relative(p, device, args.anchor, times, args.mean)
                if args.anchor
                else None
            )
            measured = times.get((device, p["kernel"]))
            rows.append(
                [
                    device,
                    p["kernel"],
                    "%.4g" % sum_ if sum_ else "-",
                    "%.4g" % rel if rel else "-",
                    "%.4g (%s)" % measured if measured else "-",
                    "%.1f%%" % (100 * p["residual"]),
                ]
            )
    table(
        rows,
        [
            "Device",
            "Kernel",
            "Additive [s, lower is better]",
            "Relative [s, lower is better]",
            "Time [s, lower is better] (source)",
            "Residual [lower is better]",
        ],
        args.markdown,
    )
    return 0


def command_validate(args, weights):
    counts, _ = read_counts(args.counts)
    rows = []
    # identity: a mix projected onto all mixes is itself; leaving it out says what the others cover
    for target in sorted(counts):
        for kernel in sorted(counts[target]):
            base, _ = split_name(kernel, args.precision)
            if base not in MIXES:
                continue
            same = projections(
                counts,
                args.backend,
                weights["exclude"],
                args.precision,
                (kernel, []),
            )
            left = projections(
                counts,
                args.backend,
                weights["exclude"],
                args.precision,
                (kernel, [base]),
            )
            for p in same:
                if p["target"] == target:
                    alpha = dict(zip(p["mixes"], p["alpha"]))
                    error = max(
                        abs(a - (1.0 if mix == base else 0.0))
                        for mix, a in alpha.items()
                    )
                    rows.append(
                        [
                            target,
                            kernel,
                            "identity",
                            "max abs(alpha - e) = %.2g" % error,
                            "%.1f%%" % (100 * p["residual"]),
                        ]
                    )
            for p in left:
                if p["target"] == target:
                    rows.append(
                        [
                            target,
                            kernel,
                            "others only",
                            mixture(p),
                            "%.1f%%" % (100 * p["residual"]),
                        ]
                    )
    table(
        rows,
        ["Target", "Mix", "Test", "Result", "Residual [lower is better]"],
        args.markdown,
    )
    if (
        args.times
    ):  # across devices: each device with a measured kernel predicted from every other as the anchor
        times, targets = times_of(read_times(args.times))
        pairs = []
        for device, target in sorted(targets.items()):
            for p in projections(
                {target: counts.get(target, {})},
                args.backend,
                weights["exclude"],
                args.precision,
            ):
                measured = times.get((device, p["kernel"]))
                for anchor in sorted(targets):
                    if (
                        anchor != device
                        and measured is not None
                        and "measured" == measured[1]
                    ):
                        rel = relative(p, device, anchor, times, args.mean)
                        if rel:
                            pairs.append(
                                (math.log(rel), math.log(measured[0]))
                            )
        if 2 < len(pairs):
            pred, meas = numpy.array(pairs).T
            rho = spearmanr(pred, meas)[0]
            # ratios aggregate geometrically: the bias is the geometric mean of predicted / measured,
            # also without the extremes (--trim of each end), so that a peak or a low does not decide it
            ratios = numpy.exp(pred - meas)
            print(
                "\nAcross devices: %i predictions, Spearman %.3f, median |log error| %.3f, "
                "predicted/measured %.3f (geometric mean; 1 is exact), %.3f (without %g%% of each end)"
                % (
                    len(pairs),
                    rho,
                    float(numpy.median(numpy.abs(pred - meas))),
                    geomean(ratios),
                    geomean(ratios, args.trim),
                    100 * args.trim,
                )
            )
        else:
            print(
                "\nAcross devices: fewer than three predictions with a measured time"
            )
        # on each device: the additive prediction against every measured kernel; the geometric
        # mean of measured / additive is the overlap factor, and what is left once it is taken out
        # is the scatter the mixtures do not explain (all ratios: lower is better for neither)
        rows = []
        for device, target in sorted(targets.items()):
            pairs = []
            for p in projections(
                {target: counts.get(target, {})},
                args.backend,
                weights["exclude"],
                args.precision,
            ):
                measured = times.get((device, p["kernel"]))
                predicted = additive(p, device, times, 1.0)
                if (
                    measured is not None
                    and "measured" == measured[1]
                    and predicted
                ):
                    pairs.append(
                        (p["kernel"], predicted, measured[0], p["residual"])
                    )
            if 2 < len(pairs):
                pred = numpy.log([pair[1] for pair in pairs])
                meas = numpy.log([pair[2] for pair in pairs])
                overlap = geomean(numpy.exp(meas - pred))
                left = numpy.abs(meas - pred - math.log(overlap))
                rows.append(
                    [
                        device,
                        target,
                        len(pairs),
                        "%.3f" % spearmanr(pred, meas)[0],
                        "%.3f" % overlap,
                        "%.3f" % geomean(numpy.exp(meas - pred), args.trim),
                        "%.2f" % math.exp(float(numpy.median(left))),
                        "%.2f" % math.exp(float(numpy.max(left))),
                    ]
                )
                if args.verbose:
                    for (kernel, p_, m_, r_), d in zip(pairs, left):
                        print(
                            "%s\t%s\tadditive %.4g ms\tmeasured %.4g ms\tx%.2f after overlap\tresidual %.1f%%"
                            % (
                                device,
                                kernel,
                                1e3 * p_ * overlap,
                                1e3 * m_,
                                math.exp(d),
                                100 * r_,
                            )
                        )
        if rows:
            print()
            table(
                rows,
                [
                    "Device",
                    "Counts of",
                    "Kernels",
                    "Spearman [higher is better]",
                    "Measured/additive (geometric mean)",
                    "Same, trimmed",
                    "Median factor off [lower is better]",
                    "Worst factor off [lower is better]",
                ],
                args.markdown,
            )
    return 0


def fit_features(c):
    """Launch, issue (lane-instructions), memory (memory lane-operations) of one launch."""
    wave = c.get("wave", 1.0)
    memory = (
        c.get("flat_read", 0.0)
        + c.get("flat_write", 0.0)
        + c.get("flat_atomic", 0.0)
    )
    return [1.0, c.get("instructions", 0.0) * wave, memory * wave]


def command_fit(args, weights):
    """Per device and donor target: time = launch + issue + memory, every kernel predicted left out."""
    counts, _ = read_counts(args.counts)
    rows, out = [], []
    times = defaultdict(dict)
    for row in read_times(args.times):
        if "measured" == row["source"]:
            times[row["device"]][row["kernel"]] = row["seconds"]
    donors = args.donors.split(",") if args.donors else sorted(counts)
    for device in sorted(times):
        for donor in donors:
            kernels = [
                k
                for k in sorted(counts.get(donor, {}))
                if k in times[device]
                and split_name(k, args.precision)[0] not in MIXES
                and "instructions" in counts[donor][k]
            ]
            left = [
                i
                for i, k in enumerate(kernels)
                if any(pattern in k for pattern in args.leave)
            ]
            fitted = [i for i in range(len(kernels)) if i not in left]
            if len(fitted) < 4:
                continue
            x = numpy.array([fit_features(counts[donor][k]) for k in kernels])
            y = numpy.array([times[device][k] for k in kernels])
            scale = x.max(axis=0)
            scale[0 == scale] = 1.0

            def solve(
                idx,
            ):  # relative error: each row divided by its measured time
                return (
                    nnls(
                        x[idx] / scale / y[idx][:, None], numpy.ones(len(idx))
                    )[0]
                    / scale
                )

            predicted = numpy.array(
                [
                    max(
                        float(x[i] @ solve([j for j in fitted if j != i])),
                        1e-12,
                    )
                    for i in range(len(kernels))
                ]
            )
            for i, k in enumerate(kernels):
                out.append(
                    [
                        device,
                        donor,
                        k,
                        "%.6g" % y[i],
                        "%.6g" % predicted[i],
                        "left" if i in left else "fitted",
                    ]
                )
            p, m = predicted[fitted], y[fitted]
            factor = numpy.exp(numpy.abs(numpy.log(p / m)))
            coef = solve(fitted)
            rows.append(
                [
                    device,
                    donor,
                    len(fitted),
                    "%.3f" % spearmanr(p, m)[0],
                    "%.2f" % numpy.median(factor),
                    "%.2f" % factor.max(),
                    "%.3f" % geomean(m / p),
                    "%.3g" % (1e3 * coef[0]),
                    "%.3g" % (1e12 * coef[1]),
                    "%.3g" % (1e12 * coef[2]),
                ]
            )
    table(
        rows,
        [
            "Device",
            "Counts of",
            "Kernels",
            "Spearman [higher is better]",
            "Median factor off [lower is better]",
            "Worst factor off",
            "Measured/predicted (geometric mean)",
            "Launch [ms]",
            "Issue [ps per lane-instruction]",
            "Memory [ps per lane-operation]",
        ],
        args.markdown,
    )
    if args.csv:
        with open(args.csv, "w") as file:
            file.write("device,counts,kernel,measured_s,predicted_s,fit\n")
            for line in out:
                file.write(
                    ",".join('"%s"' % c if "," in c else c for c in line)
                    + "\n"
                )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=["project", "predict", "validate", "fit"]
    )
    parser.add_argument(
        "--counts",
        required=True,
        help="long-form counts (target, kernel, category, value)",
    )
    parser.add_argument(
        "--times",
        help="times of mixes and kernels (device, target, kernel, seconds, source)",
    )
    parser.add_argument(
        "--anchor",
        help="device on which the kernels were measured (relative prediction)",
    )
    parser.add_argument(
        "--beta-from",
        choices=["target", "anchor"],
        default="target",
        help="whose counts give the mixture of the relative prediction",
    )
    parser.add_argument(
        "--mean",
        choices=["arithmetic", "geometric"],
        default="arithmetic",
        help="weighted mean of the relative prediction",
    )
    parser.add_argument(
        "--donors",
        help="fit: comma-separated targets whose counts are used (default: every target)",
    )
    parser.add_argument(
        "--leave",
        action="append",
        default=[],
        help="fit: kernels containing this text are predicted but never fitted (repeatable)",
    )
    parser.add_argument(
        "--csv", help="fit: write every prediction to this file"
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.1,
        help="fraction of the lowest and of the highest ratios a score leaves out",
    )
    parser.add_argument(
        "--weights",
        help="weights file (default: overlap 1, the built-in exclusions)",
    )
    parser.add_argument(
        "--weights-out",
        help="fit the weights where kernels were measured, and write them",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="double",
        help="precision of a kernel whose name has no suffix",
    )
    parser.add_argument(
        "--backend", choices=["native", "predict"], default="native"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="markdown tables instead of tab-separated",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also every kernel of a validation",
    )
    args = parser.parse_args()
    weights = dict(WEIGHTS)
    if args.weights:
        with open(args.weights) as file:
            weights.update(json.load(file))
    args.backend = Native() if "native" == args.backend else Predict()
    if args.command in ("predict", "fit") and not args.times:
        parser.error("%s needs --times" % args.command)
    return {
        "project": command_project,
        "fit": command_fit,
        "predict": command_predict,
        "validate": command_validate,
    }[args.command](args, weights)


if __name__ == "__main__":
    sys.exit(main())
