#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
"""The figures of the presentation, from the CSV files beside this script.

  scatter.png    fit.csv: B390 kernel times predicted (left out) from counts of
                 a simulated RX 7900 XTX against measured
  transfer.png   ozaki-kernels.csv, specs.csv: GH200 kernel times predicted
                 from H100 PCIe by datasheet ratios against measured
  ozaki-amd.png  ozaki-kernels.csv, specs.csv: Ozaki-2 FP64 projected for AMD
                 parts against their native FP64 matrix peak

Colors: the categorical blue and orange, and a gray for context, on the black
theme (contrast at least 3.2:1; OKLab difference at least 15.8 between any two
marks under protan, deutan, and tritan simulation).
"""

import csv
import math
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK_2, MUTED = "#191919", "#ffffff", "#c3c2b7", "#898781"
GRID, AXIS, BLUE, ORANGE, GRAY = (
    "#2c2c2a",
    "#383835",
    "#3987e5",
    "#d95926",
    "#6b6a66",
)
DONOR = "amd/rx7900xtx"


def rows(name):
    with open(os.path.join(HERE, name)) as f:
        return list(
            csv.DictReader(line for line in f if not line.startswith("#"))
        )


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 13,
        }
    )


def frame(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0, labelsize=12)


def loglog(ax, lo, hi):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    grid = numpy.geomspace(lo, hi, 50)
    ax.fill_between(
        grid, grid / 2, grid * 2, color=GRID, linewidth=0, zorder=0
    )
    ax.plot(grid, grid, color=MUTED, linewidth=1.2, zorder=1)
    ticks = [
        t
        for t in (
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1,
            3,
            10,
            30,
            100,
            300,
            1000,
        )
        if lo <= t <= hi
    ]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(plt.FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
        axis.set_minor_locator(plt.NullLocator())
    frame(ax)


def band():
    """Legend entry of the shaded band (instead of a label the points could cover)."""
    return plt.Rectangle((0, 0), 1, 1, color=GRID), "within 2x of measured"


def save(fig, name):
    fig.savefig(
        os.path.join(HERE, name), dpi=160, bbox_inches="tight", pad_inches=0.2
    )
    plt.close(fig)
    print("wrote", name)


def scatter():
    data = [r for r in rows("fit.csv") if r["counts"] == DONOR]
    fit = [
        (1e3 * float(r["measured_s"]), 1e3 * float(r["predicted_s"]))
        for r in data
        if r["fit"] == "fitted"
    ]
    left = [
        (1e3 * float(r["measured_s"]), 1e3 * float(r["predicted_s"]))
        for r in data
        if r["fit"] != "fitted"
    ]
    every = fit + left
    lo, hi = min(min(p) for p in every) / 1.6, max(max(p) for p in every) * 1.6
    fig, ax = plt.subplots(figsize=(10, 6.4))
    loglog(ax, lo, hi)
    ax.scatter(
        *zip(*fit), s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3
    )
    ax.scatter(
        *zip(*left),
        s=70,
        facecolor="none",
        edgecolor=GRAY,
        linewidth=2,
        zorder=3
    )
    m, p = numpy.array(fit).T
    off = numpy.exp(numpy.abs(numpy.log(p / m)))
    ax.text(
        0.02,
        0.97,
        "rank correlation %.2f\nmedian within %.2fx\neach kernel left out of its own fit"
        % (spearmanr(p, m)[0], numpy.median(off)),
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        linespacing=1.5,
    )
    ax.set_xlabel(
        "measured on an Intel Arc B390 [ms]", fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        "predicted from counts of a simulated RX 7900 XTX [ms]",
        fontsize=12,
        labelpad=10,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BLUE),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=GRAY,
            markeredgewidth=2,
        ),
    ]
    handles.append(band()[0])
    ax.legend(
        handles,
        [
            "SMM, DBM, Ozaki, stencil kernels",
            "Ozaki GEMM: other code path, left out",
            band()[1],
        ],
        loc="lower right",
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
    )
    save(fig, "scatter.png")


def specs():
    return {r["part"]: r for r in rows("specs.csv")}


def transfer():
    s = specs()
    int8 = float(s["GH200"]["int8_tops"]) / float(s["H100 PCIe"]["int8_tops"])
    bw = float(s["GH200"]["bandwidth_tbs"]) / float(
        s["H100 PCIe"]["bandwidth_tbs"]
    )
    gemm, other = [], []
    for r in rows("ozaki-kernels.csv"):
        g = r["kernel"].startswith("gemm") and "reduce" not in r["kernel"]
        (gemm if g else other).append(
            (
                float(r["gh200_ms"]),
                float(r["h100pcie_ms"]) / (int8 if g else bw),
            )
        )
    every = gemm + other
    lo, hi = min(min(p) for p in every) / 1.6, max(max(p) for p in every) * 1.6
    fig, ax = plt.subplots(figsize=(10, 6.4))
    loglog(ax, lo, hi)
    ax.scatter(
        *zip(*other),
        s=26,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=0.8,
        zorder=2
    )
    ax.scatter(
        *zip(*gemm),
        s=40,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=3
    )
    gm = math.exp(statistics.mean(math.log(m / p) for m, p in gemm))
    om = math.exp(statistics.mean(math.log(m / p) for m, p in other))
    ax.text(
        0.02,
        0.97,
        "int8 GEMM: measured/predicted %.2f (%i cases)\nother kernels: %.2f (%i cases)"
        % (gm, len(gemm), om, len(other)),
        transform=ax.transAxes,
        va="top",
        fontsize=13,
        linespacing=1.5,
    )
    ax.set_xlabel("measured on GH200 [ms]", fontsize=12, labelpad=10)
    ax.set_ylabel(
        "predicted from H100 PCIe by datasheet ratios [ms]",
        fontsize=12,
        labelpad=10,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=c)
        for c in (BLUE, ORANGE)
    ]
    handles.append(band()[0])
    ax.legend(
        handles,
        [
            "int8 GEMM (scaled by int8 peak)",
            "preprocessing, reduction (by bandwidth)",
            band()[1],
        ],
        loc="lower right",
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
    )
    save(fig, "transfer.png")


def projection():
    """[(part, low, high, native)]: Ozaki-2 FP64-equivalent TFLOPS at n=4096 from either anchor."""
    s, best = specs(), {}
    for column, anchor in (
        ("h100pcie_ms", "H100 PCIe"),
        ("gh200_ms", "GH200"),
    ):
        configs = defaultdict(dict)
        for r in rows("ozaki-kernels.csv"):
            if (
                r["size"] == "4096"
                and r["scheme"] == "2"
                and "x" not in r["config"]
            ):
                configs[(r["config"], r["option"])][r["kernel"]] = float(
                    r[column]
                )
        full = [
            k
            for k in configs.values()
            if any(n.startswith("gemm_crt_fused") for n in k) and len(k) >= 4
        ]
        kernels = min(full, key=lambda k: sum(k.values()))
        g = sum(
            t
            for n, t in kernels.items()
            if n.startswith("gemm") and "reduce" not in n
        )
        best[anchor] = (g, sum(kernels.values()) - g)
    out = []
    for part, r in s.items():
        if part in best:
            continue
        tf = []
        for anchor, (g, o) in best.items():
            a = s[anchor]
            t = g * float(a["int8_tops"]) / float(r["int8_tops"]) + o * float(
                a["bandwidth_tbs"]
            ) / float(r["bandwidth_tbs"])
            tf.append(2.0 * 4096**3 / (t * 1e9))
        out.append(
            (
                part,
                min(tf),
                max(tf),
                float(r["fp64_matrix"]) if r["fp64_matrix"] else None,
            )
        )
    return out


def ozaki_amd():
    order = [
        "MI355X",
        "MI325X",
        "MI300X",
        "MI250X (one GCD)",
        "RX 9070 XT",
        "RX 7900 XTX",
    ]
    data = sorted(
        projection(),
        key=lambda r: order.index(r[0]) if r[0] in order else len(order),
    )
    n = len(data)
    ys = list(range(n))[::-1]
    xmax = max(max(r[2], r[3] or 0) for r in data) * 1.35
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=12)
    h = 0.32
    for y, (part, lo, hi, native) in zip(ys, data):
        mid = math.sqrt(lo * hi)
        if native:
            ax.barh(
                y + h / 2 + 0.02, native, height=h, color=GRAY, linewidth=0
            )
        ax.barh(y - h / 2 - 0.02, mid, height=h, color=BLUE, linewidth=0)
        ax.plot([lo, hi], [y - h / 2 - 0.02] * 2, color=INK, linewidth=1.5)
        win = native and mid > native
        ax.text(
            max(hi, native or 0) + xmax * 0.02,
            y,
            (
                ("%.2gx native" % (mid / native))
                if native
                else "no FP64 figure listed"
            ),
            va="center",
            fontsize=13,
            color=INK if win else INK_2,
            fontweight="bold" if win else None,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in data], color=INK_2, fontsize=13)
    ax.set_xlabel(
        "TFLOPS: FP64 matrix peak, or FP64-equivalent of Ozaki-2 at n=4096",
        fontsize=12,
        labelpad=10,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
    ]
    ax.legend(
        handles,
        [
            "Ozaki-2, projected (range: either anchor)",
            "native FP64 matrix peak",
        ],
        loc="lower right",
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
    )
    save(fig, "ozaki-amd.png")


if __name__ == "__main__":
    style()
    scatter()
    transfer()
    ozaki_amd()
