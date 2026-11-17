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
  ozaki-amd.png  ozaki-kernels.csv, specs.csv, published.csv: Ozaki-2 FP64
                 projected for AMD parts from GH200, between AMD's published
                 GEMM efficiency and NVIDIA's, against the native FP64 peak
  channel.png    the same projection over the port's GEMM efficiency
  isa.png        isa.csv: slots per work-item of the named mixes across AMD,
                 NVIDIA, and Intel targets, relative to the best target
  isa-score.png  the same as one score per target: the geometric mean over
                 the mixes

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


def published():
    """{part: {"bw": [...], "eff": [...]}} from published.csv."""
    out = defaultdict(lambda: {"bw": [], "eff": []})
    for r in rows("published.csv"):
        if r["bandwidth_tbs"]:
            out[r["part"]]["bw"].append(float(r["bandwidth_tbs"]))
        if r["matrix_efficiency"]:
            out[r["part"]]["eff"].append(float(r["matrix_efficiency"]))
    return out


def model():
    """The projection from GH200 (the anchor with a published bandwidth):
    specs, published figures, the bandwidth fractions and GEMM efficiencies
    AMD parts have published, and tflops(part, eff, bw), Ozaki-2's
    FP64-equivalent TFLOPS at n=4096 for a GEMM efficiency relative to
    Hopper's and a bandwidth in TB/s."""
    s, pub = specs(), published()
    configs = defaultdict(dict)
    for r in rows("ozaki-kernels.csv"):
        if (
            r["size"] == "4096"
            and r["scheme"] == "2"
            and "x" not in r["config"]
        ):
            configs[(r["config"], r["option"])][r["kernel"]] = float(
                r["gh200_ms"]
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
    o = sum(kernels.values()) - g
    a, nv = s["GH200"], max(pub["Hopper"]["eff"])
    bw_a = pub["GH200"]["bw"][0]
    amd = [p for p in pub if p in s and p not in ("H100 PCIe", "GH200")]
    effs = [e for p in amd for e in pub[p]["eff"]]
    fracs = [
        b / float(s[p]["bandwidth_tbs"]) for p in amd for b in pub[p]["bw"]
    ]

    def tflops(r, eff, bw):
        t = (
            g * float(a["int8_tops"]) / float(r["int8_tops"]) / eff
            + o * bw_a / bw
        )
        return 2.0 * 4096**3 / (t * 1e9)

    return s, pub, a, bw_a, nv, effs, fracs, tflops


def projection():
    """[(part, low, high, native, own)]: Ozaki-2 FP64-equivalent TFLOPS at n=4096
    from GH200. Low: at the part's published GEMM efficiency relative to
    Hopper's, and its published bandwidth. High: as efficient as on NVIDIA.
    A part without published figures (own is False) takes the range of those
    that have them."""
    s, pub, a, bw_a, nv, effs, fracs, tflops = model()
    out = []
    for part, r in s.items():
        if part in ("H100 PCIe", "GH200"):
            continue
        own = part in pub
        eff = min(pub[part]["eff"] or effs) / nv
        datasheet = float(r["bandwidth_tbs"])
        bw = pub[part]["bw"] or [datasheet * f for f in fracs]
        out.append(
            (
                part,
                tflops(r, eff, min(bw)),
                tflops(
                    r,
                    1.0,
                    (
                        pub[part]["bw"][0]
                        if pub[part]["bw"]
                        else datasheet * bw_a / float(a["bandwidth_tbs"])
                    ),
                ),
                float(r["fp64_matrix"]) if r["fp64_matrix"] else None,
                own,
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
    xmax = max(max(r[2], r[3] or 0) for r in data) * 1.45
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
    for y, (part, lo, hi, native, own) in zip(ys, data):
        yb = y - h / 2 - 0.02
        if native:
            ax.barh(
                y + h / 2 + 0.02, native, height=h, color=GRAY, linewidth=0
            )
        ax.barh(yb, lo, height=h, color=BLUE, linewidth=0)
        ax.barh(
            yb, hi - lo, left=lo, height=h, color=BLUE, alpha=0.4, linewidth=0
        )
        win = native and hi > native
        ax.text(
            max(hi, native or 0) + xmax * 0.02,
            y,
            (
                ("%.2f-%.2fx native" % (lo / native, hi / native))
                if native
                else "no FP64 figure listed"
            ),
            va="center",
            fontsize=13,
            color=INK if win else INK_2,
            fontweight="bold" if win else None,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [r[0] + ("" if r[4] else " *") for r in data], color=INK_2, fontsize=13
    )
    ax.set_xlabel(
        "TFLOPS: FP64 matrix peak, or FP64-equivalent of Ozaki-2 at n=4096",
        fontsize=12,
        labelpad=10,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.4),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
    ]
    ax.legend(
        handles,
        [
            "Ozaki-2 at AMD's published GEMM efficiency",
            "... up to NVIDIA's",
            "native FP64 matrix peak",
        ],
        loc="lower right",
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
    )
    save(fig, "ozaki-amd.png")


def channel():
    """Ozaki-2 against the native FP64 matrix peak over the port's GEMM
    efficiency relative to NVIDIA's, a band over the bandwidth (published, or
    the datasheet at the fractions published for other parts), the shaded
    column from the lowest efficiency AMD published to NVIDIA's."""
    s, pub, a, bw_a, nv, effs, fracs, tflops = model()
    x = numpy.geomspace(0.25, 2.0, 120)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.05, 4)
    ax.axvspan(min(effs) / nv, 1.0, color=GRID, linewidth=0, zorder=0)
    ax.axhline(1, color=MUTED, linewidth=1.2, zorder=1)
    ax.text(
        0.26,
        1.07,
        "native FP64 matrix peak",
        color=INK_2,
        fontsize=12,
        va="bottom",
    )
    ax.text(
        (min(effs) / nv) ** 0.5,
        3.6,
        "AMD's published\nto NVIDIA's",
        color=INK_2,
        fontsize=12,
        ha="center",
        va="top",
    )
    for part, color, label in (
        ("MI355X", BLUE, "MI355X"),
        ("MI325X", ORANGE, "MI325X"),
        ("MI250X (one GCD)", GRAY, "MI250X (one GCD)"),
    ):
        r = s[part]
        native = float(r["fp64_matrix"])
        datasheet = float(r["bandwidth_tbs"])
        bws = pub[part]["bw"] or [datasheet * f for f in fracs]
        lo = [tflops(r, e, min(bws)) / native for e in x]
        hi = [tflops(r, e, max(bws)) / native for e in x]
        ax.fill_between(
            x, lo, hi, color=color, alpha=0.35, linewidth=0, zorder=2
        )
        ax.plot(
            x,
            [(p * q) ** 0.5 for p, q in zip(lo, hi)],
            color=color,
            linewidth=2,
            zorder=3,
        )
        ax.text(
            x[-1] * 1.03,
            (lo[-1] * hi[-1]) ** 0.5,
            label,
            color=INK_2,
            fontsize=13,
            va="center",
        )
        # the curve at the shaded column's edges, inside the column
        for e, dy, ha, va in (
            (min(effs) / nv, 0.82, "left", "top"),
            (1.0, 1.18, "right", "bottom"),
        ):
            v = (
                tflops(r, e, min(bws)) * tflops(r, e, max(bws))
            ) ** 0.5 / native
            ax.plot(
                e,
                v,
                "o",
                color=color,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=4,
            )
            ax.text(
                e * (1.02 if "left" == ha else 0.98),
                v * dy,
                "%.2fx" % v,
                color=INK_2,
                fontsize=12,
                ha=ha,
                va=va,
                zorder=4,
            )
    ax.set_xticks([0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.25", "0.5", "1", "2"])
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    frame(ax)
    ax.set_xlabel(
        "GEMM efficiency of the port, relative to NVIDIA's",
        fontsize=12,
        labelpad=10,
    )
    ax.set_ylabel("Ozaki-2 / native FP64 matrix peak", fontsize=12)
    save(fig, "channel.png")


def isa():
    """Slots of the best target over each target's own, per mix, for all
    instructions and for the work alone (without the instructions that only
    schedule). Vendors by their customary hues and by shape."""
    color = {"AMD": "#e66767", "NVIDIA": "#0ca30c", "Intel": BLUE}
    marker = {"AMD": "o", "NVIDIA": "s", "Intel": "^"}
    size = {"o": 1.0, "s": 0.9, "^": 1.25}
    kinds = ("vector", "scalar", "memory", "local", "control", "other")
    table = defaultdict(list)
    for r in rows("isa.csv"):
        b = {k: float(r[k]) for k in kinds}
        table[r["mix"]].append((r["vendor"], r["target"], b))
    mixes = (
        ("stream", "stream"),
        ("gemm", "gemm"),
        ("gather", "gather"),
        ("atomics", "atomics"),
        ("atomics_mask", "atomics, mask"),
    )
    fig, axes = plt.subplots(1, len(mixes), figsize=(23, 9), sharey=True)
    for ax, (mix, title) in zip(axes, mixes):
        data = table[mix]
        best = min(sum(b.values()) for _, _, b in data)
        bestw = min(sum(b.values()) - b["other"] for _, _, b in data)
        y, ticks, labels, prev = 0, [], [], None
        for vendor, name, b in data:
            if prev is not None and vendor != prev:
                y += 0.8
            prev = vendor
            p = best / sum(b.values())
            q = bestw / (sum(b.values()) - b["other"])
            m = marker[vendor]
            ax.plot((p, q), (y, y), color=GRID, linewidth=2, zorder=1)
            ax.plot(
                p,
                y,
                m,
                color=color[vendor],
                markersize=10 * size[m],
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=3,
            )
            ax.plot(
                q,
                y,
                m,
                color=color[vendor],
                markersize=8 * size[m],
                markerfacecolor=SURFACE,
                markeredgewidth=2,
                zorder=2,
            )
            ticks.append(y)
            labels.append(name)
            y += 1
        ax.axvline(1, color=GRID, linewidth=1, zorder=0)
        ax.set_title(title, fontsize=13, color=INK, loc="left")
        ax.set_xlim(0.25, 1.05)
        ax.set_ylim(y - 0.4, -0.6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=11)
        frame(ax)
        ax.set_xlabel("slots of the best target / own", fontsize=12)
    handles = [
        plt.Line2D(
            [],
            [],
            marker=marker[v],
            linestyle="",
            color=color[v],
            markersize=10 * size[marker[v]],
            markeredgecolor=SURFACE,
            label=v,
        )
        for v in color
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK_2,
            markersize=9,
            label="all instructions",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK_2,
            markerfacecolor=SURFACE,
            markersize=8,
            markeredgewidth=2,
            label="work alone",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
    )
    fig.subplots_adjust(bottom=0.16)
    save(fig, "isa.png")


def isa_score():
    """One score per target: the geometric mean over the mixes of the slots
    of the best target over the target's own, for all instructions and for
    the work alone, and below it the range of the per-mix ratios (all
    instructions), which the mean does not show."""
    color = {"AMD": "#e66767", "NVIDIA": "#0ca30c", "Intel": BLUE}
    marker = {"AMD": "o", "NVIDIA": "s", "Intel": "^"}
    size = {"o": 1.0, "s": 0.9, "^": 1.25}
    kinds = ("vector", "scalar", "memory", "local", "control", "other")
    table = defaultdict(list)
    for r in rows("isa.csv"):
        b = {k: float(r[k]) for k in kinds}
        table[r["mix"]].append((r["vendor"], r["target"], b))
    ratios = defaultdict(lambda: ([], []))
    order = []
    for mix, data in table.items():
        best = min(sum(b.values()) for _, _, b in data)
        bestw = min(sum(b.values()) - b["other"] for _, _, b in data)
        for vendor, name, b in data:
            if (vendor, name) not in order:
                order.append((vendor, name))
            ratios[(vendor, name)][0].append(best / sum(b.values()))
            ratios[(vendor, name)][1].append(
                bestw / (sum(b.values()) - b["other"])
            )
    fig, ax = plt.subplots(figsize=(11, 8))
    y, ticks, labels, prev = 0, [], [], None
    for vendor, name in order:
        if prev is not None and vendor != prev:
            y += 0.8
        prev = vendor
        a, w = ratios[(vendor, name)]
        p = math.exp(statistics.fmean(math.log(v) for v in a))
        q = math.exp(statistics.fmean(math.log(v) for v in w))
        m = marker[vendor]
        ax.plot((p, q), (y, y), color=GRID, linewidth=2, zorder=1)
        ax.plot(
            (min(a), max(a)),
            (y + 0.3, y + 0.3),
            color=color[vendor],
            alpha=0.55,
            linewidth=1.5,
            marker="|",
            markersize=6,
            zorder=1,
        )
        ax.plot(
            p,
            y,
            m,
            color=color[vendor],
            markersize=10 * size[m],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.plot(
            q,
            y,
            m,
            color=color[vendor],
            markersize=8 * size[m],
            markerfacecolor=SURFACE,
            markeredgewidth=2,
            zorder=2,
        )
        ticks.append(y)
        labels.append(name)
        y += 1
    ax.axvline(1, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(y - 0.2, -0.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=11)
    frame(ax)
    ax.set_xlabel(
        "slots of the best target / own, geometric mean over the mixes",
        fontsize=12,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker=marker[v],
            linestyle="",
            color=color[v],
            markersize=10 * size[marker[v]],
            markeredgecolor=SURFACE,
            label=v,
        )
        for v in color
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK_2,
            markersize=9,
            label="all instructions",
        ),
        plt.Line2D(
            [],
            [],
            marker="|",
            color=INK_2,
            alpha=0.55,
            linewidth=1.5,
            markersize=6,
            label="range over the mixes",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK_2,
            markerfacecolor=SURFACE,
            markersize=8,
            markeredgewidth=2,
            label="work alone",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        labelcolor=INK_2,
        fontsize=12,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.14)
    save(fig, "isa-score.png")


if __name__ == "__main__":
    style()
    scatter()
    transfer()
    ozaki_amd()
    channel()
    isa()
    isa_score()
