"""Render the explanation carousel from the committed benchmark records.

Run: MPLBACKEND=Agg python benchmarks/render_figures.py
Requires matplotlib and numpy; no GPU or model execution is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
CPU_PATH = ROOT / "benchmarks/results/i7-12700k-pytorch-2.11-threads1.json"
GPU_PATH = ROOT / "benchmarks/results/rtx3060-pytorch-2.11-cuda.json"
BG, INK, MUTED, GRID = "#F5F3ED", "#162D3A", "#566977", "#D8DEDE"
TEAL, BLUE, ORANGE = "#007F79", "#356BC2", "#BC572C"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": BG,
    }
)


def page(number, title, subtitle):
    fig = plt.figure(figsize=(10, 10), facecolor=BG)
    fig.text(
        0.065, 0.95, "BLOCK GNN   /   INFERENCE SYSTEMS", color=TEAL, fontsize=12, weight="bold"
    )
    fig.text(0.065, 0.895, title, fontsize=29, weight="bold", va="top")
    fig.text(0.065, 0.81, subtitle, fontsize=14, color=MUTED, va="top", linespacing=1.5)
    fig.text(
        0.065,
        0.035,
        "Hesam Damghanian   •   github.com/Sk7w4tch3r/block-gnn",
        fontsize=10,
        color=MUTED,
    )
    fig.text(0.935, 0.035, f"{number:02d} / 05", ha="right", fontsize=11, color=TEAL)
    return fig


def box(ax, x, y, width, height, text, color=TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
        )
    )
    ax.text(
        x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=13, linespacing=1.5
    )


def arrow(ax, a, b):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=MUTED, linewidth=1.5)
    )


def operator_figure():
    fig = page(
        1,
        "Train once. Change the execution.",
        "A frozen convolution is a sparse linear map at a fixed input shape.\n"
        "The learned weights are reused; no retraining is required.",
    )
    ax = fig.add_axes([0.065, 0.57, 0.87, 0.18])
    ax.axis("off")
    box(ax, 0.015, 0.25, 0.25, 0.5, "Trained Conv2d\nweights + geometry")
    box(ax, 0.375, 0.25, 0.25, 0.5, "Offline compilation\nCSR indices + values")
    box(ax, 0.735, 0.25, 0.25, 0.5, "Repeated inference\nsparse matmul + bias")
    arrow(ax, (0.28, 0.5), (0.36, 0.5))
    arrow(ax, (0.64, 0.5), (0.72, 0.5))
    palette = [
        "#F0EDE6",
        "#127F80",
        "#369E91",
        "#80C3A6",
        "#3154A2",
        "#4781BC",
        "#81B6D1",
        "#AB432C",
        "#D17B3D",
        "#E4B866",
    ]
    cmap = ListedColormap(palette)
    op = np.zeros((16, 16), dtype=int)
    for y in range(4):
        for x in range(4):
            for ky in range(3):
                for kx in range(3):
                    iy, ix = y + ky - 1, x + kx - 1
                    if 0 <= iy < 4 and 0 <= ix < 4:
                        op[y * 4 + x, iy * 4 + ix] = 1 + 3 * ky + kx
    left = fig.add_axes([0.085, 0.315, 0.23, 0.23])
    left.imshow(np.arange(1, 10).reshape(3, 3), cmap=cmap, vmin=0, vmax=9)
    for y in range(3):
        for x in range(3):
            left.text(
                x, y, f"w{3 * y + x + 1}", ha="center", va="center", color="white", weight="bold"
            )
    left.set_xticks([])
    left.set_yticks([])
    left.set_title("One 3×3 kernel", fontsize=14, pad=13)
    right = fig.add_axes([0.49, 0.235, 0.40, 0.34])
    right.imshow(op, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    right.set_title("16×16 sparse operator M", fontsize=14, pad=13)
    right.set_xlabel("Input pixel index")
    right.set_ylabel("Output pixel index")
    right.set_xticks([0, 5, 10, 15])
    right.set_yticks([0, 5, 10, 15])
    fig.text(
        0.065,
        0.185,
        "Example: 4×4 input, stride 1, zero padding 1, one channel.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(0.065, 0.142, "Same color = the same trained weight, repeated in M.", fontsize=14)
    fig.text(0.065, 0.087, r"$Y_{flat} = M X_{flat} + b$", fontsize=23, color=TEAL)
    fig.text(0.48, 0.095, "Each batch sample is a column of X.", fontsize=12, color=MUTED)
    return fig


def cuda_figure():
    fig = page(
        2,
        "Two ways to execute the stencil.",
        "The CUDA kernels compute the same weighted neighborhoods directly.\n"
        "They keep the kernel compact instead of expanding it into CSR.",
    )
    ax = fig.add_axes([0.065, 0.42, 0.39, 0.28])
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(6.5, -0.5)
    ax.axis("off")
    for y in range(7):
        for x in range(7):
            chosen = 2 <= x <= 4 and 2 <= y <= 4
            ax.add_patch(
                Rectangle(
                    (x - 0.46, y - 0.46),
                    0.92,
                    0.92,
                    facecolor="#AEDAD2" if chosen else "#E1E6E3",
                    edgecolor=BG,
                )
            )
    ax.add_patch(Rectangle((2.5, 2.5), 1, 1, facecolor=TEAL))
    ax.text(3, 3, "y", color="white", weight="bold", ha="center", va="center")
    ax.set_title("DIRECT CUDA", color=TEAL, weight="bold", fontsize=15, pad=14)
    ax2 = fig.add_axes([0.55, 0.42, 0.39, 0.28])
    ax2.set_aspect("equal")
    ax2.set_xlim(0, 18)
    ax2.set_ylim(18, 0)
    ax2.axis("off")
    for y in range(18):
        for x in range(18):
            halo = x in (0, 17) or y in (0, 17)
            ax2.add_patch(Rectangle((x, y), 0.94, 0.94, facecolor="#E4B866" if halo else "#A7C9ED"))
    ax2.set_title("TILED CUDA", color=BLUE, weight="bold", fontsize=15, pad=14)
    fig.text(0.065, 0.365, "One thread → one output", fontsize=17, weight="bold")
    fig.text(
        0.065,
        0.325,
        "Read neighbors and weights.\nAccumulate channels and offsets.\nWrite one output value.",
        fontsize=14,
        linespacing=1.65,
        va="top",
    )
    fig.text(0.55, 0.365, "One block → a 16×16 tile", fontsize=17, weight="bold")
    fig.text(
        0.55,
        0.325,
        "Load an 18×18 input patch.\nSynchronize the block.\nReuse neighbors in shared memory.",
        fontsize=14,
        linespacing=1.65,
        va="top",
    )
    fig.text(0.55, 0.205, "Gold cells = halo for a 3×3 kernel", fontsize=11, color=MUTED)
    fig.text(
        0.065, 0.135, "Current scope: stride 1 • dilation 1 • no groups", fontsize=14, color=TEAL
    )
    fig.text(
        0.065,
        0.097,
        "Tiled specialization: one input and one output channel.\n"
        "Topology is implicit in the CUDA indexing; it is not a generic graph kernel.",
        fontsize=11,
        color=MUTED,
        linespacing=1.5,
    )
    return fig


def speed_figure(records, gpu):
    fig = page(
        4 if gpu else 3,
        "CUDA: small shapes benefit." if gpu else "CPU: batching changes the result.",
        (
            "RTX 3060 • float32 • cuDNN autotuning enabled\n"
            "Direct, tiled and CSR backends compared with the same Conv2d."
            if gpu
            else "Intel i7-12700K • float32 • one CPU thread\n"
            "CSR inference compared with the complete eager Conv2d call."
        ),
    )
    ax = fig.add_axes([0.315, 0.27, 0.58, 0.42], facecolor=BG)
    ax.set_xscale("log")
    ax.set_xlim(0.045, 4.5)
    ax.axvspan(0.045, 1, color="#F1E5DC", zorder=0)
    ax.axvline(1, color=INK, linewidth=1.3, linestyle="--")
    rows = np.arange(len(records))
    series = [("csr", "CSR", ORANGE, "s", 0)]
    if gpu:
        series = [
            ("csr", "CSR", ORANGE, "s", -0.22),
            ("direct", "Direct CUDA", TEAL, "o", 0),
            ("tiled", "Tiled CUDA", BLUE, "D", 0.22),
        ]
    for key, label, color, marker, dy in series:
        vals = [r["dense"]["median_us"] / r[key]["median_us"] for r in records if key in r]
        ys = [i + dy for i, r in enumerate(records) if key in r]
        ax.scatter(vals, ys, label=label, color=color, marker=marker, s=55, zorder=3)
        if not gpu:
            for val, y in zip(vals, ys, strict=True):
                ax.text(val * 1.12, y, f"{val:.2f}×", fontsize=12, va="center", color=INK)
    labels = [
        f"B{r['batch']}  |  {r['in_channels']}→{r['out_channels']}  |  {r['size']}×{r['size']}"
        for r in records
    ]
    ax.set_yticks(rows, labels, fontsize=12)
    ax.set_ylim(8.6, -0.6)
    ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FixedFormatter(["0.05×", "0.1×", "0.25×", "0.5×", "1×", "2×", "4×"])
    )
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(axis="both", length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Speedup = baseline median / backend median", fontsize=12, labelpad=12)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=3 if gpu else 1,
        fontsize=11,
    )
    fig.text(0.32, 0.16, "← Slower than baseline", fontsize=12, color=ORANGE)
    fig.text(0.70, 0.16, "Faster →", fontsize=12, color=TEAL)
    fig.text(
        0.065,
        0.105,
        "All 9 measured shapes shown. B = batch size; channels = input → output.",
        fontsize=11,
        color=MUTED,
    )
    note = (
        "CUDA events around eager launches; host launch gaps may contribute.\n"
        "Medians of 9 timing groups; hardware-specific operator timings."
        if gpu
        else "Warm-up + 9 timing groups per backend; compilation excluded.\n"
        "Medians from saved benchmark JSON; ratios are not confidence intervals."
    )
    fig.text(0.065, 0.065, note, fontsize=10, color=MUTED, linespacing=1.5)
    return fig


def memory_figure(records):
    fig = page(
        5,
        "The cost of explicit sparsity.",
        "CSR stores repeated weights and indices at each valid output location.\n"
        "Operator storage grows even when the trained kernel stays small.",
    )
    selected = [
        next(r for r in records if r["name"] == key)
        for key in ["tiny-batch", "filter-28", "mnist-batch", "rgb-stem", "feature-map"]
    ]
    ax = fig.add_axes([0.12, 0.36, 0.80, 0.32], facecolor=BG)
    x = np.arange(len(selected))
    width = 0.32
    weights = [r["weight_bytes"] / 1024 for r in selected]
    sparse = [r["csr_bytes"] / 1024 for r in selected]
    ax.bar(x - width / 2, weights, width, color=TEAL, label="Weights + bias")
    ax.bar(x + width / 2, sparse, width, color=ORANGE, label="CSR + bias")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 45000)
    ax.set_ylabel("Persistent tensor storage (KiB, log scale)", fontsize=12)
    ax.set_xticks(
        x,
        [f"{r['in_channels']}→{r['out_channels']}\n{r['size']}×{r['size']}" for r in selected],
        fontsize=12,
    )
    for i, r in enumerate(selected):
        ax.text(
            i + width / 2,
            sparse[i] * 1.65,
            f"{r['csr_bytes'] / r['weight_bytes']:,.0f}×",
            ha="center",
            color=ORANGE,
            fontsize=12,
            weight="bold",
        )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.065, 0.25, "1→32 on 28×28: 1.25 KiB → 1.74 MiB", fontsize=21, weight="bold")
    fig.text(
        0.065,
        0.195,
        "Direct CUDA keeps the original weight tensor.\n"
        "Tiled CUDA adds a temporary shared-memory patch per block.",
        fontsize=15,
        linespacing=1.6,
    )
    fig.text(
        0.065,
        0.095,
        "Storage is calculated from tensor values and indices, not peak process memory.\n"
        "Activations, allocator caches and backend workspaces are excluded.",
        fontsize=11,
        color=MUTED,
        linespacing=1.5,
    )
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(CPU_PATH.read_text())["results"]
    gpu = json.loads(GPU_PATH.read_text())["results"]
    figures = [
        ("01_frozen_operator", operator_figure()),
        ("02_cuda_execution", cuda_figure()),
        ("03_cpu_crossover", speed_figure(cpu, False)),
        ("04_cuda_crossover", speed_figure(gpu, True)),
        ("05_memory_tradeoff", memory_figure(cpu)),
    ]
    with PdfPages(
        OUT / "block_gnn_carousel.pdf",
        metadata={
            "Title": "Block GNN: from trained convolution to sparse and CUDA inference",
            "Author": "Hesam Damghanian",
        },
    ) as pdf:
        for name, fig in figures:
            fig.savefig(OUT / f"{name}.png", dpi=160)
            fig.savefig(OUT / f"{name}.svg")
            pdf.savefig(fig)
            plt.close(fig)
    manifest = {
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (CPU_PATH, GPU_PATH)
        },
        "generator": "benchmarks/render_figures.py",
        "matplotlib": matplotlib.__version__,
        "pages": [name for name, _ in figures],
        "timing_scope": "Existing eager operator benchmark records; no new measurements.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(figures)} figures (PNG + SVG) and one carousel PDF in {OUT}")


if __name__ == "__main__":
    main()
