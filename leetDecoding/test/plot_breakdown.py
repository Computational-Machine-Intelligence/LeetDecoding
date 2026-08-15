"""Plot attention breakdown from profile_breakdown.py JSON output.

Produces:
  1. Stacked bar chart: total time broken into attn_core, attn_non_core, mlp, other
  2. Per-layer line/bar chart: attn_module, attn_core, mlp by layer index
  3. (Optional) pie chart of the breakdown

Usage:
    python -m leetDecoding.test.plot_breakdown \\
        --input outputs/profile_breakdown/retnet_FleetAttention_*.json \\
        --output outputs/profile_breakdown/retnet_FleetAttention_bd.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_stacked_breakdown(
    results: List[dict],
    labels: List[str],
    output_path: str,
    figsize: tuple = (10, 6),
) -> None:
    """Stacked bar chart: attn_core | attn_non_core | mlp | other."""
    fig, ax = plt.subplots(figsize=figsize)

    categories = ["attn_core_ms", "attn_non_core_ms", "mlp_ms", "other_ms"]
    colors = ["#2196F3", "#90CAF9", "#FF9800", "#BDBDBD"]
    legend_labels = ["Attention Core", "Attention Other", "MLP", "Other"]

    x = np.arange(len(labels))
    width = 0.55

    bottoms = np.zeros(len(labels))
    for cat, color, legend_label in zip(categories, colors, legend_labels):
        values = []
        for r in results:
            v = r["breakdown"][cat]["mean"]
            values.append(v)
        values = np.array(values)
        bars = ax.bar(x, values, width, bottom=bottoms, color=color, label=legend_label,
                      edgecolor="white", linewidth=0.5)
        bottoms += values

        # Annotate percentages on bars
        for bar, val in zip(bars, values):
            if val < 0.5:
                continue
            total = sum(r["breakdown"][c]["mean"] for c in categories)
            pct = val / total * 100
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{pct:.0f}%",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Attention Breakdown by Configuration", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Stacked breakdown saved to {output_path}")


def plot_per_layer(
    result: dict,
    output_path: str,
    figsize: tuple = (14, 7),
) -> None:
    """Per-layer line plot: attn_module, attn_core, mlp across layers."""
    per_layer = result["per_layer"]
    layer_indices = sorted(int(k) for k in per_layer.keys())

    if not layer_indices:
        print("No per-layer data to plot.")
        return

    attn_module_vals = []
    attn_core_vals = []
    attn_non_core_vals = []
    mlp_vals = []

    for lidx in layer_indices:
        ld = per_layer[str(lidx)]
        attn_mod = ld.get("attn_module", 0)
        attn_core = ld.get("attn_core", 0)
        mlp = ld.get("mlp", 0)
        attn_module_vals.append(attn_mod)
        attn_core_vals.append(attn_core)
        attn_non_core_vals.append(attn_mod - attn_core)
        mlp_vals.append(mlp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: line plot of all components
    ax1.plot(layer_indices, attn_module_vals, "o-", color="#2196F3", linewidth=2,
             markersize=5, label="Attention Module")
    ax1.plot(layer_indices, attn_core_vals, "s-", color="#1565C0", linewidth=2,
             markersize=5, label="Attention Core")
    ax1.plot(layer_indices, mlp_vals, "D-", color="#FF9800", linewidth=2,
             markersize=5, label="MLP")
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Per-Layer Timing", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: stacked bar of attn_core + attn_non_core per layer
    x = np.arange(len(layer_indices))
    width = 0.7
    ax2.bar(x, attn_core_vals, width, color="#1565C0", label="Attention Core",
            edgecolor="white", linewidth=0.3)
    ax2.bar(x, attn_non_core_vals, width, bottom=attn_core_vals, color="#90CAF9",
            label="Attention Non-Core", edgecolor="white", linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{i}" for i in layer_indices], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title("Per-Layer Attention Breakdown", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-layer plot saved to {output_path}")


def plot_pie(
    result: dict,
    output_path: str,
    figsize: tuple = (8, 8),
) -> None:
    """Pie chart of time breakdown."""
    breakdown = result["breakdown"]
    categories = ["attn_core_ms", "attn_non_core_ms", "mlp_ms", "other_ms"]
    values = [breakdown[c]["mean"] for c in categories]
    labels = ["Attention Core", "Attention Other", "MLP", "Other"]
    colors = ["#1565C0", "#90CAF9", "#FF9800", "#BDBDBD"]

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        explode=(0.03, 0.03, 0.03, 0.03),
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    total = sum(values)
    method = result["config"]["attention_method"]
    ax.set_title(
        f"Attention Breakdown\n{method}  |  Total: {total:.1f} ms",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pie chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Comparison across multiple JSON inputs
# ---------------------------------------------------------------------------

def plot_comparison(
    json_paths: List[str],
    output_dir: str,
    prefix: str = "comparison",
) -> None:
    """Load multiple JSON results and produce comparison plots."""
    results = []
    labels = []

    for jp in json_paths:
        with open(jp) as f:
            r = json.load(f)
        results.append(r)

        cfg = r["config"]
        label = f"{cfg['attention_method']}\nb{cfg['batch_size']} n{cfg['seq_len']}"
        labels.append(label)

    os.makedirs(output_dir, exist_ok=True)

    # Stacked breakdown comparison
    plot_stacked_breakdown(
        results, labels,
        os.path.join(output_dir, f"{prefix}_stacked.png"),
    )

    # Per-layer for each result
    for r, label in zip(results, labels):
        safe_label = label.replace("\n", "_").replace(" ", "")
        plot_per_layer(
            r,
            os.path.join(output_dir, f"{prefix}_perlayer_{safe_label}.png"),
        )

    # Pie for each
    for r, label in zip(results, labels):
        safe_label = label.replace("\n", "_").replace(" ", "")
        plot_pie(
            r,
            os.path.join(output_dir, f"{prefix}_pie_{safe_label}.png"),
        )

    print(f"\nAll plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot attention breakdown.")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="One or more JSON files from profile_breakdown.py")
    parser.add_argument("--output_dir", type=str, default="./outputs/profile_breakdown")
    parser.add_argument("--prefix", type=str, default="comparison")

    args = parser.parse_args()

    # Validate inputs
    for jp in args.input:
        if not os.path.exists(jp):
            raise FileNotFoundError(f"Input file not found: {jp}")

    plot_comparison(args.input, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
