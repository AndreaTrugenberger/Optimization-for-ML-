"""
plot_results.py — Generates exactly the 3 figure types requested.

FIGURE 1 — Baseline (bs=128): 3-panel row per dataset
  For MNIST and CIFAR-10 separately:
    Panel 1: Test Accuracy vs Epoch   (4 optimizer curves)
    Panel 2: Training Loss vs Epoch   (4 optimizer curves)
    Panel 3: Test Loss vs Epoch       (4 optimizer curves)

FIGURE 2 — Dataset comparison with batch size curves:
  8 subplots (2 rows × 4 cols), one per optimizer × dataset.
  Each subplot has 4 curves — one per batch size (32, 128, 512, 1024).
  (Extends the original dataset comparison: instead of 1 curve per subplot,
   now 4 curves = one per batch size.)

FIGURE 3 — Final accuracy vs batch size:
  Side-by-side: MNIST (left) and CIFAR-10 (right).
  Each panel has 4 optimizer curves across batch sizes.

Usage:
  python3 plot_results.py
  python3 plot_results.py --in_dir results --out_dir results
"""

import json, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

OPT_COLORS = {
    "adam":       "#1f77b4",
    "signsgd":    "#d62728",
    "lion":       "#2ca02c",
    "adahessian": "#ff7f0e",
}
OPT_LABELS = {
    "adam":       "Adam",
    "signsgd":    "SignSGD",
    "lion":       "Lion",
    "adahessian": "AdaHessian",
}

# Colors for the 4 batch sizes in Figure 2
BS_COLORS = {
    32:   "#1f77b4",
    128:  "#2ca02c",
    512:  "#d62728",
    1024: "#9467bd",
}
BS_LABELS = {32: "bs=32", 128: "bs=128", 512: "bs=512", 1024: "bs=1024"}


# ── Data helpers ───────────────────────────────────────────────────────────────

def load(in_dir):
    path = os.path.join(in_dir, "results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run train.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} runs from {path}")
    return data


def filter_runs(results, dataset=None, batch_size=None, opt=None):
    out = results
    if dataset    is not None: out = [r for r in out if r.get("dataset",    "mnist") == dataset]
    if batch_size is not None: out = [r for r in out if r.get("batch_size", 128)     == batch_size]
    if opt        is not None: out = [r for r in out if r["opt"].lower()              == opt.lower()]
    return out


def mean_std(runs, metric):
    """Returns (mean, std) arrays of shape (n_epochs,) across seeds."""
    arrays = np.array([r[metric] for r in runs])
    return arrays.mean(axis=0), arrays.std(axis=0)


# ── Shared panel helper ────────────────────────────────────────────────────────

def draw_curve(ax, epochs, mean, std, color, label, lw=2):
    ax.plot(epochs, mean, color=color, label=label, linewidth=lw)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Baseline: 3 panels for a single dataset at bs=128
# ══════════════════════════════════════════════════════════════════════════════

def make_figure1_baseline(results, out_dir, dataset, batch_size=128):
    """
    3-panel row: Test Accuracy | Training Loss | Test Loss
    One curve per optimizer. Saved as fig1_{dataset}_baseline.pdf/png
    """
    ds_label = dataset.upper()
    runs_ds  = filter_runs(results, dataset=dataset, batch_size=batch_size)

    if not runs_ds:
        print(f"⚠  No runs found for {ds_label} bs={batch_size} — skipping Figure 1.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Optimizer Comparison — {ds_label}, batch size {batch_size} "
        f"(mean ± std, 3 seeds)",
        fontsize=12, fontweight="bold"
    )

    metrics = [
        ("test_acc",   "Test Accuracy",      "Test Accuracy vs Epoch",   True),
        ("train_loss", "Cross-Entropy Loss",  "Training Loss vs Epoch",   False),
        ("test_loss",  "Cross-Entropy Loss",  "Test Loss vs Epoch",       False),
    ]

    for ax, (metric, ylabel, title, higher) in zip(axes, metrics):
        for opt in OPT_COLORS:
            runs = filter_runs(runs_ds, opt=opt)
            if not runs:
                continue
            n_ep   = len(runs[0][metric])
            epochs = np.arange(1, n_ep + 1)
            mean, std = mean_std(runs, metric)
            draw_curve(ax, epochs, mean, std,
                       OPT_COLORS[opt], OPT_LABELS[opt])

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right" if higher else "upper right")

    plt.tight_layout()
    name = f"fig1_{dataset}_baseline"
    _save(fig, out_dir, name)
    print(f"  → Figure 1 ({ds_label}) saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Dataset comparison with batch size curves
# 2 rows (MNIST top, CIFAR-10 bottom) × 4 cols (one per optimizer)
# Each subplot: 4 curves, one per batch size
# ══════════════════════════════════════════════════════════════════════════════

def make_figure2_dataset_batchsize(results, out_dir, batch_sizes=(32, 128, 512, 1024)):
    """
    8 subplots (2 rows × 4 cols).
    Row 0 = MNIST, Row 1 = CIFAR-10.
    Col  = optimizer (Adam, SignSGD, Lion, AdaHessian).
    Each subplot has len(batch_sizes) curves, one per batch size.
    """
    opts     = list(OPT_COLORS.keys())
    datasets = ["mnist", "cifar10"]
    ds_label = {"mnist": "MNIST", "cifar10": "CIFAR-10"}

    fig, axes = plt.subplots(2, len(opts), figsize=(5 * len(opts), 9))
    fig.suptitle(
        "Dataset Comparison: MNIST vs CIFAR-10 — Test Accuracy per Optimizer\n"
        "(mean ± std, 3 seeds; one curve per batch size)",
        fontsize=12, fontweight="bold"
    )

    for row, ds in enumerate(datasets):
        for col, opt in enumerate(opts):
            ax = axes[row, col]
            ax.set_title(f"{OPT_LABELS[opt]} — {ds_label[ds]}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Test Accuracy")

            any_data = False
            for bs in batch_sizes:
                runs = filter_runs(results, dataset=ds, batch_size=bs, opt=opt)
                if not runs:
                    continue
                any_data = True
                n_ep   = len(runs[0]["test_acc"])
                epochs = np.arange(1, n_ep + 1)
                mean, std = mean_std(runs, "test_acc")
                draw_curve(ax, epochs, mean, std,
                           BS_COLORS.get(bs, "gray"),
                           BS_LABELS.get(bs, f"bs={bs}"))

            if any_data:
                ax.legend(loc="lower right")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")

    plt.tight_layout()
    _save(fig, out_dir, "fig2_dataset_batchsize_comparison")
    print("  → Figure 2 (dataset × batch size comparison) saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Final accuracy vs batch size: MNIST (left) | CIFAR-10 (right)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure3_batchsize_sweep(results, out_dir, batch_sizes=(32, 128, 512, 1024)):
    """
    2-panel figure side by side.
    Left = MNIST, Right = CIFAR-10.
    Each panel: 4 optimizer curves across batch sizes (final test accuracy).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Effect of Batch Size on Final Test Accuracy — CNN (mean ± std, 3 seeds)",
        fontsize=12, fontweight="bold"
    )

    for ax, ds in zip(axes, ["mnist", "cifar10"]):
        ds_label = ds.upper()
        for opt in OPT_COLORS:
            means, stds = [], []
            for bs in batch_sizes:
                runs = filter_runs(results, dataset=ds, batch_size=bs, opt=opt)
                if not runs:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                final_accs = np.array([r["test_acc"][-1] for r in runs])
                means.append(final_accs.mean())
                stds.append(final_accs.std())

            ax.errorbar(
                batch_sizes, means, yerr=stds,
                color=OPT_COLORS[opt], label=OPT_LABELS[opt],
                marker="o", linewidth=2, capsize=4,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(list(batch_sizes))
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Final Test Accuracy")
        ax.set_title(f"Final Test Accuracy vs Batch Size — {ds_label}")
        ax.legend()

    plt.tight_layout()
    _save(fig, out_dir, "fig3_batchsize_sweep")
    print("  → Figure 3 (batch size sweep) saved.")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",      default="./results")
    parser.add_argument("--out_dir",     default="./results")
    parser.add_argument("--batch_sizes", nargs="+", type=int,
                        default=[32, 128, 512, 1024])
    parser.add_argument("--baseline_bs", type=int, default=128,
                        help="Batch size used for Figure 1 baseline plots")
    args = parser.parse_args()

    results     = load(args.in_dir)
    batch_sizes = tuple(args.batch_sizes)

    print("\nGenerating Figure 1a — MNIST baseline (bs=128)...")
    make_figure1_baseline(results, args.out_dir, "mnist",   args.baseline_bs)

    print("Generating Figure 1b — CIFAR-10 baseline (bs=128)...")
    make_figure1_baseline(results, args.out_dir, "cifar10", args.baseline_bs)

    print("Generating Figure 2 — Dataset × batch size comparison...")
    make_figure2_dataset_batchsize(results, args.out_dir, batch_sizes)

    print("Generating Figure 3 — Batch size sweep (MNIST + CIFAR-10)...")
    make_figure3_batchsize_sweep(results, args.out_dir, batch_sizes)

    print(f"\n✓ All figures saved to {args.out_dir}/")
    print("  Use the .pdf versions in your LaTeX report.")


if __name__ == "__main__":
    main()
