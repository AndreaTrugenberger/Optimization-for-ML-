#!/bin/bash
# run_all.sh — Runs all experiments then generates all plots.
# Usage: bash run_all.sh
#
# Experiments:
#   1. MNIST   + CIFAR-10, bs=128 (baseline)
#   2. MNIST   + CIFAR-10, bs=32 128 512 1024 (batch size sweep)
#
# All results are saved to a single results/results.json
# (train.py appends, so you can run experiments separately if needed)

set -e  # stop immediately if any command fails

echo "============================================================"
echo "  OptML CNN Experiment — GPU-accelerated"
echo "============================================================"

mkdir -p results

# ── Experiment 1: Baseline (bs=128 only, both datasets) ──────────────────────
echo ""
echo ">>> [1/2] Baseline — MNIST + CIFAR-10, batch size 128..."
python3 train.py \
    --datasets mnist cifar10 \
    --batch_sizes 128 \
    --epochs 20 \
    --seeds 0 1 2 \
    --out_dir results

# ── Experiment 2: Full batch size sweep (both datasets) ──────────────────────
echo ""
echo ">>> [2/2] Batch size sweep — MNIST + CIFAR-10, bs=32 128 512 1024..."
echo "    (bs=128 already done above — train.py will append the rest)"
python3 train.py \
    --datasets mnist cifar10 \
    --batch_sizes 32 512 1024 \
    --epochs 20 \
    --seeds 0 1 2 \
    --out_dir results

# ── Generate all plots ────────────────────────────────────────────────────────
echo ""
echo ">>> Generating all figures..."
python3 plot_results.py \
    --in_dir  results \
    --out_dir results \
    --batch_sizes 32 128 512 1024

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Done! Figures saved to results/"
echo ""
echo "  results/fig1_mnist_baseline.pdf      ← Figure 1a"
echo "  results/fig1_cifar10_baseline.pdf    ← Figure 1b"
echo "  results/fig2_dataset_batchsize_comparison.pdf  ← Figure 2"
echo "  results/fig3_batchsize_sweep.pdf     ← Figure 3"
echo "============================================================"
