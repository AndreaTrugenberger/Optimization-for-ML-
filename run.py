"""
Produces all figures for the report.
First runs train.py to generate results, then plot_results.py to generate 3 figures.

This takes a while to run, a strong GPU is recommended!
"""

import subprocess
import sys

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result

if __name__ == "__main__":
    python = sys.executable

    # Train all combinations
    run([python, "train.py",
         "--optimizers", "adam", "signsgd", "lion", "adahessian",
         "--datasets", "mnist", "cifar10",
         "--batch_sizes", "32", "128", "512", "1024",
         "--epochs", "20",
         "--seeds", "0", "1", "2",
    ])

    # Generate figures
    run([python, "plot_results.py",
         "--in_dir", "results",
         "--out_dir", "results",
         "--batch_sizes", "32", "128", "512", "1024",
    ])

    print("\nDone. Figures saved to results/")