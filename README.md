# Optimizer Comparison: Adam, SignSGD, Lion, AdaHessian
Empirical comparison of four coordinate-wise adaptive optimizers on MNIST and CIFAR-10, across four batch sizes.

## Reproduce all results
python run.py

## Requirements
pip install torch torchvision matplotlib numpy

Results and figures are saved to the results/ folder.
A strong GPU is recommended!