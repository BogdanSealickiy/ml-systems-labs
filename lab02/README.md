# Lab 2 — Neural Network from Scratch

## Overview

This lab implements a fully connected neural network from scratch using NumPy and the `load_digits` dataset from scikit-learn.

The network has two hidden layers with ReLU activation and an output layer with softmax activation.

## Dataset

- Dataset: `load_digits`
- Input features: 64
- Output classes: 10
- Input normalization: divide pixel values by 16
- Train/test split: 70% / 30%
- Random state: 42
- Stratified splitting: enabled

## Model and Training

- Hidden layer widths: 16 and 64
- Weight initialization: normal distribution with standard deviation 0.01
- Loss function: categorical cross-entropy
- Optimizer: mini-batch gradient descent
- Batch size: 32
- Training duration: 50 epochs
- Learning rates: 0.001, 0.01 and 0.1

## Results

| Hidden Width | Learning Rate | Test Accuracy |
|---:|---:|---:|
| 16 | 0.001 | 10.19% |
| 16 | 0.01 | 10.19% |
| 16 | 0.1 | 76.11% |
| 64 | 0.001 | 10.19% |
| 64 | 0.01 | 10.19% |
| 64 | 0.1 | 94.07% |

The best test accuracy was **94.07%**, achieved with a hidden width of 64 and a learning rate of 0.1.

The lower learning rates did not learn effectively within 50 epochs in these experiments.

## Gradient Check

The implementation includes a numerical gradient check that compares analytical gradients with finite-difference approximations.

## Performance

| Operation | Median Time |
|---|---:|
| Forward pass | 0.1023 ms |
| Backward pass | 0.1348 ms |

The backward pass took longer in this measurement because it calculates gradients using additional matrix multiplications.

## Files

- `lab02.py` — neural network implementation, training, gradient checking and evaluation.
- `report.md` — experiment report.
- `results.txt` — experiment results and timing measurements.
- `results/loss_curves.png` — training loss curves.
- `results/results.txt` — copy of the experiment results.

## Requirements

- Python 3.11
- NumPy
- Matplotlib
- scikit-learn

## How to Run

From the project root, activate the virtual environment and run:

```bash
python lab02/lab02.py
```

The script trains all six configurations, performs gradient checking, generates loss curves and records the results.
