# Lab 3 — Neural Network Comparison

## Goal
The goal was to compare three models on the Fashion-MNIST dataset: MLP, CNN, and Attention.

## Setup
- Dataset: Fashion-MNIST
- Training: 5 epochs
- Optimizer: Adam
- Learning rate: 0.001
- Device: CPU

## Results

| Model | Parameters | Test Accuracy |
|---|---:|---:|
| MLP | 235,146 | 87.14% |
| CNN | 20,490 | 90.05% |
| Attention | 70,794 | 84.01% |

## Computational Cost

| Model | MACs | Approx. FLOPs |
|---|---:|---:|
| MLP | 234,752 | 469,504 |
| CNN | 1,031,744 | 2,063,488 |
| Attention | 591,488 | 1,182,976 |

## Inference Latency

| Model | Batch 1 | Batch 128 |
|---|---:|---:|
| MLP | 0.088 ms | 1.080 ms |
| CNN | 0.382 ms | 96.250 ms |
| Attention | 7.374 ms | 27.366 ms |

## Conclusion

CNN achieved the highest test accuracy (90.05%) and had the fewest parameters. MLP was the fastest for the measured CPU batches. Attention achieved the lowest accuracy (84.01%) and had higher latency for a single image.

The results show that accuracy, model size, and inference speed should all be considered when choosing a model.
