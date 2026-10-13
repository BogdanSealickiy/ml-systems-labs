lab 2 — neural network from scratch

The network uses the load_digits dataset with 64 input features and 10 output classes. The inputs are divided by 16, and the data is split into 70% training and 30% testing with stratification and random_state=42.

The network has two hidden layers with ReLU activation and an output layer with softmax. I tested hidden widths of 16 and 64, learning rates of 0.001, 0.01 and 0.1, and trained each configuration for 50 epochs using mini-batches of 32.

Test accuracy:
- Width 16, learning rate 0.001: 10.19%
- Width 16, learning rate 0.01: 10.19%
- Width 16, learning rate 0.1: 76.11%
- Width 64, learning rate 0.001: 10.19%
- Width 64, learning rate 0.01: 10.19%
- Width 64, learning rate 0.1: 94.07%

The best result was 94.07% with width 64 and learning rate 0.1. The smaller learning rates did not learn effectively within 50 epochs.

The median forward-pass time was 0.1023 ms, and the backward-pass time was 0.1348 ms. Backpropagation was slower in this measurement because it computes gradients using additional matrix multiplications.

The loss curves are saved in results/loss_curves.png. The experiment results are saved in results/results.txt.
