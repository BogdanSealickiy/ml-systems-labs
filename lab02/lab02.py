import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
np.random.seed(SEED)


def prepare_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    Y = np.eye(10)[y]
    return train_test_split(
        X, Y, y, test_size=0.30, random_state=42, stratify=y
    )


def initialize_network(width):
    return [
        {"W": np.random.randn(64, width) * 0.01, "b": np.zeros(width)},
        {"W": np.random.randn(width, width) * 0.01, "b": np.zeros(width)},
        {"W": np.random.randn(width, 10) * 0.01, "b": np.zeros(10)},
    ]


def forward(X, params):
    z1 = X @ params[0]["W"] + params[0]["b"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ params[1]["W"] + params[1]["b"]
    a2 = np.maximum(0, z2)
    logits = a2 @ params[2]["W"] + params[2]["b"]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    cache = (X, z1, a1, z2, a2, probs)
    return probs, cache


def loss_fn(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def backward(Y, params, cache):
    X, z1, a1, z2, a2, probs = cache
    n = X.shape[0]
    dz3 = (probs - Y) / n
    grads = [None, None, None]

    grads[2] = {
        "W": a2.T @ dz3,
        "b": np.sum(dz3, axis=0),
    }
    da2 = dz3 @ params[2]["W"].T
    dz2 = da2 * (z2 > 0)
    grads[1] = {
        "W": a1.T @ dz2,
        "b": np.sum(dz2, axis=0),
    }
    da1 = dz2 @ params[1]["W"].T
    dz1 = da1 * (z1 > 0)
    grads[0] = {
        "W": X.T @ dz1,
        "b": np.sum(dz1, axis=0),
    }
    return grads


def gradient_check():
    np.random.seed(7)
    X = np.random.rand(4, 64)
    Y = np.eye(10)[np.array([1, 3, 5, 7])]
    params = initialize_network(16)
    probs, cache = forward(X, params)
    grads = backward(Y, params, cache)
    checks = []

    for layer in range(3):
        for _ in range(4):
            i = np.random.randint(params[layer]["W"].shape[0])
            j = np.random.randint(params[layer]["W"].shape[1])
            original = params[layer]["W"][i, j]
            eps = 1e-5

            params[layer]["W"][i, j] = original + eps
            plus = loss_fn(forward(X, params)[0], Y)
            params[layer]["W"][i, j] = original - eps
            minus = loss_fn(forward(X, params)[0], Y)
            params[layer]["W"][i, j] = original

            numerical = (plus - minus) / (2 * eps)
            analytical = grads[layer]["W"][i, j]
            relative = abs(numerical - analytical) / max(
                1e-8, abs(numerical) + abs(analytical)
            )
            checks.append(relative)

    passed = sum(err < 1e-6 for err in checks)
    print(f"Gradient check: {passed}/12 passed (relative error < 1e-6)")
    print("Maximum relative error:", f"{max(checks):.3e}")
    if passed < 9:
        raise RuntimeError("Gradient check failed; do not trust training results.")


def train(X_train, Y_train, X_test, y_test, width, lr):
    params = initialize_network(width)
    history = []
    n = len(X_train)

    for epoch in range(EPOCHS):
        order = np.random.permutation(n)
        X_shuffled, Y_shuffled = X_train[order], Y_train[order]

        for start in range(0, n, BATCH_SIZE):
            xb = X_shuffled[start:start + BATCH_SIZE]
            yb = Y_shuffled[start:start + BATCH_SIZE]
            probs, cache = forward(xb, params)
            grads = backward(yb, params, cache)
            for layer in range(3):
                params[layer]["W"] -= lr * grads[layer]["W"]
                params[layer]["b"] -= lr * grads[layer]["b"]

        train_probs = forward(X_train, params)[0]
        epoch_loss = loss_fn(train_probs, Y_train)
        history.append(epoch_loss)
        print(f"  epoch {epoch + 1:02d}/{EPOCHS}, loss={epoch_loss:.4f}")

    test_probs = forward(X_test, params)[0]
    accuracy = np.mean(np.argmax(test_probs, axis=1) == y_test)
    return history, accuracy, params


def timing_test(X, Y, params):
    # Warm-up
    for _ in range(10):
        probs, cache = forward(X, params)
        backward(Y, params, cache)

    forward_times, backward_times = [], []
    for _ in range(100):
        start = time.perf_counter()
        probs, cache = forward(X, params)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward(Y, params, cache)
        backward_times.append(time.perf_counter() - start)

    print(f"Forward median:  {np.median(forward_times) * 1000:.4f} ms")
    print(f"Backward median: {np.median(backward_times) * 1000:.4f} ms")


def main():
    X_train, X_test, Y_train, Y_test, y_train, y_test = prepare_data()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Input range:", X_train.min(), "to", X_train.max())
    print("\nChecking gradient implementation...")
    gradient_check()

    np.random.seed(SEED)
    example = initialize_network(16)
    probs, _ = forward(X_train[:32], example)
    print("\nForward output shape:", probs.shape)
    print("Maximum row-sum error:", np.max(np.abs(probs.sum(axis=1) - 1)))
    print("Initial loss:", f"{loss_fn(forward(X_train, example)[0], Y_train):.4f}")
    print("Expected initial loss is close to ln(10) = 2.3026.")

    results = []
    plt.figure(figsize=(12, 7))

    for width in WIDTHS:
        for lr in LEARNING_RATES:
            print(f"\nTraining width={width}, learning rate={lr}")
            history, accuracy, params = train(
                X_train, Y_train, X_test, y_test, width, lr
            )
            results.append((width, lr, accuracy))
            plt.plot(history, label=f"W={width}, lr={lr}")
            print(f"Final test accuracy: {accuracy:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Neural Network from Scratch: Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lab02/loss_curves.png", dpi=150)
    plt.close()

    with open("lab02/results.txt", "w", encoding="utf-8") as f:
        f.write("width,learning_rate,test_accuracy\n")
        for width, lr, accuracy in results:
            f.write(f"{width},{lr},{accuracy:.6f}\n")

    print("\nTiming forward and backward passes (100 runs)...")
    timing_params = initialize_network(64)
    timing_test(X_train[:32], Y_train[:32], timing_params)
    print("\nResults saved to lab02/loss_curves.png and lab02/results.txt")
    print("Backward propagation uses matrix multiplications to compute weight")
    print("gradients; its cost is similar to forward propagation and may be higher.")


if __name__ == "__main__":
    main()
