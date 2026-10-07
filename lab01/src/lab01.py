from pathlib import Path
import os
import statistics
import sys
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
try:
    import random
    random.seed(SEED)
except Exception:
    pass
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab01" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def median_training_time(factory, X, y):
    factory().fit(X, y)  # warm-up
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss
    times = []
    for _ in range(5):
        model = factory()
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
        peak = max(peak, process.memory_info().rss)
    return statistics.median(times), peak


def median_inference_time(model, sample):
    model.predict(sample)  # warm-up
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - start)
        peak = max(peak, process.memory_info().rss)
    return statistics.median(times), peak


def deployment_fit(memory_mb, latency_ms, size_kb):
    budgets = {
        "Cloud": lambda: memory_mb >= 1024 and latency_ms <= 100 and size_kb <= 500 * 1024,
        "Edge": lambda: 256 <= memory_mb <= 1024 and latency_ms <= 50 and size_kb <= 50 * 1024,
        "Mobile": lambda: 64 <= memory_mb <= 256 and latency_ms <= 20 and size_kb <= 10 * 1024,
        "TinyML": lambda: memory_mb <= 0.25 and latency_ms <= 10 and size_kb <= 100,
    }
    return {name: "YES" if check() else "NO" for name, check in budgets.items()}


def main():
    print("Lab 1 — Environment and First System Measurements")
    print(f"Python: {sys.version.split()[0]}")

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }

    accuracy_rows = []
    measurement_rows = []
    deployment_rows = []

    for name, factory in models.items():
        print(f"\nRunning {name}...")
        model = factory()
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        train_median, train_peak = median_training_time(factory, X_train, y_train)

        final_model = factory()
        final_model.fit(X_train, y_train)
        inference_median, inference_peak = median_inference_time(final_model, X_test[:1])

        model_path = RESULTS / f"{name}.joblib"
        joblib.dump(final_model, model_path)
        size_bytes = model_path.stat().st_size
        size_kb = size_bytes / 1024

        peak_mb = max(train_peak, inference_peak) / (1024 * 1024)
        latency_ms = inference_median * 1000

        accuracy_rows.append({"model": name, "test_accuracy": f"{accuracy:.4f}"})
        measurement_rows.append({
            "model": name,
            "training_median_seconds": f"{train_median:.8f}",
            "inference_median_ms": f"{latency_ms:.6f}",
            "model_size_bytes": size_bytes,
            "model_size_kb": f"{size_kb:.4f}",
            "peak_rss_mb": f"{peak_mb:.4f}",
        })

        for platform, fits in deployment_fit(peak_mb, latency_ms, size_kb).items():
            deployment_rows.append({"model": name, "platform": platform, "fits": fits})

        print(f"accuracy: {accuracy:.4f}")
        print(f"training median: {train_median:.8f} s")
        print(f"inference median: {latency_ms:.6f} ms")
        print(f"model size: {size_bytes} bytes ({size_kb:.4f} KB)")
        print(f"peak RSS: {peak_mb:.4f} MB")

    pd.DataFrame(accuracy_rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    pd.DataFrame(measurement_rows).to_csv(RESULTS / "measurements.csv", index=False)
    pd.DataFrame(deployment_rows).to_csv(RESULTS / "deployment_fit.csv", index=False)

    (RESULTS / "run_info.txt").write_text(
        "Dataset: Breast Cancer Wisconsin (Diagnostic) via sklearn.datasets.load_breast_cancer\n"
        "Split: 70/30 stratified, random_state=42\n"
        "Training: 1 warm-up + 5 measured runs, median reported\n"
        "Inference: 1 warm-up + 100 measured runs, median reported\n",
        encoding="utf-8"
    )
    print("\nDone. Results saved to lab01/results/")


if __name__ == "__main__":
    main()
