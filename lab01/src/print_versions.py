import importlib
import platform
import sys

PACKAGES = [
    "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory_profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
for package in PACKAGES:
    try:
        module = importlib.import_module(package)
        print(f"{package}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{package}: ERROR ({exc})")
