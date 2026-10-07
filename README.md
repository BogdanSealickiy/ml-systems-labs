# ML Systems — Lab 1

Student project for Lab 1: Environment and First System Measurements.

## Windows setup

Use Python 3.11.8.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/lab01.py
```

The required Breast Cancer Wisconsin (Diagnostic) dataset is loaded through scikit-learn; no dataset file needs to be committed.
