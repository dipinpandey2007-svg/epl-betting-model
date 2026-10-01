"""Reproducible experiments. Run each as `python -m experiments.<name>` from the repository root.

Each module exposes `run(write: bool = True) -> dict`. With write=True the
result (plus provenance) is written to results/<name>/metrics.json.
"""
