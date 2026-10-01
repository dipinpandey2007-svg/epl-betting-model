"""Write experiment results with provenance (code version, data checksum, package versions)."""

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from eplmodel.data.checksums import content_sha256
from eplmodel.paths import PROCESSED_MATCHES, PROJECT_ROOT, RESULTS_DIR

_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(data_path: Path = PROCESSED_MATCHES) -> dict:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    # Uncommitted changes outside results/ (result files written earlier in the same run don't count).
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "data_file": Path(data_path).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "data_sha256": content_sha256(data_path),
        "python": platform.python_version(),
        "packages": versions,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_results(
    experiment: str, results: dict, results_dir: Path = RESULTS_DIR, data_path: Path = PROCESSED_MATCHES
) -> Path:
    """Write results/<experiment>/metrics.json; provenance records `data_path` (default: dataset v1)."""
    out_dir = Path(results_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    payload = {"experiment": experiment, "results": _jsonable(results), "provenance": provenance(data_path)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_predictions(experiment: str, predictions: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> dict:
    """Write results/<experiment>/predictions.csv (git-ignored) and return its path, SHA-256 and row count.

    Predictions are derived from third-party match data, so they are not committed;
    the checksum, recorded in metrics.json, identifies the exact file.
    """
    out_dir = Path(results_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.csv"
    predictions.to_csv(path)
    try:
        shown = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return {"file": shown, "sha256": content_sha256(path), "n_rows": int(len(predictions))}
