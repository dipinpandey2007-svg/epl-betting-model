"""Write guards for locked results.

Once a registered stage has been run and its results locked (a config [locked] section with status "locked"), its
results directory must never be written again. A rerun or amendment needs a new protocol with its own config and
results directory. refuse_locked_overwrite() is called before a stage computes anything and again just before it
writes.
"""

from collections.abc import Mapping
from pathlib import Path

GUARDED_FILES = ("metrics.json", "predictions.csv")


class LockedResultsError(RuntimeError):
    """A write would overwrite locked results, or results already present in the target directory."""


def refuse_locked_overwrite(results_dir: Path, lock: Mapping) -> None:
    """Raise unless writing to results_dir is legitimate: the stage is not locked and no result file exists there."""
    results_dir = Path(results_dir)
    if lock.get("status") == "locked":
        raise LockedResultsError(
            f"{results_dir.name}: the stage is locked; its results must not be overwritten. A rerun or amendment "
            "needs a new protocol with its own config and results directory.")
    existing = sorted(name for name in GUARDED_FILES if (results_dir / name).exists())
    if existing:
        raise LockedResultsError(f"{results_dir.name}: {existing} already exist; refusing to overwrite them.")
