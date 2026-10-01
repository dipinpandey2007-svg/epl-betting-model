"""Controlled access to the sealed final holdout (protocol holdout_v1, docs/HOLDOUT_PROTOCOL.md).

2025-26 is *operationally sealed*: its outcomes must not be used for model
fitting, tuning, feature selection, specification selection, descriptive
analysis or any other development decision. The season is not unknown to the
world; it is sealed from this project's development process.

Holdout outcomes can be read only with a HoldoutAccess, which only
open_final_holdout() creates, and only when ALL of these hold:

1. every requested spec id is pre-registered in configs/holdout_v1.toml;
2. the access entry is authorised in configs/holdout_v1.toml for exactly
   those specs, and the same entry id appears in docs/TEST_SET_ACCESS_LOG.md;
3. the working tree is clean (outside results/) and HEAD descends from the
   freeze tag, so the registration and the code were committed before access;
4. the sealed data manifest (data/holdout_manifest.json, written when the data
   are sealed) exists, and the processed holdout file matches its checksum.

Manifest format (repository-relative POSIX paths, content SHA-256 as in
eplmodel.data.checksums):

    {"protocol_id": "holdout_v1", "seasons": ["2526"],
     "processed_file": "data/holdout/processed/<file>.csv",
     "files": {"<path>": "<sha256>", ...}}

These checks make use of the holdout impossible *by mistake*; they cannot stop
deliberate circumvention. Development code (experiments/) must never import
this module, and tests/test_holdout.py checks that.
"""

import json
import re
import subprocess
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from eplmodel.data.checksums import content_sha256
from eplmodel.data.load import _read_matches
from eplmodel.data.validate import validate_matches
from eplmodel.paths import HOLDOUT_CONFIG, HOLDOUT_MANIFEST, PROJECT_ROOT, TEST_SET_ACCESS_LOG
from eplmodel.splits import FINAL_HOLDOUT_SEASONS, NEXT_HOLDOUT_SEASONS, HoldoutAccessError

_TOKEN = object()


@dataclass(frozen=True)
class HoldoutAccess:
    """Proof that every access condition was checked. Create it only with open_final_holdout()."""

    entry_id: str
    spec_ids: tuple[str, ...]
    git_commit: str
    processed_file: Path
    sha256: str
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _TOKEN:
            raise HoldoutAccessError("HoldoutAccess can only be created by open_final_holdout().")


def load_holdout_config(path: Path = HOLDOUT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError:
        return None


def working_tree_clean(root: Path = PROJECT_ROOT) -> bool:
    """True if git reports no uncommitted changes outside results/ (False if git is unavailable)."""
    out = _git(root, "status", "--porcelain", "--", ".", ":(exclude)results")
    return out is not None and out.returncode == 0 and out.stdout.strip() == ""


def freeze_tag_is_ancestor(tag: str, root: Path = PROJECT_ROOT) -> bool:
    """True if the freeze tag exists and HEAD descends from it (or is it)."""
    out = _git(root, "merge-base", "--is-ancestor", tag, "HEAD")
    return out is not None and out.returncode == 0


def head_commit(root: Path = PROJECT_ROOT) -> str:
    out = _git(root, "rev-parse", "HEAD")
    if out is None or out.returncode != 0:
        raise HoldoutAccessError("Cannot determine the git commit; holdout access requires a git checkout.")
    return out.stdout.strip()


def _logged_entry_ids(log_path: Path) -> set[str]:
    """Entry ids in the first column of markdown table rows (e.g. '| H2 | ...')."""
    text = Path(log_path).read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*([^|\s]+)\s*\|", text, flags=re.MULTILINE))


def open_final_holdout(
    entry_id: str,
    spec_ids: Iterable[str],
    *,
    config_path: Path = HOLDOUT_CONFIG,
    log_path: Path = TEST_SET_ACCESS_LOG,
    manifest_path: Path = HOLDOUT_MANIFEST,
    root: Path = PROJECT_ROOT,
) -> HoldoutAccess:
    """Check every access condition (see module docstring) and return a HoldoutAccess, or raise."""
    cfg = load_holdout_config(config_path)
    spec_ids = tuple(spec_ids)
    if (tuple(cfg["holdout"]["seasons"]) != FINAL_HOLDOUT_SEASONS
            or tuple(cfg["holdout"]["next_holdout_seasons"]) != NEXT_HOLDOUT_SEASONS):
        raise HoldoutAccessError("configs/holdout_v1.toml disagrees with the season roles in eplmodel.splits.")

    if not spec_ids:
        raise HoldoutAccessError("No spec ids given.")
    unregistered = sorted(set(spec_ids) - set(cfg["registration"]["spec_ids"]))
    if unregistered:
        raise HoldoutAccessError(f"Specs {unregistered} are not pre-registered in {Path(config_path).name}.")

    authorised = {a["entry_id"]: a for a in cfg.get("access", [])}
    if entry_id not in authorised:
        raise HoldoutAccessError(f"Access entry {entry_id!r} is not authorised in {Path(config_path).name}.")
    if sorted(authorised[entry_id]["spec_ids"]) != sorted(spec_ids):
        raise HoldoutAccessError(f"Access entry {entry_id!r} does not authorise exactly the specs {sorted(spec_ids)}.")
    if entry_id not in _logged_entry_ids(log_path):
        raise HoldoutAccessError(f"Access entry {entry_id!r} is not recorded in {Path(log_path).name}.")

    if not working_tree_clean(root):
        raise HoldoutAccessError("The working tree has uncommitted changes; commit before accessing the holdout.")
    tag = cfg["holdout"]["freeze_tag"]
    if not freeze_tag_is_ancestor(tag, root):
        raise HoldoutAccessError(f"HEAD does not descend from the freeze tag {tag!r}.")

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise HoldoutAccessError("The holdout data have not been sealed: no manifest at "
                                 f"{manifest_path.name}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["seasons"]) != FINAL_HOLDOUT_SEASONS:
        raise HoldoutAccessError(f"Manifest seasons {manifest['seasons']} are not the holdout seasons.")
    rel = manifest["processed_file"]
    processed = Path(root) / rel
    if not processed.exists():
        raise HoldoutAccessError(f"Sealed holdout file {rel} is missing.")
    expected = manifest["files"][rel]
    if content_sha256(processed) != expected:
        raise HoldoutAccessError(f"{rel} does not match the checksum recorded when it was sealed.")

    return HoldoutAccess(entry_id, spec_ids, head_commit(root), processed, expected, _token=_TOKEN)


def load_holdout_matches(access: HoldoutAccess) -> pd.DataFrame:
    """The sealed holdout matches, for an access opened with open_final_holdout()."""
    if not isinstance(access, HoldoutAccess):
        raise HoldoutAccessError("load_holdout_matches requires a HoldoutAccess from open_final_holdout().")
    if content_sha256(access.processed_file) != access.sha256:
        raise HoldoutAccessError("The holdout file changed after access was opened.")
    df = _read_matches(access.processed_file)
    if tuple(sorted(df["Season"].unique())) != FINAL_HOLDOUT_SEASONS:
        raise HoldoutAccessError("The sealed holdout file contains seasons other than the holdout seasons.")
    validate_matches(df)
    return df
