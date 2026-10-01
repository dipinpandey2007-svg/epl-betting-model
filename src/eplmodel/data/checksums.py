"""Content checksums for data files.

Hashes are computed after normalising CRLF to LF, so the same content gives
the same checksum on Windows and Unix (pandas writes os.linesep, and git may
convert line endings on checkout).
"""

import hashlib
import json
from pathlib import Path

from eplmodel.paths import CHECKSUMS_FILE, PROJECT_ROOT


def content_sha256(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path = CHECKSUMS_FILE) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["files"]


def verify_file(path: Path, manifest_path: Path = CHECKSUMS_FILE) -> bool:
    """True if `path` matches the checksum recorded for it (keyed by repo-relative POSIX path)."""
    key = Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    expected = load_manifest(manifest_path).get(key)
    if expected is None:
        raise KeyError(f"No checksum recorded for {key}")
    return content_sha256(path) == expected
