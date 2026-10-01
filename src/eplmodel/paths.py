"""Filesystem locations used across the project, resolved relative to the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_MATCHES = PROCESSED_DIR / "matches.csv"
CHECKSUMS_FILE = DATA_DIR / "checksums.json"

# Sealed final holdout (docs/HOLDOUT_PROTOCOL.md). Data files live under the
# git-ignored HOLDOUT_DIR; the committed manifest records their checksums.
HOLDOUT_DIR = DATA_DIR / "holdout"
HOLDOUT_MANIFEST = DATA_DIR / "holdout_manifest.json"

CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
HOLDOUT_CONFIG = CONFIG_DIR / "holdout_v1.toml"
TEST_SET_ACCESS_LOG = PROJECT_ROOT / "docs" / "TEST_SET_ACCESS_LOG.md"
