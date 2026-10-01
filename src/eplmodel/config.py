"""Load frozen model specifications from configs/*.toml."""

import tomllib
from pathlib import Path

from eplmodel.paths import CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR / "baselines_v1.toml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
