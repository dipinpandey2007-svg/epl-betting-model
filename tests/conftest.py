import json
from pathlib import Path

import pytest

from eplmodel.data.checksums import content_sha256
from eplmodel.paths import PROCESSED_MATCHES

GOLDEN_FILE = Path(__file__).parent / "golden" / "golden_values.json"


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def matches(golden):
    """The processed dataset, verified to be the exact data the golden values were computed from."""
    if not PROCESSED_MATCHES.exists():
        pytest.skip("data/processed/matches.csv missing: run `python -m eplmodel.data.download` "
                    "and `python -m eplmodel.data.build`")
    actual = content_sha256(PROCESSED_MATCHES)
    if actual != golden["data_sha256"]:
        pytest.fail(f"Processed data checksum {actual} differs from the golden dataset {golden['data_sha256']}; "
                    "golden results are only defined for that dataset.")
    from eplmodel.data import load_matches
    return load_matches()
