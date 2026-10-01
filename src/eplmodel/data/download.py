"""Download raw season CSVs from football-data.co.uk.

The raw files are third-party data. They are NOT committed to the repository
and NOT covered by this project's MIT licence. Check Football-Data's current
terms before downloading. This module is the reproducible way to obtain them.

Behaviour preserved from the original download_data.py: each file is read with
pandas and re-written with ``to_csv(index=False)``, and existing files are
never overwritten. Because of the pandas round-trip, raw-file bytes can vary
with pandas version or upstream edits; the processed dataset checksum in
data/checksums.json is the authoritative reproducibility check.

Usage:  python -m eplmodel.data.download
"""

import argparse
import time
import urllib.error
from pathlib import Path

import pandas as pd

from eplmodel.paths import RAW_DIR
from eplmodel.splits import SEASON_ORDER

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUE = "E0"  # English Premier League


def raw_season_path(season_code: str, raw_dir: Path = RAW_DIR, league: str = LEAGUE) -> Path:
    return Path(raw_dir) / f"{league}_{season_code}.csv"


def download_season(
    season_code: str,
    raw_dir: Path = RAW_DIR,
    league: str = LEAGUE,
    max_retries: int = 3,
    retry_wait: float = 3.0,
) -> Path | None:
    """Download one season, skipping it if the local file already exists. Returns the path or None."""
    local_path = raw_season_path(season_code, raw_dir, league)
    if local_path.exists():
        print(f"{local_path} already exists, skipping download.")
        return local_path

    url = BASE_URL.format(season=season_code, league=league)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}) ...")
            season_df = pd.read_csv(url)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            season_df.to_csv(local_path, index=False)
            return local_path
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"  Failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_wait)
    print(f"  Giving up on {season_code} after {max_retries} attempts.")
    return None


def download_all(season_codes=SEASON_ORDER, raw_dir: Path = RAW_DIR, pause: float = 1.0) -> list[Path | None]:
    paths = []
    for code in season_codes:
        paths.append(download_season(code, raw_dir))
        time.sleep(pause)  # be polite to the server between seasons
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    print("Note: these files are third-party data from football-data.co.uk, not covered by this project's "
          "licence. Make sure your use complies with Football-Data's current terms.")
    paths = download_all(raw_dir=args.raw_dir)
    missing = [code for code, p in zip(SEASON_ORDER, paths) if p is None]
    if missing:
        raise SystemExit(f"Failed to download seasons: {missing}")


if __name__ == "__main__":
    main()
