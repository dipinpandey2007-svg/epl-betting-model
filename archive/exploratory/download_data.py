import pandas as pd
import os
import time
import urllib.error

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

SEASON_CODES = ['1415', '1516', '1617', '1718', '1819',
                 '1920', '2021', '2122', '2223', '2324']

def get_season_file(season_code, league='E0', max_retries=3):
    local_path = f"{RAW_DIR}/{league}_{season_code}.csv"
    if os.path.exists(local_path):
        print(f"{local_path} already exists, skipping download.")
        return local_path

    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}) ...")
            season_df = pd.read_csv(url)
            season_df.to_csv(local_path, index=False)
            return local_path
        except urllib.error.HTTPError as e:
            print(f"  Failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"  Giving up on {season_code} after {max_retries} attempts.")
    return None

for code in SEASON_CODES:
    get_season_file(code)
    time.sleep(1)  # be polite to the server between different seasons