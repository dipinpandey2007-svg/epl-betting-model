import pandas as pd
import glob

RAW_DIR = "data/raw"

def load_and_clean_season(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')

    essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    df = df[essential_cols].copy()

    before = len(df)
    df = df.dropna(subset=essential_cols)
    dropped = before - len(df)
    if dropped > 0:
        print(f"{path}: dropped {dropped} incomplete row(s)")

    return df

files = sorted(glob.glob(f"{RAW_DIR}/E0_*.csv"))
print(files)  # sanity check: should list all 10, in a sensible order

all_seasons = []
for f in files:
    season_df = load_and_clean_season(f)
    season_label = f.split("E0_")[1].replace(".csv", "")  # e.g. '1415'
    season_df['Season'] = season_label
    all_seasons.append(season_df)

matches = pd.concat(all_seasons, ignore_index=True)
matches = matches.sort_values('Date').reset_index(drop=True)

print(matches.shape)
print(matches['Season'].value_counts().sort_index())
print(matches.head())
print(matches.tail())

import os

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

matches.to_csv(f"{PROCESSED_DIR}/matches.csv", index=False)
print(f"Saved {len(matches)} rows to {PROCESSED_DIR}/matches.csv")

# Now reload it fresh, as if in a brand new script, and check the dtype
reloaded = pd.read_csv(f"{PROCESSED_DIR}/matches.csv")
print(reloaded['Date'].dtype)   # <-- watch this: expect 'object', NOT datetime64

reloaded['Date'] = pd.to_datetime(reloaded['Date'], format='mixed')
print(reloaded['Date'].dtype)   # <-- now should be datetime64 again

import pandas as pd

matches = pd.read_csv("data/processed/matches.csv")
matches['Date'] = pd.to_datetime(matches['Date'], format='mixed')

home_teams = set(matches['HomeTeam'].unique())
away_teams = set(matches['AwayTeam'].unique())

all_teams = home_teams | away_teams
print(f"Number of unique teams: {len(all_teams)}")
print(sorted(all_teams))

# Sanity check: any team that only ever appears as home, or only ever as away?
only_home = home_teams - away_teams
only_away = away_teams - home_teams
print(f"Teams only ever Home: {only_home}")
print(f"Teams only ever Away: {only_away}")

INITIAL_RATING = 1500

ratings = {team: INITIAL_RATING for team in all_teams}

print(len(ratings))
print(ratings['Arsenal'])
print(ratings['Bournemouth'])