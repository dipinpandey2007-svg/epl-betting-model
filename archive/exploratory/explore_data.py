import pandas as pd
df = pd.read_csv("data/raw/E0_2324.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print(df['Date'].dtype)
print(df['Date'].min(), df['Date'].max())

df_sorted = df.sort_values('Date')
print(df_sorted[['Date', 'HomeTeam', 'AwayTeam']].head())
print(df_sorted[['Date', 'HomeTeam', 'AwayTeam']].tail())
essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
matches = df_sorted[essential_cols].reset_index(drop=True)

print(matches.head())
print(matches.isna().sum())

import numpy as np
implied_result = np.where(
    matches['FTHG'] > matches['FTAG'], 'H',
    np.where(matches['FTHG'] < matches['FTAG'], 'A', 'D')
)

mismatches = matches[matches['FTR'] != implied_result]
print(f"Number of mismatches: {len(mismatches)}")
print(mismatches)