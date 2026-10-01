import pandas as pd
import numpy as np

matches = pd.read_csv("data/processed/matches.csv")
matches['Date'] = pd.to_datetime(matches['Date'], format='mixed')
matches['Season'] = matches['Season'].astype(str)


all_teams = set(matches['HomeTeam'].unique()) | set(matches['AwayTeam'].unique())

INITIAL_RATING = 1500
ratings = {team: INITIAL_RATING for team in all_teams}

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Known-answer checks -- reason about each one BEFORE running it
print(expected_score(1500, 1500))   # equal ratings -> ?
print(expected_score(1600, 1400))   # A stronger -> should be > 0.5
print(expected_score(1400, 1600))   # A weaker -> should be < 0.5, and mirror the line above
print(expected_score(1600, 1400) + expected_score(1400, 1600))  # sanity identity

def actual_score(home_goals, away_goals):
    if home_goals > away_goals:
        return 1.0
    elif home_goals < away_goals:
        return 0.0
    else:
        return 0.5

print(actual_score(3, 1))  # predict first
print(actual_score(0, 2))  # predict first
print(actual_score(1, 1))  # predict first

K = 20

def update_ratings(rating_home, rating_away, home_goals, away_goals, k=K):
    exp_home = expected_score(rating_home, rating_away)
    exp_away = expected_score(rating_away, rating_home)
    act_home = actual_score(home_goals, away_goals)
    act_away = 1 - act_home

    new_rating_home = rating_home + k * (act_home - exp_home)
    new_rating_away = rating_away + k * (act_away - exp_away)

    return new_rating_home, new_rating_away

# Hypothetical: two evenly matched teams (both 1500), home team wins 2-0
new_home, new_away = update_ratings(1500, 1500, 2, 0)
print(new_home, new_away)

# Check the zero-sum property yourself
print((new_home - 1500) + (new_away - 1500))  # predict this before running -- what must it equal, and why?

INITIAL_RATING = 1500
ratings = {team: INITIAL_RATING for team in all_teams}

history = []

for row in matches.itertuples():
    home_rating_before = ratings[row.HomeTeam]
    away_rating_before = ratings[row.AwayTeam]

    new_home, new_away = update_ratings(
        home_rating_before, away_rating_before, row.FTHG, row.FTAG
    )

    ratings[row.HomeTeam] = new_home
    ratings[row.AwayTeam] = new_away

    history.append({
        'Date': row.Date,
        'HomeTeam': row.HomeTeam,
        'AwayTeam': row.AwayTeam,
        'FTHG': row.FTHG,
        'FTAG': row.FTAG,
        'FTR': row.FTR,
        'EloHome': home_rating_before,   # pre-match -- this is the leakage-safe value
        'EloAway': away_rating_before,   # pre-match -- this is the leakage-safe value
    })

elo_history = pd.DataFrame(history)
print(len(elo_history))
print(elo_history.head())

final_ratings = pd.Series(ratings).sort_values(ascending=False)
print(final_ratings.head(10))
print(final_ratings.tail(10))

elo_history['ExpHome'] = expected_score(elo_history['EloHome'], elo_history['EloAway'])
elo_history['ActHome'] = elo_history['FTR'].map({'H': 1.0, 'D': 0.5, 'A': 0.0})
elo_history['Surprise'] = elo_history['ActHome'] - elo_history['ExpHome']

print(elo_history['Surprise'].mean())

import math

mean_home_actual = elo_history['ActHome'].mean()
print(mean_home_actual)

# Algebra: solve expected_score(HOME_ADV, 0) = mean_home_actual for HOME_ADV
HOME_ADV = 400 * math.log10(mean_home_actual / (1 - mean_home_actual))
print(HOME_ADV)

# Sanity check: plugging it back in should return you to where you started
print(expected_score(HOME_ADV, 0))

def update_ratings(rating_home, rating_away, home_goals, away_goals, k=K, home_adv=HOME_ADV):
    exp_home = expected_score(rating_home + home_adv, rating_away)
    exp_away = expected_score(rating_away, rating_home + home_adv)
    act_home = actual_score(home_goals, away_goals)
    act_away = 1 - act_home

    new_rating_home = rating_home + k * (act_home - exp_home)
    new_rating_away = rating_away + k * (act_away - exp_away)

    return new_rating_home, new_rating_away

elo_history['ExpHome'] = expected_score(elo_history['EloHome'] + HOME_ADV, elo_history['EloAway'])
elo_history['Surprise'] = elo_history['ActHome'] - elo_history['ExpHome']
print(elo_history['Surprise'].mean())

ratings = {team: INITIAL_RATING for team in all_teams}
history = []

for row in matches.itertuples():
    home_rating_before = ratings[row.HomeTeam]
    away_rating_before = ratings[row.AwayTeam]

    new_home, new_away = update_ratings(
        home_rating_before, away_rating_before, row.FTHG, row.FTAG
    )  # this now applies HOME_ADV internally on every single call

    ratings[row.HomeTeam] = new_home
    ratings[row.AwayTeam] = new_away

    history.append({
        'Date': row.Date, 'HomeTeam': row.HomeTeam, 'AwayTeam': row.AwayTeam,
        'FTHG': row.FTHG, 'FTAG': row.FTAG, 'FTR': row.FTR,
        'EloHome': home_rating_before, 'EloAway': away_rating_before,
    })

elo_history = pd.DataFrame(history)

elo_history['ExpHome'] = expected_score(elo_history['EloHome'] + HOME_ADV, elo_history['EloAway'])
elo_history['ActHome'] = elo_history['FTR'].map({'H': 1.0, 'D': 0.5, 'A': 0.0})
elo_history['Surprise'] = elo_history['ActHome'] - elo_history['ExpHome']
print(elo_history['Surprise'].mean())

final_ratings = pd.Series(ratings).sort_values(ascending=False)
print(final_ratings.head(10))
print(final_ratings.tail(10))

SEASON_ORDER = ['1415', '1516', '1617', '1718', '1819',
                 '1920', '2021', '2122', '2223', '2324']

TEST_SEASONS = ['2223', '2324']
TRAIN_SEASONS = [s for s in SEASON_ORDER if s not in TEST_SEASONS]

print(TRAIN_SEASONS)
print(TEST_SEASONS)

elo_history['Season'] = elo_history['Date'].map(
    dict(zip(matches['Date'], matches['Season']))
)  # bring the Season label across from `matches` into `elo_history`

train_mask = elo_history['Season'].isin(TRAIN_SEASONS)
test_mask = elo_history['Season'].isin(TEST_SEASONS)

print(train_mask.sum(), test_mask.sum())

print(matches['Season'].dtype)
print(matches['Season'].head())
print(type(matches['Season'].iloc[0]))

mean_home_actual_train = elo_history.loc[train_mask, 'ActHome'].mean()
print(mean_home_actual_train)

HOME_ADV_TRAIN = 400 * math.log10(mean_home_actual_train / (1 - mean_home_actual_train))
print(HOME_ADV_TRAIN)

def update_ratings(rating_home, rating_away, home_goals, away_goals, k=K, home_adv=HOME_ADV_TRAIN):
    exp_home = expected_score(rating_home + home_adv, rating_away)
    exp_away = expected_score(rating_away, rating_home + home_adv)
    act_home = actual_score(home_goals, away_goals)
    act_away = 1 - act_home
    new_rating_home = rating_home + k * (act_home - exp_home)
    new_rating_away = rating_away + k * (act_away - exp_away)
    return new_rating_home, new_rating_away

ratings = {team: INITIAL_RATING for team in all_teams}
history = []

for row in matches.itertuples():
    home_rating_before = ratings[row.HomeTeam]
    away_rating_before = ratings[row.AwayTeam]
    new_home, new_away = update_ratings(home_rating_before, away_rating_before, row.FTHG, row.FTAG)
    ratings[row.HomeTeam] = new_home
    ratings[row.AwayTeam] = new_away
    history.append({
        'Date': row.Date, 'Season': row.Season, 'HomeTeam': row.HomeTeam, 'AwayTeam': row.AwayTeam,
        'FTHG': row.FTHG, 'FTAG': row.FTAG, 'FTR': row.FTR,
        'EloHome': home_rating_before, 'EloAway': away_rating_before,
    })

elo_history = pd.DataFrame(history)

elo_history['ExpHome'] = expected_score(elo_history['EloHome'] + HOME_ADV_TRAIN, elo_history['EloAway'])
elo_history['ActHome'] = elo_history['FTR'].map({'H': 1.0, 'D': 0.5, 'A': 0.0})
elo_history['Surprise'] = elo_history['ActHome'] - elo_history['ExpHome']

train_mask = elo_history['Season'].isin(TRAIN_SEASONS)
test_mask = elo_history['Season'].isin(TEST_SEASONS)

print("Train surprise mean:", elo_history.loc[train_mask, 'Surprise'].mean())
print("Test surprise mean:", elo_history.loc[test_mask, 'Surprise'].mean())

elo_history['EloDiff'] = (elo_history['EloHome'] + HOME_ADV_TRAIN) - elo_history['EloAway']

X_train = elo_history.loc[train_mask, ['EloDiff']]
y_train = elo_history.loc[train_mask, 'FTR']

X_test = elo_history.loc[test_mask, ['EloDiff']]
y_test = elo_history.loc[test_mask, 'FTR']

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.classes_)  # important: check the ORDER sklearn assigned to H/D/A internally

probs_test = model.predict_proba(X_test)
print(probs_test[:5])       # first 5 matches' predicted probability triples
print(probs_test[:5].sum(axis=1))   # should be 1.0 for every row -- verify, don't assume

print(X_test.head())

from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize

# Your model's log loss
model_logloss = log_loss(y_test, probs_test, labels=model.classes_)
print("Model log loss:", model_logloss)

# Baseline: every match gets the exact same fixed train-set proportions
baseline_probs_row = y_train.value_counts(normalize=True).reindex(model.classes_).values
baseline_probs = np.tile(baseline_probs_row, (len(y_test), 1))

baseline_logloss = log_loss(y_test, baseline_probs, labels=model.classes_)
print("Baseline log loss:", baseline_logloss)

# Brier score -- sklearn has no built-in multiclass version, so we build it directly
y_test_binarized = label_binarize(y_test, classes=model.classes_)

model_brier = ((probs_test - y_test_binarized) ** 2).sum(axis=1).mean()
baseline_brier = ((baseline_probs - y_test_binarized) ** 2).sum(axis=1).mean()

print("Model Brier:", model_brier)
print("Baseline Brier:", baseline_brier)

import matplotlib.pyplot as plt

home_col_idx = list(model.classes_).index('H')
home_pred_prob = probs_test[:, home_col_idx]
home_actual = (y_test == 'H').astype(int)

bins = pd.qcut(home_pred_prob, q=10, duplicates='drop')
calib_df = pd.DataFrame({'pred': home_pred_prob, 'actual': home_actual, 'bin': bins})

calib_summary = calib_df.groupby('bin', observed=True).agg(
    mean_pred=('pred', 'mean'),
    mean_actual=('actual', 'mean'),
    n=('actual', 'size')
)
print(calib_summary)

plt.figure()
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
plt.plot(calib_summary['mean_pred'], calib_summary['mean_actual'], marker='o', label='Model (Home win)')
plt.xlabel('Mean predicted P(Home)')
plt.ylabel('Actual frequency of Home win')
plt.legend()
plt.title('Calibration: Home win probability')
plt.savefig('calibration_home.png')
plt.show()

folds = [
    (['1415', '1516', '1617'], '1718'),
    (['1415', '1516', '1617', '1718'], '1819'),
    (['1415', '1516', '1617', '1718', '1819'], '1920'),
    (['1415', '1516', '1617', '1718', '1819', '1920'], '2021'),
    (['1415', '1516', '1617', '1718', '1819', '1920', '2021'], '2122'),
]

for train_seasons, val_season in folds:
    n_train = elo_history['Season'].isin(train_seasons).sum()
    n_val = (elo_history['Season'] == val_season).sum()
    print(f"Train seasons {train_seasons[0]}-{train_seasons[-1]} ({n_train} matches) -> validate on {val_season} ({n_val} matches)")

def simulate_elo(matches_df, k, home_adv):
    ratings = {team: INITIAL_RATING for team in all_teams}
    history = []

    for row in matches_df.itertuples():
        home_before = ratings[row.HomeTeam]
        away_before = ratings[row.AwayTeam]

        exp_home = expected_score(home_before + home_adv, away_before)
        exp_away = expected_score(away_before, home_before + home_adv)
        act_home = actual_score(row.FTHG, row.FTAG)
        act_away = 1 - act_home

        ratings[row.HomeTeam] = home_before + k * (act_home - exp_home)
        ratings[row.AwayTeam] = away_before + k * (act_away - exp_away)

        history.append({
            'Date': row.Date, 'Season': row.Season,
            'HomeTeam': row.HomeTeam, 'AwayTeam': row.AwayTeam,
            'FTHG': row.FTHG, 'FTAG': row.FTAG, 'FTR': row.FTR,
            'EloHome': home_before, 'EloAway': away_before,
        })

    return pd.DataFrame(history)


# Regression test: this MUST match Step 17's numbers exactly
check_history = simulate_elo(matches, k=20, home_adv=HOME_ADV_TRAIN)
check_history['EloDiff'] = (check_history['EloHome'] + HOME_ADV_TRAIN) - check_history['EloAway']

final_check_ratings = {}
for row in matches.itertuples():
    pass  # not needed -- instead, pull final ratings directly from the function's last state

# Simplest reliable check: compare a specific known value from Step 17 directly
print(check_history[['Date','HomeTeam','AwayTeam','EloHome','EloAway']].tail(5))
check_slice = check_history.iloc[3040:3045].copy()
check_slice['EloDiff'] = (check_slice['EloHome'] + HOME_ADV_TRAIN) - check_slice['EloAway']
print(check_slice[['Date', 'HomeTeam', 'AwayTeam', 'EloDiff']])
from sklearn.metrics import log_loss as sk_log_loss

def evaluate_k_on_fold(k, train_seasons, val_season):
    hist = simulate_elo(matches, k=k, home_adv=0)  # home_adv=0 for now -- we add it back in below, fit per-fold

    fold_train_mask = hist['Season'].isin(train_seasons)
    fold_val_mask = hist['Season'] == val_season

    # Fit HOME_ADV using only this fold's training rows
    fold_train_actual = hist.loc[fold_train_mask, 'FTR'].map({'H': 1.0, 'D': 0.5, 'A': 0.0})
    fold_mean_home_actual = fold_train_actual.mean()
    fold_home_adv = 400 * math.log10(fold_mean_home_actual / (1 - fold_mean_home_actual))

    hist['EloDiff'] = (hist['EloHome'] + fold_home_adv) - hist['EloAway']

    X_fold_train = hist.loc[fold_train_mask, ['EloDiff']]
    y_fold_train = hist.loc[fold_train_mask, 'FTR']
    X_fold_val = hist.loc[fold_val_mask, ['EloDiff']]
    y_fold_val = hist.loc[fold_val_mask, 'FTR']

    fold_model = LogisticRegression()
    fold_model.fit(X_fold_train, y_fold_train)

    val_probs = fold_model.predict_proba(X_fold_val)
    return sk_log_loss(y_fold_val, val_probs, labels=fold_model.classes_)


# Sanity check: run just ONE fold, ONE K value, and look at the number before trusting the loop
test_score = evaluate_k_on_fold(k=20, train_seasons=folds[0][0], val_season=folds[0][1])
print(test_score)
CANDIDATE_KS = [10, 15, 20, 25, 30, 40, 50]

results = []
for k in CANDIDATE_KS:
    fold_scores = [evaluate_k_on_fold(k, train_seasons, val_season) for train_seasons, val_season in folds]
    avg_score = sum(fold_scores) / len(fold_scores)
    results.append({'K': k, 'avg_log_loss': avg_score, 'fold_scores': fold_scores})
    print(f"K={k}: avg log loss = {avg_score:.4f}, folds = {[round(s, 4) for s in fold_scores]}")

results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df['avg_log_loss'].idxmin(), 'K']
print(f"\nBest K: {best_k}")
BEST_K = 25

elo_history = simulate_elo(matches, k=BEST_K, home_adv=0)

train_mask = elo_history['Season'].isin(TRAIN_SEASONS)
test_mask = elo_history['Season'].isin(TEST_SEASONS)

train_actual = elo_history.loc[train_mask, 'FTR'].map({'H': 1.0, 'D': 0.5, 'A': 0.0})
mean_home_actual_train = train_actual.mean()
HOME_ADV_FINAL = 400 * math.log10(mean_home_actual_train / (1 - mean_home_actual_train))
print("HOME_ADV_FINAL:", HOME_ADV_FINAL)

elo_history['EloDiff'] = (elo_history['EloHome'] + HOME_ADV_FINAL) - elo_history['EloAway']

X_train = elo_history.loc[train_mask, ['EloDiff']]
y_train = elo_history.loc[train_mask, 'FTR']
X_test = elo_history.loc[test_mask, ['EloDiff']]
y_test = elo_history.loc[test_mask, 'FTR']

final_model = LogisticRegression()
final_model.fit(X_train, y_train)
probs_test = final_model.predict_proba(X_test)

final_logloss = log_loss(y_test, probs_test, labels=final_model.classes_)
print("Final (K=25) model log loss:", final_logloss)

baseline_probs_row = y_train.value_counts(normalize=True).reindex(final_model.classes_).values
baseline_probs = np.tile(baseline_probs_row, (len(y_test), 1))
baseline_logloss = log_loss(y_test, baseline_probs, labels=final_model.classes_)
print("Baseline log loss:", baseline_logloss)

y_test_binarized = label_binarize(y_test, classes=final_model.classes_)
final_brier = ((probs_test - y_test_binarized) ** 2).sum(axis=1).mean()
baseline_brier = ((baseline_probs - y_test_binarized) ** 2).sum(axis=1).mean()
print("Final (K=25) model Brier:", final_brier)
print("Baseline Brier:", baseline_brier)
home_col_idx = list(final_model.classes_).index('H')
home_pred_prob = probs_test[:, home_col_idx]
home_actual = (y_test == 'H').astype(int)

bins = pd.qcut(home_pred_prob, q=10, duplicates='drop')
calib_df = pd.DataFrame({'pred': home_pred_prob, 'actual': home_actual, 'bin': bins})

calib_summary = calib_df.groupby('bin', observed=True).agg(
    mean_pred=('pred', 'mean'),
    mean_actual=('actual', 'mean'),
    n=('actual', 'size')
)
calib_summary['gap'] = calib_summary['mean_actual'] - calib_summary['mean_pred']
print(calib_summary)

plt.figure()
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
plt.plot(calib_summary['mean_pred'], calib_summary['mean_actual'], marker='o', label='Model (K=25, Home win)')
plt.xlabel('Mean predicted P(Home)')
plt.ylabel('Actual frequency of Home win')
plt.legend()
plt.title('Calibration: Home win probability (K=25, tuned)')
plt.savefig('calibration_home_k25.png')
plt.show()
home_goals = matches['FTHG']
away_goals = matches['FTAG']

print("Home goals - mean:", home_goals.mean(), "variance:", home_goals.var())
print("Away goals - mean:", away_goals.mean(), "variance:", away_goals.var())

print(home_goals.value_counts().sort_index())
print(away_goals.value_counts().sort_index())
from scipy.stats import poisson

n_matches = len(matches)
max_goals = 9

home_lambda = home_goals.mean()
away_lambda = away_goals.mean()

for g in range(max_goals + 1):
    observed_home = (home_goals == g).sum()
    predicted_home = poisson.pmf(g, home_lambda) * n_matches
    observed_away = (away_goals == g).sum()
    predicted_away = poisson.pmf(g, away_lambda) * n_matches
    print(f"{g} goals -- Home: observed={observed_home}, predicted={predicted_home:.1f} | Away: observed={observed_away}, predicted={predicted_away:.1f}")

home_rows = matches[['HomeTeam', 'AwayTeam', 'FTHG']].copy()
home_rows.columns = ['Team', 'Opponent', 'Goals']
home_rows['IsHome'] = 1

away_rows = matches[['AwayTeam', 'HomeTeam', 'FTAG']].copy()
away_rows.columns = ['Team', 'Opponent', 'Goals']
away_rows['IsHome'] = 0

goal_model_data = pd.concat([home_rows, away_rows], ignore_index=True)

print(goal_model_data.shape)     # expect exactly double the match count
print(goal_model_data.head())
print(goal_model_data.tail())

print(goal_model_data.iloc[[0, 3800]])

import statsmodels.api as sm
import statsmodels.formula.api as smf

poisson_model = smf.glm(
    formula="Goals ~ Team + Opponent + IsHome",
    data=goal_model_data,
    family=sm.families.Poisson()
).fit()

print(poisson_model.summary())

intercept = poisson_model.params['Intercept']
team_mancity = poisson_model.params['Team[T.Man City]']
opp_sheffieldutd = poisson_model.params['Opponent[T.Sheffield United]']
is_home_coef = poisson_model.params['IsHome']

log_lambda_home = intercept + team_mancity + opp_sheffieldutd + is_home_coef
lambda_home_manual = np.exp(log_lambda_home)
print("Manual calc, Man City (home) expected goals:", lambda_home_manual)

# Now verify using the model's own predict() on a matching hypothetical row
hypothetical = pd.DataFrame({'Team': ['Man City'], 'Opponent': ['Sheffield United'], 'IsHome': [1]})
lambda_home_predicted = poisson_model.predict(hypothetical)
print("model.predict() Man City (home) expected goals:", lambda_home_predicted.values)

away_hypothetical = pd.DataFrame({'Team': ['Sheffield United'], 'Opponent': ['Man City'], 'IsHome': [0]})
lambda_away_predicted = poisson_model.predict(away_hypothetical)
print("Sheffield United (away) expected goals:", lambda_away_predicted.values)

lambda_home = lambda_home_predicted.values[0]
lambda_away = lambda_away_predicted.values[0]

max_g = 10
score_probs = np.zeros((max_g + 1, max_g + 1))
for h in range(max_g + 1):
    for a in range(max_g + 1):
        score_probs[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)

print("Total probability mass captured:", score_probs.sum())  # should be very close to 1.0

p_home_win = np.tril(score_probs, k=-1).sum()
p_draw = np.trace(score_probs)
p_away_win = np.triu(score_probs, k=1).sum()

print(f"P(Home win) = {p_home_win:.4f}")
print(f"P(Draw)     = {p_draw:.4f}")
print(f"P(Away win) = {p_away_win:.4f}")
print(f"Sum check: {p_home_win + p_draw + p_away_win:.4f}")

print("\nMost likely individual scorelines:")
top_scores_idx = np.dstack(np.unravel_index(np.argsort(-score_probs.ravel()), score_probs.shape))[0][:5]
for h, a in top_scores_idx:
    print(f"{h}-{a}: {score_probs[h, a]:.4f}")

home_rows = matches[['HomeTeam', 'AwayTeam', 'FTHG', 'Season']].copy()
home_rows.columns = ['Team', 'Opponent', 'Goals', 'Season']
home_rows['IsHome'] = 1

away_rows = matches[['AwayTeam', 'HomeTeam', 'FTAG', 'Season']].copy()
away_rows.columns = ['Team', 'Opponent', 'Goals', 'Season']
away_rows['IsHome'] = 0

goal_model_data = pd.concat([home_rows, away_rows], ignore_index=True)

goal_train_mask = goal_model_data['Season'].isin(TRAIN_SEASONS)
goal_test_mask = goal_model_data['Season'].isin(TEST_SEASONS)

print(goal_train_mask.sum(), goal_test_mask.sum())  # expect 6080, 1520 -- double the match-level 3040/760

poisson_model_train = smf.glm(
    formula="Goals ~ Team + Opponent + IsHome",
    data=goal_model_data[goal_train_mask],
    family=sm.families.Poisson()
).fit()

print("IsHome (train only):", poisson_model_train.params['IsHome'])
print("IsHome (full data, from before):", poisson_model.params['IsHome'])

def match_probs(team_home, team_away, model, max_g=10):
    lam_home = model.predict(pd.DataFrame({'Team': [team_home], 'Opponent': [team_away], 'IsHome': [1]})).values[0]
    lam_away = model.predict(pd.DataFrame({'Team': [team_away], 'Opponent': [team_home], 'IsHome': [0]})).values[0]

    grid = np.zeros((max_g + 1, max_g + 1))
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            grid[h, a] = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)

    p_home = np.tril(grid, k=-1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total  # renormalize so the tiny truncated tail doesn't cause probs to not sum to 1

# Sanity check on one known match first, before running the whole test set
ph, pd_, pa = match_probs('Man City', 'Sheffield United', poisson_model_train)
print(ph, pd_, pa)

train_teams = set(matches.loc[matches['Season'].isin(TRAIN_SEASONS), 'HomeTeam']) | \
              set(matches.loc[matches['Season'].isin(TRAIN_SEASONS), 'AwayTeam'])
test_teams = set(matches.loc[matches['Season'].isin(TEST_SEASONS), 'HomeTeam']) | \
             set(matches.loc[matches['Season'].isin(TEST_SEASONS), 'AwayTeam'])
unseen_teams = test_teams - train_teams
print("Unseen teams:", unseen_teams)

test_matches = matches[matches['Season'].isin(TEST_SEASONS)].copy()
affected_mask = test_matches['HomeTeam'].isin(unseen_teams) | test_matches['AwayTeam'].isin(unseen_teams)
print(f"Matches involving an unseen team: {affected_mask.sum()} out of {len(test_matches)}")

clean_test_matches = test_matches[~affected_mask].copy()
print(f"Evaluating Poisson on {len(clean_test_matches)} matches (excluded {affected_mask.sum()})")

poisson_probs = np.array([
    match_probs(row.HomeTeam, row.AwayTeam, poisson_model_train)
    for row in clean_test_matches.itertuples()
])
print(poisson_probs.shape)
print(poisson_probs[:5])

poisson_logloss = log_loss(clean_test_matches['FTR'], poisson_probs, labels=['H', 'D', 'A'])
print("Poisson model log loss:", poisson_logloss)

y_test_binarized_poisson = label_binarize(clean_test_matches['FTR'], classes=['H', 'D', 'A'])
poisson_brier = ((poisson_probs - y_test_binarized_poisson) ** 2).sum(axis=1).mean()
print("Poisson model Brier:", poisson_brier)

# A fake, obvious case: home team wins, and we're highly confident it's a home win
toy_probs_as_written = np.array([[0.9, 0.05, 0.05]])  # intended as (P_home, P_draw, P_away)
toy_true = ['H']

print("Passing labels=['H','D','A']:", log_loss(toy_true, toy_probs_as_written, labels=['H', 'D', 'A']))

# Same 90% confidence, but columns physically reordered to alphabetical (A, D, H)
toy_probs_reordered = np.array([[0.05, 0.05, 0.9]])
print("Passing labels=['A','D','H'], columns reordered:", log_loss(toy_true, toy_probs_reordered, labels=['A', 'D', 'H']))

poisson_probs_ordered = poisson_probs[:, [2, 1, 0]]  # (H,D,A) columns -> (A,D,H) columns

poisson_logloss = log_loss(clean_test_matches['FTR'], poisson_probs_ordered, labels=['A', 'D', 'H'])
print("Poisson model log loss:", poisson_logloss)

y_test_binarized_poisson = label_binarize(clean_test_matches['FTR'], classes=['A', 'D', 'H'])
poisson_brier = ((poisson_probs_ordered - y_test_binarized_poisson) ** 2).sum(axis=1).mean()
print("Poisson model Brier:", poisson_brier)

elo_test_probs_full = final_model.predict_proba(X_test)  # X_test/y_test from Milestone 1, still in scope, all 760 rows

clean_mask_within_test = ~affected_mask.values  # aligns with test_matches / X_test's row order

elo_probs_matched = elo_test_probs_full[clean_mask_within_test]
y_test_matched = y_test[clean_mask_within_test]

elo_logloss_matched = log_loss(y_test_matched, elo_probs_matched, labels=final_model.classes_)
y_test_binarized_matched = label_binarize(y_test_matched, classes=final_model.classes_)
elo_brier_matched = ((elo_probs_matched - y_test_binarized_matched) ** 2).sum(axis=1).mean()

print("Elo log loss (same 648 matches):", elo_logloss_matched)
print("Elo Brier (same 648 matches):", elo_brier_matched)

def tau_correction(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lam * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

# Check 1: rho = 0 must reduce to "no correction at all", for ANY scoreline and ANY lambda/mu
print(tau_correction(0, 0, lam=1.5, mu=1.2, rho=0))
print(tau_correction(1, 0, lam=1.5, mu=1.2, rho=0))
print(tau_correction(2, 3, lam=1.5, mu=1.2, rho=0))

# Check 2: a concrete, hand-computable case with a real rho
lam, mu, rho = 1.5, 1.2, -0.1
print("tau(0,0):", tau_correction(0, 0, lam, mu, rho))   # predict by hand first
print("tau(1,0):", tau_correction(1, 0, lam, mu, rho))   # predict by hand first
print("tau(0,1):", tau_correction(0, 1, lam, mu, rho))   # predict by hand first
print("tau(1,1):", tau_correction(1, 1, lam, mu, rho))   # predict by hand first
print("tau(2,2):", tau_correction(2, 2, lam, mu, rho))   # should just be 1.0, untouched

def dc_log_likelihood(rho, matches_df, model):
    home_lams = model.predict(pd.DataFrame({
        'Team': matches_df['HomeTeam'], 'Opponent': matches_df['AwayTeam'], 'IsHome': 1
    })).values
    away_lams = model.predict(pd.DataFrame({
        'Team': matches_df['AwayTeam'], 'Opponent': matches_df['HomeTeam'], 'IsHome': 0
    })).values

    total_log_lik = 0.0
    for i, row in enumerate(matches_df.itertuples()):
        lam, mu = home_lams[i], away_lams[i]
        x, y = row.FTHG, row.FTAG
        tau = tau_correction(x, y, lam, mu, rho)
        prob = tau * poisson.pmf(x, lam) * poisson.pmf(y, mu)
        total_log_lik += np.log(prob)

    return total_log_lik


train_matches_only = matches[matches['Season'].isin(TRAIN_SEASONS)].copy()

# Sanity check: log-likelihood at rho=0 should equal what plain independent Poisson gives
# (since tau=1 everywhere when rho=0) -- compute it, don't assume
ll_at_zero = dc_log_likelihood(0.0, train_matches_only, poisson_model_train)
print("Log-likelihood at rho=0:", ll_at_zero)

ll_at_small_negative = dc_log_likelihood(-0.05, train_matches_only, poisson_model_train)
print("Log-likelihood at rho=-0.05:", ll_at_small_negative)
candidate_rhos = np.arange(-0.30, 0.30, 0.02)

ll_results = []
for rho in candidate_rhos:
    ll = dc_log_likelihood(rho, train_matches_only, poisson_model_train)
    ll_results.append(ll)
    print(f"rho={rho:.2f}: log-likelihood={ll:.3f}")

best_idx = np.argmax(ll_results)
best_rho = candidate_rhos[best_idx]
print(f"\nBest rho: {best_rho:.2f}, log-likelihood: {ll_results[best_idx]:.3f}")

plt.figure()
plt.plot(candidate_rhos, ll_results, marker='o')
plt.axvline(best_rho, color='red', linestyle='--', label=f'Best rho = {best_rho:.2f}')
plt.xlabel('rho')
plt.ylabel('Training log-likelihood')
plt.title('Dixon-Coles rho search (train data only)')
plt.legend()
plt.savefig('rho_search.png')
plt.show()

def dc_match_probs(team_home, team_away, model, rho, max_g=10):
    lam_home = model.predict(pd.DataFrame({'Team': [team_home], 'Opponent': [team_away], 'IsHome': [1]})).values[0]
    lam_away = model.predict(pd.DataFrame({'Team': [team_away], 'Opponent': [team_home], 'IsHome': [0]})).values[0]

    grid = np.zeros((max_g + 1, max_g + 1))
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            tau = tau_correction(h, a, lam_home, lam_away, rho)
            grid[h, a] = tau * poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)

    p_home = np.tril(grid, k=-1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

# Sanity check against the plain Poisson result for the same fixture
BEST_RHO = -0.04
print("Poisson (rho=0)      :", match_probs('Man City', 'Sheffield United', poisson_model_train))
print("Dixon-Coles (rho=-0.04):", dc_match_probs('Man City', 'Sheffield United', poisson_model_train, BEST_RHO))

BEST_RHO = -0.04

dc_probs = np.array([
    dc_match_probs(row.HomeTeam, row.AwayTeam, poisson_model_train, BEST_RHO)
    for row in clean_test_matches.itertuples()
])
print(dc_probs.shape)  # expect (648, 3)

dc_logloss = log_loss(clean_test_matches['FTR'], dc_probs, labels=['H', 'D', 'A'])
print("Dixon-Coles model log loss:", dc_logloss)

y_test_binarized_dc = label_binarize(clean_test_matches['FTR'], classes=['H', 'D', 'A'])
dc_brier = ((dc_probs - y_test_binarized_dc) ** 2).sum(axis=1).mean()
print("Dixon-Coles model Brier:", dc_brier)

dc_probs_ordered = dc_probs[:, [2, 1, 0]]  # (H,D,A) -> (A,D,H)

dc_logloss = log_loss(clean_test_matches['FTR'], dc_probs_ordered, labels=['A', 'D', 'H'])
print("Dixon-Coles model log loss:", dc_logloss)

y_test_binarized_dc = label_binarize(clean_test_matches['FTR'], classes=['A', 'D', 'H'])
dc_brier = ((dc_probs_ordered - y_test_binarized_dc) ** 2).sum(axis=1).mean()
print("Dixon-Coles model Brier:", dc_brier)


matches = pd.read_csv("data/processed/matches.csv")
matches['Date'] = pd.to_datetime(matches['Date'], format='mixed')
matches['Season'] = matches['Season'].astype(str)

SEASON_ORDER = ['1415', '1516', '1617', '1718', '1819',
                 '1920', '2021', '2122', '2223', '2324']

TRAIN_SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122']

# Candidate validation seasons: every training season except the first
# (2014-15 can't be validated against -- no prior history exists yet)
candidate_validation_seasons = TRAIN_SEASONS[1:]

fold_summaries = []

for val_season in candidate_validation_seasons:
    idx = TRAIN_SEASONS.index(val_season)
    fold_train_seasons = TRAIN_SEASONS[:idx]  # everything strictly before val_season

    train_mask = matches['Season'].isin(fold_train_seasons)
    val_mask = matches['Season'] == val_season

    train_teams = set(matches.loc[train_mask, 'HomeTeam']) | set(matches.loc[train_mask, 'AwayTeam'])
    val_teams = set(matches.loc[val_mask, 'HomeTeam']) | set(matches.loc[val_mask, 'AwayTeam'])

    newly_appeared = val_teams - train_teams

    val_matches = matches[val_mask]
    affected_mask = val_matches['HomeTeam'].isin(newly_appeared) | val_matches['AwayTeam'].isin(newly_appeared)
    n_affected = affected_mask.sum()
    n_total = len(val_matches)

    fold_summaries.append({
        'fold_train_seasons': f"{fold_train_seasons[0]}-{fold_train_seasons[-1]}" if fold_train_seasons else "NONE",
        'val_season': val_season,
        'newly_appeared_teams': sorted(newly_appeared),
        'n_newly_appeared': len(newly_appeared),
        'n_affected_matches': n_affected,
        'n_total_val_matches': n_total,
    })

for f in fold_summaries:
    print(f"Train: {f['fold_train_seasons']:>12}  ->  Validate: {f['val_season']}")
    print(f"  Newly appeared teams ({f['n_newly_appeared']}): {f['newly_appeared_teams']}")
    print(f"  Affected matches: {f['n_affected_matches']} out of {f['n_total_val_matches']}")
    print()

summary_df = pd.DataFrame(fold_summaries)
print(summary_df[['fold_train_seasons', 'val_season', 'n_newly_appeared', 'n_affected_matches', 'n_total_val_matches']])

import pandas as pd

matches = pd.read_csv("data/processed/matches.csv")
matches['Date'] = pd.to_datetime(matches['Date'], format='mixed')
matches['Season'] = matches['Season'].astype(str)

SEASON_ORDER = ['1415', '1516', '1617', '1718', '1819',
                 '1920', '2021', '2122', '2223', '2324']
TRAIN_SEASONS = SEASON_ORDER[:8]

# External, pre-cutoff historical fact table.
# seasons_since_last_top_flight: None means "never played in the PL before"
TEAM_HISTORY = {
    'Bournemouth':      None,
    'Norwich':          1,
    'Watford':          15,
    'Middlesbrough':    7,
    'Brighton':         None,
    'Huddersfield':     None,
    'Cardiff':          4,
    'Fulham':           4,
    'Wolves':           6,
    'Sheffield United': 12,
    'Leeds':            16,
    'Brentford':        None,
}

YOYO_THRESHOLD = 2  # seasons; adjustable

def categorize(team):
    seasons_out = TEAM_HISTORY.get(team)
    if seasons_out is None:
        return 'long_absence_or_newcomer'
    return 'recent_yoyo' if seasons_out <= YOYO_THRESHOLD else 'long_absence_or_newcomer'

candidate_validation_seasons = TRAIN_SEASONS[1:]
fold_rows = []

for val_season in candidate_validation_seasons:
    idx = TRAIN_SEASONS.index(val_season)
    fold_train_seasons = TRAIN_SEASONS[:idx]

    train_mask = matches['Season'].isin(fold_train_seasons)
    val_mask = matches['Season'] == val_season

    train_teams = set(matches.loc[train_mask, 'HomeTeam']) | set(matches.loc[train_mask, 'AwayTeam'])
    val_teams = set(matches.loc[val_mask, 'HomeTeam']) | set(matches.loc[val_mask, 'AwayTeam'])
    newly_appeared = val_teams - train_teams

    val_matches = matches[val_mask].copy()

    def match_category(row):
        home_new = row['HomeTeam'] in newly_appeared
        away_new = row['AwayTeam'] in newly_appeared
        if not home_new and not away_new:
            return None
        cats = set()
        if home_new:
            cats.add(categorize(row['HomeTeam']))
        if away_new:
            cats.add(categorize(row['AwayTeam']))
        if len(cats) > 1:
            return 'mixed'
        return cats.pop()

    val_matches['fold_category'] = val_matches.apply(match_category, axis=1)
    is_fold1 = (fold_train_seasons == TRAIN_SEASONS[:1])

    counts = val_matches['fold_category'].value_counts(dropna=True)
    fold_rows.append({
        'fold_train_seasons': f"{fold_train_seasons[0]}-{fold_train_seasons[-1]}",
        'val_season': val_season,
        'low_confidence_fold': is_fold1,
        'newly_appeared_teams': sorted(newly_appeared),
        'n_recent_yoyo': counts.get('recent_yoyo', 0),
        'n_long_absence_or_newcomer': counts.get('long_absence_or_newcomer', 0),
        'n_mixed': counts.get('mixed', 0),
        'n_total_affected': int(val_matches['fold_category'].notna().sum()),
    })

summary_df = pd.DataFrame(fold_rows)
print(summary_df.to_string(index=False))

print("\nPooled totals across all 7 folds:")
print(summary_df[['n_recent_yoyo', 'n_long_absence_or_newcomer', 'n_mixed', 'n_total_affected']].sum())