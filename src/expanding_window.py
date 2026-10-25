"""Expanding-window comparison: baselines and random forest on the same folds.

For each test season t (3, 4, 5): train on seasons < t, predict season t.
Methods:
  previous value         : the player's own ln market value in season t-1 (returning players only)
  previous value + drift : previous value plus the average year-on-year change in ln value,
                           computed only from seasons before t (no look-ahead)
  OLS                    : Base model without season dummies (a season dummy for t cannot be estimated)
  OLS + trend            : same, plus a linear time trend (season_num), which can extrapolate inflation
  RF                     : random forest on the Base features (no season information)
  RF + trend             : random forest on ln value minus the OLS time trend; the trend is added back
                           for the test season (a forest cannot extrapolate market inflation by itself)

All preprocessing lives inside a scikit-learn Pipeline, so it is fitted on training data only.
Random forest hyperparameters are chosen inside each training window with player-grouped
cross-validation (GroupKFold), so the same player never sits in both the training and validation
folds and the test season is never used for tuning.
Metrics are reported in log scale and in euros, on all test rows, on European leagues only,
on returning players, and on new players (no previous value).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE

FIRST_TEST = 3  # first test season (train on seasons 1 and 2), as in the thesis
LAST_SEASON = 5
SEED = 42

df = apply_filters(add_features(load_data()))

# Base formula without the season dummies, with and without a linear trend
SEASON_DUMMIES_TERM = " + s2122 + s2223 + s2324 + s2425"
FORMULA_NO_SEASON = FORMULA_BASE.replace(SEASON_DUMMIES_TERM, "")
FORMULA_TREND = FORMULA_NO_SEASON + " + season_num"
assert FORMULA_NO_SEASON != FORMULA_BASE  # the replace must have found the season dummies

# Forest features = the terms of the formula, without age_sq (a forest does not need it).
# Taking them from the formula keeps OLS and forest on the same variables.
FEATURES = [term.strip() for term in FORMULA_NO_SEASON.split("~")[1].split("+")]
FEATURES = [f for f in FEATURES if f != "age_sq"]

# Mean ln value per player and season (a player with two clubs in a season has two rows)
prev_map = df.groupby(["player_id", "season_num"])["ln_mv"].mean()
wide = prev_map.unstack("season_num")  # one column per season, one row per player


def make_forest():
    """Pipeline: median imputation (fitted on train only), then the random forest."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                         n_jobs=-1, random_state=SEED)),
    ])


def tune_forest(train, target):
    """Choose forest hyperparameters with player-grouped CV, using the training window only.

    `target` is passed in because the RF + trend variant trains on a detrended ln value.
    """
    grid = {"forest__min_samples_leaf": [1, 2, 3, 5],
            "forest__max_features": [0.1, 0.2, 0.3, 0.5]}
    search = GridSearchCV(make_forest(), grid,
                          cv=GroupKFold(n_splits=5),                   # same player never in train and validation
                          scoring="neg_root_mean_squared_error")       # higher is better, so it is minus the RMSE
    search.fit(train[FEATURES], target, groups=train["player_id"])
    return search  # already refitted on the whole training window with the best combination


def metrics(y, pred):
    """Errors in log and in euros. Positive bias = the method predicts too low."""
    err = y - pred
    abs_eur = (np.exp(y) - np.exp(pred)).abs()
    return {
        "N": len(y),
        "RMSE (log)": np.sqrt((err ** 2).mean()),
        "MAE (log)": err.abs().mean(),
        "bias (log)": err.mean(),
        "R2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "median abs error (EUR)": abs_eur.median(),
        "median abs % error": 100 * (abs_eur / np.exp(y)).median(),
    }


PRED_COLUMNS = {"previous value": "pred_prev",
                "previous value + drift": "pred_drift",
                "OLS": "pred_ols",
                "OLS + trend": "pred_trend",
                "RF": "pred_rf",
                "RF + trend": "pred_rf_trend"}


def evaluate(test, label):
    """Metrics for each method on all rows, European leagues, returning players and new players."""
    rows = []
    subsets = {"all test rows": pd.Series(True, index=test.index),
               "European leagues only": (test["d_argentina"] + test["d_brasileirao"]) == 0,
               "returning players": test["pred_prev"].notna(),
               "new players": test["pred_prev"].isna()}
    for subset, mask in subsets.items():
        for method, col in PRED_COLUMNS.items():
            keep = mask & test[col].notna()
            if keep.sum() == 0:
                continue
            rows.append({"test": label, "subset": subset, "method": method,
                         **metrics(test.loc[keep, "ln_mv"], test.loc[keep, col])})
    return rows


all_rows, test_frames = [], []
for t in range(FIRST_TEST, LAST_SEASON + 1):
    train = df[df["season_num"] < t]
    test = df[df["season_num"] == t].copy()

    # Previous value: look up the same player in season t-1
    idx = pd.MultiIndex.from_arrays([test["player_id"], test["season_num"] - 1])
    test["pred_prev"] = prev_map.reindex(idx).to_numpy()

    # Drift: average change in ln value between consecutive seasons, using only seasons < t
    changes = pd.concat([wide[s] - wide[s - 1] for s in range(2, t)]).dropna()
    test["pred_drift"] = test["pred_prev"] + changes.mean()

    # OLS fitted on the training seasons only, then predict the test season
    test["pred_ols"] = smf.ols(FORMULA_NO_SEASON, data=train).fit().predict(test)
    ols_trend = smf.ols(FORMULA_TREND, data=train).fit()
    test["pred_trend"] = ols_trend.predict(test)

    # Random forest on ln value, hyperparameters tuned inside the training window
    search = tune_forest(train, train["ln_mv"])
    test["pred_rf"] = search.predict(test[FEATURES])
    print(f"season {t} RF best params:", search.best_params_)

    # Random forest on ln value minus the time trend; add the trend back for season t
    b = ols_trend.params["season_num"]  # estimated growth per season, controlling for the features
    search_t = tune_forest(train, train["ln_mv"] - b * train["season_num"])
    test["pred_rf_trend"] = search_t.predict(test[FEATURES]) + b * t
    print(f"season {t} RF + trend best params:", search_t.best_params_)

    all_rows += evaluate(test, f"season {t}")
    test_frames.append(test)

pooled = pd.concat(test_frames)
all_rows += evaluate(pooled, "pooled 3-5")

table = pd.DataFrame(all_rows).set_index(["test", "subset", "method"])
pd.set_option("display.width", 200)
print(table.loc["pooled 3-5"].round(3).to_string())  # per-season detail is in the CSV

# Per-season RF on European leagues only (compare with Table 34 of the thesis)
eu = table.xs("European leagues only", level="subset")
print(eu.loc[eu.index.get_level_values("method") == "RF"].round(3).to_string())

Path("outputs").mkdir(exist_ok=True)
table.to_csv("outputs/expanding_window_metrics.csv")

# Out-of-time predictions per player-season, for residual analysis later
id_cols = [c for c in ["player_id", "name", "club", "season_clean", "season_num", "position",
                       "d_argentina", "d_brasileirao"] if c in pooled.columns]
pooled[id_cols + ["ln_mv"] + list(PRED_COLUMNS.values())].to_csv("outputs/expanding_window_predictions.csv", index=False)