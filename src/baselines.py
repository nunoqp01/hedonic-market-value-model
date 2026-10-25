"""Expanding-window baselines (no random forest yet).

For each test season t (3, 4, 5): train on seasons < t, predict season t.
Methods:
  previous value : the player's own ln market value in season t-1 (only for returning players)
  OLS            : Base model without season dummies (a season dummy for t cannot be estimated)
  OLS + trend    : same, plus a linear time trend (season_num), which can extrapolate inflation

Metrics are reported in log scale and in euros, on all test rows and on returning players only
(the only rows where all three methods can be compared fairly).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE

FIRST_TEST = 3  # first test season (train on seasons 1 and 2), as in the thesis
LAST_SEASON = 5

df = apply_filters(add_features(load_data()))

# Base formula without the season dummies, with and without a linear trend
SEASON_DUMMIES_TERM = " + s2122 + s2223 + s2324 + s2425"
FORMULA_NO_SEASON = FORMULA_BASE.replace(SEASON_DUMMIES_TERM, "")
FORMULA_TREND = FORMULA_NO_SEASON + " + season_num"
assert FORMULA_NO_SEASON != FORMULA_BASE  # the replace must have found the season dummies

# Mean ln value per player and season (a player with two clubs in a season has two rows)
prev_map = df.groupby(["player_id", "season_num"])["ln_mv"].mean()


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


def evaluate(test, label):
    """Metrics for each method on all rows and on returning players only."""
    rows = []
    preds = {"previous value": test["pred_prev"],
             "OLS": test["pred_ols"],
             "OLS + trend": test["pred_trend"]}
    subsets = {"all test rows": pd.Series(True, index=test.index),
               "returning players": test["pred_prev"].notna()}
    for subset, mask in subsets.items():
        for method, pred in preds.items():
            keep = mask & pred.notna()
            if keep.sum() == 0:
                continue
            rows.append({"test": label, "subset": subset, "method": method,
                         **metrics(test.loc[keep, "ln_mv"], pred[keep])})
    return rows


all_rows, test_frames = [], []
for t in range(FIRST_TEST, LAST_SEASON + 1):
    train = df[df["season_num"] < t]
    test = df[df["season_num"] == t].copy()

    # Previous value: look up the same player in season t-1
    idx = pd.MultiIndex.from_arrays([test["player_id"], test["season_num"] - 1])
    test["pred_prev"] = prev_map.reindex(idx).to_numpy()

    # OLS fitted on the training seasons only, then predict the test season
    test["pred_ols"] = smf.ols(FORMULA_NO_SEASON, data=train).fit().predict(test)
    test["pred_trend"] = smf.ols(FORMULA_TREND, data=train).fit().predict(test)

    all_rows += evaluate(test, f"season {t}")
    test_frames.append(test)

# All test seasons pooled
all_rows += evaluate(pd.concat(test_frames), "pooled 3-5")

table = pd.DataFrame(all_rows).set_index(["test", "subset", "method"])
pd.set_option("display.width", 200)
print(table.round(3).to_string())

Path("outputs").mkdir(exist_ok=True)
table.to_csv("outputs/baselines.csv")