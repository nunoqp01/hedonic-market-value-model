"""Out-of-time undervaluation flags and ex-post check.

Residual = ln market value minus a prediction made only with earlier seasons (expanding window).
Flag: residual below -1.5 SD in BOTH models (OLS + trend and RF + trend). Strong tier: below -2 SD
in both. The SD is that of the test season's own out-of-time residuals (a mild look-ahead on the
scale of the residuals only, not on the predictions).
Ex post: change in ln value from season t to t+1 for flagged vs other players. Only seasons 3 and 4
have a next season in the data, and only players still in the sample next season are observed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features

THRESHOLD = 1.5
STRONG = 2.0

pred = pd.read_csv("outputs/expanding_window_predictions.csv")
pred = pred[(pred["d_argentina"] + pred["d_brasileirao"]) == 0].copy()  # European leagues only

# Standardised out-of-time residual per model and season (negative = value below prediction)
MODELS = {"z_ols": "pred_trend", "z_rf": "pred_rf_trend"}
for z_col, pred_col in MODELS.items():
    resid = pred["ln_mv"] - pred[pred_col]
    pred[z_col] = resid / resid.groupby(pred["season_num"]).transform("std")

# Consensus: the weaker of the two z-scores must still be below the threshold
pred["z_worst"] = pred[["z_ols", "z_rf"]].max(axis=1)
pred["tier"] = np.select([pred["z_worst"] < -STRONG, pred["z_worst"] < -THRESHOLD],
                         ["strong", "moderate"], default="none")

print("Flags per season (consensus tiers):")
print(pred.groupby(["season_num", "tier"]).size().unstack(fill_value=0))
print("\nSingle-model undervalued counts:")
print(pd.DataFrame({m: (pred[m] < -THRESHOLD).groupby(pred["season_num"]).sum() for m in MODELS}))

# Ex post: change in ln value from season t to t+1 (mean per player-season, as in expanding_window.py)
value = pred.groupby(["player_id", "season_num"])["ln_mv"].mean()
idx = pd.MultiIndex.from_arrays([pred["player_id"], pred["season_num"] + 1])
pred["next_ln_mv"] = value.reindex(idx).to_numpy()
pred["change"] = pred["next_ln_mv"] - pred["ln_mv"]

raw = add_features(load_data())
cur_age = raw.groupby(["player_id", "season_num"])["age"].min()
pred["age_t"] = cur_age.reindex(pd.MultiIndex.from_arrays([pred["player_id"], pred["season_num"]])).to_numpy()

# Players aged 27+ in season t leave the file next season by the age < 28 rule, so exclude them
exp = pred[(pred["season_num"] < 5) & (pred["age_t"] < 27)]
print("\nShare still observed next season, by tier (attrition):")
print(exp.groupby("tier")["change"].apply(lambda s: s.notna().mean()).round(3))

obs = exp[exp["change"].notna()].copy()
obs["excess"] = obs["change"] - obs.groupby("season_num")["change"].transform("mean")  # vs average change that season
summary = obs.groupby("tier").agg(n=("change", "size"),
                                  mean_change=("change", "mean"),
                                  mean_excess=("excess", "mean"),
                                  share_up=("change", lambda s: (s > 0).mean()))
print("\nChange in ln value next season, by tier:")
print(summary.round(3))

# Dose-response: does a lower average z predict a larger next-season rise? (cf. Stata: reg ..., vce(cluster player_id))
obs["z_mean"] = obs[["z_ols", "z_rf"]].mean(axis=1)
fit = smf.ols("change ~ z_mean + C(season_num)", data=obs).fit(
    cov_type="cluster", cov_kwds={"groups": obs["player_id"].to_numpy()})
print("\nDose-response (cluster-robust by player):")
print(fit.summary().tables[1])

# Control for value momentum/inertia: change in ln value from t-1 to t (players with a previous season)
mean_ln = raw.groupby(["player_id", "season_num"])["ln_mv"].mean()
prev_ln = mean_ln.reindex(pd.MultiIndex.from_arrays([obs["player_id"], obs["season_num"] - 1])).to_numpy()
obs["lag_change"] = obs["ln_mv"] - prev_ln
sub = obs[obs["lag_change"].notna()]  # same sample for both regressions, so they are comparable

for label, formula in [("without lag", "change ~ z_mean + C(season_num)"),
                       ("with lag", "change ~ z_mean + lag_change + C(season_num)")]:
    f = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["player_id"].to_numpy()})
    print(f"\nDose-response {label} (N = {int(f.nobs)}):")
    print(f.summary().tables[1])
Path("outputs").mkdir(exist_ok=True)
pred.to_csv("outputs/oot_flags.csv", index=False)