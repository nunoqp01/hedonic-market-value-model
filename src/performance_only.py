"""How much of market value does performance alone explain?

Three specifications on the Extended sample (rows with all advanced metrics):
  A. performance only (no league), B. performance + league, C. full Extended (adds status variables).
"Status" = senior and youth international caps, cumulative fees and minutes played.
For each: in-sample R2, and out-of-time (expanding window, linear time trend) OLS and random forest.
The forest uses fixed, untuned settings (min_samples_leaf = 1, max_features = 0.5): the settings tuned for
the full Base are not necessarily right for specs with fewer variables.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_EXT

SEED = 42
FIRST_TEST, LAST_SEASON = 3, 5
STATUS = {"int_A", "int_youth", "ln_fees_cum", "minutes"}

# Split the Extended formula into groups of terms (season dummies dropped: we use a trend instead)
terms = [t.strip() for t in FORMULA_EXT.split("~")[1].split("+")]
terms = [t for t in terms if not t.startswith("s2")]
league = [t for t in terms if t.startswith("d_")]
performance = [t for t in terms if t not in STATUS and t not in league]  # includes age, age_sq, Winger as controls

SPECS = {
    "0. Age and position only": ["age", "age_sq", "Winger"],
    "A. Performance only": performance,
    "B. Performance + league": performance + league,
    "C. Full Extended": terms,
}
for name, spec in SPECS.items():
    print(f"{name} ({len(spec)} terms): {spec}")

df = apply_filters(add_features(load_data())).dropna(subset=terms)  # Extended sample
print(f"\nSample: {len(df)} rows\n")


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


rows = []
for name, spec in SPECS.items():
    formula = "ln_mv ~ " + " + ".join(spec) + " + season_num"
    in_sample_r2 = smf.ols(formula, data=df).fit().rsquared
    features = [f for f in spec if f != "age_sq"]

    y_all, p_ols_all, p_rf_all = [], [], []
    for t in range(FIRST_TEST, LAST_SEASON + 1):
        train, test = df[df["season_num"] < t], df[df["season_num"] == t]
        fit = smf.ols(formula, data=train).fit()
        b = fit.params["season_num"]  # growth per season, to detrend the forest target

        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=0.5,
                                       n_jobs=-1, random_state=SEED)
        forest.fit(train[features], train["ln_mv"] - b * train["season_num"])

        y_all.append(test["ln_mv"])
        p_ols_all.append(fit.predict(test))
        p_rf_all.append(pd.Series(forest.predict(test[features]) + b * t, index=test.index))

    y, p_ols, p_rf = pd.concat(y_all), pd.concat(p_ols_all), pd.concat(p_rf_all)
    rows.append({"model": name, "N test": len(y), "in-sample R2": in_sample_r2,
                 "OLS + trend RMSE": rmse(y, p_ols), "OLS + trend R2": r2(y, p_ols),
                 "RF + trend RMSE": rmse(y, p_rf), "RF + trend R2": r2(y, p_rf)})

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))