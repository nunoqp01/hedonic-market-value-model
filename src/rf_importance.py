"""Permutation importance of the tuned random forest, out of time.

Train on seasons 1 to 4, measure importance on season 5 (never seen in training): for each variable,
shuffle its values in the test season and see how much the RMSE (log) gets worse. A big increase means
the forest relies on that variable. Hyperparameters are the ones chosen by the tuning in
expanding_window.py (min_samples_leaf = 1, max_features = 0.3).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE

SEED = 42
TEST_SEASON = 5

df = apply_filters(add_features(load_data()))

# Same features as in expanding_window.py (formula without season dummies, without age_sq)
FORMULA_NO_SEASON = FORMULA_BASE.replace(" + s2122 + s2223 + s2324 + s2425", "")
FEATURES = [t.strip() for t in FORMULA_NO_SEASON.split("~")[1].split("+")]
FEATURES = [f for f in FEATURES if f != "age_sq"]

train = df[df["season_num"] < TEST_SEASON]
test = df[df["season_num"] == TEST_SEASON]

forest = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=0.3,
                                     n_jobs=-1, random_state=SEED)),
]).fit(train[FEATURES], train["ln_mv"])

rmse = np.sqrt(((test["ln_mv"] - forest.predict(test[FEATURES])) ** 2).mean())
print(f"Out-of-time RMSE (log), season {TEST_SEASON}, no shuffling: {rmse:.3f}\n")

# Shuffle each variable 30 times; the importance is the average increase in RMSE
result = permutation_importance(forest, test[FEATURES], test["ln_mv"],
                                scoring="neg_root_mean_squared_error",
                                n_repeats=30, random_state=SEED, n_jobs=-1)

table = pd.DataFrame({"increase in RMSE": result.importances_mean,
                      "std over shuffles": result.importances_std},
                     index=FEATURES).sort_values("increase in RMSE", ascending=False)
print(table.round(4).to_string())