"""Why do flagged players disappear from the sample the next season?

For rows of seasons 3 and 4 in outputs/oot_flags.csv, look the player up in the full data (before
filters) in season t+1 and assign ONE reason, checked in this order:
  observed, not in the data, only in Brazil/Argentina, age 28+, under 450 minutes, no market value, other.
Age and minutes are taken from the player's rows in t+1 (min age, max minutes across clubs), so the
split between the last categories is approximate.
"""

import numpy as np
import pandas as pd
from load_data import load_data
from features import add_features

MIN_MINUTES = 450
MAX_AGE = 28  # filters keep age < 28

oot = pd.read_csv("outputs/oot_flags.csv")
oot = oot[oot["season_num"] < 5].copy()  # seasons 3 and 4 have a next season in the data

raw = add_features(load_data())  # no filters: we want to see who was filtered out
raw["european"] = (raw["d_argentina"] + raw["d_brasileirao"]) == 0
nxt = raw.groupby(["player_id", "season_num"]).agg(
    minutes=("minutes", "max"),
    age=("age", "min"),
    has_value=("tm_mv", lambda s: (s > 0).any()),
    european=("european", "any"),
)

# Look up each flagged row's player in season t+1
idx = pd.MultiIndex.from_arrays([oot["player_id"], oot["season_num"] + 1])
look = nxt.reindex(idx)
look.index = oot.index  # align with oot rows

in_raw = look["minutes"].notna()
european = look["european"].fillna(False).astype(bool)
has_value = look["has_value"].fillna(False).astype(bool)

conditions = [
    oot["change"].notna(),
    ~in_raw,
    in_raw & ~european,
    look["age"] >= MAX_AGE,
    look["minutes"] < MIN_MINUTES,
    ~has_value,
]
choices = ["observed", "not in data", "only Brazil/Argentina", "age 28+", "under 450 min", "no market value"]
oot["reason"] = np.select(conditions, choices, default="other")

order = [c for c in choices + ["other"] if c in set(oot["reason"])]
counts = pd.crosstab(oot["tier"], oot["reason"])[order]
print("Counts by tier:")
print(counts)
print("\nRow percentages by tier:")
print((100 * pd.crosstab(oot["tier"], oot["reason"], normalize="index")[order]).round(1))

# Age at season t: is attrition mechanical (players near 28 leave the file the next season)?
cur_age = raw.groupby(["player_id", "season_num"])["age"].min()
idx_t = pd.MultiIndex.from_arrays([oot["player_id"], oot["season_num"]])
oot["age_t"] = cur_age.reindex(idx_t).to_numpy()
oot["age_group"] = pd.cut(oot["age_t"], [0, 21, 23, 25, 27, 28], labels=["<=21", "21-23", "23-25", "25-27", "27+"])

print("\nShare observed next season, by age at t:")
print(oot.groupby("age_group", observed=True)["reason"].apply(lambda s: (s == "observed").mean()).round(3))
print("\nAge composition by tier (counts):")
print(pd.crosstab(oot["tier"], oot["age_group"]))