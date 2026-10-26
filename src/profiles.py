"""Step 1 of the comparables: a standardised style profile per player-season.

Profile = per-90 style metrics (not outcomes like goals), standardised within position group
(centre-forwards and wingers separately), clipped at +-3. One season, minimum minutes for reliable per-90 numbers.
"""

import numpy as np
import pandas as pd
from load_data import load_data
from features import add_features
from filters import apply_filters

SEASON = 5
MIN_MINUTES_PROFILE = 900
PROFILE = ["npxg_p90", "xa_p90", "shots_p90", "dribbles_p90", "keypasses_p90", "aerial_p90"]


def build_profiles(season=SEASON, min_minutes=MIN_MINUTES_PROFILE):
    """Return one row per player-season with the profile metrics and their within-position z-scores."""
    df = apply_filters(add_features(load_data()))
    pool = df[(df["season_num"] == season) & (df["minutes"] >= min_minutes)].dropna(subset=PROFILE).copy()
    pool["position_group"] = np.where(pool["Winger"] == 1, "Winger", "Centre-forward")

    z = pool.groupby("position_group")[PROFILE].transform(lambda s: (s - s.mean()) / s.std())
    z = z.clip(-3, 3)
    z.columns = [c + "_z" for c in PROFILE]
    return pool.join(z)


if __name__ == "__main__":
    pool = build_profiles()
    print(f"Season {SEASON}, at least {MIN_MINUTES_PROFILE} minutes: {len(pool)} player-seasons")
    print(pool["position_group"].value_counts().to_string())

    print("\nCorrelation of the raw profile metrics:")
    print(pool[PROFILE].corr().round(2).to_string())

    z_cols = [c + "_z" for c in PROFILE]
    print("\nz-scores by position (mean should be about 0, std about 1; clipping makes std slightly lower):")
    print(pool.groupby("position_group")[z_cols].agg(["mean", "std"]).round(2).T.to_string())

    show = [c for c in ["name", "club"] if c in pool.columns] + ["position_group"] + z_cols
    print("\nFirst rows:")
    print(pool[show].head(5).round(2).to_string(index=False))