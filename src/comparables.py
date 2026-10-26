"""Step 2 of the comparables: nearest neighbours by style profile, within the same position group.

Weighted Euclidean distance on the within-position z-scores of the profile metrics.
The two correlated pairs (shooting: npxG + shots; creation: xA + key passes) get half weight each,
so that each of the four dimensions (shooting, creation, dribbling, aerial) counts about once.
Usage: python src/comparables.py "part of a player name"
"""

import sys

import numpy as np
import pandas as pd
from profiles import build_profiles, PROFILE

WEIGHTS = {"npxg_p90": 0.5, "shots_p90": 0.5, "xa_p90": 0.5, "keypasses_p90": 0.5,
           "dribbles_p90": 1.0, "aerial_p90": 1.0}
N_NEIGHBOURS = 10


def find_similar(pool, name, n=N_NEIGHBOURS):
    """Return the target row and its n closest players in the same position group."""
    matches = pool[pool["name"].str.contains(name, case=False, regex=False)]
    if len(matches) != 1:
        raise ValueError(f"'{name}' matches {len(matches)} players: {matches['name'].tolist()}")
    target = matches.iloc[0]

    others = pool[(pool["position_group"] == target["position_group"]) & (pool.index != target.name)].copy()
    z_cols = [c + "_z" for c in PROFILE]
    weights = np.array([WEIGHTS[c] for c in PROFILE])
    diff = others[z_cols].to_numpy(dtype=float) - target[z_cols].to_numpy(dtype=float)
    others["distance"] = np.sqrt((weights * diff ** 2).sum(axis=1))
    return target, others.sort_values("distance").head(n)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/comparables.py "part of a player name"')

    pool = build_profiles()
    target, neighbours = find_similar(pool, sys.argv[1])

    show = [c for c in ["name", "club", "age", "minutes", "tm_mv"] if c in pool.columns]
    print(f"Target: {target['name']} ({target['position_group']})")
    print(target[show].to_string())
    print(f"\n{N_NEIGHBOURS} most similar players (same position group, season, at least 900 minutes):")
    print(neighbours[show + ["distance"]].round(2).to_string(index=False))