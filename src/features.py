"""Derived variables for the hedonic model.

Python version of Section 4 of the Stata do-file: log of market value,
age squared, log of cumulative fees, the Winger dummy, a numeric player id
and the season and league dummies.
"""

import numpy as np
import pandas as pd

# Season label in the data -> name of the dummy column.
# 2020/2021 is the reference season, so it has no dummy.
SEASON_DUMMIES = {
    "2021/2022": "s2122",
    "2022/2023": "s2223",
    "2023/2024": "s2324",
    "2024/2025": "s2425",
}

# League label in the data -> name of the dummy column.
# Liga Portugal is the reference league, so it has no dummy.
LEAGUE_DUMMIES = {
    "Premier League": "d_premier",
    "La Liga": "d_laliga",
    "Serie A": "d_seriea",
    "Bundesliga": "d_bundesliga",
    "Ligue 1": "d_ligue1",
    "Eredivisie": "d_eredivisie",
    "Championship": "d_championship",
    "Segunda Division": "d_segunda",
    "Primera Division": "d_argentina",
    "BR Serie A": "d_brasileirao",
}


def add_features(df):
    """Add the derived variables to the players table and return it.

    Expects the columns tm_mv, age, fees_cum_raw, position, name,
    season_harmonized and league, as returned by load_data().
    """
    # Stata: gen ln_mv = ln(tm_mv)
    df["ln_mv"] = np.log(df["tm_mv"])
    # Stata: gen age_sq = age * age
    df["age_sq"] = df["age"] ** 2
    # The +1 keeps players with zero fees defined (log of zero does not exist)
    df["ln_fees_cum"] = np.log1p(df["fees_cum_raw"] / 1_000_000)
    # Left and right wingers pooled; centre-forwards are the reference
    df["Winger"] = df["position"].isin(["Left Winger", "Right Winger"]).astype(int)
    # Stata: encode name, gen(player_id). Used later to cluster standard errors
    df["player_id"] = pd.factorize(df["name"])[0]

    # One 0/1 column per season, built from season_harmonized.
    # The raw "season" column mixes text and numbers, so it is not used.
    for season, dummy in SEASON_DUMMIES.items():
        df[dummy] = (df["season_harmonized"] == season).astype(int)

    # One 0/1 column per league, built from the league column
    for league, dummy in LEAGUE_DUMMIES.items():
        df[dummy] = (df["league"] == league).astype(int)

    return df