import numpy as np
import pandas as pd

SEASON_DUMMIES = {
    "2021/2022": "s2122",
    "2022/2023": "s2223",
    "2023/2024": "s2324",
    "2024/2025": "s2425",
}


def add_features(df):
    """Create the derived variables used in the model."""
    df["ln_mv"] = np.log(df["tm_mv"])
    df["age_sq"] = df["age"] ** 2
    df["ln_fees_cum"] = np.log1p(df["fees_cum_raw"] / 1_000_000)
    df["Winger"] = df["position"].isin(["Left Winger", "Right Winger"]).astype(int)
    df["player_id"] = pd.factorize(df["name"])[0]

    for label, name in SEASON_DUMMIES.items():
        df[name] = (df["season_harmonized"] == label).astype(int)

    return df