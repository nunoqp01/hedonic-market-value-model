"""Filters for the data used in the hedonic model.

in stata: drop if tm_mv == . | tm_mv <= 0
          drop if age >= 28
          drop if minutes < 450
"""

# Players whose position label is wrong in the source data (checked by hand).
# Excluding them changes N, so the switch is OFF by default to keep the thesis
# replication (N = 3,373) reproducible. Set to True for the cleaned sample.
EXCLUDE_NON_FORWARDS = False
NON_FORWARD_IDS = {
    1347,  # Francisco Sierralta Carvallo: centre-back labelled Right Winger
}


def apply_filters(df, min_minutes=450):
    """Sample filters of the thesis. min_minutes is a parameter so it can be varied."""
    df = df.dropna(subset=["tm_mv"])
    df = df[df["tm_mv"] > 0]
    df = df[df["age"] < 28]
    df = df[df["minutes"] >= min_minutes]
    if EXCLUDE_NON_FORWARDS:
        df = df[~df["player_id"].isin(NON_FORWARD_IDS)]
    return df