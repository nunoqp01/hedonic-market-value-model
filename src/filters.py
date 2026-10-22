"""Filters for the data used in the hedonic model.

in stata: drop if tm_mv == . | tm_mv <= 0
drop if age >= 28
drop if minutes < 450
"""
def apply_filters(df):
    df = df.dropna(subset=["tm_mv"])
    df = df[df["tm_mv"] > 0]
    df = df[df["age"] < 28]
    df = df[df["minutes"] >= 450]
    return df