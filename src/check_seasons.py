"""Check the corrected season coding: counts per season and remaining duplicates."""

from load_data import load_data
from features import add_features
from filters import apply_filters

df = apply_filters(add_features(load_data()))
print("Rows:", len(df))
print("\nRows per season:")
print(df.groupby(["season_num", "season_clean"]).size())

dup = df.duplicated(subset=["player_id", "season_clean"], keep=False)
print("\nPlayer-season duplicate rows (two clubs in one season):", int(dup.sum()))