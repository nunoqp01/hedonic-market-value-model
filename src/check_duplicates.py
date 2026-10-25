"""Find player-season duplicates and show where they come from (by league)."""

from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import LEAGUES

df = apply_filters(add_features(load_data()))

# Which column holds the harmonised season? (the raw data has 'season_harmonized')
season_col = "season_harmonized" if "season_harmonized" in df.columns else "season"
print("Season column used:", season_col)

dup = df.duplicated(subset=["player_id", season_col], keep=False)
print(f"\nRows involved in player-season duplicates: {int(dup.sum())}")
print("\nBy league dummy:")
print(df.loc[dup, LEAGUES].sum())

cols = [c for c in ["player_id", "name", "club", "season", season_col, "minutes", "tm_mv"]
        if c in df.columns]
print("\nExamples:")
print(df.loc[dup, cols].sort_values(["player_id", "season"]).head(20).to_string(index=False))

# --- Test: map calendar years (e.g. "2025") to the season Y/(Y+1) ---
def map_calendar_year(s):
    """'2025' -> '2025/2026'; European seasons like '2024/2025' stay unchanged."""
    s = str(s)
    return f"{s}/{int(s) + 1}" if s.isdigit() and len(s) == 4 else s


df["season_new"] = df["season"].map(map_calendar_year)
dup_new = df.duplicated(subset=["player_id", "season_new"], keep=False)
print(f"\nDuplicate rows with the new mapping: {int(dup_new.sum())} (was {int(dup.sum())})")
print(df["season_new"].value_counts().sort_index())