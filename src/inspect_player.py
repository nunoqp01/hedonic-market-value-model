"""Print all rows of one player so data problems can be spotted.

Usage:  python src/inspect_player.py 1347
(the number after the script name is the player_id; sys.argv[1] reads it)
"""

import sys
from load_data import load_data
from features import add_features
from filters import apply_filters

df = apply_filters(add_features(load_data()))
player_id = int(sys.argv[1])

wanted = ["player_id", "name", "club", "season", "age", "position", "minutes",
          "goals_p90", "assists_p90", "tm_mv"]
cols = [c for c in wanted if c in df.columns]  # keep only columns that exist
print(df.loc[df["player_id"] == player_id, cols].to_string(index=False))