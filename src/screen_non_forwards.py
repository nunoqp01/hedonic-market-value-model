"""Screen for players labelled as forwards whose stats look like a defender's.

Heuristic, inferred from one known case (Sierralta: high aerial duels, almost no shots,
dribbles or expected goals). The thresholds are NOT validated: treat the output as a
list of candidates to check by hand, not as a decision.
"""

from load_data import load_data
from features import add_features
from filters import apply_filters

# Screening thresholds (adjust after looking at the distributions printed below)
MIN_MINUTES = 900
MAX_NPXG = 0.05    # non-penalty xG per 90
MAX_SHOTS = 0.25   # shots per 90
MIN_AERIAL = 2.5   # aerial duels per 90

df = apply_filters(add_features(load_data()))

# 1. How do these stats look for each position? (to judge whether thresholds are sensible)
print(df.groupby("position")[["npxg_p90", "shots_p90", "aerial_p90"]]
      .describe(percentiles=[0.05, 0.5, 0.95]).T.round(2).to_string())

# 2. Candidates, position-specific and based on percentiles of each position group
adv = df.dropna(subset=["npxg_p90", "shots_p90", "aerial_p90"])
adv = adv[adv["minutes"] >= MIN_MINUTES]
wing = adv["position"].isin(["Left Winger", "Right Winger"])
cf = adv["position"] == "Centre-Forward"

# Wingers: aerial duels at or above ~95th percentile of wingers, and shots below the winger median
shots_med = df.loc[df["position"].isin(["Left Winger", "Right Winger"]), "shots_p90"].median()
cand_w = adv[wing & (adv["aerial_p90"] >= 2.0) & (adv["shots_p90"] <= shots_med)]

# Centre-forwards: npxG and shots both at or below the 5th percentile of centre-forwards
cf_all = df[df["position"] == "Centre-Forward"]
q_npxg = cf_all["npxg_p90"].quantile(0.05)
q_shots = cf_all["shots_p90"].quantile(0.05)
cand_cf = adv[cf & (adv["npxg_p90"] <= q_npxg) & (adv["shots_p90"] <= q_shots)]

cols = [c for c in ["player_id", "name", "club", "season", "age", "position", "minutes",
                    "npxg_p90", "shots_p90", "aerial_p90", "tm_mv"] if c in df.columns]
for label, cand in [("Wingers", cand_w), ("Centre-forwards", cand_cf)]:
    print(f"\n{label}: {len(cand)} candidate rows ({cand['player_id'].nunique()} players)")
    print(cand[cols].sort_values(["player_id", "season"]).to_string(index=False))