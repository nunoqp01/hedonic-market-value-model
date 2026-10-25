"""Strength score and tiers for undervalued / overvalued players (two-OLS consensus).

For each player-season on the Extended sample:
  z_base, z_ext  = residual / that model's own residual SD
  under_score    = min(-z_base, -z_ext)   -> how far BELOW the prediction in the
                                             LEAST extreme model (>0 only if both agree)
  over_score     = min(z_base, z_ext)     -> same, ABOVE the prediction

Tier cut-offs (fixed BEFORE looking at any ex-post result, to avoid picking cut-offs that
flatter the findings):
  strong    : score >= 2.0
  moderate  : 1.5 <= score < 2.0
  watch     : 1.0 <= score < 1.5   (for analysis only, not for the final list)
  none      : score < 1.0

The random forest can be added later by extending the min() to three models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED

BINS = [-np.inf, 1.0, 1.5, 2.0, np.inf]
LABELS = ["none", "watch", "moderate", "strong"]

df = apply_filters(add_features(load_data()))
df_ext = df.dropna(subset=ADVANCED)  # Extended sample


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


res_base = fit(FORMULA_BASE, df)
res_ext = fit(FORMULA_EXT, df_ext)

z_base = res_base.resid / res_base.resid.std()
z_ext = res_ext.resid / res_ext.resid.std()

# Work on a copy of the Extended sample; z_base.loc[...] aligns Base z-scores by row label
out = df_ext.copy()
out["z_base"] = z_base.loc[out.index]
out["z_ext"] = z_ext
out["under_score"] = np.minimum(-out["z_base"], -out["z_ext"])
out["over_score"] = np.minimum(out["z_base"], out["z_ext"])

# Tiers: right=False makes the intervals [a, b), so a score of exactly 2.0 is "strong"
out["under_tier"] = pd.cut(out["under_score"], bins=BINS, labels=LABELS, right=False)
out["over_tier"] = pd.cut(out["over_score"], bins=BINS, labels=LABELS, right=False)

print("Undervalued tiers:")
print(out["under_tier"].value_counts().reindex(LABELS[::-1]))
print("\nOvervalued tiers:")
print(out["over_tier"].value_counts().reindex(LABELS[::-1]))

# --- Export ---
# Keep only the identification columns that exist in the dataset
id_cols = [c for c in ["player_id", "player", "name", "squad", "team", "club",
                       "season", "age", "position", "tm_mv"] if c in out.columns]
score_cols = ["z_base", "z_ext", "under_score", "under_tier", "over_score", "over_tier"]

Path("outputs").mkdir(exist_ok=True)
with pd.ExcelWriter("outputs/tiers_ols.xlsx") as writer:
    (out[out["under_score"] >= 1.5][id_cols + score_cols]
     .sort_values("under_score", ascending=False)
     .to_excel(writer, sheet_name="undervalued", index=False))
    (out[out["over_score"] >= 1.5][id_cols + score_cols]
     .sort_values("over_score", ascending=False)
     .to_excel(writer, sheet_name="overvalued", index=False))
    out[id_cols + score_cols].to_excel(writer, sheet_name="all", index=False)  # for ex-post validation

print("\nSaved outputs/tiers_ols.xlsx")
print("\nTop 10 undervalued:")
print(out.sort_values("under_score", ascending=False)[id_cols + ["under_score", "under_tier"]].head(10).to_string(index=False))