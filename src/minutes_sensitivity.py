"""Sensitivity to the minimum-minutes filter (thesis uses 450).

Per-90 metrics are noisy for players with few minutes, so they may produce extreme
residuals. We re-estimate Base and Extended with higher cut-offs and check:
  - how many players the two-OLS consensus flags,
  - how many of the flags at 450 minutes had fewer than 900 minutes,
  - how many of the 450-minute flags survive (among those still in the sample).
"""

import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED

CUTOFFS = [450, 900, 1350]
K = 1.5  # SD threshold, as in the thesis
raw = add_features(load_data())  # loaded once; filtered differently below


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


def run(min_minutes):
    """Estimate both models for one cut-off and return the consensus flags (Series by row label)."""
    d = apply_filters(raw, min_minutes=min_minutes)
    d_ext = d.dropna(subset=ADVANCED)
    res_b, res_e = fit(FORMULA_BASE, d), fit(FORMULA_EXT, d_ext)
    z_b = res_b.resid / res_b.resid.std()
    z_e = res_e.resid / res_e.resid.std()
    z_b_ext = z_b.loc[d_ext.index]
    return {
        "d_ext": d_ext, "res_b": res_b, "res_e": res_e,
        "under": (z_b_ext < -K) & (z_e < -K),
        "over": (z_b_ext > K) & (z_e > K),
    }


runs = {m: run(m) for m in CUTOFFS}
base_flagged = runs[450]["under"][runs[450]["under"]].index  # row labels flagged at 450 minutes
base_minutes = runs[450]["d_ext"].loc[base_flagged, "minutes"]

rows = []
for m, r in runs.items():
    still_in = base_flagged.intersection(r["under"].index)  # baseline flags still in this sample
    rows.append({
        "min minutes": m,
        "N Base": int(r["res_b"].nobs),
        "N Ext": int(r["res_e"].nobs),
        "R2 Base": round(r["res_b"].rsquared, 3),
        "R2 Ext": round(r["res_e"].rsquared, 3),
        "Consensus under": int(r["under"].sum()),
        "Consensus over": int(r["over"].sum()),
        "450-flags still in sample": len(still_in),
        "...of which flagged again": int(r["under"].loc[still_in].sum()),
    })
print(pd.DataFrame(rows).set_index("min minutes").to_string())

print(f"\nUndervalued flags at 450 minutes: {len(base_flagged)}")
print(f"  with fewer than 900 minutes: {int((base_minutes < 900).sum())} "
      f"({100 * (base_minutes < 900).mean():.0f}%)")
print(f"  median minutes of flagged players: {base_minutes.median():.0f}")