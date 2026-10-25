"""Sensitivity of the undervalued/overvalued lists to the SD threshold.

The thesis uses 1.5 SD and says it was not tested against 1.0, 1.25 or 2.0
(Section 5.2.6). This script counts the flags of each OLS model and of the
two-model consensus (Base and Extended OLS) at each threshold.

A player is flagged by a model when the residual is below -k SD (undervalued) or
above +k SD (overvalued), with the SD taken from that model's own residuals.
The random forest is not included yet, so this is a two-OLS consensus, not the
thesis three-model consensus (71 undervalued / 113 overvalued at 1.5 SD).
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED

THRESHOLDS = [1.0, 1.25, 1.5, 2.0]

df = apply_filters(add_features(load_data()))
df_ext = df.dropna(subset=ADVANCED)  # Extended sample


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


res_base = fit(FORMULA_BASE, df)
res_ext = fit(FORMULA_EXT, df_ext)

# Standardised residuals: each model uses its own residual SD
z_base = res_base.resid / res_base.resid.std()
z_ext = res_ext.resid / res_ext.resid.std()

# The consensus only exists on the Extended sample, so restrict the Base z-scores to it.
# Both Series share the same index (row labels), so comparisons below line up by player-season.
z_base_in_ext = z_base.loc[df_ext.index]

rows = []
for k in THRESHOLDS:
    rows.append({
        "threshold (SD)": k,
        "normal tail %": round(100 * (1 - norm.cdf(k)), 1),  # expected share in one tail if residuals were normal
        "Base under": int((z_base < -k).sum()),
        "Base over": int((z_base > k).sum()),
        "Ext under": int((z_ext < -k).sum()),
        "Ext over": int((z_ext > k).sum()),
        "Consensus under": int(((z_base_in_ext < -k) & (z_ext < -k)).sum()),
        "Consensus over": int(((z_base_in_ext > k) & (z_ext > k)).sum()),
    })

table = pd.DataFrame(rows).set_index("threshold (SD)")
print(table)

# Size of each list relative to the 1.5 SD list (how fast the list shrinks as the cut-off tightens)
print("\nList size relative to 1.5 SD (1.00 = same size):")
print((table.drop(columns="normal tail %") / table.loc[1.5].drop("normal tail %")).round(2))

# Save the table for the repository / README
Path("outputs").mkdir(exist_ok=True)
table.to_csv("outputs/threshold_sensitivity.csv")