"""Reproduce the thesis numbers exactly (thesis coding, N = 3,373).

Uses the full sample (max_season_num=6) and the season dummies the thesis used: built from
the raw 'season' column, so Brazilian/Argentine rows (plain years) get all four season
dummies equal to zero. The rest of the project uses the corrected season coding.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED

df = apply_filters(add_features(load_data()), max_season_num=6)  # all 3,373 rows

# Thesis (Stata) season dummies: built from the raw label, no calendar-year mapping
for col, label in [("s2122", "2021/2022"), ("s2223", "2022/2023"),
                   ("s2324", "2023/2024"), ("s2425", "2024/2025")]:
    df[col] = (df["season"] == label).astype(int)
df_ext = df.dropna(subset=ADVANCED)


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


def n_flagged(res, sign):
    """Number of residuals beyond 1.5 SD (sign = -1 undervalued, +1 overvalued)."""
    z = res.resid / res.resid.std()
    return int((z * sign > 1.5).sum())


def bp_fitted(res):
    """Breusch-Pagan against fitted values (Stata's estat hettest default)."""
    return het_breuschpagan(res.resid, sm.add_constant(res.fittedvalues), robust=False)[0]


res_b, res_e = fit(FORMULA_BASE, df), fit(FORMULA_EXT, df_ext)

# VIF of d_argentina in the Base model
model_b = smf.ols(FORMULA_BASE, data=df)
idx = model_b.exog_names.index("d_argentina")
vif_arg = variance_inflation_factor(model_b.exog, idx)

# Wald test LW = RW (Winger split into two dummies)
d = df.copy()
d["d_lw"] = (d["position"] == "Left Winger").astype(int)
d["d_rw"] = (d["position"] == "Right Winger").astype(int)
res_lr = fit(FORMULA_BASE.replace("Winger", "d_lw + d_rw"), d)
p_lw_rw = float(res_lr.wald_test("d_lw = d_rw", use_f=False, scalar=True).pvalue)

rows = [
    ("N Base", res_b.nobs, 3373),
    ("N Extended", res_e.nobs, 2694),
    ("R2 Base", res_b.rsquared, 0.802),
    ("R2 Extended", res_e.rsquared, 0.821),
    ("Undervalued Base (1.5 SD)", n_flagged(res_b, -1), 199),
    ("Undervalued Extended (1.5 SD)", n_flagged(res_e, -1), 165),
    ("Breusch-Pagan Base", bp_fitted(res_b), 41.75),
    ("Breusch-Pagan Extended", bp_fitted(res_e), 45.61),
    ("VIF d_argentina (Base)", vif_arg, 2.46),
    ("Wald LW = RW, p-value", p_lw_rw, 0.973),
]
out = pd.DataFrame(rows, columns=["metric", "python", "thesis"]).set_index("metric")
print(out.round(3))