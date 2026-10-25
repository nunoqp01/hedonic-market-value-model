"""Sample effect vs variable effect: Base vs Extended on the same observations.

The Extended model differs from the Base in two ways at once: it adds advanced
metrics, and it loses the rows where those metrics are missing. Comparing them
directly mixes both. Three regressions separate them:

  A. Base on its full sample                 (N = 3373)
  B. Base on the Extended sample             (same rows as C)
  C. Extended on the Extended sample

A vs B = effect of the sample.   B vs C = effect of the added variables.
"""

import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED, LEAGUES

df = apply_filters(add_features(load_data()))
df_ext = df.dropna(subset=ADVANCED)  # Extended sample

# --- Which rows does the Extended sample lose? ---
lost = df.loc[~df.index.isin(df_ext.index)]
print(f"Rows in Base sample: {len(df)} | in Extended sample: {len(df_ext)} | lost: {len(lost)}")
print("\nLost rows by league dummy (compare with the full sample):")
print(pd.DataFrame({"lost": lost[LEAGUES].sum(), "full sample": df[LEAGUES].sum()}))

# --- Three regressions ---
# d_argentina is removed from B: there are no Argentine rows in the Extended sample,
# so the dummy would be always zero (perfect collinearity with the constant).
formula_base_ext = FORMULA_BASE.replace(" + d_argentina", "")


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


res_a = fit(FORMULA_BASE, df)
res_b = fit(formula_base_ext, df_ext)
res_c = fit(FORMULA_EXT, df_ext)

# --- Summary ---
summary = pd.DataFrame({
    "A: Base, full": [res_a.nobs, res_a.rsquared, res_a.rsquared_adj, res_a.resid.std()],
    "B: Base, ext. sample": [res_b.nobs, res_b.rsquared, res_b.rsquared_adj, res_b.resid.std()],
    "C: Extended": [res_c.nobs, res_c.rsquared, res_c.rsquared_adj, res_c.resid.std()],
}, index=["N", "R2", "R2 adj", "Residual SD"]).round(3)
print("\n", summary)

# --- Coefficients of variables present in all three models ---
coefs = pd.DataFrame({"A": res_a.params, "B": res_b.params, "C": res_c.params})
common = ["assists_p90", "minutes", "age", "age_sq", "Winger", "int_A", "int_youth",
          "ln_fees_cum", "s2122", "s2223", "s2324", "s2425"] + [l for l in LEAGUES if l != "d_argentina"]
print("\nCoefficients (common variables):")
print(coefs.loc[common].round(3))