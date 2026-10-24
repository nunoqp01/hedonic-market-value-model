"""OLS Extended model with player-clustered standard errors.

Python version of Section 12 of the Stata do-file.
"""

# Standard library, then third-party libraries, then our own modules.
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from load_data import load_data
from features import add_features
from filters import apply_filters

# Stata: use data, then Section 4 (gen) and Section 5 (drop if), in that order.
df = apply_filters(add_features(load_data()))
print("Base sample:", df.shape)

# ---------------------------------------------------------------------
# Extended sample: players with advanced metrics
# ---------------------------------------------------------------------
ADVANCED = [
    "npxg_p90", "xa_p90", "shots_p90", "dribbles_p90",
    "keypasses_p90", "aerial_p90", "finishing",
]

# Stata: misstable summarize. Missing values per advanced metric.
print(df[ADVANCED].isna().sum())

# Stata drops rows with missing regressors on its own inside "reg".
# Here we do it explicitly, so the clusters stay aligned with the data.
df = df.dropna(subset=ADVANCED)

# Should be (2694, 47), the N of the Extended model.
print("Extended sample:", df.shape)

# Which leagues have advanced metrics? Argentina should be absent.
print(df["league"].value_counts())

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
# goals_p90 is replaced by npxg_p90 (multicollinearity, VIF of 18.53).
# d_argentina is left out because no Argentine row has advanced metrics
# (Stata omits it automatically; here we drop it ourselves).
# Reference categories: 2020/2021 season and Liga Portugal.
formula = (
    "ln_mv ~ assists_p90 + minutes + age + age_sq"
    " + Winger + int_A + int_youth + ln_fees_cum"
    " + npxg_p90 + xa_p90 + shots_p90 + dribbles_p90"
    " + keypasses_p90 + aerial_p90 + finishing"
    " + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_brasileirao"
)

# Stata: reg ln_mv ..., vce(cluster player_id)
model = smf.ols(formula, data=df)
result = model.fit(cov_type="cluster", cov_kwds={"groups": df["player_id"]})

print(result.summary())
print("R-squared:", round(result.rsquared, 4))  # Stata table: 0.821

# ---------------------------------------------------------------------
# Save outputs (the outputs/ folder is in .gitignore)
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

(OUTPUT_DIR / "ols_extended_summary.txt").write_text(result.summary().as_text())

coef_table = result.params.to_frame("coef")
coef_table["std_err"] = result.bse
coef_table.to_csv(OUTPUT_DIR / "ols_extended_coefficients.csv")

# ---------------------------------------------------------------------
# Replication check against the Stata table (tabela_hedonic.doc)
# ---------------------------------------------------------------------
# The Stata do-file builds the season dummies from the raw "season" column,
# which uses plain years ("2021" to "2025") for the Brazilian league. Those
# rows get all four season dummies equal to 0 in Stata. We rebuild the
# dummies the same way and compare. The main model above is unchanged
# (it uses season_harmonized).

# Work on a copy so the main df is untouched (Stata: preserve / restore).
df_stata = df.copy()

SEASON_LABELS = {
    "s2122": "2021/2022",
    "s2223": "2022/2023",
    "s2324": "2023/2024",
    "s2425": "2024/2025",
}
for dummy, label in SEASON_LABELS.items():
    df_stata[dummy] = (df_stata["season"] == label).astype(int)

result_stata = smf.ols(formula, data=df_stata).fit(
    cov_type="cluster", cov_kwds={"groups": df_stata["player_id"]}
)

# Extended model coefficients from the Stata table (3 or 4 significant
# figures, so small differences are rounding).
stata_coefs = {
    "Intercept": 21.14, "assists_p90": 0.486, "minutes": 0.000500,
    "age": -0.653, "age_sq": 0.0114, "Winger": 0.0477,
    "int_A": 0.315, "int_youth": 0.185, "ln_fees_cum": 0.292,
    "npxg_p90": 1.845, "xa_p90": 1.594, "shots_p90": 0.259,
    "dribbles_p90": 0.0486, "keypasses_p90": 0.149, "aerial_p90": -0.0318,
    "finishing": 0.924, "s2122": -0.0136, "s2223": 0.0810,
    "s2324": 0.0370, "s2425": 0.167, "d_premier": 1.594,
    "d_laliga": 1.264, "d_seriea": 1.051, "d_bundesliga": 1.100,
    "d_ligue1": 0.918, "d_eredivisie": -0.282, "d_championship": 0.155,
    "d_segunda": -0.319, "d_brasileirao": 0.517,
}

comparison = pd.DataFrame({"stata": pd.Series(stata_coefs)})
comparison["python_stata_style"] = result_stata.params
comparison["python_harmonized"] = result.params
comparison["diff_stata_style"] = comparison["python_stata_style"] - comparison["stata"]
comparison["diff_harmonized"] = comparison["python_harmonized"] - comparison["stata"]

print(comparison.round(5))
print("Largest absolute difference, Stata-style dummies:",
      round(comparison["diff_stata_style"].abs().max(), 4))
print("Largest absolute difference, harmonized dummies:",
      round(comparison["diff_harmonized"].abs().max(), 4))

# ---------------------------------------------------------------------
# Fitted values and residuals (main model, harmonized seasons)
# ---------------------------------------------------------------------
# Stata: predict ln_mv_hat_ext, xb   and   predict resid_extended, residuals
df["ln_mv_hat_ext"] = result.fittedvalues
df["resid_ext"] = result.resid

# Stata: gen mv_predicted_ext = exp(ln_mv_hat_ext)
df["mv_predicted_ext"] = np.exp(df["ln_mv_hat_ext"])

# Sanity checks, as for the Base model.
print("Mean of residuals:", round(df["resid_ext"].mean(), 6))
print("SD of residuals:", round(df["resid_ext"].std(), 4))
print("Squared correlation:", round(df["ln_mv"].corr(df["ln_mv_hat_ext"]) ** 2, 4))

# ---------------------------------------------------------------------
# Undervalued and overvalued players (rule from the do-file)
# ---------------------------------------------------------------------
sd_resid = df["resid_ext"].std()

df["undervalued_ext"] = (df["resid_ext"] < -1.5 * sd_resid).astype(int)
df["overvalued_ext"] = (df["resid_ext"] > 1.5 * sd_resid).astype(int)

print("Undervalued (harmonized seasons):", df["undervalued_ext"].sum())
print("Overvalued (harmonized seasons):", df["overvalued_ext"].sum())

# Same rule for the Stata-style model: this is the count to find in Stata.
sd_stata = result_stata.resid.std()
under_stata = result_stata.resid < -1.5 * sd_stata
over_stata = result_stata.resid > 1.5 * sd_stata
print("Undervalued (Stata-style seasons):", under_stata.sum())
print("Overvalued (Stata-style seasons):", over_stata.sum())

both = (df["undervalued_ext"] == 1) & under_stata
print("Flagged undervalued in both versions:", both.sum())

# ---------------------------------------------------------------------
# Export the list of undervalued players
# ---------------------------------------------------------------------
columns = ["name", "club", "league", "season_harmonized",
           "tm_mv", "mv_predicted_ext", "resid_ext"]

undervalued = df.loc[df["undervalued_ext"] == 1, columns].sort_values("resid_ext")
undervalued.to_excel(OUTPUT_DIR / "undervalued_extended.xlsx", index=False)

print(undervalued.head(10).round(2))