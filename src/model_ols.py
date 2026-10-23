"""OLS Base model with player-clustered standard errors.

Python version of Section 8 of the Stata do-file.
"""

# Stata: calling helper do-files. Here we import the functions we wrote.
from load_data import load_data
from features import add_features
from filters import apply_filters
# pandas is needed now to build the comparison table.
import pandas as pd
# numpy gives us exp(), the inverse of the log.
import numpy as np

# Stata: use data, then Section 4 (gen) and Section 5 (drop if), in that order.
df = apply_filters(add_features(load_data()))

# Quick check: should print (3373, 47), the same N as the Base model.
print(df.shape)

# At the top, with the other imports.
# Stata: loading the regression engine. smf.ols is the equivalent of "reg".
import statsmodels.formula.api as smf

# Model formula, as text. Stata: reg ln_mv goals_p90 assists_p90 ...
# "~" separates the dependent variable from the regressors, and regressors
# are joined with "+" instead of spaces. Season and league dummies are
# already columns in df, so they go in like any other variable.
# Reference categories (left out): 2020/2021 season and Liga Portugal.
formula = (
    "ln_mv ~ goals_p90 + assists_p90 + minutes + age + age_sq"
    " + Winger + int_A + int_youth + ln_fees_cum"
    " + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_argentina + d_brasileirao"
)

# Define the model (nothing is estimated yet).
model = smf.ols(formula, data=df)

# Estimate it. Stata: , vce(cluster player_id)
# Standard errors are clustered by player, because the same player
# appears in several seasons.
result = model.fit(cov_type="cluster", cov_kwds={"groups": df["player_id"]})

# Stata: the regression output table.
print(result.summary())

# Stata: outreg2 / export excel. Here the script saves its own outputs.
# Path is a tool for file paths. mkdir creates the folder if it does not exist yet.
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Full regression table as plain text (what print(result.summary()) shows).
(OUTPUT_DIR / "ols_base_summary.txt").write_text(result.summary().as_text())

# Coefficients and clustered standard errors as a CSV, easy to compare with Stata.
# result.params = coefficients (Stata: e(b)); result.bse = standard errors.
coef_table = result.params.to_frame("coef")
coef_table["std_err"] = result.bse
coef_table.to_csv(OUTPUT_DIR / "ols_base_coefficients.csv")

# ---------------------------------------------------------------------
# Replication check against the Stata table (tabela_hedonic.doc)
# ---------------------------------------------------------------------
# In the Stata do-file the season dummies are built from the raw "season"
# column (season == "2021/2022", etc.). In that column the Brazilian and
# Argentine leagues use plain years ("2021" to "2025"), so those rows get
# all four season dummies equal to 0, and Stata treats them as if they
# belonged to the reference season (2020/2021).
# Here we rebuild the dummies the same way, to see whether our code
# reproduces the Stata coefficients. The main model above is unchanged
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

# Same formula, same clustered standard errors, different season dummies.
result_stata = smf.ols(formula, data=df_stata).fit(
    cov_type="cluster", cov_kwds={"groups": df_stata["player_id"]}
)

# Base model coefficients from the Stata table. The table shows only
# 3 or 4 significant figures, so small differences are rounding.
stata_coefs = {
    "Intercept": 20.82, "goals_p90": 1.731, "assists_p90": 1.642,
    "minutes": 0.000525, "age": -0.602, "age_sq": 0.0103,
    "Winger": 0.226, "int_A": 0.305, "int_youth": 0.200,
    "ln_fees_cum": 0.353, "s2122": 0.0226, "s2223": 0.113,
    "s2324": 0.123, "s2425": 0.193, "d_premier": 1.422,
    "d_laliga": 1.131, "d_seriea": 0.850, "d_bundesliga": 1.003,
    "d_ligue1": 0.796, "d_eredivisie": -0.284, "d_championship": 0.0638,
    "d_segunda": -0.388, "d_argentina": 0.0185, "d_brasileirao": 0.608,
}

# Build the comparison table. Assigning a Series to a column aligns the
# rows by name (like merging on the variable name).
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
# Stata: predict ln_mv_hat, xb
# The fitted value is what the model predicts for ln(market value).
df["ln_mv_hat"] = result.fittedvalues

# Stata: predict resid_base, residuals
# Residual = actual ln(market value) minus predicted. A negative residual
# means the player is valued below what the model predicts.
df["resid_base"] = result.resid

# Stata: gen mv_predicted = exp(ln_mv_hat)
# Back from logs to euros. Note: exp of a log prediction tends to
# underestimate the average value, but we follow the do-file here.
df["mv_predicted"] = np.exp(df["ln_mv_hat"])

# Sanity checks (all three should look right before we trust the residuals).
# 1) With an intercept, OLS residuals average to (almost exactly) zero.
print("Mean of residuals:", round(df["resid_base"].mean(), 6))
# 2) Residual standard deviation. Stata: summarize resid_base (uses n-1,
#    and so does pandas by default).
print("SD of residuals:", round(df["resid_base"].std(), 4))
# 3) The squared correlation between actual and fitted equals the R-squared.
#    Stata: corr ln_mv ln_mv_hat, then r(rho)^2.
print("Squared correlation:", round(df["ln_mv"].corr(df["ln_mv_hat"]) ** 2, 4))

# ---------------------------------------------------------------------
# Undervalued and overvalued players (rule from the do-file)
# ---------------------------------------------------------------------
# Stata: summarize resid_base, then scalar sd_base = r(sd)
sd_resid = df["resid_base"].std()

# Stata: gen undervalued_base = (resid_base < -1.5 * sd_base)
# The comparison gives True/False for each row; astype(int) turns it into 1/0.
df["undervalued_base"] = (df["resid_base"] < -1.5 * sd_resid).astype(int)
df["overvalued_base"] = (df["resid_base"] > 1.5 * sd_resid).astype(int)

# Stata: count if undervalued_base == 1
# Summing a column of 1s and 0s counts the 1s.
print("Undervalued (harmonized seasons):", df["undervalued_base"].sum())
print("Overvalued (harmonized seasons):", df["overvalued_base"].sum())

# Same rule applied to the Stata-style model. This is the count you should
# find in Stata, so it is the one to compare with your Stata output.
sd_stata = result_stata.resid.std()
under_stata = result_stata.resid < -1.5 * sd_stata
over_stata = result_stata.resid > 1.5 * sd_stata
print("Undervalued (Stata-style seasons):", under_stata.sum())
print("Overvalued (Stata-style seasons):", over_stata.sum())

# How many players are flagged by both versions? "&" means "and".
both = (df["undervalued_base"] == 1) & under_stata
print("Flagged undervalued in both versions:", both.sum())