"""OLS Base model with player-clustered standard errors.

Python version of Section 8 of the Stata do-file.
"""

# Stata: calling helper do-files. Here we import the functions we wrote.
from load_data import load_data
from features import add_features
from filters import apply_filters

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