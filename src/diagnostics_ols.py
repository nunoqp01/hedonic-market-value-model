"""OLS diagnostics for the hedonic market value model (Stata equivalents in brackets).

1. Variance inflation factors, Base and Extended           [estat vif]
2. Residuals by age group (is the age curve well specified?)
3. Flag rates by age group, single vs group-specific SD cut-off
4. Breusch-Pagan heteroskedasticity test                   [estat hettest]
5. Check that the thesis used "Stata-style" season dummies
6. Joint Wald tests (seasons, leagues)                     [testparm]
7. Wald test LW = RW                                       [test d_lw = d_rw]
"""

# --- Imports ---
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from load_data import load_data
from features import add_features
from filters import apply_filters

# --- Data and formulas ---
df = apply_filters(add_features(load_data()))

FORMULA_BASE = (
    "ln_mv ~ goals_p90 + assists_p90 + minutes + age + age_sq + Winger"
    " + int_A + int_youth + ln_fees_cum + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_argentina + d_brasileirao"
)

ADVANCED = ["npxg_p90", "xa_p90", "shots_p90", "dribbles_p90",
            "keypasses_p90", "aerial_p90", "finishing"]
df_ext = df.dropna(subset=ADVANCED)  # Extended sample (Argentina drops out here)

FORMULA_EXT = (
    "ln_mv ~ assists_p90 + minutes + age + age_sq + Winger + int_A + int_youth"
    " + ln_fees_cum + npxg_p90 + xa_p90 + shots_p90 + dribbles_p90"
    " + keypasses_p90 + aerial_p90 + finishing + s2122 + s2223 + s2324 + s2425"
    " + d_premier + d_laliga + d_seriea + d_bundesliga + d_ligue1"
    " + d_eredivisie + d_championship + d_segunda + d_brasileirao"
)


def vif_table(model):
    """VIF of each regressor (like estat vif, without the constant).

    model.exog is the X matrix statsmodels built from the formula (constant and
    dummies included); model.exog_names holds the column names.
    """
    X = model.exog
    out = pd.DataFrame({
        "variable": model.exog_names,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
    })
    out = out[out["variable"] != "Intercept"]
    return out.sort_values("VIF", ascending=False).round(2)


# --- 1. VIF ---
model_base = smf.ols(FORMULA_BASE, data=df)
model_ext = smf.ols(FORMULA_EXT, data=df_ext)

print("VIF Base (thesis: age and age_sq ~340; next highest 2.46, d_argentina)")
print(vif_table(model_base).to_string(index=False))

print("\nVIF Extended (thesis: highest excluding age/age_sq 4.69, xa_p90)")
print(vif_table(model_ext).to_string(index=False))

# --- 2. Mean residual by age group (Base model) ---
# If the age + age_sq curve is well specified, the mean residual in each group
# should be close to zero. Means far from zero indicate systematic error by age.
result_base = model_base.fit(cov_type="cluster", cov_kwds={"groups": df["player_id"]})
df["resid"] = result_base.resid

df["age_group"] = pd.cut(df["age"], bins=[15, 19, 21, 23, 25, 28])
print("\nMean residual by age group (Base)")
print(df.groupby("age_group", observed=True)["resid"].agg(["mean", "count"]).round(3))

# Residual SD and standard error of the mean by group (SE ignores player clustering)
print("\nResidual SD (Base):", round(df["resid"].std(), 3))
stats = df.groupby("age_group", observed=True)["resid"].agg(["mean", "std", "count"])
stats["se_mean"] = stats["std"] / stats["count"] ** 0.5
print(stats.round(3))

# Base model plus age-group dummies: if these are significant, the parabola is not enough.
# C(age_group) creates one dummy per group (the first group is the reference).
model_age = smf.ols(FORMULA_BASE + " + C(age_group)", data=df)
result_age = model_age.fit(cov_type="cluster", cov_kwds={"groups": df["player_id"]})
print("\nBase + age-group dummies (coefficients of the age groups)")
print(result_age.summary2().tables[1].filter(like="age_group", axis=0).round(3))
print("Adjusted R2 Base:", round(result_base.rsquared_adj, 4),
      "| Base + age groups:", round(result_age.rsquared_adj, 4))

# --- 3. Flag rates by age group ---
# If young players are flagged much more often, a single 1.5 SD cut-off favours
# the groups with more dispersed residuals.
sd = df["resid"].std()
df["undervalued"] = df["resid"] < -1.5 * sd
df["overvalued"] = df["resid"] > 1.5 * sd
print("\nFlag rates by age group (Base, single SD cut-off)")
print(df.groupby("age_group", observed=True)[["undervalued", "overvalued"]].mean().round(3))

# Same flags, but each age group uses its own residual SD
sd_group = df.groupby("age_group", observed=True)["resid"].transform("std")
df["z_group"] = df["resid"] / sd_group
df["undervalued_g"] = df["z_group"] < -1.5
df["overvalued_g"] = df["z_group"] > 1.5

print("\nUndervalued: single cut-off =", int(df["undervalued"].sum()),
      "| group-specific cut-off =", int(df["undervalued_g"].sum()),
      "| in both =", int((df["undervalued"] & df["undervalued_g"]).sum()))
print("\nFlag rates with group-specific SD")
print(df.groupby("age_group", observed=True)[["undervalued_g", "overvalued_g"]].mean().round(3))

# --- 4. Breusch-Pagan (thesis: 41.75 Base, 45.61 Extended) ---
# The test regresses the squared residual on a set of variables and tests for a relation.
# Variants: explanatory variables = fitted values (what Stata's estat hettest does by
# default) or all regressors; classical (robust=False) or robust (robust=True).
result_ext = model_ext.fit(cov_type="cluster", cov_kwds={"groups": df_ext["player_id"]})

for name, res, mod in [("Base", result_base, model_base), ("Extended", result_ext, model_ext)]:
    fitted = sm.add_constant(res.fittedvalues)  # constant + fitted values
    print(f"\nBreusch-Pagan {name}")
    for desc, exog, robust in [
        ("fitted values, classical", fitted, False),
        ("fitted values, robust", fitted, True),
        ("all regressors, classical", mod.exog, False),
        ("all regressors, robust", mod.exog, True),
    ]:
        lm, lm_p, _, _ = het_breuschpagan(res.resid, exog, robust=robust)
        print(f"  {desc}: chi2 = {lm:.2f}, p = {lm_p:.4f}")

# --- 5. "Stata-style" season dummies (as used in the thesis) ---
# Built from the raw 'season' column: Brazilian and Argentine players, whose 'season'
# is a plain year (e.g. "2023"), get all four season dummies equal to zero.
# If this reproduces the thesis numbers, the thesis used this coding.
df_stata = df.copy()
for col, label in [("s2122", "2021/2022"), ("s2223", "2022/2023"),
                   ("s2324", "2023/2024"), ("s2425", "2024/2025")]:
    df_stata[col] = (df_stata["season"] == label).astype(int)
print("\nObservations per season dummy (Stata-style):")
print(df_stata[["s2122", "s2223", "s2324", "s2425"]].sum())

model_st = smf.ols(FORMULA_BASE, data=df_stata)
res_st = model_st.fit(cov_type="cluster", cov_kwds={"groups": df_stata["player_id"]})

vif_st = vif_table(model_st)
print("\nVIF d_argentina (thesis 2.46):",
      float(vif_st.loc[vif_st["variable"] == "d_argentina", "VIF"].iloc[0]))

lm, lm_p, _, _ = het_breuschpagan(res_st.resid, sm.add_constant(res_st.fittedvalues), robust=False)
print(f"Breusch-Pagan Base, Stata-style dummies (thesis 41.75): chi2 = {lm:.2f}")

# --- 6. Joint Wald tests (like Stata's testparm) ---
# H0: all coefficients in the group are zero. Rejecting H0 means the group matters.
# Note: after vce(cluster), Stata's testparm reports an F; here we use chi2
# (use_f=False), so p-values will be close but not identical.
SEASONS = ["s2122", "s2223", "s2324", "s2425"]
LEAGUES = ["d_premier", "d_laliga", "d_seriea", "d_bundesliga", "d_ligue1",
           "d_eredivisie", "d_championship", "d_segunda", "d_argentina", "d_brasileirao"]


def wald_joint(res, names):
    """Wald test that all coefficients in 'names' are zero."""
    restriction = ", ".join(f"{n} = 0" for n in names)
    w = res.wald_test(restriction, use_f=False, scalar=True)
    return float(w.statistic), float(w.pvalue)


print("\nJoint Wald tests (Base)")
for label, res in [("harmonised", result_base), ("Stata-style", res_st)]:
    for group, names in [("seasons", SEASONS), ("leagues", LEAGUES)]:
        stat, p = wald_joint(res, names)
        print(f"  {label:12s} {group:8s} chi2 = {stat:8.2f}, p = {p:.4f}")

# --- 7. Wald test LW = RW (thesis reports p = 0.973) ---
# Split the Winger dummy into left and right, then test whether the coefficients are equal.
# If H0 is not rejected, pooling LW and RW into one dummy is justified.
print("\nValues of the position column:")
print(df["position"].value_counts())

# These must match the labels in the 'position' column exactly
LW_LABEL = "Left Winger"
RW_LABEL = "Right Winger"

formula_lw_rw = FORMULA_BASE.replace("Winger", "d_lw + d_rw")  # replace Winger by two dummies

for label, data in [("Stata-style", df_stata), ("harmonised", df)]:
    d = data.copy()
    d["d_lw"] = (d["position"] == LW_LABEL).astype(int)
    d["d_rw"] = (d["position"] == RW_LABEL).astype(int)
    # Check: d_lw + d_rw must equal Winger, otherwise the labels are wrong
    assert (d["d_lw"] + d["d_rw"] == d["Winger"]).all(), "LW/RW labels do not match Winger"
    res = smf.ols(formula_lw_rw, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["player_id"]})
    w = res.wald_test("d_lw = d_rw", use_f=False, scalar=True)
    print(f"LW = RW ({label}): coef LW = {res.params['d_lw']:.3f}, coef RW = {res.params['d_rw']:.3f}, "
          f"chi2 = {float(w.statistic):.3f}, p = {float(w.pvalue):.3f}")