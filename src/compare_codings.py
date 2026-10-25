"""Do the consensus lists change with the corrected season coding?

Builds the two-OLS consensus (1.5 SD) twice: with the thesis season coding (N = 3,373) and
with the corrected coding (N = 3,276), then compares which player-seasons are flagged.
Both samples share the same row labels, so rows are matched by index.
"""

import pandas as pd
import statsmodels.formula.api as smf
from load_data import load_data
from features import add_features
from filters import apply_filters
from formulas import FORMULA_BASE, FORMULA_EXT, ADVANCED

K = 1.5
raw = add_features(load_data())


def fit(formula, data):
    """OLS with standard errors clustered by player."""
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["player_id"]})


def consensus(df):
    """Row labels flagged by both models as undervalued / overvalued."""
    d_ext = df.dropna(subset=ADVANCED)
    rb, re = fit(FORMULA_BASE, df), fit(FORMULA_EXT, d_ext)
    zb = (rb.resid / rb.resid.std()).loc[d_ext.index]
    ze = re.resid / re.resid.std()
    return set(d_ext.index[(zb < -K) & (ze < -K)]), set(d_ext.index[(zb > K) & (ze > K)])


# Thesis coding: full sample, season dummies from the raw label
old = apply_filters(raw.copy(), max_season_num=6)
for col, label in [("s2122", "2021/2022"), ("s2223", "2022/2023"),
                   ("s2324", "2023/2024"), ("s2425", "2024/2025")]:
    old[col] = (old["season"] == label).astype(int)

# Corrected coding: default filters (seasons 1 to 5), dummies from season_clean
new = apply_filters(raw.copy())

for name, i in [("undervalued", 0), ("overvalued", 1)]:
    a, b = consensus(old)[i], consensus(new)[i]
    print(f"\n{name}: thesis coding {len(a)} | corrected {len(b)} | in both {len(a & b)}")
    cols = [c for c in ["name", "club", "season", "age", "tm_mv"] if c in raw.columns]
    print("  only in thesis coding:")
    print(raw.loc[sorted(a - b), cols].to_string())
    print("  only in corrected coding:")
    print(raw.loc[sorted(b - a), cols].to_string())