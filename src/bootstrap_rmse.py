"""Player-level bootstrap for RMSE differences between methods (out-of-time predictions).

Reads outputs/expanding_window_predictions.csv. In each replication we resample PLAYERS with
replacement (all rows of a drawn player come along), so repeated seasons of the same player are
not treated as independent. The same resample is used for both methods in a comparison.
Difference = RMSE(first method) - RMSE(second method): negative means the first is better.
"""

import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 2000

pred = pd.read_csv("outputs/expanding_window_predictions.csv")
pred["european"] = (pred["d_argentina"] + pred["d_brasileirao"]) == 0

rng = np.random.default_rng(SEED)


def boot_diff(data, col_a, col_b):
    """Bootstrap the RMSE difference between two prediction columns, resampling players."""
    d = data.dropna(subset=[col_a, col_b]).copy()
    d["se_a"] = (d["ln_mv"] - d[col_a]) ** 2
    d["se_b"] = (d["ln_mv"] - d[col_b]) ** 2
    g = d.groupby("player_id").agg(sse_a=("se_a", "sum"), sse_b=("se_b", "sum"), n=("se_a", "size"))
    sse_a, sse_b, n = g["sse_a"].to_numpy(), g["sse_b"].to_numpy(), g["n"].to_numpy()

    n_players = len(g)
    idx = rng.integers(0, n_players, size=(N_BOOT, n_players))  # N_BOOT resamples of player indices
    rmse_a = np.sqrt(sse_a[idx].sum(axis=1) / n[idx].sum(axis=1))
    rmse_b = np.sqrt(sse_b[idx].sum(axis=1) / n[idx].sum(axis=1))
    diff = rmse_a - rmse_b

    point = np.sqrt(sse_a.sum() / n.sum()) - np.sqrt(sse_b.sum() / n.sum())
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"rows": int(n.sum()), "players": n_players,
            "diff (RMSE first - second)": point, "CI 2.5%": low, "CI 97.5%": high,
            "first better in % of resamples": 100 * (diff < 0).mean()}


european = pred[pred["european"]]
returning = european[european["pred_prev"].notna()]
new = european[european["pred_prev"].isna()]

comparisons = [
    ("European, all rows", european, "pred_rf_trend", "pred_trend"),
    ("European, all rows", european, "pred_rf_trend", "pred_rf"),
    ("European, all rows", european, "pred_trend", "pred_ols"),
    ("European, returning", returning, "pred_rf_trend", "pred_trend"),
    ("European, returning", returning, "pred_rf_trend", "pred_prev"),
    ("European, returning", returning, "pred_drift", "pred_rf_trend"),
    ("European, new players", new, "pred_ols", "pred_rf"),
]

rows = []
for subset, data, a, b in comparisons:
    rows.append({"subset": subset, "first": a, "second": b, **boot_diff(data, a, b)})

pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(3).to_string(index=False))