# Hedonic model of forward market values: Stata to Python, and out-of-time validation

A Python replication of the hedonic market-value model from my master's dissertation (University of Aveiro, 2026), originally built in Stata, extended with an out-of-time validation that the dissertation left as future work.

## What this project does

The model explains the log of the last Transfermarkt market value of the season for forwards (centre-forwards and wingers) in European and peripheral leagues, using performance, age, league and career variables. Players whose value sits well below what the model expects (residual below -1.5 standard deviations) are flagged as undervalued.

1. **Replication.** The Python code reproduces the Stata results of the dissertation (sample sizes, R2, flag counts, diagnostics).
2. **Corrections.** Re-examining the pipeline showed that the season coding in the original estimation treated Brazilian and Argentine rows differently from European ones, which leaks future information into the rolling-window validation. The corrected coding keeps the lists of flagged players almost unchanged (97% of undervalued and 92% of overvalued cases kept).
3. **Out-of-time validation.** Expanding window: train on seasons before t, predict season t, with baselines (previous value, previous value plus drift), OLS with and without a time trend, and a random forest whose hyperparameters are tuned inside each training window using player-grouped cross-validation. Differences between methods are tested with a player-level bootstrap.

## Main results

European leagues only, test seasons 2022/23 to 2024/25 (N = 1,697 player-seasons):

| Method | RMSE (log) | R2 | Median abs. % error |
|---|---|---|---|
| OLS + time trend | 0.667 | 0.805 | 41.8 |
| Random forest + time trend (tuned) | 0.667 | 0.805 | 40.2 |
| Random forest (tuned) | 0.680 | 0.797 | 39.4 |

Baselines apply only to players with a value in the previous season (N = 1,053): previous value 0.636 / 0.803 / 30.0; previous value plus average drift 0.614 / 0.816 / 34.3.

- Out-of-time R2 is about 0.80, close to the in-sample figure, so the valuation structure is consistent across seasons.
- A tuned random forest with a time trend ties the linear model with a trend (RMSE difference 0.000, 95% player-level bootstrap interval -0.024 to 0.023). It does not improve prediction over OLS.
- For returning players, previous value (with or without drift) is numerically better than the models, but the difference is within sampling noise; for new players plain OLS is significantly better than the random forest.
- The bootstrap resamples players, not seasons, so it does not capture the uncertainty from having only three test seasons.
- Market values are estimates, not transaction prices, and the median error is about 40%. The model is useful to rank and screen players, not to price a transfer.

Full log of experiments and decisions: [`docs/experiments.md`](docs/experiments.md). Planned next steps: [`docs/ideas.md`](docs/ideas.md).

## Repository structure

`src/`
- **Data and features:** `load_data.py`, `features.py`, `filters.py`, `formulas.py` (model formulas shared by all scripts).
- **Data checks:** `check_features.py`, `check_filters.py`, `check_seasons.py`, `check_duplicates.py`.
- **Models:** `model_ols_base.py`, `model_ols_extended.py`, `diagnostics_ols.py`.
- **Thesis replication:** `replicate_thesis.py` (original season coding), `compare_codings.py` (original vs corrected coding), `compare_samples.py` (Base vs Extended on the same sample).
- **Sensitivity analyses:** `threshold_sensitivity.py`, `minutes_sensitivity.py`, `screen_non_forwards.py`.
- **Validation:** `expanding_window.py` (baselines, OLS and random forest, expanding window), `bootstrap_rmse.py` (player-level bootstrap of RMSE differences).
- **Flags and inspection:** `tiers.py` (strength tiers), `inspect_player.py`.

`docs/`: experiment log (`experiments.md`) and ideas (`ideas.md`). `outputs/`: generated tables (not tracked).

Tag `replication-v1` marks the version that reproduces the dissertation with its original coding.

## Data

The data are not included. They combine FBref, FootyStats and Transfermarkt, they are the basis of academic papers in preparation, and the sources have their own terms of use. The code expects a file at `data/raw/Base.xlsx` with a sheet called `Data`, so it cannot be run without access to the data.

## How to run

Requires Python 3.15 (tested with 3.15.1). Packages are pinned in `requirements.txt`: pandas, numpy, openpyxl, statsmodels and scikit-learn.

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/expanding_window.py
python src/bootstrap_rmse.py
```

Windows (PowerShell):

```
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/expanding_window.py
python src/bootstrap_rmse.py
```

## Dissertation

"Financial Sustainability and Player Trading in European Football" (MSc dissertation, University of Aveiro, 2026). Not public: it is being adapted into two academic articles. A short write-up of the results: [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7518582515645726721/).