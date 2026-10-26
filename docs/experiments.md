# Experiment log

Each entry: question, what was changed, result, decision. Baseline = replicated thesis model (tag `replication-v1`, once created).

## 1. Is the age curve (age + age_sq) well specified?
- **Change:** residuals by age group (Base); Base plus age-group dummies.
- **Result:** mean residual about 0 for ages 21 to 28; +0.207 (n = 54) up to age 19 and -0.086 (n = 296) at ages 20 to 21, neither significant (SEs 0.128 and 0.046, not clustered). Age-group dummies not significant (smallest p = 0.100); adjusted R2 0.801 vs 0.8014.
- **Decision:** keep age + age_sq. Note: the fitted curve has its minimum around age 29, so within the sample (age < 28) it is decreasing, not an inverted U.

## 2. Does a single 1.5 SD cut-off favour some age groups?
- **Change:** flag rates by age group; flags using each group's own residual SD.
- **Result:** residual SD falls with age (0.94 up to 19, 0.62 at 25 to 28). Single cut-off: 201 undervalued; group-specific: 206; 184 in both. Undervalued rate at ages 20 to 21 falls from 13.5% to 10.8% but stays about twice the 25 to 28 rate (5.7%).
- **Decision:** keep the single cut-off in the main replication; report the group-specific cut-off as a robustness variant. Settle whether the 20 to 21 excess is real with the ex-post validation.

## 3. Replication of thesis diagnostics
- **VIF:** Base age/age_sq 339/337 (thesis about 340); Extended max (excl. age) 4.69 xa_p90 (thesis 4.69).
- **Breusch-Pagan** (fitted values, classical): Extended 45.65 (thesis 45.61). Base 40.15 with harmonised seasons, 41.75 with Stata-style seasons (thesis 41.75). VIF of d_argentina: 2.00 harmonised, 2.46 Stata-style (thesis 2.46).
- **Conclusion:** the thesis used season dummies built from the raw `season` column, which sets all four season dummies to 0 for the 602 Brazilian and Argentine rows.
- **Open:** how Brazil/Argentina calendar years map to seasons (2024 and 2025 both map to 2024/2025 in `season_harmonized`).

## 4. Joint and pooling tests
- Season dummies jointly significant (chi2 37.2 harmonised, 28.6 Stata-style); league dummies jointly significant (chi2 about 920 to 940). All p < 0.0001.
- Wald LW = RW: p = 0.973 (Stata-style, same as thesis), p = 0.927 (harmonised). Pooling into one Winger dummy is justified.

## 5. Sample effect vs variable effect (Base vs Extended)
- **Change:** estimated the Base on the Extended sample (N = 2,694, Argentina removed from the formula) and compared with the Base (N = 3,373) and the Extended.
- **Result:** R2 0.802 (Base) -> 0.809 (Base, Extended sample) -> 0.821 (Extended). About 37% of the R2 gain comes from the sample, 63% from the added variables. The 679 lost rows are Argentina (370) plus peripheral leagues (Brazil 57, Segunda 57, Championship 47, Eredivisie 42); only 4 Big Five rows are lost, so the Big Five share rises from 41.7% to 52.1%. Winger coefficient falls from 0.22 to 0.045 once advanced metrics enter; assists_p90 from 1.73 to 0.49.
- **Decision:** compare Base and Extended on the same sample from now on (B vs C). Report the Extended gain as +0.012 R2, not +0.019.

## 6. Threshold sensitivity (two-OLS consensus)
- **Change:** flags at 1.0, 1.25, 1.5 and 2.0 SD for Base, Extended and their consensus (Extended sample).
- **Result:** consensus undervalued / overvalued = 305/316 (1.0), 184/215 (1.25), 116/150 (1.5), 38/64 (2.0). The 1.5 SD list is about 2.6 times smaller than at 1.0 and three times bigger than at 2.0; about two thirds of the 1.5 SD undervalued cases are below 2 SD in at least one model. Flag counts at 1.5 SD are close to the normal expectation (6.7%), but the right tail is heavier at 2.0 SD (104 vs about 77 expected, Base).
- **Decision:** report the consensus in tiers (strong: both models beyond 2 SD; moderate: 1.5 to 2 SD) instead of one binary list, and test in the ex-post validation whether stronger flags are followed by larger value increases (dose-response). Add the random forest to the consensus once available.

## 7. Minimum-minutes filter (450 vs 900 vs 1,350)
- **Change:** re-estimated Base and Extended with higher minimum minutes; two-OLS consensus at 1.5 SD.
- **Result:** N Base 3,373 / 2,495 / 1,771; R2 Base 0.802 / 0.823 / 0.838 (different samples, not comparable). Of the 116 consensus undervalued at 450 min, 43 (37%) have under 900 min. Among the 73 still in the sample at 900 min, 69 (95%) are flagged again; at 1,350 min, 42 of 51 (82%).
- **Decision:** keep 450 as the main cut-off. Treat flags with under 900 minutes as lower confidence (add as a modifier to the tiers) and check them separately in the ex-post validation. Consider minutes-weighted regression or shrinkage of per-90 metrics later.

## 8. Duplicate player-seasons and Brazil/Argentina season mapping
- **Finding:** 223 rows are in player-season duplicates. 82 European rows are players who changed club mid-season (legitimate, each row has its own club, minutes and value date). Of the 141 Brazil/Argentina rows, about 110 (55 players) were the same player in calendar years 2024 and 2025, both labelled 2024/2025.
- **Change:** map calendar year Y to season Y/(Y+1) (2025 -> 2025/2026).
- **Result:** duplicates fall from 223 to 104 (all two-club cases). 2025/2026 contains only 97 Brazil/Argentina rows.
- **Decision:** exclude the 97 rows from the expanding window and OLS until European 2025/26 data is added; keep the original coding in the tagged thesis replication (`replication-v1`).

## 9. Thesis replication script
- **Change:** `src/replicate_thesis.py` reproduces the thesis with its original season coding (N = 3,373, dummies from the raw `season` label).
- **Result:** N, R2 (Base 0.802, Extended 0.821), undervalued counts (199 and 165), Breusch-Pagan (41.75, 45.61), VIF of d_argentina (2.46) and Wald LW = RW (p = 0.973) all match the thesis.
- **Note:** experiments 1 to 7 above were run with the earlier season coding (N = 3,373). They are rerun on the corrected sample (N = 3,276) in the next step; the numbers may differ slightly.

## 10. Do the consensus lists change with the corrected season coding?
- **Change:** two-OLS consensus (1.5 SD) with the thesis coding (N = 3,373) vs the corrected coding (N = 3,276).
- **Result:** undervalued 119 vs 116, 115 in both (97% of the thesis list kept). Overvalued 153 vs 146, 141 in both (92%). The differences are Brazilian rows (2025 rows leave the sample; 2021 rows get a proper season dummy) and players close to the 1.5 SD cut-off.
- **Decision:** adopt the corrected coding as the project default; the thesis lists are robust to it.

## 11. Random forest in the expanding window (untuned)
- **Change:** `src/expanding_window.py`: RF (300 trees, min_samples_leaf=3, max_features=0.5, median imputer in a Pipeline) on the Base features without season information, and a variant fitted on ln value minus the OLS time trend. "Trend" = one coefficient on `season_num` (constant growth in ln value per season) instead of season dummies, which cannot be estimated for an unseen test season. "Drift" = mean year-on-year change in ln value, computed from seasons before the test season only.
- **Result (pooled test seasons 3 to 5, N = 2,083):** OLS + trend is the best hedonic model (RMSE 0.691, R2 0.792); RF 0.719 / 0.775; RF + trend 0.708 / 0.782. On returning players RF + trend ties OLS + trend on RMSE (0.689 vs 0.690) with a lower median % error (37.8% vs 41.6%); previous value and previous value + drift remain best (0.644 and 0.620). On new players the trend hurts (bias -0.175) and plain OLS is best (RMSE 0.674).
- **Reading:** returning players carry unobserved quality that the hedonic features miss (bias +0.16 even with the trend); new players are priced below the trend. Differences of 0.01 to 0.02 RMSE are probably within noise (three test seasons). The thesis used season dummies and no trend, so its expanding-window RF could not extrapolate market growth.
- **Next:** European-only subset to compare with Table 34 of the thesis; hyperparameter tuning with player-grouped validation inside the training window; out-of-time residuals and flags.

## 12. Hyperparameter tuning of the forest with player-grouped CV
- **Change:** `tune_forest` in `src/expanding_window.py`: GridSearchCV with GroupKFold (5 folds, grouped by player) inside each training window, so the test season is never used for tuning. First grid (min_samples_leaf in {3, 5, 10, 20}, max_features in {0.3, 0.5, 0.8}) picked the smallest value of both parameters in every season (grid edge); second grid: min_samples_leaf in {1, 2, 3, 5}, max_features in {0.1, 0.2, 0.3, 0.5}.
- **Result:** the tuned forest uses min_samples_leaf = 1 and max_features 0.2 to 0.3 in every window. European leagues only, pooled: RF RMSE 0.706 -> 0.680 (R2 0.782 -> 0.797); RF + trend 0.694 -> 0.667 (R2 0.789 -> 0.805), a tie with OLS + trend (0.667, 0.805). Per-season RF R2 0.785 / 0.809 / 0.799 (thesis Table 34: 0.774 / 0.799 / 0.792). Returning players: RF + trend is the best hedonic model (RMSE 0.667 vs 0.690 OLS + trend) but previous value (0.644) and previous value + drift (0.620) still win. New players: plain OLS best (0.674 vs 0.704 RF); trend variants biased by about -0.17.
- **Caveat:** the test seasons were inspected several times (untuned, narrow grid, wide grid); with three test seasons the tie between RF + trend and OLS + trend is not tested for significance (player-level bootstrap pending).
- **Decision:** keep the tuned RF + trend as the forest benchmark; next, out-of-time residual flags and a player-level bootstrap for the RMSE differences.

## 13. Player-level bootstrap of RMSE differences
- **Change:** `src/bootstrap_rmse.py`: 2,000 resamples of players (all rows of a drawn player included) on the out-of-time predictions; European leagues only; same resample for both methods; difference = RMSE(first) - RMSE(second).
- **Result (95% percentile intervals):** RF + trend vs OLS + trend 0.000 (-0.024, 0.023), a tie; RF + trend vs RF -0.013 (-0.020, -0.005); OLS + trend vs OLS -0.006 (-0.012, -0.000), marginal; new players OLS vs RF -0.037 (-0.070, -0.005). Returning players (1,053 rows, 591 players): RF + trend vs OLS + trend -0.017 (-0.048, 0.010); RF + trend vs previous value +0.026 (-0.032, 0.082); previous value + drift vs RF + trend -0.048 (-0.106, 0.008). All returning-player intervals include zero.
- **Reading:** the earlier statements that RF + trend is the best hedonic model for returning players, and that previous value and drift still win, are not supported at 95%. What holds: the trend helps the forest, forest and OLS with trend tie overall, and plain OLS beats the forest for new players.
- **Caveats:** the resampling is over players, not seasons, so the uncertainty from having only three test seasons is not captured; seven comparisons were made without correction.

## 14. Out-of-time flags and ex-post check
- **Change:** `src/oot_flags.py`: standardised out-of-time residuals (OLS + trend, RF + trend; SD of each test season's own residuals), consensus flag below -1.5 SD in both models, strong tier below -2 SD in both. Ex post: change in ln value from t to t+1, European leagues only, seasons 3 and 4 (no European 2025/26 data yet).
- **Result:** consensus flags 22 / 19 / 17 per test season (about 3% to 4% of rows); single-model counts 25 to 33. Share still observed next season: strong 29%, moderate 52%, others 62%. Change in ln value next season: moderate +0.465 (n = 14, 50% rise) vs others +0.167 (n = 680, 50% rise); strong n = 4. Dose-response, change on mean z with season effect and player-clustered SE: -0.128 (SE 0.026, p < 0.001, N = 698).
- **Reading:** tiers are too small to interpret. The continuous result says lower residuals are followed by larger rises, but cannot separate unrecognised quality from mean reversion of noisy, sticky Transfermarkt values. Attrition is much higher for flagged players, so the comparison is on a selected group.
- **Next:** diagnose who disappears (age cap, minutes, leaving the sample); test the dose-response with lagged value change as control; add European 2025/26 for a longer ex-post window.

## 15. Attrition in the ex-post check
- **Finding:** `Base.xlsx` is already filtered (minutes >= 450, age < 28, market value >= 100,000), so players who fail these limits the next season are simply absent; no other reason can be recovered. Players aged 27+ in season t almost never reappear (0.7%) because of the age rule; below 27 the share observed next season is 61% to 73%.
- **Result:** flagged players are over-represented among the 27+ (strong 4/14, moderate 5/27, others 141/1,092). Restricting to age < 27 at t: share observed strong 4/10 (40%), moderate 14/22 (64%), others about 71%. Strong vs others one-sided binomial p about 0.04 (post hoc, n = 10); moderate p about 0.28.
- **Reading:** most of the extra attrition of flagged players is mechanical (age). A weak residual signal remains for the strong tier (tiny n); the reasons are not recoverable from the file.
- **Decision:** redo the ex-post tiers and the dose-response restricted to age < 27 at t; do not present ex-post results as evidence of success until a better outcome source exists (e.g. next-season Transfermarkt values for players who left the sample).

## 16. Does the dose-response survive a control for value momentum?
- **Change:** in `src/oot_flags.py`, added the change in ln value from t-1 to t as a control (players with a previous season; same sample for both regressions; European leagues, age < 27 at t, seasons 3 and 4).
- **Result:** coefficient of the mean out-of-time z on next-season change in ln value: -0.103 (SE 0.031) without the control, -0.109 (SE 0.031) with it (N = 450). The control itself is +0.055 (SE 0.039, p = 0.15): momentum, not reversion.
- **Reading:** the predictive power of the residual does not reduce to rebound after a recent fall. A one-SD lower residual goes with about 0.10 more ln-value growth the next season (average yearly change about 0.17). This is a partial control only: noise and inertia in Transfermarkt values, attrition and having only two outcome seasons remain open.
- **Next:** extend the outcome window with European 2025/26 data; consider collecting next-season values for players who left the sample.

## 17. How much of market value does performance explain?
- **Change:** `src/performance_only.py`: four nested specifications on the Extended sample (N = 2,655 rows): 0 age and position only; A performance metrics (assists, npxG, xA, shots, dribbles, key passes, aerial, finishing) plus age and position; B adds league dummies; C is the full Extended (adds minutes, senior and youth caps, cumulative fees). OLS and random forest, both with a linear time trend, expanding window, test seasons 3 to 5 (N test = 1,560).
- **Result (out-of-time R2, OLS + trend):** 0.000, 0.302, 0.668, 0.807 for 0, A, B, C. In-sample R2: 0.023, 0.372, 0.692, 0.822. The forest ties OLS in A (0.305) and is worse in B (0.618) and C (0.782); in model 0 it is below zero (-0.452), an artefact of fixed untuned settings with two variables.
- **Reading:** age and position alone explain nothing in this sample (age < 28); performance metrics explain about 30% out of time, league adds a lot (the largest block), and the remaining status variables add about 14 points. Increments depend on the order in which blocks are added, since league and performance are correlated. Performance-based value is a weak but real signal; market value is a poor definition of "quality".
- **Next:** a separate performance score (percentile ranks by position and league) as a quality measure; cross it with the market-based expected value.

## 18. Permutation importance of the random forest, out of time
- **Change:** `src/rf_importance.py`: forest (300 trees, min_samples_leaf = 1, max_features = 0.3) trained on seasons 1 to 4, permutation importance (30 shuffles per variable) measured on season 5 as the increase in RMSE (log).
- **Result:** baseline out-of-time RMSE 0.693. Largest increases: ln_fees_cum 0.455, minutes 0.170, goals_p90 0.099, d_premier 0.088, int_A 0.074, age 0.063, d_laliga 0.060, assists_p90 0.058. Normalised to the maximum (as in thesis Table 35): int_A 0.162, d_premier 0.194, int_youth 0.102, ln_fees_cum 1.000 (thesis: 1.000, 0.769, 0.455, 0.309).
- **Reading:** out of time the forest relies mostly on cumulative past fees and playing time; int_A, first in the thesis table, is fifth. The measures are not necessarily comparable (the thesis does not say whether its importance is impurity-based or permutation-based, and its model is pooled and untuned). League effects are split over several dummies, so each looks small; a grouped permutation would show the joint weight of league. Heavy reliance on past fees suggests "undervalued" may partly mean "worth less than what was paid before".
- **Next:** compute impurity-based importance on the same forest to separate the effect of the measure from that of the model; test flags without ln_fees_cum.