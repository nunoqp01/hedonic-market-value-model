# Roadmap

Model comparison table (same expanding window, same metrics): previous value, previous value + drift, OLS, OLS + trend, random forest, gradient boosting, lasso/ridge, nearest-neighbour comparables.

## Planned
1. Drift baseline.
2. Random forest (sklearn Pipeline, player-grouped validation).
3. Gradient boosting with native missing values: Extended metrics on the full sample (N = 3,276 instead of dropping rows without advanced metrics).
4. Regularised linear models (ridge, lasso, elastic net) for the multicollinearity among npxG, xA, shots and finishing.
5. SHAP and partial dependence: explain each flagged player; check the age curve.
6. Prediction intervals (quantile regression, conformal) instead of the fixed 1.5 SD cut-off.
7. Nearest-neighbour comparables (neighbours from training seasons only).

## Later
- Player random effects (persistence of residuals), Bayesian hierarchical model, contract length (only if recorded at observation time).