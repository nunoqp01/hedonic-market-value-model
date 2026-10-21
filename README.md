# Hedonic pricing model for player market values (Python)

**Work in progress.** A Python replication of a model first built in Stata. No model has been estimated in Python yet.

## Objective

This repository replicates in Python the hedonic market-value model from my master's dissertation (University of Aveiro, 2026), originally built in Stata. The model covers forwards (centre-forwards and wingers), with the log of the last Transfermarkt market value of the season as the dependent variable. The dissertation estimates OLS and random forest models and flags undervalued players where three models agree.

## Status

**Done**
- `src/load_data.py` reads the Excel file (3,373 rows, 28 columns) and repairs text-encoding errors in player and club names.
- `src/features.py` builds `ln_mv` (log of market value), `age_sq`, `ln_fees_cum` = `ln(1 + cumulative fees / 1,000,000)`, `Winger` (1 for left and right wingers, 0 for centre-forwards) and `player_id`. Checked on sample rows and group summaries.

**To do**
- Season filter
- Season and league dummies
- OLS with cluster-robust standard errors, and diagnostics
- Random forest
- Expanding-window validation
- SQL queries on the same data

## Data

The data are not included. They combine FBref, FootyStats and Transfermarkt, they are the basis of academic papers in preparation, and the sources have their own terms of use. The code expects a file at `data/raw/Base.xlsx` with a sheet called `Data`, so it cannot be run without access to the data.

## How to run

Requires Python 3.14. Currently uses pandas, numpy and openpyxl (see `requirements.txt`). statsmodels and scikit-learn are installed for the models still to come.

[PREENCHER: commands to set up the environment and run the loading and features modules, with the real output]

## Dissertation

"Financial Sustainability and Player Trading in European Football" (MSc dissertation, University of Aveiro, 2026). Not public: it is being adapted into two academic articles. A short write-up of the results: [https://www.linkedin.com/feed/update/urn:li:activity:7518582515645726721/].