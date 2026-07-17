# MLB Home Run Predictor

A reproducible baseline that estimates whether each expected starting hitter hits at least one home run in a specified MLB game. Each training observation is one batter-game; the binary target is one when that batter homered at least once.

The model is logistic regression with sigmoid probability calibration. Numeric values are median-imputed and standardized; categorical values are imputed and one-hot encoded. Training, calibration, and test sets are split on whole game dates in chronological order (70% / 15% / 15%). Test metrics are Brier score, log loss, and ROC AUC.

## Leakage policy

Statcast pitches are aggregated to player-game totals. Batter and opposing starting-pitcher rates are expanding cumulative rates shifted by one complete game. A row dated `D` therefore only uses earlier games. No outcome from the predicted game enters its features, and every game date belongs to exactly one evaluation partition.

Features are batter HR/PA, barrel rate, hard-hit rate, fly-ball rate and handedness; starter HR allowed/BF, barrel rate allowed and handedness; platoon matchup; ballpark; and expected batting-order position. Historical batting order is reconstructed from each hitter's first plate appearance. Prediction uses the announced MLB lineup.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Collect historical Statcast data

Use complete prior seasons plus the current season through yesterday. Collection can take several minutes.

```bash
mlb-hr collect --start 2022-03-01 --end 2025-11-01 \
  --output data/raw/statcast.parquet
```

Dates are inclusive. Training also accepts an existing Statcast `.parquet` or `.csv` file.
When the output file already exists, `collect` keeps it and downloads only dates after
the latest stored game. Use `--replace` only when you intentionally want a fresh full download.

## Train and evaluate

```bash
mlb-hr train \
  --data data/raw/statcast.parquet \
  --model artifacts/hr_logistic.joblib \
  --features-output data/processed/player_games.parquet
```

This prints held-out metrics, writes `artifacts/hr_logistic.metrics.json`, and stores preprocessing, logistic regression, and the calibrator in one artifact. Early-season and new-player missing rates are imputed using training data only.

## Predict every expected hitter in a game

Find the MLB `gamePk` in an MLB game URL. After lineups and starters are announced:

```bash
mlb-hr predict-game \
  --game-pk 777001 \
  --data data/raw/statcast.parquet \
  --model artifacts/hr_logistic.joblib \
  --output predictions/777001.csv
```

The command reads both batting orders and opposing starters from MLB's live game feed, calculates rates from Statcast games strictly before the official game date, and prints calibrated probabilities. It fails clearly when a lineup or starter is unavailable. Keep the history file current through the day before prediction.

## Tests

```bash
pytest
```

Tests cover the player-game target, starting-pitcher selection, prior-game shifting, unseen-player handling, and chronological boundaries.

## Baseline limitations

This is a first model, not a betting system. It does not yet include weather, scratches, bullpen exposure, normalized park factors, handedness splits, injuries, or pitch mix. Retrain on a rolling schedule and monitor discrimination and calibration by season.
