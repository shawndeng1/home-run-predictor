from __future__ import annotations

import pandas as pd

from mlb_hr_predictor.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from mlb_hr_predictor.model import chronological_split, load_artifact, train_and_evaluate


def test_chronological_split_has_strictly_ordered_dates() -> None:
    frame = pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=20), "target": [0, 1] * 10})
    train, calibration, test = chronological_split(frame)
    assert train.game_date.max() < calibration.game_date.min()
    assert calibration.game_date.max() < test.game_date.min()
    assert len(train) + len(calibration) + len(test) == len(frame)


def test_training_writes_loadable_calibrated_artifact(tmp_path) -> None:
    rows = []
    for day in range(1, 41):
        for player in range(8):
            row = {
                "game_date": pd.Timestamp("2024-01-01") + pd.offsets.Day(day),
                "target": int((day + player) % 9 == 0),
            }
            row.update({name: float((day + player) % 7) / 7 for name in NUMERIC_FEATURES})
            row.update({name: "category_a" if player % 2 else "category_b" for name in CATEGORICAL_FEATURES})
            rows.append(row)
    path = tmp_path / "model.joblib"
    artifact = train_and_evaluate(pd.DataFrame(rows), path)
    assert path.exists()
    assert path.with_suffix(".metrics.json").exists()
    assert 0 <= artifact.metrics["brier_score"] <= 1
    assert load_artifact(path).cutoff_train == artifact.cutoff_train
