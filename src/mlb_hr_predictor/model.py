"""Chronological training, calibration, evaluation, and persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    model: CalibratedClassifierCV
    cutoff_train: str
    cutoff_calibration: str
    metrics: dict[str, float]


def make_baseline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, calibration_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique game dates so no date appears in more than one partition."""
    dates = np.array(sorted(pd.to_datetime(frame["game_date"]).dt.normalize().unique()))
    if len(dates) < 3:
        raise ValueError("At least three unique game dates are required")
    train_idx = max(1, int(len(dates) * train_fraction))
    cal_idx = max(train_idx + 1, int(len(dates) * (train_fraction + calibration_fraction)))
    cal_idx = min(cal_idx, len(dates) - 1)
    normalized = pd.to_datetime(frame["game_date"]).dt.normalize()
    return (
        frame[normalized.isin(dates[:train_idx])].copy(),
        frame[normalized.isin(dates[train_idx:cal_idx])].copy(),
        frame[normalized.isin(dates[cal_idx:])].copy(),
    )


def train_and_evaluate(frame: pd.DataFrame, output: Path) -> ModelArtifact:
    train, calibration, test = chronological_split(frame)
    baseline = make_baseline()
    baseline.fit(train[FEATURES], train["target"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(baseline), method="sigmoid")
    calibrated.fit(calibration[FEATURES], calibration["target"])
    probabilities = calibrated.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "brier_score": float(brier_score_loss(test["target"], probabilities)),
        "log_loss": float(log_loss(test["target"], probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(test["target"], probabilities))
        if test["target"].nunique() == 2 else float("nan"),
        "test_rows": float(len(test)),
    }
    artifact = ModelArtifact(
        model=calibrated,
        cutoff_train=str(train["game_date"].max().date()),
        cutoff_calibration=str(calibration["game_date"].max().date()),
        metrics=metrics,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("Saved calibrated model to %s; metrics=%s", output, metrics)
    return artifact


def load_artifact(path: Path) -> ModelArtifact:
    return joblib.load(path)
