"""Specified-game feature assembly and probability prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES
from .data_collection import fetch_expected_hitters, load_statcast
from .features import PA_EVENTS, _prepare_pitches
from .model import load_artifact

LOGGER = logging.getLogger(__name__)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def build_game_features(game_pk: int, history: pd.DataFrame) -> pd.DataFrame:
    hitters = fetch_expected_hitters(game_pk)
    as_of = hitters[0].game_date
    p = _prepare_pitches(history)
    p = p[p["game_date"].lt(as_of)]
    rows: list[dict[str, object]] = []
    for hitter in hitters:
        batter = p[p["batter"].eq(hitter.player_id)]
        pitcher = p[p["pitcher"].eq(hitter.pitcher_id)]
        batter_pa = batter["events"].isin(PA_EVENTS).sum()
        pitcher_bf = pitcher["events"].isin(PA_EVENTS).sum()
        batter_bip = int(batter["is_batted_ball"].sum())
        pitcher_bip = int(pitcher["is_batted_ball"].sum())
        rows.append({
            "game_pk": hitter.game_pk,
            "game_date": hitter.game_date,
            "player_id": hitter.player_id,
            "player_name": hitter.player_name,
            "pitcher_id": hitter.pitcher_id,
            "pitcher_name": hitter.pitcher_name,
            "batter_hr_per_pa": _safe_ratio(batter["is_hr"].sum(), batter_pa),
            "batter_barrel_rate": _safe_ratio(batter["is_barrel"].sum(), batter_bip),
            "batter_hard_hit_rate": _safe_ratio(batter["is_hard_hit"].sum(), batter_bip),
            "batter_fly_ball_rate": _safe_ratio(batter["is_fly_ball"].sum(), batter_bip),
            "batter_hand": hitter.batter_hand,
            "pitcher_hr_per_bf": _safe_ratio(pitcher["is_hr"].sum(), pitcher_bf),
            "pitcher_barrel_rate_allowed": _safe_ratio(pitcher["is_barrel"].sum(), pitcher_bip),
            "pitcher_hand": hitter.pitcher_hand,
            "platoon_matchup": "switch" if hitter.batter_hand == "S" else (
                "same" if hitter.batter_hand == hitter.pitcher_hand else "opposite"
            ),
            "ballpark": hitter.ballpark,
            "expected_batting_order": hitter.batting_order,
        })
    return pd.DataFrame(rows)


def predict_game(game_pk: int, history_path: Path, model_path: Path) -> pd.DataFrame:
    features = build_game_features(game_pk, load_statcast(history_path))
    artifact = load_artifact(model_path)
    features["home_run_probability"] = artifact.model.predict_proba(features[FEATURES])[:, 1]
    return features[[
        "game_pk", "game_date", "player_id", "player_name", "pitcher_name",
        "expected_batting_order", "home_run_probability",
    ]].sort_values("home_run_probability", ascending=False)

