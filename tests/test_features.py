from __future__ import annotations

import logging

import pandas as pd
import pytest

from mlb_hr_predictor.features import _prepare_pitches, build_player_games


def pitch(game: int, day: str, batter: int, pitcher: int, event: str, at_bat: int,
          speed: float = 90, angle: int = 3, bb_type: str = "ground_ball") -> dict[str, object]:
    return {
        "game_pk": game, "game_date": day, "batter": batter, "pitcher": pitcher,
        "events": event, "stand": "R", "p_throws": "L", "launch_speed": speed,
        "launch_speed_angle": angle, "bb_type": bb_type, "at_bat_number": at_bat,
        "pitch_number": 1, "home_team": "TOR", "inning_topbot": "Top",
    }


def test_features_are_shifted_by_game() -> None:
    raw = pd.DataFrame([
        pitch(1, "2024-04-01", 10, 20, "home_run", 1, 101, 6, "fly_ball"),
        pitch(2, "2024-04-02", 10, 20, "field_out", 1, 98, 4, "fly_ball"),
        pitch(3, "2024-04-03", 10, 20, "field_out", 1, 80, 2, "ground_ball"),
    ])
    rows = build_player_games(raw).sort_values("game_date")
    assert pd.isna(rows.iloc[0]["batter_hr_per_pa"])
    assert rows.iloc[1]["batter_hr_per_pa"] == pytest.approx(1.0)
    assert rows.iloc[2]["batter_hr_per_pa"] == pytest.approx(0.5)
    assert rows.iloc[1]["pitcher_hr_per_bf"] == pytest.approx(1.0)
    assert rows["target"].tolist() == [1, 0, 0]


def test_one_row_per_player_game_and_starting_pitcher() -> None:
    raw = pd.DataFrame([
        pitch(1, "2024-04-01", 10, 99, "field_out", 8),
        pitch(1, "2024-04-01", 10, 20, "single", 1),
    ])
    rows = build_player_games(raw)
    assert len(rows) == 1
    assert rows.iloc[0]["pitcher"] == 20
    assert rows.iloc[0]["plate_appearances"] == 2


def test_doubleheader_games_cannot_leak_into_each_other() -> None:
    raw = pd.DataFrame([
        pitch(1, "2024-04-01", 10, 20, "home_run", 1),
        pitch(2, "2024-04-01", 10, 20, "field_out", 1),
        pitch(3, "2024-04-02", 10, 20, "field_out", 1),
    ])
    rows = build_player_games(raw).sort_values(["game_date", "game_pk"])
    assert rows.iloc[:2]["batter_hr_per_pa"].isna().all()
    assert rows.iloc[2]["batter_hr_per_pa"] == pytest.approx(0.5)


def test_large_row_count_logging_is_valid(caplog) -> None:
    caplog.set_level(logging.INFO)
    raw = pd.DataFrame([pitch(1, "2024-04-01", 10, 20, "field_out", 1)])
    build_player_games(raw)
    assert "Built 1 player-game rows" in caplog.text


def test_nullable_statcast_measurements_are_treated_as_zero() -> None:
    raw = pd.DataFrame([pitch(1, "2024-04-01", 10, 20, "field_out", 1)])
    raw["launch_speed"] = pd.Series([pd.NA], dtype="Float64")
    raw["launch_speed_angle"] = pd.Series([pd.NA], dtype="Int64")
    prepared = _prepare_pitches(raw)
    assert prepared.iloc[0]["is_batted_ball"] == 0
    assert prepared.iloc[0]["is_barrel"] == 0
    assert prepared.iloc[0]["is_hard_hit"] == 0
