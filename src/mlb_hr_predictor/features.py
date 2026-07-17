"""Leakage-safe player-game feature engineering from Statcast pitches."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
HIT_EVENTS = {"single", "double", "triple", "home_run"}
PA_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "force_out",
    "grounded_into_double_play", "field_error", "fielders_choice",
    "fielders_choice_out", "strikeout", "strikeout_double_play", "walk",
    "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf",
}


def _first_non_null(values: pd.Series) -> object:
    present = values.dropna()
    return present.iloc[0] if not present.empty else np.nan


def _prepare_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    required = {"game_pk", "game_date", "batter", "pitcher", "events", "stand", "p_throws"}
    missing = required.difference(pitches.columns)
    if missing:
        raise ValueError(f"Statcast data is missing columns: {sorted(missing)}")
    frame = pitches.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame["is_pa"] = frame["events"].isin(PA_EVENTS).fillna(False).astype("int8")
    frame["is_hr"] = frame["events"].eq("home_run").fillna(False).astype("int8")
    frame["is_batted_ball"] = frame["launch_speed"].notna().astype("int8") if "launch_speed" in frame else 0
    if "launch_speed_angle" in frame:
        frame["is_barrel"] = frame["launch_speed_angle"].eq(6).fillna(False).astype("int8")
    else:
        frame["is_barrel"] = 0
    frame["is_hard_hit"] = (
        frame.get("launch_speed", pd.Series(index=frame.index, dtype=float))
        .ge(95).fillna(False).astype("int8")
    )
    frame["is_fly_ball"] = (
        frame.get("bb_type", pd.Series(index=frame.index, dtype=object))
        .eq("fly_ball").fillna(False).astype("int8")
    )
    sort_columns = [column for column in ("game_date", "game_pk", "at_bat_number", "pitch_number") if column in frame]
    return frame.sort_values(sort_columns)


def build_player_games(pitches: pd.DataFrame) -> pd.DataFrame:
    """Build one row per batter-game with only expanding statistics from earlier games."""
    p = _prepare_pitches(pitches)
    switch_hitters = set(
        p.groupby("batter")["stand"].nunique().loc[lambda counts: counts > 1].index
    )
    pa = p[p["is_pa"].eq(1)].copy()
    if pa.empty:
        raise ValueError("No completed plate appearances found")

    game_keys = ["game_date", "game_pk", "batter"]
    batter_games = pa.groupby(game_keys, as_index=False).agg(
        pitcher=("pitcher", "first"),
        batter_hand=("stand", _first_non_null),
        pitcher_hand=("p_throws", _first_non_null),
        plate_appearances=("is_pa", "sum"),
        home_runs=("is_hr", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        fly_balls=("is_fly_ball", "sum"),
        at_bat_number=("at_bat_number", "min"),
        home_team=("home_team", _first_non_null),
    )
    batter_games["target"] = batter_games["home_runs"].gt(0).astype(int)
    batter_games.loc[batter_games["batter"].isin(switch_hitters), "batter_hand"] = "S"
    batter_games["expected_batting_order"] = (
        batter_games.groupby(["game_pk", "batter_hand"], sort=False)["at_bat_number"]
        .rank(method="first").clip(upper=9)
    )
    # In Statcast, each half-inning's first nine PA sequence gives the starting order.
    first_pa = pa.groupby(game_keys, as_index=False)["at_bat_number"].min()
    first_pa["expected_batting_order"] = first_pa.groupby("game_pk")["at_bat_number"].rank(method="dense")
    # Rank separately by batting team using top/bottom half when available.
    if "inning_topbot" in pa:
        halves = pa.groupby(game_keys, as_index=False)["inning_topbot"].first()
        first_pa = first_pa.drop(columns="expected_batting_order").merge(halves, on=game_keys)
        first_pa["expected_batting_order"] = first_pa.groupby(
            ["game_pk", "inning_topbot"]
        )["at_bat_number"].rank(method="dense")
        batter_games = batter_games.drop(columns="expected_batting_order").merge(
            first_pa[game_keys + ["expected_batting_order"]], on=game_keys
        )

    batter_games = batter_games.sort_values(["game_date", "game_pk", "batter"])
    batter_games = _add_prior_rates(
        batter_games, "batter",
        {"batter_hr_per_pa": ("home_runs", "plate_appearances"),
         "batter_barrel_rate": ("barrels", "batted_balls"),
         "batter_hard_hit_rate": ("hard_hits", "batted_balls"),
         "batter_fly_ball_rate": ("fly_balls", "batted_balls")},
    )

    pitcher_game = batter_games.groupby(["game_date", "game_pk", "pitcher"], as_index=False).agg(
        batters_faced=("plate_appearances", "sum"),
        hr_allowed=("home_runs", "sum"),
        batted_balls_allowed=("batted_balls", "sum"),
        barrels_allowed=("barrels", "sum"),
    )
    pitcher_game = _add_prior_rates(
        pitcher_game, "pitcher",
        {"pitcher_hr_per_bf": ("hr_allowed", "batters_faced"),
         "pitcher_barrel_rate_allowed": ("barrels_allowed", "batted_balls_allowed")},
    )
    features = batter_games.merge(
        pitcher_game[["game_pk", "pitcher", "pitcher_hr_per_bf", "pitcher_barrel_rate_allowed"]],
        on=["game_pk", "pitcher"], how="left",
    )
    features["platoon_matchup"] = np.where(
        features["batter_hand"].eq("S"), "switch",
        np.where(features["batter_hand"].eq(features["pitcher_hand"]), "same", "opposite"),
    )
    features["ballpark"] = features["home_team"].fillna("UNKNOWN").astype(str)
    LOGGER.info("Built %s player-game rows", f"{len(features):,}")
    return features


def _add_prior_rates(
    frame: pd.DataFrame,
    entity: str,
    rates: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Add expanding ratios shifted by one whole date for an entity.

    Treating a date atomically prevents doubleheaders from leaking based on an
    unreliable game-pk ordering.
    """
    out = frame.sort_values(["game_date", "game_pk"]).copy()
    component_columns = sorted({column for pair in rates.values() for column in pair})
    daily = out.groupby([entity, "game_date"], as_index=False)[component_columns].sum()
    daily = daily.sort_values([entity, "game_date"])
    grouped = daily.groupby(entity, sort=False)
    for name, (numerator, denominator) in rates.items():
        prior_num = grouped[numerator].cumsum() - daily[numerator]
        prior_den = grouped[denominator].cumsum() - daily[denominator]
        daily[name] = prior_num.div(prior_den.replace(0, np.nan))
    out = out.merge(daily[[entity, "game_date", *rates]], on=[entity, "game_date"], how="left")
    return out


def latest_rates_before(
    pitches: pd.DataFrame, entity_ids: Iterable[int], as_of: pd.Timestamp, entity: str
) -> pd.DataFrame:
    """Return cumulative component totals before a prediction date."""
    p = _prepare_pitches(pitches)
    p = p[p["game_date"].lt(pd.Timestamp(as_of).normalize())]
    ids = set(int(value) for value in entity_ids)
    p = p[p[entity].isin(ids)]
    return p
