"""Collection and normalization of Statcast and MLB game data."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
MLB_API = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


@dataclass(frozen=True)
class ExpectedHitter:
    player_id: int
    player_name: str
    batter_hand: str
    pitcher_id: int
    pitcher_name: str
    pitcher_hand: str
    batting_order: int
    ballpark: str
    game_date: pd.Timestamp
    game_pk: int


def collect_statcast(
    start_date: str, end_date: str, output: Path, *, replace: bool = False, retries: int = 3
) -> Path:
    """Download Statcast pitches, incrementally updating an existing parquet file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYBASEBALL_CACHE", str(output.parent / ".pybaseball-cache"))
    # Lazy import keeps pybaseball's cache side effect out of other commands.
    from pybaseball import cache, statcast

    cache.enable()
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    existing: pd.DataFrame | None = None
    download_start = requested_start
    if output.exists() and not replace:
        stored_dates = pd.read_parquet(output, columns=["game_date"])["game_date"]
        latest = pd.to_datetime(stored_dates).max().normalize()
        download_start = max(requested_start, latest + pd.offsets.Day(1))
        if download_start > requested_end:
            LOGGER.info("Statcast file is already current through %s", latest.date())
            return output
        LOGGER.info(
            "Updating existing Statcast file; only downloading %s through %s",
            download_start.date(), requested_end.date(),
        )
        existing = load_statcast(output)
    else:
        LOGGER.info("Downloading Statcast data from %s through %s", start_date, end_date)

    frame: pd.DataFrame | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = statcast(
                start_dt=str(download_start.date()), end_dt=str(requested_end.date()), parallel=True
            )
            break
        except Exception:
            if attempt == retries:
                raise
            delay = attempt * 2
            LOGGER.warning("Statcast request failed; retrying in %s seconds (%s/%s)", delay, attempt, retries)
            time.sleep(delay)
    assert frame is not None
    if frame.empty:
        LOGGER.info("Statcast returned no new pitches; existing file was left unchanged")
        return output
    if existing is not None:
        frame = pd.concat([existing, frame], ignore_index=True)
        keys = [key for key in ("game_pk", "at_bat_number", "pitch_number") if key in frame]
        frame = frame.drop_duplicates(subset=keys or None, keep="last")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    LOGGER.info("Saved %s pitches to %s", f"{len(frame):,}", output)
    return output


def load_statcast(path: Path) -> pd.DataFrame:
    """Load collected Statcast pitches."""
    LOGGER.info("Loading Statcast data from %s", path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def fetch_expected_hitters(game_pk: int, timeout: int = 30) -> list[ExpectedHitter]:
    """Read announced lineups and probable/actual starters from MLB's live game feed."""
    response = requests.get(MLB_API.format(game_pk=game_pk), timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    game_data = payload["gameData"]
    live_data = payload["liveData"]
    game_date = pd.Timestamp(game_data["datetime"]["officialDate"])
    # Statcast history identifies a park by its home-team code.
    venue = game_data["teams"]["home"]["abbreviation"]
    players = game_data["players"]
    box_teams = live_data["boxscore"]["teams"]
    hitters: list[ExpectedHitter] = []

    for side, opponent in (("away", "home"), ("home", "away")):
        lineup = box_teams[side].get("battingOrder", [])
        probable = game_data.get("probablePitchers", {}).get(opponent)
        pitcher_id = probable.get("id") if probable else None
        if pitcher_id is None:
            pitcher_ids = box_teams[opponent].get("pitchers", [])
            pitcher_id = pitcher_ids[0] if pitcher_ids else None
        if not lineup or pitcher_id is None:
            raise ValueError(
                f"Expected lineup or starting pitcher is not yet available for game {game_pk}"
            )
        pitcher = players[f"ID{pitcher_id}"]
        for order, player_id in enumerate(lineup[:9], start=1):
            batter = players[f"ID{player_id}"]
            hitters.append(
                ExpectedHitter(
                    player_id=int(player_id),
                    player_name=batter["fullName"],
                    batter_hand=batter.get("batSide", {}).get("code", "U"),
                    pitcher_id=int(pitcher_id),
                    pitcher_name=pitcher["fullName"],
                    pitcher_hand=pitcher.get("pitchHand", {}).get("code", "U"),
                    batting_order=order,
                    ballpark=venue,
                    game_date=game_date,
                    game_pk=game_pk,
                )
            )
    return hitters
