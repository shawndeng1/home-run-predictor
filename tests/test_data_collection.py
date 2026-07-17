from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from mlb_hr_predictor.data_collection import collect_statcast


class FakeCache:
    def __init__(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True


def test_collect_incrementally_updates_and_retries(tmp_path, monkeypatch) -> None:
    output = tmp_path / "statcast.parquet"
    pd.DataFrame([{
        "game_date": pd.Timestamp("2024-04-01"), "game_pk": 1,
        "at_bat_number": 1, "pitch_number": 1,
    }]).to_parquet(output, index=False)
    calls: list[tuple[str, str]] = []
    cache = FakeCache()

    def fake_statcast(start_dt: str, end_dt: str, parallel: bool) -> pd.DataFrame:
        calls.append((start_dt, end_dt))
        if len(calls) == 1:
            raise pd.errors.ParserError("temporary malformed response")
        return pd.DataFrame([{
            "game_date": pd.Timestamp("2024-04-02"), "game_pk": 2,
            "at_bat_number": 1, "pitch_number": 1,
        }])

    monkeypatch.setitem(sys.modules, "pybaseball", SimpleNamespace(cache=cache, statcast=fake_statcast))
    monkeypatch.setattr("mlb_hr_predictor.data_collection.time.sleep", lambda _: None)
    collect_statcast("2024-04-01", "2024-04-03", output)
    saved = pd.read_parquet(output)
    assert calls == [("2024-04-02", "2024-04-03"), ("2024-04-02", "2024-04-03")]
    assert saved["game_pk"].tolist() == [1, 2]
    assert cache.enabled

