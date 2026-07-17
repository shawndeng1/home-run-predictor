from mlb_hr_predictor.predict import _safe_ratio


def test_safe_ratio_for_unseen_player() -> None:
    assert _safe_ratio(0, 0) != _safe_ratio(0, 0)
    assert _safe_ratio(2, 10) == 0.2

