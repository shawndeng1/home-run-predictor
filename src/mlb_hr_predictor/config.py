"""Shared feature definitions."""

NUMERIC_FEATURES = [
    "batter_hr_per_pa",
    "batter_barrel_rate",
    "batter_hard_hit_rate",
    "batter_fly_ball_rate",
    "pitcher_hr_per_bf",
    "pitcher_barrel_rate_allowed",
    "expected_batting_order",
]

CATEGORICAL_FEATURES = [
    "batter_hand",
    "pitcher_hand",
    "platoon_matchup",
    "ballpark",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

