"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .data_collection import collect_statcast, load_statcast
from .features import build_player_games
from .model import train_and_evaluate
from .predict import predict_day, predict_game


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mlb-hr")
    root.add_argument("--log-level", default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="download Statcast pitch history")
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    collect.add_argument("--output", type=Path, default=Path("data/raw/statcast.parquet"))
    collect.add_argument("--replace", action="store_true", help="replace instead of incrementally updating")
    train = commands.add_parser("train", help="engineer features and train calibrated model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    train.add_argument("--features-output", type=Path)
    predict = commands.add_parser("predict-game", help="predict all expected hitters in a game")
    predict.add_argument("--game-pk", type=int, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    predict_day_parser = commands.add_parser("predict-day", help="rank expected hitters across a day's games")
    predict_day_parser.add_argument("--date", required=True, help="game date in YYYY-MM-DD format")
    predict_day_parser.add_argument("--data", type=Path, required=True)
    predict_day_parser.add_argument("--model", type=Path, required=True)
    predict_day_parser.add_argument("--output", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "collect":
        collect_statcast(args.start, args.end, args.output, replace=args.replace)
    elif args.command == "train":
        features = build_player_games(load_statcast(args.data))
        if args.features_output:
            args.features_output.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(args.features_output, index=False)
        artifact = train_and_evaluate(features, args.model)
        print(artifact.metrics)
    elif args.command == "predict-game":
        predictions = predict_game(args.game_pk, args.data, args.model)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(args.output, index=False)
        print(predictions.to_string(index=False))
    elif args.command == "predict-day":
        predictions = predict_day(args.date, args.data, args.model)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(args.output, index=False)
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
