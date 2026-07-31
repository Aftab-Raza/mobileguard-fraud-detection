"""Command-line entry point for the detection engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mobileguard.data import dataset_profile, load_training_data, write_json
from mobileguard.download import download_paysim
from mobileguard.modeling import SUPPORTED_MODELS, train_engine
from mobileguard.predictor import MobileGuardPredictor


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobileguard",
        description="PaySim mobile money fraud detection engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-data", help="Download the complete public PaySim Parquet dataset"
    )
    download.add_argument(
        "--output", default="data/raw/paysim.parquet", help="Destination file"
    )
    download.add_argument("--overwrite", action="store_true")

    profile = subparsers.add_parser(
        "profile", help="Validate the dataset and write a compact EDA profile"
    )
    profile.add_argument("--data", default="data/raw/paysim.parquet")
    profile.add_argument("--output", default="reports/dataset_profile.json")

    train = subparsers.add_parser(
        "train", help="Train, compare, tune, evaluate, and save the detector"
    )
    train.add_argument("--data", default="data/raw/paysim.parquet")
    train.add_argument("--artifact-dir", default="artifacts")
    train.add_argument("--report-dir", default="reports")
    train.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    train.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
        help="Stratified training cap; use 0 for all historical training rows",
    )
    train.add_argument("--review-target-recall", type=float, default=0.90)
    train.add_argument("--block-target-precision", type=float, default=0.75)
    train.add_argument("--random-state", type=int, default=42)

    predict = subparsers.add_parser(
        "predict", help="Score one transaction from a JSON file"
    )
    predict.add_argument("--input", required=True, help="Transaction JSON file")
    predict.add_argument(
        "--model", default="artifacts/mobile_fraud_model.joblib"
    )
    predict.add_argument(
        "--thresholds", default="artifacts/fraud_thresholds.json"
    )

    batch = subparsers.add_parser(
        "predict-batch", help="Score transaction rows from CSV or Parquet"
    )
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--model", default="artifacts/mobile_fraud_model.joblib")
    batch.add_argument("--thresholds", default="artifacts/fraud_thresholds.json")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "download-data":
        path = download_paysim(args.output, overwrite=args.overwrite)
        _print_json({"dataset": str(path), "bytes": path.stat().st_size})
        return

    if args.command == "profile":
        profile = dataset_profile(load_training_data(args.data))
        path = write_json(profile, args.output)
        _print_json({"profile": str(path), **profile["class_distribution"]})
        return

    if args.command == "train":
        result = train_engine(
            args.data,
            artifact_dir=args.artifact_dir,
            report_dir=args.report_dir,
            model_names=tuple(args.models),
            max_train_rows=(
                None if args.max_train_rows == 0 else args.max_train_rows
            ),
            review_target_recall=args.review_target_recall,
            block_target_precision=args.block_target_precision,
            random_state=args.random_state,
        )
        _print_json(result)
        return

    predictor = MobileGuardPredictor(args.model, args.thresholds)
    if args.command == "predict":
        transaction = json.loads(Path(args.input).read_text(encoding="utf-8"))
        _print_json(predictor.predict(transaction))
        return

    source = Path(args.input)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Batch input must be a CSV or Parquet file")
    results = pd.DataFrame(predictor.predict_many(frame))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        results.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        results.to_csv(output, index=False)
    else:
        raise ValueError("Batch output must be a CSV or Parquet file")
    _print_json({"output": str(output), "rows": len(results)})


if __name__ == "__main__":
    main()

