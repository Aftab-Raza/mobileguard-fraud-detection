"""Dataset loading, validation, profiling, and chronological splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mobileguard.schema import (
    FULL_DATASET_COLUMNS,
    TRAINING_COLUMNS,
    DataValidationError,
    require_columns,
    validate_transaction_frame,
)

READ_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
}


def _dataset_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(path).schema.names
    if suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    raise DataValidationError("Dataset must be a .parquet, .pq, or .csv file")


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load only model columns after validating the complete PaySim schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    require_columns(_dataset_columns(source), FULL_DATASET_COLUMNS)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source, columns=TRAINING_COLUMNS)
        for column, dtype in READ_DTYPES.items():
            frame[column] = frame[column].astype(dtype)
    else:
        frame = pd.read_csv(source, usecols=TRAINING_COLUMNS, dtype=READ_DTYPES)

    frame = validate_transaction_frame(frame, require_target=True)
    if not frame["step"].is_monotonic_increasing:
        frame = frame.sort_values("step", kind="stable").reset_index(drop=True)
    return frame


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split past-to-future without allowing one step into two partitions."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no room for a test set")

    ordered = frame
    if not ordered["step"].is_monotonic_increasing:
        ordered = ordered.sort_values("step", kind="stable").reset_index(drop=True)

    steps = ordered["step"].to_numpy()
    train_target = int(len(ordered) * train_fraction)
    validation_target = int(len(ordered) * (train_fraction + validation_fraction))
    # The row at each target belongs to the later partition. Moving to the
    # beginning of its step prevents leakage and avoids consuming the final
    # step into validation when the target lands exactly on that step.
    train_end = int(np.searchsorted(steps, steps[train_target], side="left"))
    validation_end = int(
        np.searchsorted(steps, steps[validation_target], side="left")
    )

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise DataValidationError("Chronological split produced an empty partition")
    return train, validation, test


def stratified_training_sample(
    frame: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Cap training cost while preserving the fraud prevalence."""

    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame

    fraud = frame[frame["isFraud"] == 1]
    legitimate = frame[frame["isFraud"] == 0]
    fraud_rows = max(1, round(max_rows * len(fraud) / len(frame)))
    fraud_rows = min(fraud_rows, len(fraud))
    legitimate_rows = min(max_rows - fraud_rows, len(legitimate))

    sampled = pd.concat(
        [
            fraud.sample(n=fraud_rows, random_state=random_state),
            legitimate.sample(n=legitimate_rows, random_state=random_state),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the EDA facts most relevant to a detection model."""

    label_counts = frame["isFraud"].value_counts().sort_index()
    type_table = pd.crosstab(frame["type"], frame["isFraud"])
    amount_stats = (
        frame.groupby("isFraud", observed=True)["amount"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(4)
    )
    fraud_by_step = (
        frame.groupby("step", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
    )

    return {
        "rows": int(len(frame)),
        "step_range": [int(frame["step"].min()), int(frame["step"].max())],
        "class_distribution": {
            "legitimate": int(label_counts.get(0, 0)),
            "fraud": int(label_counts.get(1, 0)),
            "fraud_rate": float(frame["isFraud"].mean()),
        },
        "fraud_by_transaction_type": {
            str(index): {
                "legitimate": int(row.get(0, 0)),
                "fraud": int(row.get(1, 0)),
            }
            for index, row in type_table.iterrows()
        },
        "amount_statistics": {
            ("fraud" if int(index) == 1 else "legitimate"): {
                key: (int(value) if key == "count" else float(value))
                for key, value in row.items()
            }
            for index, row in amount_stats.iterrows()
        },
        "fraud_over_time": {
            "steps_with_fraud": int((fraud_by_step["fraud_count"] > 0).sum()),
            "peak_fraud_step": int(fraud_by_step["fraud_count"].idxmax()),
            "peak_fraud_count": int(fraud_by_step["fraud_count"].max()),
        },
    }


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
