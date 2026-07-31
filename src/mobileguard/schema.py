"""Input schemas and validation for PaySim transactions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

VALID_TRANSACTION_TYPES = frozenset(
    {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
)

RAW_MODEL_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

FULL_DATASET_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TRAINING_COLUMNS = [*RAW_MODEL_COLUMNS, "isFraud"]
NUMERIC_RAW_COLUMNS = [column for column in RAW_MODEL_COLUMNS if column != "type"]
BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


class DataValidationError(ValueError):
    """Raised when transaction data violates the MobileGuard input contract."""


def require_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    present = set(columns)
    missing = sorted(set(required) - present)
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")


def validate_transaction_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Validate and normalize raw transaction rows.

    The returned frame is a defensive copy. Numeric strings are accepted when
    they can be converted without loss of meaning.
    """

    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame")
    if frame.empty:
        raise DataValidationError("Transaction data is empty")

    required = TRAINING_COLUMNS if require_target else RAW_MODEL_COLUMNS
    require_columns(frame.columns, required)

    normalized = frame.loc[:, required].copy()
    normalized["type"] = normalized["type"].astype("string").str.strip().str.upper()

    invalid_types = sorted(
        set(normalized["type"].dropna().unique()) - VALID_TRANSACTION_TYPES
    )
    if normalized["type"].isna().any() or invalid_types:
        details = ", ".join(invalid_types) if invalid_types else "missing value"
        raise DataValidationError(f"Invalid transaction type(s): {details}")

    numeric_columns = [*NUMERIC_RAW_COLUMNS]
    if require_target:
        numeric_columns.append("isFraud")

    for column in numeric_columns:
        converted = pd.to_numeric(normalized[column], errors="coerce")
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise DataValidationError(f"Column '{column}' contains missing/non-finite values")
        normalized[column] = converted

    if (normalized["step"] < 0).any():
        raise DataValidationError("Column 'step' cannot contain negative values")
    if (normalized["amount"] < 0).any():
        raise DataValidationError("Column 'amount' cannot contain negative values")
    for column in BALANCE_COLUMNS:
        if (normalized[column] < 0).any():
            raise DataValidationError(f"Column '{column}' cannot contain negative values")

    if require_target:
        labels = set(normalized["isFraud"].unique())
        if not labels.issubset({0, 1}):
            raise DataValidationError("Column 'isFraud' must contain only 0 and 1")
        normalized["isFraud"] = normalized["isFraud"].astype("int8")

    return normalized

