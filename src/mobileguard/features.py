"""Leakage-safe feature engineering used by both training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mobileguard.schema import RAW_MODEL_COLUMNS, validate_transaction_frame

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "hour",
    "day",
    "log_amount",
    "origin_balance_error",
    "destination_balance_error",
    "abs_origin_balance_error",
    "abs_destination_balance_error",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_change",
    "destination_balance_change",
    "origin_zero_before",
    "origin_zero_after",
    "destination_zero_before",
    "destination_zero_after",
    "amount_equals_origin_balance",
]
MODEL_FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic PaySim fraud indicators from raw values."""

    raw = validate_transaction_frame(frame)
    engineered = raw.loc[:, RAW_MODEL_COLUMNS].copy()

    engineered["hour"] = (raw["step"] % 24).astype("int16")
    engineered["day"] = (raw["step"] // 24).astype("int32")
    engineered["log_amount"] = np.log1p(raw["amount"])

    engineered["origin_balance_error"] = (
        raw["oldbalanceOrg"] - raw["amount"] - raw["newbalanceOrig"]
    )
    engineered["destination_balance_error"] = (
        raw["oldbalanceDest"] + raw["amount"] - raw["newbalanceDest"]
    )
    engineered["abs_origin_balance_error"] = engineered[
        "origin_balance_error"
    ].abs()
    engineered["abs_destination_balance_error"] = engineered[
        "destination_balance_error"
    ].abs()

    # A floor of 1 keeps the ratio finite when a reported balance is zero.
    engineered["amount_to_origin_balance"] = raw["amount"] / np.maximum(
        raw["oldbalanceOrg"].abs(), 1.0
    )
    engineered["amount_to_destination_balance"] = raw["amount"] / np.maximum(
        raw["oldbalanceDest"].abs(), 1.0
    )

    engineered["origin_balance_change"] = (
        raw["oldbalanceOrg"] - raw["newbalanceOrig"]
    )
    engineered["destination_balance_change"] = (
        raw["newbalanceDest"] - raw["oldbalanceDest"]
    )

    engineered["origin_zero_before"] = (raw["oldbalanceOrg"] == 0).astype("int8")
    engineered["origin_zero_after"] = (raw["newbalanceOrig"] == 0).astype("int8")
    engineered["destination_zero_before"] = (raw["oldbalanceDest"] == 0).astype(
        "int8"
    )
    engineered["destination_zero_after"] = (raw["newbalanceDest"] == 0).astype(
        "int8"
    )
    engineered["amount_equals_origin_balance"] = np.isclose(
        raw["amount"],
        raw["oldbalanceOrg"],
        rtol=1e-5,
        atol=0.01,
    ).astype("int8")

    return engineered.loc[:, MODEL_FEATURES]


class PaySimFeatureEngineer(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer that keeps training and inference identical."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "PaySimFeatureEngineer":
        engineer_features(X.iloc[:1] if len(X) > 1 else X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)

