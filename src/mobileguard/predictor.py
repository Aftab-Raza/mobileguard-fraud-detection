"""Load-once prediction interface for single or batched transactions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mobileguard.schema import RAW_MODEL_COLUMNS, validate_transaction_frame


class MobileGuardPredictor:
    """Detection-only facade over the saved scikit-learn pipeline."""

    def __init__(
        self,
        model_path: str | Path = "artifacts/mobile_fraud_model.joblib",
        thresholds_path: str | Path = "artifacts/fraud_thresholds.json",
    ) -> None:
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError("Invalid MobileGuard model artifact")
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle.get("metadata", {})

        thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        self.review_threshold = float(thresholds["review"])
        self.block_threshold = float(thresholds["block"])
        if not 0 <= self.review_threshold < self.block_threshold <= 1:
            raise ValueError("Invalid review/block thresholds")

    def predict_many(
        self, transactions: Sequence[Mapping[str, Any]] | pd.DataFrame
    ) -> list[dict[str, Any]]:
        if isinstance(transactions, pd.DataFrame):
            frame = transactions.copy()
        else:
            frame = pd.DataFrame(list(transactions))
        frame = validate_transaction_frame(frame)
        probabilities = self.pipeline.predict_proba(frame.loc[:, RAW_MODEL_COLUMNS])[
            :, 1
        ]

        results: list[dict[str, Any]] = []
        for probability in probabilities:
            score = float(probability)
            if score >= self.block_threshold:
                risk_level, decision = "HIGH", "BLOCK"
            elif score >= self.review_threshold:
                risk_level, decision = "MEDIUM", "REVIEW"
            else:
                risk_level, decision = "LOW", "APPROVE"
            results.append(
                {
                    "fraud_probability": round(score, 6),
                    "risk_level": risk_level,
                    "decision": decision,
                    "model_version": self.metadata.get("engine_version", "unknown"),
                }
            )
        return results

    def predict(self, transaction: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_many([transaction])[0]

