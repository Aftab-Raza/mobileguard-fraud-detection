"""Fraud-focused evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    *,
    review_threshold: float,
    block_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    money = np.asarray(amounts, dtype=np.float64)
    detected = scores >= review_threshold
    blocked = scores >= block_threshold
    matrix = confusion_matrix(labels, detected, labels=[0, 1])

    total_fraud_amount = float(money[labels == 1].sum())
    captured_fraud_amount = float(money[(labels == 1) & detected].sum())
    missed_fraud_amount = float(money[(labels == 1) & ~detected].sum())

    return {
        "rows": int(len(labels)),
        "fraud_rows": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "review_threshold": float(review_threshold),
        "block_threshold": float(block_threshold),
        "precision_at_review": float(
            precision_score(labels, detected, zero_division=0)
        ),
        "recall_at_review": float(recall_score(labels, detected, zero_division=0)),
        "f1_at_review": float(f1_score(labels, detected, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "decisions": {
            "approve": int((scores < review_threshold).sum()),
            "review": int(
                ((scores >= review_threshold) & (scores < block_threshold)).sum()
            ),
            "block": int(blocked.sum()),
        },
        "fraud_amount": {
            "total": total_fraud_amount,
            "captured": captured_fraud_amount,
            "missed": missed_fraud_amount,
            "captured_fraction": (
                captured_fraud_amount / total_fraud_amount
                if total_fraud_amount
                else 0.0
            ),
        },
    }

