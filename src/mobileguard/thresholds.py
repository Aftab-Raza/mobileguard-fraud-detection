"""Validation-only threshold selection for review and block decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class DecisionThresholds:
    review: float
    block: float
    review_target_recall: float
    block_target_precision: float
    validation_review_precision: float
    validation_review_recall: float
    validation_block_precision: float
    validation_block_recall: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _point_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[float, float]:
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & (y_true == 1)))
    false_positive = int(np.sum(predicted & (y_true == 0)))
    false_negative = int(np.sum(~predicted & (y_true == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return precision, recall


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    review_target_recall: float = 0.90,
    block_target_precision: float = 0.75,
) -> DecisionThresholds:
    """Tune both decision boundaries using validation labels only."""

    labels = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and probabilities must be non-empty and equally sized")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both legitimate and fraud rows")

    precision, recall, raw_thresholds = precision_recall_curve(labels, scores)
    point_precision = precision[:-1]
    point_recall = recall[:-1]

    review_candidates = np.flatnonzero(point_recall >= review_target_recall)
    if len(review_candidates):
        best_precision = point_precision[review_candidates].max()
        candidates = review_candidates[
            point_precision[review_candidates] == best_precision
        ]
        review_index = int(candidates[-1])
    else:
        review_index = int(np.argmax(point_recall))
    review_threshold = float(raw_thresholds[review_index])

    block_candidates = np.flatnonzero(
        (raw_thresholds > review_threshold)
        & (point_precision >= block_target_precision)
    )
    if len(block_candidates):
        best_recall = point_recall[block_candidates].max()
        candidates = block_candidates[point_recall[block_candidates] == best_recall]
        block_index = int(candidates[0])
    else:
        above_review = np.flatnonzero(raw_thresholds > review_threshold)
        if len(above_review):
            best_precision = point_precision[above_review].max()
            candidates = above_review[point_precision[above_review] == best_precision]
            block_index = int(candidates[0])
        else:
            block_index = review_index

    block_threshold = float(raw_thresholds[block_index])
    if block_threshold <= review_threshold:
        block_threshold = float(np.nextafter(review_threshold, 1.0))

    review_precision, review_recall = _point_metrics(
        labels, scores, review_threshold
    )
    block_precision, block_recall = _point_metrics(labels, scores, block_threshold)
    return DecisionThresholds(
        review=review_threshold,
        block=block_threshold,
        review_target_recall=review_target_recall,
        block_target_precision=block_target_precision,
        validation_review_precision=review_precision,
        validation_review_recall=review_recall,
        validation_block_precision=block_precision,
        validation_block_recall=block_recall,
    )

