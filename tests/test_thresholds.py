import numpy as np

from mobileguard.thresholds import select_thresholds


def test_thresholds_are_ordered_and_data_derived():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.01, 0.05, 0.10, 0.40, 0.50, 0.70, 0.90, 0.99])

    thresholds = select_thresholds(
        labels,
        scores,
        review_target_recall=0.75,
        block_target_precision=1.0,
    )

    assert 0 <= thresholds.review < thresholds.block <= 1
    assert thresholds.validation_review_recall >= 0.75
    assert thresholds.validation_block_precision == 1.0

