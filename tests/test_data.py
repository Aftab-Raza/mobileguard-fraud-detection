import pandas as pd

from mobileguard.data import chronological_split, stratified_training_sample


def test_chronological_split_keeps_steps_separate():
    frame = pd.DataFrame(
        {
            "step": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
            "isFraud": [0, 1, 0, 0, 0] * 4,
        }
    )
    train, validation, test = chronological_split(
        frame, train_fraction=0.50, validation_fraction=0.25
    )

    assert set(train["step"]).isdisjoint(validation["step"])
    assert set(validation["step"]).isdisjoint(test["step"])
    assert train["step"].max() < validation["step"].min() < test["step"].min()


def test_training_sample_preserves_both_classes():
    frame = pd.DataFrame(
        {
            "step": range(1_000),
            "isFraud": [1] * 20 + [0] * 980,
        }
    )
    sample = stratified_training_sample(frame, max_rows=100, random_state=42)

    assert len(sample) == 100
    assert sample["isFraud"].sum() == 2

