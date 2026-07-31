import pandas as pd
import pytest

from mobileguard.features import engineer_features
from mobileguard.schema import DataValidationError


def transaction(**overrides):
    row = {
        "step": 25,
        "type": "TRANSFER",
        "amount": 250_000.0,
        "oldbalanceOrg": 250_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 10_000.0,
        "newbalanceDest": 10_000.0,
    }
    row.update(overrides)
    return row


def test_expected_balance_and_time_features():
    features = engineer_features(pd.DataFrame([transaction()])).iloc[0]

    assert features["hour"] == 1
    assert features["day"] == 1
    assert features["origin_balance_error"] == 0
    assert features["destination_balance_error"] == 250_000
    assert features["amount_equals_origin_balance"] == 1
    assert features["origin_zero_after"] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"type": "WIRE"}, "Invalid transaction type"),
        ({"amount": -1}, "amount"),
        ({"oldbalanceOrg": None}, "oldbalanceOrg"),
    ],
)
def test_invalid_input_is_rejected(change, message):
    with pytest.raises(DataValidationError, match=message):
        engineer_features(pd.DataFrame([transaction(**change)]))

