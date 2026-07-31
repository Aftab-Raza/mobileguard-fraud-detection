import numpy as np
import pandas as pd

from mobileguard.modeling import build_pipeline


def test_pipeline_returns_probabilities():
    rows = []
    labels = []
    for index in range(80):
        fraud = index % 10 == 0
        amount = 100_000.0 if fraud else 100.0 + index
        old_origin = amount if fraud else amount * 10
        rows.append(
            {
                "step": index + 1,
                "type": "TRANSFER" if fraud else "PAYMENT",
                "amount": amount,
                "oldbalanceOrg": old_origin,
                "newbalanceOrig": 0.0 if fraud else old_origin - amount,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0 if fraud else amount,
            }
        )
        labels.append(int(fraud))

    pipeline = build_pipeline("logistic", random_state=42)
    pipeline.fit(pd.DataFrame(rows), np.asarray(labels))
    probabilities = pipeline.predict_proba(pd.DataFrame(rows[:3]))[:, 1]

    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

