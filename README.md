# MobileGuard

MobileGuard is a detection-only machine-learning engine for suspicious mobile
money transfers. It trains on the synthetic PaySim dataset and returns a fraud
probability, risk level, and one of three decisions:

- `APPROVE` — probability is below the validation-tuned review threshold.
- `REVIEW` — probability is between the review and block thresholds.
- `BLOCK` — probability is at or above the block threshold.

This repository intentionally contains no frontend, web API, payment logic, or
automatic enforcement. It estimates risk; it does not prove that a person
committed fraud.

## What is implemented

- PaySim schema and value validation
- No-login download of the complete 6,362,620-row PaySim Parquet dataset
- Leakage-safe feature engineering shared by training and inference
- Strict past-to-future train/validation/test splitting
- Class-weighted Logistic Regression baseline
- Class-weighted Random Forest comparison
- Model selection by validation Average Precision (PR-AUC)
- Review threshold tuned for a target fraud recall
- Block threshold tuned for a target fraud precision
- Final one-time test evaluation
- Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, decision counts,
  captured fraud amount, and missed fraud amount
- Saved preprocessing/model pipeline and threshold JSON
- Single-transaction and batch prediction CLIs
- Automated tests and a GitHub Actions workflow

Raw account identifiers and `isFlaggedFraud` are validated in the source
dataset but excluded from model inputs. This prevents account memorization and
leakage from PaySim's existing rule flag.

## 1. Open in VS Code

Open this `mobileguard` folder, then open **Terminal > New Terminal**. All
commands below assume the terminal is at the repository root.

## 2. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Command Prompt:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

In VS Code, select `.venv\Scripts\python.exe` using **Python: Select
Interpreter**.

Python 3.11 or newer is required. If editable installation is not desired, use
`python -m pip install -r requirements.txt` and run commands as
`python -m mobileguard ...` from an installed package context.

## 3. Download and validate PaySim

```powershell
mobileguard download-data
mobileguard profile
```

The dataset is stored at `data/raw/paysim.parquet` (about 274 MB). The compact
EDA output is stored at `reports/dataset_profile.json`. Data files and generated
reports are ignored by Git.

The downloader uses the public Hugging Face Parquet mirror of the
[PaySim dataset](https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection).
The original simulator is maintained in the
[PaySim repository](https://github.com/EdgarLopezPhD/PaySim), and the dataset is
also published on
[Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). Review the dataset's
terms before redistributing it; the dataset is not bundled with this code.

## 4. Train the detection engine

Recommended laptop run:

```powershell
mobileguard train
```

This compares both supported models. It preserves the fraud rate while capping
the historical training partition at 300,000 rows, then evaluates against the
complete chronological validation and test partitions.

Use every historical training row when memory and runtime allow:

```powershell
mobileguard train --max-train-rows 0
```

Run only the fast, interpretable baseline:

```powershell
mobileguard train --models logistic
```

Generated files:

```text
artifacts/
  mobile_fraud_model.joblib
  fraud_thresholds.json
reports/
  dataset_profile.json
  model_comparison.json
  test_metrics.json
```

The test set is not used for model choice or threshold tuning. A simulation
`step` is never split across two partitions. See [MODEL_CARD.md](MODEL_CARD.md)
for the recorded run's validation/test metrics and their interpretation.

## 5. Detect one transaction

```powershell
mobileguard predict --input examples/transaction.json
```

Output shape:

```json
{
  "decision": "BLOCK",
  "fraud_probability": 0.92,
  "model_version": "0.1.0",
  "risk_level": "HIGH"
}
```

The exact probability and decision depend on the trained model and
validation-derived thresholds.

Use the Python interface:

```python
from mobileguard import MobileGuardPredictor

detector = MobileGuardPredictor()
result = detector.predict(
    {
        "step": 25,
        "type": "TRANSFER",
        "amount": 250000,
        "oldbalanceOrg": 250000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 10000,
        "newbalanceDest": 10000,
    }
)
print(result)
```

Load `MobileGuardPredictor` once when a process starts. Do not reload the model
for every transaction.

For a CSV or Parquet file containing the seven required input columns:

```powershell
mobileguard predict-batch --input transactions.csv --output decisions.csv
```

## Input contract

| Field | Type | Rule |
|---|---:|---|
| `step` | integer | Non-negative PaySim hour index |
| `type` | string | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, or `TRANSFER` |
| `amount` | number | Non-negative |
| `oldbalanceOrg` | number | Sender balance before; non-negative |
| `newbalanceOrig` | number | Sender balance after; non-negative |
| `oldbalanceDest` | number | Receiver balance before; non-negative |
| `newbalanceDest` | number | Receiver balance after; non-negative |

## Feature engineering

The saved pipeline creates:

- hour and day from `step`
- `log1p(amount)`
- signed and absolute sender/receiver balance errors
- amount-to-balance ratios (with a denominator floor of 1 for zero balances)
- sender and receiver balance changes
- four zero-balance indicators
- full-origin-balance transfer indicator
- one-hot encoded transaction type

Logistic Regression receives standardized numeric values. Random Forest uses
the same deterministic preprocessing so the saved winning pipeline can accept
the same raw input.

## Tests

```powershell
pytest
```

## Future software integration

The safest next step is a small internal prediction service that loads the two
artifact files once and calls `MobileGuardPredictor`. Keep the payment
application responsible for authentication, limits, audit logs, idempotency,
and the final business action. See [docs/INTEGRATION.md](docs/INTEGRATION.md)
for the future request/response contract and rollout plan. No integration code
is included in this detection-only version.

## Important limitations

1. PaySim is synthetic and does not prove performance on real transactions.
2. The model uses transaction-level values, not device, IP, location, login, or
   historical customer behavior.
3. PaySim balance-error features may partly reflect simulator mechanics.
4. Fraud behavior and input distributions change; production use needs drift
   monitoring, labeled outcomes, threshold review, and retraining.
5. Class weighting improves rare-class learning but raw scores should not be
   treated as perfectly calibrated real-world probabilities.
6. Review and block thresholds depend on operational capacity and the cost of
   false positives/false negatives.
7. A fraud model should support authentication and business rules, never
   replace them.
8. A `BLOCK` result should normally mean a temporary hold or stronger
   verification—not an irreversible accusation.

## License

The MobileGuard source code is licensed under the MIT License. Dataset terms
are separate and are controlled by the dataset publisher.
