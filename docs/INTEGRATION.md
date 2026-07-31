# Future integration guide

This document describes the next phase only. The current repository remains a
detection engine and does not ship an API or payment-system adapter.

## Recommended boundary

```text
Mobile/web client
        |
        v
Payment backend  -- authentication, limits, ledger, audit, idempotency
        |
        v
Internal fraud service -- MobileGuard model + thresholds
        |
        v
APPROVE / REVIEW / BLOCK recommendation
```

The payment backend, not the model service, owns the final transaction state.
Treat `BLOCK` as a temporary hold or step-up-verification recommendation.

## Request contract

The future internal endpoint can accept:

```json
{
  "transaction_id": "tx_123",
  "step": 25,
  "type": "TRANSFER",
  "amount": 250000,
  "oldbalanceOrg": 250000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 10000,
  "newbalanceDest": 10000
}
```

`transaction_id` is for tracing and idempotency; it is not a model feature.

## Response contract

```json
{
  "transaction_id": "tx_123",
  "fraud_probability": 0.92,
  "risk_level": "HIGH",
  "decision": "BLOCK",
  "model_version": "0.1.0"
}
```

## Implementation sequence

1. Freeze and version `mobile_fraud_model.joblib`,
   `fraud_thresholds.json`, and the seven-field input schema together.
2. Wrap one process-wide `MobileGuardPredictor` in an internal FastAPI service.
3. Add health/readiness checks that confirm both artifacts load successfully.
4. Require service authentication, TLS, request limits, structured audit logs,
   and correlation IDs.
5. Add a short timeout and a documented fail-safe policy in the payment backend.
6. Start in shadow mode: score transactions but do not change user outcomes.
7. Compare predictions with confirmed outcomes and measure precision, recall,
   captured amount, false-positive cost, and latency.
8. Introduce `REVIEW` first, then temporary `BLOCK` only after operational
   approval and monitoring are mature.
9. Record model version, thresholds, inputs, output, and eventual outcome for
   every scored transaction without logging unnecessary personal data.
10. Monitor data quality, score distributions, drift, queue capacity, and model
    performance; establish rollback and retraining procedures.

## Production features to add later

- transaction velocity over 10 minutes and 24 hours
- new-receiver and unique-receiver counts
- deviation from a sender's normal amount
- time since previous transaction
- device, IP, and geographic changes
- failed authentication and OTP attempts

These require a database or online feature store and point-in-time-correct
feature computation. They must not be computed using information that became
available after the transaction being scored.

