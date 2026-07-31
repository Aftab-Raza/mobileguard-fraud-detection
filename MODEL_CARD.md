# MobileGuard model card

## Intended use

MobileGuard 0.1.0 is a learning and prototyping detector for PaySim-format
mobile money transactions. It ranks transaction risk and recommends `APPROVE`,
`REVIEW`, or temporary `BLOCK`.

It is not approved for autonomous real-world financial decisions. A score is
not evidence of wrongdoing.

## Training data

- Dataset: PaySim synthetic mobile money transactions
- Total rows: 6,362,620
- Fraud rows: 8,213
- Fraud prevalence: 0.1291%
- Model inputs: simulation step, transaction type, amount, and four
  before/after balance values
- Excluded inputs: raw account identifiers and `isFlaggedFraud`

The source rows were ordered by `step`. Entire simulation steps were assigned
to one partition only.

## Reproducible run recorded on 2026-07-31

| Partition | Rows | Use |
|---|---:|---|
| Historical training partition | 4,433,703 | Available for fitting |
| Stratified training sample | 300,000 | Used for this laptop run |
| Validation | 973,173 | Model choice and thresholds |
| Future test | 955,744 | Final evaluation only |

Candidate validation results:

| Model | Average Precision | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.996389 | 0.999999 |
| Random Forest | 0.999935 | 1.000000 |

Random Forest was selected by validation Average Precision. Validation selected
a review threshold of `0.95` and a block threshold of `0.956172`.

Held-out future test results at the review threshold:

| Metric | Result |
|---|---:|
| Average Precision | 0.999991 |
| ROC-AUC | 1.000000 |
| Precision | 1.000000 |
| Recall | 0.886783 |
| F1 | 0.939995 |
| True positives | 3,556 |
| False positives | 0 |
| False negatives | 454 |
| True negatives | 951,734 |
| Fraud amount captured | 82.54% |

The test recall is lower than the 90% validation target, illustrating why a
future-time test is necessary.

## Why these results require caution

The near-perfect ranking is not evidence that real mobile money fraud is easy.
PaySim is synthetic, and balance consistency is strongly connected to its
simulator mechanics. Features such as sender and receiver balance errors can
therefore separate PaySim fraud much more cleanly than they may separate real
fraud.

Raw class-weighted model scores are also not guaranteed to be calibrated
real-world probabilities. Before deployment, the model must be validated and
calibrated on representative, point-in-time real data.

## Required production safeguards

- shadow-mode evaluation before any customer impact
- human or step-up-authentication path for reviews and temporary blocks
- data-quality, drift, latency, and score-distribution monitoring
- labeled-outcome collection and recurring threshold review
- model/version audit trail and rollback capability
- business rules, authentication, and transaction limits alongside the model
- fairness and disparate-impact assessment on legally appropriate attributes

