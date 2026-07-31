"""Model construction, comparison, threshold tuning, and artifact export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mobileguard import __version__
from mobileguard.data import (
    chronological_split,
    load_training_data,
    stratified_training_sample,
    write_json,
)
from mobileguard.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PaySimFeatureEngineer,
)
from mobileguard.metrics import evaluate_probabilities
from mobileguard.schema import RAW_MODEL_COLUMNS
from mobileguard.thresholds import select_thresholds

SUPPORTED_MODELS = ("logistic", "random_forest")


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "transaction_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "logistic":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
            solver="lbfgs",
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    return Pipeline(
        steps=[
            ("features", PaySimFeatureEngineer()),
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return (
        frame.loc[:, RAW_MODEL_COLUMNS],
        frame["isFraud"].to_numpy(dtype=np.int8),
    )


def train_engine(
    data_path: str | Path,
    *,
    artifact_dir: str | Path = "artifacts",
    report_dir: str | Path = "reports",
    model_names: tuple[str, ...] = SUPPORTED_MODELS,
    max_train_rows: int | None = 300_000,
    review_target_recall: float = 0.90,
    block_target_precision: float = 0.75,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train candidates, select on validation AP, then evaluate test once."""

    unknown = sorted(set(model_names) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}")
    if not model_names:
        raise ValueError("At least one model must be requested")

    full = load_training_data(data_path)
    train, validation, test = chronological_split(full)
    sampled_train = stratified_training_sample(
        train, max_train_rows, random_state=random_state
    )
    del full

    X_train, y_train = _xy(sampled_train)
    X_validation, y_validation = _xy(validation)
    comparison: dict[str, Any] = {}
    candidates: dict[str, tuple[Pipeline, Any]] = {}

    for model_name in model_names:
        pipeline = build_pipeline(model_name, random_state=random_state)
        pipeline.fit(X_train, y_train)
        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        thresholds = select_thresholds(
            y_validation,
            validation_scores,
            review_target_recall=review_target_recall,
            block_target_precision=block_target_precision,
        )
        comparison[model_name] = {
            "average_precision": float(
                average_precision_score(y_validation, validation_scores)
            ),
            "roc_auc": float(roc_auc_score(y_validation, validation_scores)),
            "thresholds": thresholds.to_dict(),
        }
        candidates[model_name] = (pipeline, thresholds)

    best_name = max(
        model_names,
        key=lambda name: (
            comparison[name]["average_precision"],
            comparison[name]["roc_auc"],
        ),
    )
    best_pipeline, best_thresholds = candidates[best_name]

    X_test, y_test = _xy(test)
    test_scores = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(
        y_test,
        test_scores,
        test["amount"].to_numpy(dtype=np.float64),
        review_threshold=best_thresholds.review,
        block_threshold=best_thresholds.block,
    )

    created_at = datetime.now(UTC).isoformat()
    metadata = {
        "engine_version": __version__,
        "created_at_utc": created_at,
        "model_name": best_name,
        "input_columns": RAW_MODEL_COLUMNS,
        "training_rows_available": int(len(train)),
        "training_rows_used": int(len(sampled_train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "random_state": random_state,
        "selection_metric": "validation_average_precision",
    }

    artifact_path = Path(artifact_dir) / "mobile_fraud_model.joblib"
    threshold_path = Path(artifact_dir) / "fraud_thresholds.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "metadata": metadata,
        },
        artifact_path,
        compress=3,
    )
    write_json(
        {
            "schema_version": 1,
            "model_name": best_name,
            **best_thresholds.to_dict(),
        },
        threshold_path,
    )

    report_root = Path(report_dir)
    write_json(
        {
            "selected_model": best_name,
            "selection_metric": "validation_average_precision",
            "models": comparison,
        },
        report_root / "model_comparison.json",
    )
    write_json(
        {
            "metadata": metadata,
            "thresholds": best_thresholds.to_dict(),
            "test_metrics": test_metrics,
        },
        report_root / "test_metrics.json",
    )

    return {
        "selected_model": best_name,
        "artifact": str(artifact_path),
        "thresholds": str(threshold_path),
        "model_comparison": comparison,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }

