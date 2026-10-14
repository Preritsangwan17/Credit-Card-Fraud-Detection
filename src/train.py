"""Train, select, evaluate, and save a credit-card fraud model."""

from __future__ import annotations

import argparse
import json
import hashlib
import sklearn
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/creditcard.csv"),
        help="Path to the Kaggle creditcard.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/creditcard_fraud_model.joblib"),
        help="Destination for the serialized model bundle.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/metrics.json"),
        help="Destination for the machine-readable evaluation report.",
    )
    return parser.parse_args()


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Return the validation threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.argmax(f1_values))])


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data}. See data/README.md for setup."
        )

    data = pd.read_csv(args.data).drop_duplicates().copy()
    if TARGET not in data.columns:
        raise ValueError(f"Expected a '{TARGET}' target column.")

    missing = [name for name in FEATURES if name not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite numeric values")
    y = data[TARGET]
    if (
        not y.isin([0, 1]).all()
        or y.value_counts().reindex([0, 1], fill_value=0).min() < 5
    ):
        raise ValueError("Class must be 0/1, with at least 5 rows of each class")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )

    preprocess = ColumnTransformer(
        [("scale", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )
    logistic_regression = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }
    validation_scores: dict[str, float] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_probabilities = model.predict_proba(X_val)[:, 1]
        validation_scores[name] = float(
            average_precision_score(y_val, val_probabilities)
        )

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = models[best_name]
    validation_probabilities = best_model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, validation_probabilities)

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    with args.data.open("rb") as data_file:
        data_sha256 = hashlib.file_digest(data_file, "sha256").hexdigest()
    metrics = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "data_sha256": data_sha256,
        "environment": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "split_rows": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "validation_average_precision": validation_scores,
        "threshold": threshold,
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "test_recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "test_average_precision": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "threshold": threshold,
            "features": list(X.columns),
            "metadata": {
                "selected_model": best_name,
                "data_sha256": data_sha256,
                "environment": metrics["environment"],
                "random_state": RANDOM_STATE,
            },
        },
        args.output,
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model bundle to {args.output}")


if __name__ == "__main__":
    main()
