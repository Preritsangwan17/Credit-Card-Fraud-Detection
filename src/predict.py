"""Run fraud predictions with the saved model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/creditcard_fraud_model.joblib"),
        help="Saved Joblib model bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV. If omitted, only a summary is printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = joblib.load(args.model)
    model = saved["model"]
    threshold = float(saved["threshold"])
    features = list(saved["features"])

    transactions = pd.read_csv(args.input)
    missing = [column for column in features if column not in transactions.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    probabilities = model.predict_proba(transactions[features])[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = transactions.copy()
    results["fraud_probability"] = probabilities
    results["predicted_class"] = predictions

    print(f"Transactions scored: {len(results):,}")
    print(f"Predicted frauds: {int(predictions.sum()):,}")
    print(f"Decision threshold: {threshold:.6f}")
    print(results[["fraud_probability", "predicted_class"]].head(10).to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

