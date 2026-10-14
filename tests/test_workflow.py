import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.train import choose_threshold

ROOT = Path(__file__).resolve().parents[1]


def test_threshold_maximizes_observed_validation_f1():
    y, scores = pd.Series([0, 0, 1, 1, 0, 1]), np.array([0.05, 0.3, 0.4, 0.9, 0.6, 0.7])
    threshold = choose_threshold(y, scores)
    assert f1_score(y, scores >= threshold) == max(
        f1_score(y, scores >= t) for t in scores
    )


def test_training_prediction_and_schema_errors(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(120, 30)),
        columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
    )
    frame["Class"] = [0, 1] * 60
    data, artifact, report = (
        tmp_path / "data.csv",
        tmp_path / "model.joblib",
        tmp_path / "metrics.json",
    )
    frame.to_csv(data, index=False)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/train.py"),
            "--data",
            str(data),
            "--output",
            str(artifact),
            "--metrics",
            str(report),
        ],
        check=True,
        capture_output=True,
    )
    metrics = json.loads(report.read_text())
    assert metrics["split_rows"] == {"train": 72, "validation": 24, "test": 24}
    out = tmp_path / "prediction.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/predict.py"),
            "--input",
            str(data),
            "--model",
            str(artifact),
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    result = pd.read_csv(out)
    assert len(result) == 120 and result["fraud_probability"].between(0, 1).all()
    frame.drop(columns="V1").to_csv(data, index=False)
    failed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/predict.py"),
            "--input",
            str(data),
            "--model",
            str(artifact),
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "Missing required feature" in failed.stderr
