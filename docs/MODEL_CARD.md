# Model card: Credit card fraud detection

## Purpose and scope

Research classification of anonymized transactions using the ULB/Kaggle fraud benchmark.
The positive class is `Class=1` (fraud). It is a portfolio experiment, not a deployed bank decision system.

## Inputs and artifact

30 finite numeric features: `Time`, `V1`–`V28` and `Amount`. Keep the original feature semantics and units.
PCA features are already anonymized by the dataset provider. The saved bundle contains the fitted model,
feature order and threshold. Fresh training additionally records environment versions, source fingerprint,
selected model and seed. The committed original model predates these metadata fields.

## Evaluation

Exact duplicate rows removed, then stratified 60/20/20 train/validation/test split with seed 42.
Logistic Regression and Random Forest are compared by validation Average Precision; the selected model's
threshold maximizes validation F1. Test data is used only after those choices are fixed.
The independently reproduced test precision is 0.8846, recall 0.7263 and F1 0.7977.
See [the recorded run](../reports/metrics.json) and the README for the full confusion matrix and counts.

## Interpretation and limits

`fraud_probability` is an uncalibrated model score. A default F1 threshold does not express bank loss,
customer friction or review capacity. Random row splitting does not measure future-time performance.
The dataset's rare-fraud rate may differ from deployment; changing prevalence changes precision.
The historical notebook includes exploratory split reuse; the CLI is the consistent benchmark path.

## Reproduction

```bash
python src/train.py --data data/creditcard.csv
python src/predict.py --input examples/transactions.csv --output predictions.csv
```

The example rows are illustrative schema examples, not evaluation evidence. Only load trusted Joblib files.
