# Credit Card Fraud Detection

An end-to-end machine-learning project for detecting fraudulent credit-card transactions in a highly imbalanced dataset. The project compares Logistic Regression and Random Forest, selects the stronger model using validation Average Precision, tunes the classification threshold on validation data, and evaluates the final model once on a held-out test set.

## Project highlights

- Removes exact duplicate rows before model development.
- Uses stratified train/validation/test splits so every split preserves the rare fraud class.
- Compares a class-weighted Logistic Regression pipeline with a class-weighted Random Forest.
- Selects the model using Average Precision, which is more informative than accuracy for rare-event detection.
- Tunes the decision threshold by maximizing validation F1 score.
- Saves the trained model, threshold, and required feature order together in one Joblib artifact.
- Includes reusable command-line scripts for training and batch prediction.

## Model-development architecture

```text
creditcard.csv
      |
      v
Remove exact duplicates
      |
      v
Stratified split: 60% train / 20% validation / 20% final test
      |
      +--> Logistic Regression (scale Time and Amount)
      |
      +--> Random Forest (class-balanced subsampling)
                  |
                  v
     Select highest validation Average Precision
                  |
                  v
     Tune threshold on validation precision-recall curve
                  |
                  v
       Evaluate once on untouched final test data
                  |
                  v
     Save model + threshold + feature names
```

## Results from the completed notebook

### Validation model comparison

| Model | Validation Average Precision |
|---|---:|
| Logistic Regression | 0.7868 |
| Random Forest | **0.8488** |

Random Forest was selected. The validation-tuned decision threshold was `0.4796291900916881`.

### Final held-out test results

| Metric | Score |
|---|---:|
| Accuracy | 0.999383 |
| Precision | **0.884615** |
| Recall | **0.726316** |
| F1 score | **0.797688** |
| Average Precision | **0.769861** |

Final confusion matrix:

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actual genuine | 56,642 | 9 |
| Actual fraud | 26 | 69 |

Accuracy is shown for completeness, but it is not the main success metric because genuine transactions greatly outnumber frauds. Precision, recall, F1, Average Precision, and the confusion matrix give the more meaningful picture.

## Repository structure

```text
.
├── data/
│   └── README.md
├── models/
│   ├── README.md
│   └── creditcard_fraud_model.joblib
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── src/
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project uses the public **Credit Card Fraud Detection** dataset published on Kaggle by the Machine Learning Group at Université Libre de Bruxelles. It contains anonymized PCA features `V1`–`V28`, plus `Time`, `Amount`, and the target `Class` (`0` = genuine, `1` = fraud).

Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the extracted file at:

```text
data/creditcard.csv
```

The CSV is not committed because it is approximately 151 MB and exceeds GitHub's normal 100 MB per-file limit.

## Quick start

```bash
git clone https://github.com/Preritsangwan17/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After placing `creditcard.csv` inside `data/`, train and save a fresh model:

```bash
python src/train.py
```

Run predictions on a CSV containing the 30 input features:

```bash
python src/predict.py --input data/creditcard.csv --output predictions.csv
```

To explore the work step by step:

```bash
jupyter lab notebooks/credit_card_fraud_detection.ipynb
```

## Important model note

The included Joblib file was created with scikit-learn 1.6.1. Only load model files from trusted sources because serialized Python artifacts can execute code during loading.

## Author

**Prerit Sangwan**  
B.Tech CSE (AI/ML) student  
GitHub: [@Preritsangwan17](https://github.com/Preritsangwan17)

