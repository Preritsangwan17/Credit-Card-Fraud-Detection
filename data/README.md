# Dataset setup

Download the Credit Card Fraud Detection dataset from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Extract `creditcard.csv` into this directory so the final path is:

```text
data/creditcard.csv
```

The full CSV is intentionally excluded from Git because it is approximately 151 MB, above GitHub's normal 100 MB per-file limit.

Expected columns:

```text
Time, V1, V2, ..., V28, Amount, Class
```

`Class = 0` means genuine and `Class = 1` means fraud.

