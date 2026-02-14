# Bank Marketing Classification (Logistic Regression)

## Objective
Predict whether a customer will subscribe to a term deposit using Logistic Regression and analyze performance using ROC-AUC and threshold optimization.

## Dataset
Bank Marketing Dataset (bank.csv)

Target variable: `y`

## Approach

- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- Logistic Regression in a Pipeline
- Train/Test split with stratification
- ROC-AUC evaluation
- Threshold optimization using Youden’s J statistic

## Metrics

- Confusion Matrix
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- ROC-AUC

## Key Features

- Probability-based predictions
- Optimized classification threshold
- Business-oriented interpretation of Recall vs Precision tradeoff

## Output

- probabilities.csv – predicted probabilities and labels

## How to Run

```bash
python bank_logistic.py
Key Concepts
Logistic Regression

ROC Curve & ROC-AUC

Threshold tuning

Classification metrics

Business decision tradeoffs

yaml
Copy code
