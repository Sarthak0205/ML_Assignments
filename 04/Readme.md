# Bike Demand Forecasting (Ensemble Regression)

## Objective
Predict hourly bike rental demand using ensemble regression techniques and compare model performance using 5-Fold Cross Validation.

## Dataset
UCI Bike Sharing Dataset (hour.csv)

Target variable: `cnt`

## Models Used

- Random Forest (Bagging)
- BaggingRegressor (Subagging)
- Gradient Boosting

## Evaluation

Metrics:
- RMSE
- MAE

Cross-validation:
- K = 5

## Results

| Model | RMSE | MAE |
|------|------|-----|
| Random Forest | 45.26 | 27.50 |
| Subagging | 50.81 | 32.03 |
| Boosting | 71.44 | 48.56 |

Random Forest generalized best due to effective variance reduction and robustness to noisy hourly demand patterns.

## Files

- Bike_Demand_Forecasting.ipynb – main notebook
- cv_regression_results.csv – cross-validation metrics
- final_predictions.csv – actual vs predicted values

## How to Run

Open the notebook:

```bash
jupyter notebook Bike_Demand_Forecasting.ipynb
Key Concepts
Bagging vs Subagging vs Boosting

Bias–Variance Tradeoff

K-Fold Cross Validation

Feature Importance

yaml
Copy code
