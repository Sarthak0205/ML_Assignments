# ML_Assignments

Collection of Machine Learning assignments organized by problem domain.

Each folder contains implementation code, outputs, and a short report for the respective assignment.

## Assignment Folders

### 🩺 Breast Cancer Prediction
Binary classification project to predict malignant vs benign tumors using supervised learning techniques.

Key concepts:
- Classification
- Feature scaling
- Model evaluation

---

### 🏦 Bank Logistics / Marketing Prediction
Predicts customer subscription using Logistic Regression with ROC-AUC analysis and optimized classification thresholds.

Key concepts:
- Logistic Regression
- ROC Curve & ROC-AUC
- Threshold tuning
- Precision–Recall tradeoff

---

### 🚲 Bike Demand Forecasting
Forecasts hourly bike rental demand using ensemble regression methods.

Models used:
- Random Forest (Bagging)
- Subagging (BaggingRegressor)
- Gradient Boosting

Evaluated using 5-Fold Cross Validation.

Random Forest achieved the best performance due to effective variance reduction and robustness to noisy demand patterns.

Key concepts:
- Bagging, Subagging, Boosting
- Random Forest
- K-Fold Cross Validation
- Bias–Variance Tradeoff
- RMSE & MAE

---

## Usage

Navigate to any assignment folder and run the notebook or script provided.

Example:

```bash
cd 04
jupyter notebook Bike_Demand_Forecasting.ipynb
Author: Sarthak