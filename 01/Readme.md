

# Tumor Classification: Machine Learning Workflow

This project demonstrates a complete machine learning pipeline for classifying breast tumors as **Malignant (M)** or **Benign (B)**. Using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, the project compares the performance of a linear model (**Logistic Regression**) against a non-linear model (**Decision Tree**), with a focus on error analysis and model generalization.

## 📂 Dataset Overview

* **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
* **Target Variable:** `diagnosis` (M = Malignant, B = Benign).
* **Features:** 30 real-valued features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. These describe characteristics of the cell nuclei present in the image, such as:
* Radius, Texture, Perimeter, Area, Smoothness
* Compactness, Concavity, Concave points, Symmetry, Fractal dimension
* *Note: For each characteristic, the Mean, Standard Error (SE), and "Worst" (mean of the three largest values) were calculated.*



## 🛠️ Workflow Steps

1. **Data Loading & Parsing:**
* Loaded raw data and applied column definitions based on `wdbc.names`.
* Separated features (`X`) from the target (`y`) and dropped identifiers (`ID`).


2. **Preprocessing:**
* **Encoding:** Mapped target variable to numerical values (Malignant=1, Benign=0).
* **Splitting:** Divided data into Training (80%) and Testing (20%) sets.
* **Scaling:** Applied `StandardScaler` to normalize features (crucial for Logistic Regression).


3. **Model Training:**
* **Logistic Regression:** Trained as a baseline linear model.
* **Decision Tree:** Trained as a non-linear model to test predictive complexity.


4. **Evaluation:**
* Metrics: Accuracy, Precision, Recall, F1-Score.
* Visualization: Confusion Matrices.
* Analysis: Bias-Variance tradeoff and Overfitting detection.



## 📊 Results Summary

The models were evaluated on a held-out test set (20% of data).

| Metric | Logistic Regression | Decision Tree |
| --- | --- | --- |
| **Test Accuracy** | **97.37%** | 94.74% |
| **Precision** | **0.9762** | 0.9302 |
| **Recall** | **0.9535** | 0.9302 |
| **F1-Score** | **0.9647** | 0.9302 |
| **Training Accuracy** | 98.68% | 100.00% |

### Key Findings

1. **Best Performer:** Logistic Regression outperformed the Decision Tree on all test metrics.
2. **Overfitting:** The Decision Tree achieved **100% training accuracy** but dropped to **~94% testing accuracy**, indicating significant overfitting (it memorized the training noise). Logistic Regression showed a much smaller generalization gap.
3. **Error Analysis:**
* **False Negatives (Critical):** Logistic Regression missed only **2** malignant cases, whereas the Decision Tree missed **3**. In medical diagnostics, minimizing False Negatives is often prioritized over Precision.
* **False Positives:** Logistic Regression had only **1** false alarm, compared to **3** for the Decision Tree.



## 🚀 How to Run

### Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install pandas scikit-learn matplotlib

```

### Execution

1. Download the `ML_Assgn_01.ipynb` file.
2. Ensure the dataset file `wdbc.data` is in the correct path (as specified in the notebook).
3. Run the notebook cells sequentially to reproduce the analysis.

## 📝 Conclusion

For this specific dataset, the simpler **Logistic Regression** model proved to be more robust and generalized better than the unpruned Decision Tree. The linear decision boundary was sufficient to separate the classes effectively. Future improvements could include hyperparameter tuning (pruning) for the Decision Tree or implementing Ensemble methods (Random Forest, Gradient Boosting) to reduce variance.