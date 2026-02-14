import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


DATA_PATH = "bank.csv"     # change path if needed
TARGET = "y"
TEST_SIZE = 0.25
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5


df = pd.read_csv(DATA_PATH, sep=";")

print("\nDataset shape:", df.shape)

X = df.drop(TARGET, axis=1)
y = df[TARGET].map({"yes": 1, "no": 0})


cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

model = Pipeline([
    ("prep", preprocessor),
    ("lr", LogisticRegression(max_iter=3000))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]


def evaluate(threshold):

    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("\n============================")
    print(f"Threshold: {threshold:.3f}")
    print("============================")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")



evaluate(DEFAULT_THRESHOLD)



roc_auc = roc_auc_score(y_test, probs)
print("\nROC-AUC:", round(roc_auc, 3))



fpr, tpr, thresholds = roc_curve(y_test, probs)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

evaluate(best_threshold)

print("\nOptimized Threshold:", round(best_threshold, 3))



output = pd.DataFrame({
    "RecordId": X_test.index,
    "Probability_yes": probs,
    "PredictedLabel": (probs >= best_threshold).astype(int)
})

output.to_csv("probabilities.csv", index=False)

print("\nprobabilities.csv generated successfully")
print("\n================ INTERPRETATION =================")

print("\nWhy ROC?")
print("ROC evaluates classifier performance across ALL thresholds and is robust to class imbalance.")
print("ROC-AUC summarizes the model’s ability to separate subscribers from non-subscribers independent of any cutoff.")

print("\nWhat happens when threshold changes?")
print("Lowering the threshold increases Recall (Sensitivity) but reduces Precision.")
print("Raising the threshold increases Precision but reduces Recall.")
print("This controls the business tradeoff between missing potential customers and wasting outreach on unlikely clients.")

print("\nMarketing Insight:")
print("Optimized threshold favors higher recall to capture more potential subscribers,")
print("accepting higher false positives since missed buyers are usually more costly than extra contact attempts.")

print("\n================================================")
