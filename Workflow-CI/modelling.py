import argparse
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Argument parsing (untuk MLflow Project/CLI)
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=200)
parser.add_argument('--max_depth', type=int, default=20)
parser.add_argument('--min_samples_split', type=int, default=2)
args = parser.parse_args()

# Load data
DATA_DIR = "drybean_preprocessing"
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Logging manual param & metric, artefak:
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("max_depth", args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

# Simpan artefak model
joblib.dump(model, "best_model.pkl")
mlflow.log_artifact("best_model.pkl")

# Simpan confusion matrix png
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
mlflow.log_artifact("confusion_matrix.png")

# Simpan classification report txt
with open("classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))
mlflow.log_artifact("classification_report.txt")

print("Akurasi:", acc)
print("Param:", args)
print("Training selesai dengan MLflow Project & param CLI.")

