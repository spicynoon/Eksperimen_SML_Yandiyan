import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path ke hasil preprocessing (pastikan folder ini ada di project)
DATA_DIR = "drybean_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Tracking MLflow (default lokal)
mlflow.set_experiment("drybean_rf_baseline")
mlflow.sklearn.autolog()  # Aktifkan autolog SEBELUM start_run

with mlflow.start_run(run_name="RF_baseline"):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    print("Mulai training...")
    model.fit(X_train, y_train)
    print("Training selesai.")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Akurasi test:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # --- Artefak tambahan (confusion matrix, classification report) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

    # (Tidak perlu log_model manual, sudah otomatis oleh autolog)
    # mlflow.sklearn.log_model(model, "model")

print("\nTraining selesai. Jalankan 'mlflow ui' dan buka http://localhost:5000 untuk cek hasil MLflow.")
