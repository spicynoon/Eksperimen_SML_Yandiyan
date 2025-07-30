import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ==== DagsHub Integration ====
from dagshub import dagshub_logger
import dagshub
dagshub.init(repo_owner="spicynoon", repo_name="Eksperimen_SML_Yandiyan", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/spicynoon/Eksperimen_SML_Yandiyan.mlflow")
mlflow.set_experiment("drybean_rf_tuning")

# ==== Data Path ====
DATA_DIR = "drybean_preprocessing"
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

with mlflow.start_run(run_name="RF_tuning_manual_logging"):
    # 1. Hyperparameter Tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    print("Mulai GridSearch...")
    grid.fit(X_train, y_train)
    print("GridSearch selesai.")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # 2. Manual Logging Parameter dan Metric
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])

    # 3. Log Model as artifact (manual, NOT using log_model)
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")

    # 4. Artefak: Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    # 5. Artefak: Classification Report
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

    print("Best params:", grid.best_params_)
    print("Accuracy test:", acc)
    print("Run ID:", mlflow.active_run().info.run_id)

print("\nTuning, manual logging, dan artefak sudah tercatat di MLflow DagsHub")
