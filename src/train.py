import mlflow
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os


def train_model(X, y, preprocessor, optimized=False, best_params=None):
    """Trainiert ein Machine Learning Modell und speichert es mit MLflow."""

    mlflow.set_experiment("income_classification")

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Falls Hyperparameter-Optimierung genutzt wird
    if optimized and best_params:
        if "n_estimators" in best_params:
            model = RandomForestClassifier(n_estimators=int(best_params["n_estimators"]),
                                           max_depth=int(best_params["max_depth"]),
                                           random_state=42)
        elif "C" in best_params:
            model = LogisticRegression(C=float(best_params["C"]), max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)  # Standardwerte

    # Pipeline mit Preprocessing und Modell
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Vorhersagen & Evaluierung
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # MLflow Logging
    with mlflow.start_run():
        mlflow.log_param("model", type(model).__name__)
        if optimized:
            mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Speichern der Confusion Matrix als Bild
        cm_filename = "results/confusion_matrix.png"
        os.makedirs("results", exist_ok=True)
        plot_confusion_matrix(cm, cm_filename)
        mlflow.log_artifact(cm_filename)

        # Modell in MLflow speichern
        mlflow.sklearn.log_model(pipeline, "best_model")

    # Modell lokal speichern
    model_filename = "models/final_model.joblib" if optimized else "models/baseline_model.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Modell gespeichert unter {model_filename} mit Accuracy: {accuracy:.4f}")


def plot_confusion_matrix(cm, filename):
    """Erstellt eine Confusion Matrix und speichert sie als Bild."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
