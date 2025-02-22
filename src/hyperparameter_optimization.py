import optuna
import mlflow
import mlflow.sklearn
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
from mlflow.models.signature import infer_signature
import os
from data_preprocessing import load_data, preprocess_data


def optimize_hyperparameters(X, y, preprocessor):
    """Führt Hyperparameter-Optimierung mit Optuna durch und gibt die besten Werte zurück."""

    mlflow.set_experiment("income_classification_hyperopt")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        """Optuna-Ziel: Hyperparameter-Optimierung für das Modell"""
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # MLflow Logging
        with mlflow.start_run():
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Speichern der Confusion Matrix als Bild
            cm_filename = "results/confusion_matrix_optuna.png"
            os.makedirs("results", exist_ok=True)
            plot_confusion_matrix(cm, cm_filename)
            mlflow.log_artifact(cm_filename)

            # **Neuer Code: Input Beispiel für MLflow**
            input_example = X_train.iloc[:1]  # Eine Beispiel-Zeile aus den Trainingsdaten
            signature = infer_signature(X_train, pipeline.predict(X_train))

            # **Modell in MLflow mit Signature speichern**
            mlflow.sklearn.log_model(pipeline, "best_model", signature=signature, input_example=input_example)

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"🏆 Beste Hyperparameter: {study.best_params}")
    return study.best_params


def plot_confusion_matrix(cm, filename):
    """Erstellt eine Confusion Matrix und speichert sie als Bild."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
