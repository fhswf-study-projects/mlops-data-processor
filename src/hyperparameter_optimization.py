import optuna
import mlflow
import mlflow.sklearn
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def optimized_model_exists(model_name: str) -> bool:
    """
    Checks if an optimized model already exists.

    Parameters:
        model_name (str): Name of the model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    model_path = f"models/{model_name}_optimized_model.joblib"
    return os.path.exists(model_path)


def optimize_hyperparameters(X, y, preprocessor, model_type="random_forest"):
    """
    Runs hyperparameter optimization using Optuna and saves the best model.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        preprocessor (Pipeline): Preprocessing pipeline.
        model_type (str): Model to optimize. Options: "random_forest", "logistic_regression", "svm", "xgboost".

    Returns:
        dict: The best hyperparameters found by Optuna.
    """
    optimized_model_name = f"{model_type}_optimized_model.joblib"

    # Check if the optimized model already exists
    if model_exists(optimized_model_name):
        print(f"Skipping hyperparameter optimization for {model_type}, already exists.")
        return {}

    mlflow.set_experiment(f"income_classification_hyperopt_{model_type}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def objective(trial):
        """Defines the objective function for Optuna hyperparameter tuning."""
        params = {}
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
            }
            model = RandomForestClassifier(**params, random_state=42)

        elif model_type == "logistic_regression":
            params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True), "max_iter": 1000}
            model = LogisticRegression(**params)

        elif model_type == "svm":
            params = {
                "C": trial.suggest_float("C", 0.1, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }
            model = SVC(**params, probability=True)

        elif model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
            }
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log Confusion Matrix to MLflow
        with mlflow.start_run():
            cm_filename = save_confusion_matrix(cm, f"optuna_{model_type}")
            mlflow.log_artifact(cm_filename)

        return accuracy

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print(f"Best hyperparameters for {model_type}: {best_params}")

    # Train the final optimized model
    final_model = None
    if model_type == "random_forest":
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_type == "logistic_regression":
        final_model = LogisticRegression(**best_params)
    elif model_type == "svm":
        final_model = SVC(**best_params, probability=True)
    elif model_type == "xgboost":
        final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Unsupported model type")

    # Train final model
    final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", final_model)])
    final_pipeline.fit(X_train, y_train)

    # Save final model
    model_filename = f"models/{optimized_model_name}_model.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_pipeline, model_filename)
    print(f"Optimized model saved as {model_filename}")

    return best_params


def save_confusion_matrix(cm, model_name):
    """
    Saves the confusion matrix as an image.

    Parameters:
        cm (array): Confusion matrix.
        model_name (str): Name of the model.

    Returns:
        str: Path to the saved confusion matrix image.
    """
    filename = f"results/confusion_matrix_{model_name}.png"
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(filename)
    plt.close()
    return filename
