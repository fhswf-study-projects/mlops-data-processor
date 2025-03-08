import mlflow
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


def model_exists(model_name: str) -> bool:
    """
    Checks if a trained model already exists in the models/ directory.

    Parameters:
        model_name (str): Name of the model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    model_path = f"models/{model_name}_model.joblib"
    return os.path.exists(model_path)


def train_model(X, y, preprocessor, model_type="random_forest", optimized=False, best_params=None):
    """
    Trains a machine learning model and logs results with MLflow.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        preprocessor (Pipeline): Preprocessing pipeline.
        model_type (str): Model to train ("random_forest", "logistic_regression", "svm", "xgboost").
        optimized (bool): Whether to use optimized hyperparameters.
        best_params (dict): Hyperparameter dictionary.

    Returns:
        None
    """
    model_name = f"{model_type}_optimized" if optimized else model_type

    # Check if the model already exists
    if model_exists(model_name):
        print(f"Skipping training for {model_name}, already exists.")
        return

    mlflow.set_experiment("income_classification_2")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_type == "random_forest":
        model = RandomForestClassifier(**best_params) if optimized else RandomForestClassifier(n_estimators=100)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**best_params) if optimized else LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        model = SVC(**best_params) if optimized else SVC(kernel="linear", probability=True)
    elif model_type == "xgboost":
        model = XGBClassifier(**best_params) if optimized else XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Unsupported model type")

    # Create pipeline and train model
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log results to MLflow
    with mlflow.start_run():
        mlflow.log_param("model", model_type)
        if optimized:
            mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_artifact(save_confusion_matrix(cm, model_type))

        # Save model
        model_filename = f"models/{model_name}_model.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, model_filename)
        mlflow.sklearn.log_model(pipeline, "best_model")

    # Save the trained preprocessing pipeline
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print("Preprocessing pipeline saved successfully.")


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
