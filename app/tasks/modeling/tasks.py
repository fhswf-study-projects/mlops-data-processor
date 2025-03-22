"""Every celery task is concepted in very similar fashion as
1. Reading data/objects from the previous task in the chain
2. Performing any operations on/with help of it
3. Give the objects to the very next task, as a serializable object.
So, typically the every infos are stored in s3 and only locations will be transfered via return.
"""

import logging

import mlflow
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient
from app.utils import plot_confusion_matrix


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

N_ESTIMATORS_BASE = 100
EXPERIMENT_NAME = "income_classification"
RANDOM_STATE = 42


@current_app.task(name="modeling.train_base_model")
def train_base_model(*args, **kwargs):
    logger.info("Training the base model with no optimizing the model parameters...")
    data_paths = args[0]
    dvc_client = DVCClient()

    # Read data
    preprocessor = dvc_client.read_data_from(data_paths["pipeline"])
    X = dvc_client.read_data_from(data_paths["features"])
    y = dvc_client.read_data_from(data_paths["dependent"])
    ###

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_BASE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("model", type(model).__name__)
        mlflow.log_metric("accuracy", accuracy)  # type: ignore
        mlflow.log_metric("precision", precision)  # type: ignore
        mlflow.log_metric("recall", recall)  # type: ignore
        mlflow.log_metric("f1_score", f1)  # type: ignore
        mlflow.log_metric("train_shape_rows", X_train.shape[0])  # type: ignore
        mlflow.log_metric("train_shape_cols", X_train.shape[1])  # type: ignore

        cm_filename = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(cm, cm_filename)
        mlflow.log_artifact(cm_filename)

        mlflow.sklearn.log_model(pipeline, "best_model")

    logger.info(
        f"Model saved under income_classification with Accuracy: {accuracy:.4f}"
    )


@current_app.task(name="modeling.train_optimized_model")
def optimize_hyperparameters(*args, **kwargs):
    logger.info("Training the model with hyperparameter optimization...")
    data_paths = args[0]
    dvc_client = DVCClient()

    # Read data
    preprocessor = dvc_client.read_data_from(data_paths["pipeline"])
    X = dvc_client.read_data_from(data_paths["features"])
    y = dvc_client.read_data_from(data_paths["dependent"])
    ###

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    def objective(trial):
        """Optuna-Target: Hyperparameter-Optimierung for the Model"""
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        with mlflow.start_run():
            mlflow.log_param("model", type(model).__name__)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)  # type: ignore
            mlflow.log_metric("precision", precision)  # type: ignore
            mlflow.log_metric("recall", recall)  # type: ignore
            mlflow.log_metric("f1_score", f1)  # type: ignore
            mlflow.log_metric("train_shape_rows", X_train.shape[0])  # type: ignore
            mlflow.log_metric("train_shape_cols", X_train.shape[1])  # type: ignore

            cm_filename = "artifacts/confusion_matrix_optuna.png"
            plot_confusion_matrix(cm, cm_filename)
            mlflow.log_artifact(cm_filename)

            # WHY IS THIS CODE HERE?
            # input_example = X_train.iloc[:1]
            # signature = mlflow.models.signature.infer_signature(
            #     X_train, pipeline.predict(X_train)
            # )

            # For what using signatures?
            mlflow.sklearn.log_model(
                pipeline,
                "best_model",  # , signature=signature, input_example=input_example
            )

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # type: ignore

    logger.info("Model saved under income_classification_hyperopt")


@current_app.task(name="modeling.predict")
def predict(*args, **kwargs):
    logger.info("Making Income Prediction...")
    data_paths = args[0]
    dvc_client = DVCClient()

    # Read data
    X = dvc_client.read_data_from(data_paths["features"])
    ###

    best_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="",
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if len(best_runs) != 1:
        logger.warning(f"Experiment {EXPERIMENT_NAME} not found in MlFlow.")
        return None

    best_run = best_runs.iloc[0]  # type: ignore

    model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/best_model")
    predictions = model.predict(X)
    interpertation = (
        f"Predicted Income Class: {'>50K' if predictions[0] == 1 else '<=50K'}"
    )

    return {"prediction": int(predictions[0]), "meaning": interpertation}
