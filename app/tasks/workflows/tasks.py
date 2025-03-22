"""Workflow tasks serve purpose of gathering (actually chaining) small steps
making it like a pipeline of celery tasks.
"""

import logging

import pandas as pd
from celery import chain, signature

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@current_app.task(name="workflows.model_training", acks_late=True)
def model_training_workflow(*args, **kwargs):
    optimize = kwargs["body"].get("optimize", False)
    include_user_data = kwargs["body"].get("include_user_data", False)
    tasks = [
        signature(
            "data_processing.load_data",
            kwargs={
                "body": {
                    "filepath": kwargs["body"]["filepath"],
                    "bucket": kwargs["body"]["bucket"],
                    "include_user_data": include_user_data,
                    "feedback_path": kwargs["body"]["feedback_path"],
                }
            },
        ),
        signature("data_processing.clean_data"),
        signature(
            "data_transformation.engineer_features", kwargs={"body": {"mode": "train"}}
        ),
        signature(
            "data_transformation.encode_data", kwargs={"body": {"mode": "train"}}
        ),
        signature("modeling.train_optimized_model")
        if optimize
        else signature("modeling.train_base_model"),
    ]
    task_chain = chain(tasks)

    logger.info("Start Machine Learning Pipeline...")

    return {"result_task_id": task_chain.apply_async().id}  # type: ignore


@current_app.task(name="workflows.make_prediction", acks_late=True)
def prediction_workflow(*args, **kwargs):
    dvc_client = DVCClient()

    data_json = kwargs["body"]["data"]
    if not isinstance(data_json, list):
        data_json = [data_json]

    tasks = [
        signature(
            "data_processing.clean_data",
            args=(
                {
                    "df": dvc_client.save_data_to(
                        pd.DataFrame(data_json), "prediction_samples/data.joblib"
                    )
                },
            ),
        ),
        signature(
            "data_transformation.engineer_features",
            kwargs={"body": {"mode": "predict"}},
        ),
        signature(
            "data_transformation.encode_data", kwargs={"body": {"mode": "predict"}}
        ),
        signature("modeling.predict"),
    ]
    task_chain = chain(tasks)

    logger.info("Start Prediction Pipeline...")

    return {"result_task_id": task_chain.apply_async().id}  # type: ignore
