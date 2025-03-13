import logging

import pandas as pd
from celery import chain, signature

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient


logger = logging.getLogger(__name__)


@current_app.task(name="workflows.model_training", bind=True, acks_late=True)
def model_training_workflow(self, *args, **kwargs):
    optimize = kwargs["body"].get("optimize", False)
    tasks = [
        signature(
            "data_processing.load_data",
            kwargs={
                "body": {
                    "filepath": kwargs["body"]["filepath"],
                    "bucket": kwargs["body"]["bucket"],
                }
            },
        ),
        signature(
            "feature_engineering.clean_and_tranform", kwargs={"body": {"mode": "train"}}
        ),
        signature("modeling.train_optimized_model")
        if optimize
        else signature("modeling.train_base_model"),
    ]
    task_chain = chain(tasks)

    # self.request.chain.append(group(task_chain))
    logger.info("Starte die Machine Learning Pipeline...")

    return {"result_task_id": task_chain.apply_async().id}  # type: ignore


@current_app.task(name="workflows.make_prediction", bind=True, acks_late=True)
def prediction_workflow(self, *args, **kwargs):
    dvc_client = DVCClient()

    data_json = kwargs["body"]["data"]
    if not isinstance(data_json, list):
        data_json = [data_json]

    tasks = [
        signature(
            "feature_engineering.clean_and_tranform",
            args=(
                {
                    "df": dvc_client.save_data_to(
                        pd.DataFrame(data_json), "prediction_samples/data.joblib"
                    )
                },
            ),
            kwargs={"body": {"mode": "predict"}},
        ),
        signature("modeling.predict"),
    ]
    task_chain = chain(tasks)

    logger.info("Starte die Prediction Pipeline...")

    #     self.request.chain.append(group(task_chain))

    return {"result_task_id": task_chain.apply_async().id}  # type: ignore
