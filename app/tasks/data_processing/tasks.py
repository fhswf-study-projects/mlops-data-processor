"""Every celery task is concepted in very similar fashion as
1. Reading data/objects from the previous task in the chain
2. Performing any operations on/with help of it
3. Give the objects to the very next task, as a serializable object.
So, typically the every infos are stored in s3 and only locations will be transfered via return.
"""

import logging

import pandas as pd

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@current_app.task(name="data_processing.load_data")
def load_data(*args, **kwargs):
    logger.info("Loading initial data...")

    filepath = kwargs["body"]["filepath"]
    bucket = kwargs["body"]["bucket"]
    use_feeback_data = kwargs["body"]["use_feeback_data"]

    dvc_client = DVCClient()

    df = dvc_client.read_data_from(source=filepath, bucket_name=bucket)

    if use_feeback_data:
        df_feedback = dvc_client.read_data_from(
            source=kwargs["body"]["feedback_path"], bucket_name=bucket
        )  # type: ignore
        df = pd.concat([df, df_feedback], ignore_index=True).drop_duplicates()  # type: ignore

    return {
        "df": dvc_client.save_data_to(df, "load_data/df.jobib"),
    }


@current_app.task(name="data_processing.clean_data")
def clean_data(*args, **kwargs):
    """
    Handles missing data and shaping the raw dataset.

    Raises:
        TypeError: raise TypeError when the loaded object is not a pandas pd.DataFrame.

    Returns:
        dict: Bucket location of the cleaned dataset
    """
    logger.info("Start Data Cleaning...")
    # pick first non empty
    data_paths = list(filter(None, args))[0]
    dvc_client = DVCClient()

    # Read data
    df = dvc_client.read_data_from(data_paths["df"])
    ###

    if not isinstance(df, pd.DataFrame):
        logger.error("Provided path can't be read as pd.DataFrame.")
        raise TypeError("df meant to be a pd.DataFrame, but it is not")

    # Remove 'fnlwgt' only if it exists
    if "fnlwgt" in df.columns:
        df.drop(columns=["fnlwgt"], inplace=True)

    # Replace missing values in categorical columns with "Unknown"
    df.fillna(
        {"workclass": "Unknown", "occupation": "Unknown", "native_country": "Unknown"},
        inplace=True,
    )

    return {
        "df": dvc_client.save_data_to(df, "clean_data/df.jobib"),
    }
