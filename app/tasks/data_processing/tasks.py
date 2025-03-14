import logging

from pandas import DataFrame

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient


logger = logging.getLogger(__name__)


@current_app.task(name="data_processing.load_data")
def load_data(*args, **kwargs):
    logger.info("Loading initial data...")

    filepath = kwargs["body"]["filepath"]
    bucket = kwargs["body"]["bucket"]

    dvc_client = DVCClient()

    df = dvc_client.read_data_from(source=filepath, bucket_name=bucket)

    return {
        "df": dvc_client.save_data_to(df, "load_data/df.jobib"),
    }


@current_app.task(name="data_processing.clean_data")
def clean_data(*args, **kwargs):
    logger.info("Start Data Cleaning...")
    # pick first non empty
    data_paths = list(filter(None, args))[0]
    dvc_client = DVCClient()

    # Read data
    df = dvc_client.read_data_from(data_paths["df"])
    ###

    if not isinstance(df, DataFrame):
        logger.error("Provided path can't be read as dataframe.")
        raise TypeError("df meant to be a DataFrame, but it is not")

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
