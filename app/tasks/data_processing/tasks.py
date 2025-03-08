import logging

from app.core.celery.app import current_app
from app.core.dvc_client import DVCClient


logger = logging.getLogger(__name__)


@current_app.task(name="data_processing.load_data")
def load_data(*args, **kwargs):
    logger.info("Received")

    filepath = kwargs["body"]["filepath"]
    bucket = kwargs["body"]["bucket"]

    dvc_client = DVCClient()

    df = dvc_client.read_data_from(source=filepath, bucket_name=bucket)

    return {
        "df": dvc_client.save_data_to(df, "load_data/df.jobib"),
    }
