import logging

from celery import Celery

from app.core.celery.celeryconfig import Config


logger = logging.getLogger(__name__)
logger.info("Setting up Celery app...")

app = Celery("data-processor-worker")
app.config_from_object(Config)
app.autodiscover_tasks(
    [
        "app.tasks.data_processing",
        "app.tasks.feature_engineering",
        "app.tasks.modeling",
        "app.tasks.workflows",
    ]
)
