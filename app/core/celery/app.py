import logging

from celery import Celery

from app.core.celery.celeryconfig import Config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Setting up Celery app...")

current_app = Celery("data-processor-worker")
current_app.config_from_object(Config)
current_app.autodiscover_tasks(
    [
        "app.tasks.data_processing",
        "app.tasks.data_transformation",
        "app.tasks.modeling",
        "app.tasks.workflows",
    ]
)
