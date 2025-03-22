"""File serving all needed celery config variables in form of python class"""

import os

from app.constants import EnvConfig


class Config:
    broker_url = os.getenv(EnvConfig.CELERY_BROKER_CONNECTION.value)
    broker_pool_limit = 0

    result_backend = os.getenv(EnvConfig.CELERY_BACKEND_CONNECTION.value)
    result_extended = True

    task_default_queue = os.getenv(EnvConfig.CELERY_DEFAULT_QUEUE.value)
    task_acks_late = True
    task_send_sent_event = True

    worker_hijack_root_logger = False
    worker_prefetch_multiplier = 1
    worker_send_task_event = True
