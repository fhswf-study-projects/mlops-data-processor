from enum import Enum


class EnvConfig(Enum):
    CELERY_BACKEND_CONNECTION = "CELERY_BACKEND_CONNECTION"
    CELERY_BROKER_CONNECTION = "CELERY_BROKER_CONNECTION"
    CELERY_DEFAULT_QUEUE = "CELERY_DEFAULT_QUEUE"
    S3_ENDPOINT_URL = "S3_ENDPOINT_URL"
    S3_BUCKET_NAME = "S3_BUCKET_NAME"
    S3_ACCESS_KEY_ID = "S3_ACCESS_KEY_ID"
    S3_SECRET_ACCESS_KEY = "S3_SECRET_ACCESS_KEY"
