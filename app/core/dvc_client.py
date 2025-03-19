import io
import os
import logging
from typing import Any, Union

import joblib
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from app.constants import EnvConfig


logger = logging.getLogger()


class DVCClient:
    """Handles DVC operations with S3-like remote, using temporary files."""

    _instance = None

    def __new__(cls):
        """Returns the singleton instance or creates a new one if not existend"""
        if cls._instance is None:
            cls._instance = super(DVCClient, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.environ[EnvConfig.S3_ACCESS_KEY_ID.value],
            aws_secret_access_key=os.environ[EnvConfig.S3_SECRET_ACCESS_KEY.value],
            endpoint_url=os.environ[
                EnvConfig.S3_ENDPOINT_URL.value
            ],  # Use None for AWS S3, set URL for MinIO
        )

    def read_data_from(self, source: str, bucket_name=None) -> Union[Any, None]:
        obj = None
        if not bucket_name:
            bucket_name = os.environ[EnvConfig.S3_BUCKET_NAME.value]

        try:
            response = self.client.get_object(Bucket=bucket_name, Key=source)
            buffer = io.BytesIO(response["Body"].read())  # Read data into buffer
            obj = joblib.load(buffer)  # Deserialize object
            logger.info(f"Downloaded {source}")
        except ClientError as e:
            logger.error("Error in downloading file:", e)
        return obj

    def save_data_to(self, obj, destination: str, bucket_name=None) -> str:
        if not bucket_name:
            bucket_name = os.environ[EnvConfig.S3_BUCKET_NAME.value]

        try:
            # Check if the bucket exists
            existing_buckets = self.client.list_buckets()
            bucket_names = [
                bucket["Name"] for bucket in existing_buckets.get("Buckets", [])
            ]

            if bucket_name not in bucket_names:
                self.client.create_bucket(Bucket=bucket_name)
                logger.info(f"Created bucket: {bucket_name}")

                # Enable versioning
                self.client.put_bucket_versioning(
                    Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
                )
                logger.info(f"Enabled versioning on {bucket_name}")
            else:
                logger.info(f"Bucket {bucket_name} already exists")

            # Serialize object into RAM, avoid temporary file creation
            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            buffer.seek(0)

            self.client.put_object(
                Bucket=bucket_name, Key=destination, Body=buffer.getvalue()
            )

            logger.info(f"Uploaded {destination} to {bucket_name}")
        except (NoCredentialsError, ClientError) as e:
            logger.error("Error in file upload:", e)
        return destination
