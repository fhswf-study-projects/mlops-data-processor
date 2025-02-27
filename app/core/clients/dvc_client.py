import os
import logging

import pandas as pd
import dvc.repo as data_repo

from app.constants import EnvConfig


logger = logging.getLogger()


class DVCStorageManager:
    """Handles DVC operations with MinIO, using temporary files."""

    def __init__(self, remote_name="minio") -> None:
        self.dvc_remote = remote_name
        self.repo = data_repo.Repo()
        self._setup_dvc_remote()

    def _setup_dvc_remote(self) -> None:
        """Ensures MinIO is configured as a DVC remote."""
        try:
            if self.dvc_remote not in self.repo.config["remote"]:
                self.repo.run_cmd(["dvc", "remote", "add", "-d", self.dvc_remote, f"s3://{os.getenv(EnvConfig.S3_BUCKET_NAME.value)}"])
                self.repo.run_cmd(["dvc", "remote", "modify", self.dvc_remote, "endpointurl", os.getenv(EnvConfig.S3_ENDPOINT_URL.value)])
                self.repo.run_cmd(["dvc", "remote", "modify", self.dvc_remote, "access_key_id", os.getenv(EnvConfig.S3_ACCESS_KEY_ID.value)])
                self.repo.run_cmd(["dvc", "remote", "modify", self.dvc_remote, "secret_access_key", os.getenv(EnvConfig.S3_SECRET_ACCESS_KEY.value)])
                logger.info(f"DVC remote '{self.dvc_remote}' successfully configured!")
        except Exception as e:
            logger.error(f"Error setting up DVC remote: {e}")
