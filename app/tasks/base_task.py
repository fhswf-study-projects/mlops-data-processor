import os
import uuid
import tempfile

import pandas as pd
from celery import Task

from app.core.clients.dvc_client import DVCStorageManager


class DataTask(Task):

    def __init__(self) -> None:
        super().__init__()
        self.dvc_client = DVCStorageManager("minio")

    def save_dataframe(self, df: pd.DataFrame, task_name: str) -> str:
        """Saves a Pandas DataFrame to MinIO via DVC, using a temporary file."""
        file_id = task_name + str(uuid.uuid4()) + ".parquet"

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True, delete_on_close=False) as temp_file:
            temp_path = temp_file.name
            df.to_parquet(temp_path, engine="pyarrow")

            # Add to DVC, push, and remove temp file
            self.repo.add(temp_path)
            self.repo.scm.add([temp_path + ".dvc"])
            self.repo.scm.commit(f"Track {file_id}")
            self.repo.push(remote=self.dvc_remote)
        
            temp_file.close()
        return file_id

    def load_dataframe(self, file_id: str) -> pd.DataFrame:
        """Loads a DataFrame from MinIO via DVC into memory (without saving it to disk)."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True, delete_on_close=False) as temp_file:
            temp_path = temp_file.name

            self.repo.pull(file_id, remote=self.dvc_remote)
            df = pd.read_parquet(temp_path, engine="pyarrow")
        return df
