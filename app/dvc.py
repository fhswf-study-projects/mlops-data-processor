from minio import Minio
from minio.versioningconfig import VersioningConfig
from minio.commonconfig import ENABLED
from minio.error import S3Error

ACCESS_KEY = "UpHwDaN7JHRbTWy4hQPB"
SECRET_KEY = "5hmyRmR7jSj09U7w5mBXyaB2I5l333lH8mz1oL2d"


class DVC:

    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key

    def get_version(self, bucket_name, object_name, file_path, version_id):
        client = Minio(
            "localhost:9000",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,  # SET TRUE IN PRODUCTION
        )

        client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            version_id=version_id
        )

        # LOGGING HERE

    def write_file_to_bucket(self, source_file, destination_file, bucket_name):
        client = Minio(
            "localhost:9000",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,  # SET TRUE IN PRODUCTION
        )

        # Make the bucket if it doesn't exist.
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            client.set_bucket_versioning("raw-data", VersioningConfig(ENABLED))
            print("Created bucket", bucket_name)  # SHOULD BE LOGGED HERE
        else:
            print("Bucket", bucket_name, "already exists")  # SHOULD BE LOGGED HERE

        # Upload the file, renaming it in the process
        result = client.fput_object(
            bucket_name, destination_file, source_file,
        )

        print(
            result.object_name, result.etag, result.version_id
        )  # SHOULD BE LOGGED HERE


if __name__ == "__main__":
    try:
        dvc_instance = DVC(ACCESS_KEY, SECRET_KEY)

        dvc_instance.write_file_to_bucket(
            source_file="testfile.txt",
            destination_file="testfile.txt",
            bucket_name="raw-data"
        )

        dvc_instance.get_version(
            bucket_name="raw-data",
            object_name="testfile.txt",
            file_path="testfile_get.txt",
            version_id="eedd020f-c864-4780-8bff-0bb139ae16fd"
        )
    except S3Error as exc:
        print("error occurred.", exc)
