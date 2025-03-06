import pytest
from unittest.mock import patch, MagicMock
from minio.error import S3Error
from minio.versioningconfig import VersioningConfig
from minio.commonconfig import ENABLED
from app.dvc import DVC

ACCESS_KEY = "test-access-key"
SECRET_KEY = "test-secret-key"


@pytest.fixture
def dvc():
    """Fixture to create a DVC instance for testing."""
    return DVC(ACCESS_KEY, SECRET_KEY)


@patch("app.dvc.Minio")  # Mock the Minio client
def test_write_file_to_bucket(mock_minio, dvc):
    """Test writing a file to MinIO bucket."""

    # Mock MinIO instance
    mock_client = MagicMock()
    mock_minio.return_value = mock_client

    # Configure mock responses
    mock_client.bucket_exists.return_value = False
    mock_client.fput_object.return_value = MagicMock(
        object_name="testfile.txt", etag="12345", version_id="67890"
    )

    # Call the method
    dvc.write_file_to_bucket("testfile.txt", "testfile.txt", "test-bucket")

    # Assertions
    mock_client.bucket_exists.assert_called_once_with("test-bucket")
    mock_client.make_bucket.assert_called_once_with("test-bucket")
    mock_client.set_bucket_versioning("test-bucket", VersioningConfig(ENABLED))

    mock_client.fput_object.assert_called_once_with("test-bucket", "testfile.txt", "testfile.txt")


@patch("app.dvc.Minio")
def test_get_version(mock_minio, dvc):
    """Test retrieving a specific version of an object from MinIO."""

    # Mock MinIO instance
    mock_client = MagicMock()
    mock_minio.return_value = mock_client

    # Call the method
    dvc.get_version("test-bucket", "testfile.txt", "downloaded.txt", "12345")

    # Assertions
    mock_client.fget_object.assert_called_once_with(
        bucket_name="test-bucket",
        object_name="testfile.txt",
        file_path="downloaded.txt",
        version_id="12345"
    )


@patch("app.dvc.Minio")
def test_write_file_to_existing_bucket(mock_minio, dvc):
    """Test writing a file when the bucket already exists."""

    mock_client = MagicMock()
    mock_minio.return_value = mock_client

    # Simulate bucket exists
    mock_client.bucket_exists.return_value = True

    # Call method
    dvc.write_file_to_bucket("testfile.txt", "testfile.txt", "test-bucket")

    # Check that bucket creation was NOT called
    mock_client.make_bucket.assert_not_called()


@patch("app.dvc.Minio")
def test_write_file_raises_s3_error(mock_minio, dvc):
    """Test exception handling when MinIO raises an S3Error."""

    mock_client = MagicMock()
    mock_minio.return_value = mock_client

    # Simulate S3Error
    mock_client.fput_object.side_effect = S3Error(
        code="NoSuchKey",
        message="Test S3Error",
        resource="/testfile.txt",
        request_id="12345",
        host_id="host-id-example",
        response=None
    )

    with pytest.raises(S3Error, match="Test S3Error"):
        dvc.write_file_to_bucket("testfile.txt", "testfile.txt", "test-bucket")
