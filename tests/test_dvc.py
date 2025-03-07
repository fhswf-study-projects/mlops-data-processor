import pytest

from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError

from app.core.dvc_client import DVCClient


@pytest.fixture
def dvc():
    """Fixture to create a DVC instance for testing."""
    return DVCClient()


@patch("app.core.dvc_client.boto3.client")  # Mock the Boto3 client
def test_write_file_to_bucket(mock_boto3, dvc):
    """Test writing a file to S3/MinIO bucket."""

    # Mock S3 client
    mock_client = MagicMock()
    mock_boto3.return_value = mock_client

    # Configure mock responses
    mock_client.list_buckets.return_value = {"Buckets": []}  # Simulate no buckets exist

    # Call the method
    dvc.save_data_to("testfile.txt", "testfile.txt", "test-bucket")

    # Assertions
    mock_client.create_bucket.assert_called_once_with(Bucket="test-bucket")
    mock_client.put_bucket_versioning.assert_called_once_with(
        Bucket="test-bucket", VersioningConfiguration={"Status": "Enabled"}
    )
    mock_client.upload_file.assert_called_once_with(
        "testfile.txt", "test-bucket", "testfile.txt"
    )


@patch("app.core.dvc_client.boto3.client")
def test_get_version(mock_boto3, dvc):
    """Test retrieving a specific version of an object from S3/MinIO."""

    # Mock S3 client
    mock_client = MagicMock()
    mock_boto3.return_value = mock_client

    # Mock file writing
    mock_file = MagicMock()

    # Call the method
    with patch("builtins.open", return_value=mock_file):
        dvc.read_data_from("test-bucket", "testfile.txt", "downloaded.txt", "12345")

    # Assertions
    mock_client.download_fileobj.assert_called_once_with(
        Bucket="test-bucket",
        Key="testfile.txt",
        ExtraArgs={"VersionId": "12345"},
        Fileobj=mock_file,
    )


@patch("app.core.dvc_client.boto3.client")
def test_write_file_to_existing_bucket(mock_boto3, dvc):
    """Test writing a file when the bucket already exists."""

    mock_client = MagicMock()
    mock_boto3.return_value = mock_client

    # Simulate existing bucket
    mock_client.list_buckets.return_value = {"Buckets": [{"Name": "test-bucket"}]}

    # Call method
    dvc.save_data_to("testfile.txt", "testfile.txt", "test-bucket")

    # Ensure bucket creation is NOT called
    mock_client.create_bucket.assert_not_called()


@patch("app.core.dvc_client.boto3.client")
def test_write_file_raises_s3_error(mock_boto3, dvc):
    """Test exception handling when S3 raises a ClientError."""

    mock_client = MagicMock()
    mock_boto3.return_value = mock_client

    # Simulate S3 error
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "Test S3Error"}}
    mock_client.upload_file.side_effect = ClientError(error_response, "PutObject")

    with pytest.raises(ClientError, match="Test S3Error"):
        dvc.save_data_to("testfile.txt", "testfile.txt", "test-bucket")
