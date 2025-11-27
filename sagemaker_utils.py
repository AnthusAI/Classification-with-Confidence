#!/usr/bin/env python3
"""
AWS SageMaker Training Utilities

This module provides helper functions for SageMaker training operations:
- AWS account and IAM role discovery
- S3 upload/download operations
- CloudWatch log streaming
- Resource management

These utilities support the optional SageMaker training mode in fine_tune_model.py.
"""

import boto3
import os
import time
from pathlib import Path
from typing import Optional
import tarfile
import tempfile


def get_account_id() -> str:
    """
    Get the AWS account ID for the current credentials.

    Returns:
        AWS account ID as a string

    Raises:
        Exception: If AWS credentials are not configured
    """
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        return identity['Account']
    except Exception as e:
        raise Exception(f"Failed to get AWS account ID. Please run 'aws configure': {e}")


def discover_iam_role(role_name: str = 'LlamaSageMakerExecutionRole') -> str:
    """
    Find the existing LlamaSageMakerExecutionRole IAM role.

    This role should have been created by scripts/deploy_sagemaker.py with permissions for:
    - SageMaker training jobs
    - S3 bucket access
    - CloudWatch logging

    Args:
        role_name: Name of the IAM role to find

    Returns:
        ARN of the IAM role

    Raises:
        Exception: If role doesn't exist
    """
    try:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName=role_name)
        role_arn = role['Role']['Arn']
        print(f"✓ Found IAM role: {role_name}")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        raise Exception(
            f"IAM role '{role_name}' not found. "
            f"Please create it first by running: python scripts/deploy_sagemaker.py"
        )
    except Exception as e:
        raise Exception(f"Failed to access IAM role: {e}")


def discover_s3_bucket(region: str = 'us-east-1') -> str:
    """
    Find existing S3 bucket or create a new one.

    Uses the naming pattern from scripts/deploy_sagemaker.py:
    llama-sagemaker-models-{account_id}

    Args:
        region: AWS region for bucket creation

    Returns:
        S3 bucket name

    Raises:
        Exception: If bucket operations fail
    """
    try:
        s3 = boto3.client('s3', region_name=region)
        account_id = get_account_id()

        # Try the standard naming pattern
        bucket_name = f"llama-sagemaker-models-{account_id}"

        try:
            # Check if bucket exists
            s3.head_bucket(Bucket=bucket_name)
            print(f"✓ Found S3 bucket: {bucket_name}")
            return bucket_name
        except:
            # Bucket doesn't exist, create it
            print(f"Creating S3 bucket: {bucket_name}")

            if region == 'us-east-1':
                # us-east-1 doesn't allow LocationConstraint
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )

            print(f"✓ Created S3 bucket: {bucket_name}")
            return bucket_name

    except Exception as e:
        raise Exception(f"Failed to access or create S3 bucket: {e}")


def upload_to_s3(local_path: str, s3_uri: str, show_progress: bool = True) -> None:
    """
    Upload a file or directory to S3.

    Args:
        local_path: Local file or directory path
        s3_uri: S3 URI (s3://bucket/path)
        show_progress: Whether to print progress messages

    Raises:
        Exception: If upload fails
    """
    try:
        # Parse S3 URI
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ''

        s3 = boto3.client('s3')
        local_path = Path(local_path)

        if local_path.is_file():
            # Upload single file
            s3_key = s3_prefix if s3_prefix else local_path.name
            if show_progress:
                print(f"  Uploading {local_path.name}...")
            s3.upload_file(str(local_path), bucket, s3_key)

        elif local_path.is_dir():
            # Upload directory recursively
            file_count = 0
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate S3 key maintaining directory structure
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')

                    if show_progress:
                        print(f"  Uploading {relative_path}...")

                    s3.upload_file(str(file_path), bucket, s3_key)
                    file_count += 1

            if show_progress:
                print(f"✓ Uploaded {file_count} files")
        else:
            raise ValueError(f"Path does not exist: {local_path}")

    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")


def download_from_s3(s3_uri: str, local_path: str, show_progress: bool = True) -> None:
    """
    Download a file from S3.

    Args:
        s3_uri: S3 URI (s3://bucket/path/file)
        local_path: Local destination path
        show_progress: Whether to print progress messages

    Raises:
        Exception: If download fails
    """
    try:
        # Parse S3 URI
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        s3_key = parts[1] if len(parts) > 1 else ''

        if show_progress:
            print(f"  Downloading from {s3_uri}...")

        s3 = boto3.client('s3')

        # Ensure parent directory exists
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(bucket, s3_key, str(local_path))

        if show_progress:
            print(f"✓ Downloaded to {local_path}")

    except Exception as e:
        raise Exception(f"Failed to download from S3: {e}")


def stream_cloudwatch_logs(
    log_group: str,
    log_stream_prefix: str,
    region: str = 'us-east-1',
    wait_time: int = 300,
    poll_interval: int = 5
) -> None:
    """
    Stream CloudWatch logs from a SageMaker training job.

    CloudWatch logs may not be immediately available when a training job starts.
    This function waits up to wait_time seconds for logs to appear, then streams them.

    Args:
        log_group: CloudWatch log group name (e.g., /aws/sagemaker/TrainingJobs)
        log_stream_prefix: Log stream prefix to search for (e.g., job-name/algo-1)
        region: AWS region
        wait_time: Maximum seconds to wait for logs to appear
        poll_interval: Seconds between log polling attempts

    Note:
        If logs don't appear within wait_time, this function returns gracefully
        to allow job monitoring to continue via status polling.
    """
    try:
        logs = boto3.client('logs', region_name=region)

        # Wait for log stream to be created
        log_stream_name = None
        wait_start = time.time()

        print(f"Waiting for CloudWatch logs (up to {wait_time}s)...")

        while time.time() - wait_start < wait_time:
            try:
                # Find log streams matching prefix
                response = logs.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=log_stream_prefix
                )

                if response.get('logStreams'):
                    log_stream_name = response['logStreams'][0]['logStreamName']
                    print(f"✓ Found log stream: {log_stream_name}")
                    break

            except logs.exceptions.ResourceNotFoundException:
                pass

            time.sleep(poll_interval)

        if not log_stream_name:
            print(f"Warning: CloudWatch logs not available after {wait_time}s")
            print("Training is still running, monitoring via status polling...")
            return

        # Stream logs
        print("\n" + "=" * 60)
        print("Training Logs")
        print("=" * 60 + "\n")

        next_token = None
        last_timestamp = 0

        while True:
            try:
                kwargs = {
                    'logGroupName': log_group,
                    'logStreamName': log_stream_name,
                    'startFromHead': True
                }

                if next_token:
                    kwargs['nextToken'] = next_token

                response = logs.get_log_events(**kwargs)

                # Print new log events
                for event in response['events']:
                    if event['timestamp'] > last_timestamp:
                        # Format timestamp
                        timestamp = time.strftime(
                            '%Y-%m-%d %H:%M:%S',
                            time.localtime(event['timestamp'] / 1000)
                        )
                        print(f"[{timestamp}] {event['message']}")
                        last_timestamp = event['timestamp']

                # Check if we've reached the end
                next_token = response.get('nextForwardToken')
                if not response['events']:
                    # No new events, sleep before next poll
                    time.sleep(poll_interval)

            except Exception as e:
                print(f"Warning: Error streaming logs: {e}")
                break

    except Exception as e:
        print(f"Warning: Could not stream CloudWatch logs: {e}")
        print("Training will continue, monitoring via status polling...")


def validate_prerequisites(check_dataset: bool = True) -> None:
    """
    Validate all prerequisites before starting SageMaker training.

    Checks:
    - AWS credentials configured
    - IAM role exists
    - Dataset directory exists (optional)

    Args:
        check_dataset: Whether to validate dataset/ directory exists

    Raises:
        Exception: If any prerequisite check fails
    """
    print("Validating prerequisites...")
    checks = []

    # Check 1: AWS credentials
    try:
        get_account_id()
        checks.append(("AWS credentials", True, ""))
    except Exception as e:
        checks.append(("AWS credentials", False, str(e)))

    # Check 2: IAM role
    try:
        discover_iam_role()
        checks.append(("IAM role", True, ""))
    except Exception as e:
        checks.append(("IAM role", False, str(e)))

    # Check 3: Dataset (optional)
    if check_dataset:
        if Path('dataset').exists() and Path('dataset').is_dir():
            file_count = len(list(Path('dataset').glob('*.txt')))
            checks.append(("Dataset", True, f"{file_count} files found"))
        else:
            checks.append(("Dataset", False, "dataset/ directory not found"))

    # Print results
    print("\nPrerequisite Check Results:")
    for name, passed, message in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if message:
            prefix = "   " if passed else "    Error: "
            print(f"{prefix}{message}")

    # Raise exception if any check failed
    failed_checks = [name for name, passed, _ in checks if not passed]
    if failed_checks:
        print("\n❌ Prerequisite validation failed!")
        print("\nTo fix:")
        if "AWS credentials" in failed_checks:
            print("  - Run: aws configure")
        if "IAM role" in failed_checks:
            print("  - Run: python scripts/deploy_sagemaker.py (to create IAM role)")
        if "Dataset" in failed_checks:
            print("  - Ensure dataset/ directory exists with training data")

        raise Exception("Prerequisite validation failed")

    print("\n✓ All prerequisites validated!")


def create_requirements_file() -> str:
    """
    Create a requirements.txt file with exact versions for SageMaker training.

    Returns:
        Contents of requirements.txt as a string
    """
    return """transformers>=4.43.0
torch==2.3.0
peft==0.7.1
datasets==2.16.1
accelerate>=0.32.0
bitsandbytes==0.41.3
"""


if __name__ == "__main__":
    # Test utility functions
    print("Testing SageMaker Utilities")
    print("=" * 60)

    try:
        # Test 1: Get account ID
        account_id = get_account_id()
        print(f"✓ Account ID: {account_id}")

        # Test 2: Discover IAM role
        role_arn = discover_iam_role()
        print(f"✓ IAM Role: {role_arn}")

        # Test 3: Discover S3 bucket
        bucket = discover_s3_bucket()
        print(f"✓ S3 Bucket: {bucket}")

        # Test 4: Validate prerequisites
        validate_prerequisites(check_dataset=False)

        print("\n✓ All utilities working correctly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
