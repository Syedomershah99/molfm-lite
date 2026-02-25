"""AWS utilities for MolFM-Lite"""

import os
import json
import boto3
from pathlib import Path
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError


class S3Manager:
    """Manager for S3 operations"""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client("s3", region_name=region)
        self.s3_resource = boto3.resource("s3", region_name=region)

    def create_bucket(self) -> bool:
        """Create S3 bucket if it doesn't exist"""
        try:
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )
            print(f"Created bucket: {self.bucket_name}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                print(f"Bucket already exists: {self.bucket_name}")
                return True
            else:
                print(f"Error creating bucket: {e}")
                return False

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3"""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            print(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str) -> bool:
        """Upload a directory to S3"""
        local_dir = Path(local_dir)
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                s3_key = f"{s3_prefix}/{file_path.relative_to(local_dir)}"
                self.upload_file(str(file_path), s3_key)
        return True

    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            print(f"Error listing objects: {e}")
            return []

    def delete_object(self, s3_key: str) -> bool:
        """Delete an object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            print(f"Error deleting object: {e}")
            return False

    def get_s3_uri(self, s3_key: str) -> str:
        """Get S3 URI for a key"""
        return f"s3://{self.bucket_name}/{s3_key}"


class SageMakerManager:
    """Manager for SageMaker operations"""

    def __init__(self, region: str = "us-east-1", role_arn: Optional[str] = None):
        self.region = region
        self.sm_client = boto3.client("sagemaker", region_name=region)
        self.role_arn = role_arn or self._get_execution_role()

    def _get_execution_role(self) -> str:
        """Get SageMaker execution role ARN"""
        try:
            iam = boto3.client("iam")
            roles = iam.list_roles()["Roles"]
            for role in roles:
                if "SageMaker" in role["RoleName"]:
                    return role["Arn"]
        except Exception:
            pass
        return os.environ.get("SAGEMAKER_ROLE_ARN", "")

    def create_training_job(
        self,
        job_name: str,
        image_uri: str,
        instance_type: str,
        instance_count: int,
        s3_input: str,
        s3_output: str,
        hyperparameters: Dict[str, str],
        max_runtime: int = 86400,
        spot_instances: bool = True,
        code_s3_uri: str = "s3://molfm-lite-data/code/molfm-code.tar.gz",
    ) -> str:
        """Create a SageMaker training job"""
        # Add SageMaker-required hyperparameters for entry point
        full_hyperparameters = {
            **hyperparameters,
            "sagemaker_program": "sagemaker_entry.py",
            "sagemaker_submit_directory": code_s3_uri,
        }

        training_config = {
            "TrainingJobName": job_name,
            "RoleArn": self.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
            },
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": 100,
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": s3_input,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": s3_output},
            "HyperParameters": full_hyperparameters,
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        }

        if spot_instances:
            training_config["EnableManagedSpotTraining"] = True
            training_config["StoppingCondition"]["MaxWaitTimeInSeconds"] = (
                max_runtime * 2
            )

        try:
            response = self.sm_client.create_training_job(**training_config)
            print(f"Created training job: {job_name}")
            return job_name
        except ClientError as e:
            print(f"Error creating training job: {e}")
            return ""

    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a training job"""
        try:
            response = self.sm_client.describe_training_job(TrainingJobName=job_name)
            return {
                "status": response["TrainingJobStatus"],
                "secondary_status": response.get("SecondaryStatus", ""),
                "failure_reason": response.get("FailureReason", ""),
            }
        except ClientError as e:
            print(f"Error getting job status: {e}")
            return {}

    def wait_for_training_job(self, job_name: str) -> bool:
        """Wait for training job to complete"""
        waiter = self.sm_client.get_waiter("training_job_completed_or_stopped")
        try:
            waiter.wait(TrainingJobName=job_name)
            status = self.get_training_job_status(job_name)
            return status.get("status") == "Completed"
        except Exception as e:
            print(f"Error waiting for job: {e}")
            return False

    def stop_training_job(self, job_name: str) -> bool:
        """Stop a training job"""
        try:
            self.sm_client.stop_training_job(TrainingJobName=job_name)
            return True
        except ClientError as e:
            print(f"Error stopping job: {e}")
            return False


def setup_aws_infrastructure(bucket_name: str, region: str = "us-east-1") -> Dict[str, Any]:
    """Set up AWS infrastructure for MolFM-Lite"""
    s3_manager = S3Manager(bucket_name, region)

    # Create bucket
    bucket_created = s3_manager.create_bucket()

    # Create folder structure
    folders = ["data/raw", "data/processed", "checkpoints", "results", "logs"]
    for folder in folders:
        # Create empty marker file to establish folder structure
        s3_manager.s3_client.put_object(
            Bucket=bucket_name, Key=f"{folder}/.keep", Body=b""
        )

    return {
        "bucket_name": bucket_name,
        "bucket_created": bucket_created,
        "region": region,
        "s3_uri": f"s3://{bucket_name}",
    }
