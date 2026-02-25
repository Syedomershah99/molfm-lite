#!/usr/bin/env python
"""
Clean up AWS resources to avoid charges.

This script will:
1. Stop any running SageMaker training jobs
2. Delete SageMaker endpoints (if any)
3. Optionally delete S3 bucket contents
4. List any remaining resources

Usage:
    python scripts/cleanup_aws.py --bucket molfm-lite-data
    python scripts/cleanup_aws.py --bucket molfm-lite-data --delete-s3  # Also delete S3 data
"""

import os
import sys
import argparse
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def stop_sagemaker_jobs(region: str):
    """Stop any running SageMaker training jobs"""
    sm = boto3.client("sagemaker", region_name=region)

    print("\n[1] Checking SageMaker training jobs...")

    try:
        # List training jobs
        response = sm.list_training_jobs(
            StatusEquals="InProgress",
            MaxResults=100
        )

        jobs = response.get("TrainingJobSummaries", [])

        if not jobs:
            print("  No running training jobs found.")
            return

        print(f"  Found {len(jobs)} running job(s):")

        for job in jobs:
            job_name = job["TrainingJobName"]
            print(f"    - {job_name}")

            # Ask for confirmation
            confirm = input(f"  Stop job '{job_name}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    sm.stop_training_job(TrainingJobName=job_name)
                    print(f"    ✓ Stopped {job_name}")
                except Exception as e:
                    print(f"    ✗ Error stopping {job_name}: {e}")
            else:
                print(f"    Skipped {job_name}")

    except Exception as e:
        print(f"  Error listing jobs: {e}")


def delete_sagemaker_endpoints(region: str):
    """Delete any SageMaker endpoints"""
    sm = boto3.client("sagemaker", region_name=region)

    print("\n[2] Checking SageMaker endpoints...")

    try:
        response = sm.list_endpoints(MaxResults=100)
        endpoints = response.get("Endpoints", [])

        if not endpoints:
            print("  No endpoints found.")
            return

        print(f"  Found {len(endpoints)} endpoint(s):")

        for endpoint in endpoints:
            name = endpoint["EndpointName"]
            print(f"    - {name}")

            confirm = input(f"  Delete endpoint '{name}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    sm.delete_endpoint(EndpointName=name)
                    print(f"    ✓ Deleted {name}")
                except Exception as e:
                    print(f"    ✗ Error deleting {name}: {e}")

    except Exception as e:
        print(f"  Error listing endpoints: {e}")


def delete_s3_bucket(bucket_name: str, region: str, delete_contents: bool):
    """Delete S3 bucket or just list contents"""
    s3 = boto3.resource("s3", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    print(f"\n[3] Checking S3 bucket: {bucket_name}...")

    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  Bucket '{bucket_name}' does not exist.")
            return
        else:
            print(f"  Error accessing bucket: {e}")
            return

    # List objects
    try:
        bucket = s3.Bucket(bucket_name)
        objects = list(bucket.objects.all())

        if not objects:
            print("  Bucket is empty.")
        else:
            print(f"  Found {len(objects)} object(s)")

            # Calculate total size
            total_size = sum(obj.size for obj in objects)
            print(f"  Total size: {total_size / (1024*1024):.2f} MB")

            # Show sample of objects
            print("  Sample objects:")
            for obj in objects[:10]:
                print(f"    - {obj.key} ({obj.size / 1024:.1f} KB)")
            if len(objects) > 10:
                print(f"    ... and {len(objects) - 10} more")

        if delete_contents:
            confirm = input(f"\n  DELETE all objects in '{bucket_name}'? This cannot be undone! (yes/NO): ")
            if confirm.lower() == 'yes':
                print("  Deleting objects...")
                bucket.objects.all().delete()
                print("  ✓ All objects deleted")

                # Delete bucket
                confirm_bucket = input(f"  Also delete the bucket itself? (yes/NO): ")
                if confirm_bucket.lower() == 'yes':
                    bucket.delete()
                    print(f"  ✓ Bucket '{bucket_name}' deleted")
            else:
                print("  Skipped deletion.")
        else:
            print("\n  To delete S3 contents, run with --delete-s3 flag")

    except Exception as e:
        print(f"  Error: {e}")


def list_remaining_resources(region: str):
    """List any remaining AWS resources that might incur charges"""
    print("\n[4] Checking for remaining resources...")

    # Check EC2 instances
    print("\n  EC2 Instances:")
    ec2 = boto3.client("ec2", region_name=region)
    try:
        response = ec2.describe_instances(
            Filters=[{"Name": "instance-state-name", "Values": ["running", "pending"]}]
        )
        instances = []
        for reservation in response["Reservations"]:
            instances.extend(reservation["Instances"])

        if instances:
            print(f"    Found {len(instances)} running instance(s):")
            for inst in instances:
                print(f"      - {inst['InstanceId']} ({inst['InstanceType']})")
        else:
            print("    No running instances.")
    except Exception as e:
        print(f"    Error: {e}")

    # Check SageMaker notebook instances
    print("\n  SageMaker Notebook Instances:")
    sm = boto3.client("sagemaker", region_name=region)
    try:
        response = sm.list_notebook_instances(StatusEquals="InService")
        notebooks = response.get("NotebookInstances", [])

        if notebooks:
            print(f"    Found {len(notebooks)} running notebook(s):")
            for nb in notebooks:
                print(f"      - {nb['NotebookInstanceName']} ({nb['InstanceType']})")
        else:
            print("    No running notebooks.")
    except Exception as e:
        print(f"    Error: {e}")

    # Check for any pending charges estimate
    print("\n  Cost estimate:")
    print("    Check AWS Cost Explorer for detailed breakdown:")
    print(f"    https://{region}.console.aws.amazon.com/cost-management/home")


def estimate_current_costs(region: str):
    """Try to get current month's costs"""
    print("\n[5] Estimating current costs...")

    try:
        ce = boto3.client("ce", region_name="us-east-1")  # Cost Explorer is global

        from datetime import datetime, timedelta
        end = datetime.now()
        start = end.replace(day=1)

        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start.strftime('%Y-%m-%d'),
                'End': end.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost']
        )

        for result in response['ResultsByTime']:
            cost = float(result['Total']['BlendedCost']['Amount'])
            print(f"  Current month spend: ${cost:.2f}")

    except Exception as e:
        print(f"  Could not fetch cost data: {e}")
        print("  Check AWS Cost Explorer manually.")


def main():
    parser = argparse.ArgumentParser(description="Clean up AWS resources")
    parser.add_argument("--bucket", type=str, default="molfm-lite-data",
                        help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--delete-s3", action="store_true",
                        help="Delete S3 bucket contents")
    parser.add_argument("--auto-yes", action="store_true",
                        help="Auto-confirm all deletions (use with caution!)")
    args = parser.parse_args()

    print("="*60)
    print("MolFM-Lite AWS Cleanup")
    print("="*60)
    print(f"Region: {args.region}")
    print(f"Bucket: {args.bucket}")
    print("="*60)

    # Verify AWS credentials
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print(f"\nAWS Account: {identity['Account']}")
        print(f"User ARN: {identity['Arn']}")
    except Exception as e:
        print(f"\nError: Cannot connect to AWS. Check your credentials.")
        print(f"Run 'aws configure' to set up credentials.")
        return

    # Run cleanup steps
    stop_sagemaker_jobs(args.region)
    delete_sagemaker_endpoints(args.region)
    delete_s3_bucket(args.bucket, args.region, args.delete_s3)
    list_remaining_resources(args.region)
    estimate_current_costs(args.region)

    print("\n" + "="*60)
    print("Cleanup complete!")
    print("="*60)
    print("\nRemember to:")
    print("  1. Delete/rotate your AWS access keys if no longer needed")
    print("  2. Check AWS Cost Explorer for any remaining charges")
    print("  3. Set up billing alerts for future projects")


if __name__ == "__main__":
    main()
