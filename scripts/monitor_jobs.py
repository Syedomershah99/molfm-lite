#!/usr/bin/env python3
"""Monitor SageMaker jobs and collect results"""

import os
import sys
import json
import time
import boto3
from pathlib import Path

# AWS credentials - set via environment variables or AWS CLI
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret
# export AWS_DEFAULT_REGION=us-east-1
if not os.environ.get('AWS_ACCESS_KEY_ID'):
    print("Warning: AWS_ACCESS_KEY_ID not set. Configure AWS CLI or set environment variables.")
if not os.environ.get('AWS_DEFAULT_REGION'):
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

BUCKET = 'molfm-lite-data'
JOBS = [
    'molfm-ft-bbbp-1767877702',
    'molfm-ft-bace-1767877702',
    'molfm-ft-tox21-1767877702',
    'molfm-ft-lipophilicity-1767877702'
]

def get_job_status(sm_client, job_name):
    """Get job status and details"""
    try:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        return {
            'status': response['TrainingJobStatus'],
            'secondary': response.get('SecondaryStatus', 'N/A'),
            'failure': response.get('FailureReason', ''),
            'start_time': response.get('TrainingStartTime'),
            'end_time': response.get('TrainingEndTime'),
        }
    except Exception as e:
        return {'status': 'Error', 'error': str(e)}

def download_results(s3_client, job_name, output_dir):
    """Download results for a completed job"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download model.tar.gz
    s3_key = f"output/{job_name}/output/model.tar.gz"
    local_path = output_dir / f"{job_name}_model.tar.gz"

    try:
        s3_client.download_file(BUCKET, s3_key, str(local_path))
        print(f"Downloaded: {local_path}")

        # Extract
        import tarfile
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(output_dir / job_name)
        print(f"Extracted to: {output_dir / job_name}")
        return True
    except Exception as e:
        print(f"Error downloading {job_name}: {e}")
        return False

def monitor_jobs(interval=60, output_dir='results'):
    """Monitor jobs until all complete"""
    sm_client = boto3.client('sagemaker', region_name='us-east-1')
    s3_client = boto3.client('s3', region_name='us-east-1')

    completed = set()

    while len(completed) < len(JOBS):
        print(f"\n{'='*60}")
        print(f"Status check at {time.strftime('%H:%M:%S')}")
        print('='*60)

        for job in JOBS:
            if job in completed:
                continue

            status = get_job_status(sm_client, job)
            dataset = job.split('-')[2].upper()

            if status['status'] == 'Completed':
                print(f"✓ {dataset:15} COMPLETED")
                completed.add(job)
                download_results(s3_client, job, output_dir)
            elif status['status'] == 'Failed':
                print(f"✗ {dataset:15} FAILED: {status.get('failure', 'Unknown')[:50]}")
                completed.add(job)
            else:
                runtime = ''
                if status.get('start_time'):
                    import datetime
                    now = datetime.datetime.now(status['start_time'].tzinfo)
                    mins = (now - status['start_time']).total_seconds() / 60
                    runtime = f" ({mins:.0f} min)"
                print(f"  {dataset:15} {status['status']} ({status['secondary']}){runtime}")

        if len(completed) < len(JOBS):
            print(f"\nWaiting {interval}s for next check...")
            time.sleep(interval)

    print(f"\n{'='*60}")
    print("All jobs finished!")
    print('='*60)
    return completed

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    monitor_jobs(args.interval, args.output)
