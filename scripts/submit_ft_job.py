#!/usr/bin/env python
"""Submit fine-tuning jobs using SageMaker SDK"""

import os
import sys
import time
import argparse
from pathlib import Path

import boto3
from sagemaker import Session

# Change to project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

from sagemaker.pytorch import PyTorch

def submit_job(dataset):
    """Submit a fine-tuning job for a dataset"""

    timestamp = int(time.time())
    job_name = f"molfm-ft-{dataset}-{timestamp}"

    # Create SageMaker session with region
    boto_session = boto3.Session(region_name="us-east-1")
    sagemaker_session = Session(boto_session=boto_session)

    # Create PyTorch estimator
    # source_dir is project root, entry_point is at root level
    estimator = PyTorch(
        entry_point="sagemaker_entry.py",
        source_dir=".",
        role="arn:aws:iam::529964558252:role/MolFMLiteSageMakerRole",
        sagemaker_session=sagemaker_session,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters={
            "mode": "finetune",
            "dataset": dataset,
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 5e-5,
            "hidden_dim": 256,
            "num_layers": 4,
            "patience": 15,
        },
        base_job_name=f"molfm-ft-{dataset}",
        output_path="s3://molfm-lite-data/output/",
        code_location="s3://molfm-lite-data/code/",
        max_run=86400,
        disable_profiler=True,
    )

    # Submit training job
    estimator.fit(
        inputs={"training": "s3://molfm-lite-data/data/"},
        job_name=job_name,
        wait=False,
    )

    print(f"Submitted job: {job_name}")
    return job_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset to finetune on")
    args = parser.parse_args()

    submit_job(args.dataset)


if __name__ == "__main__":
    main()
