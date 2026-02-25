#!/usr/bin/env python
"""SageMaker training script for MolFM-Lite"""

import os
import sys
import json
import argparse
import boto3
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.aws import S3Manager, SageMakerManager


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--bucket", type=str, default="molfm-lite-data")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--spot", action="store_true", help="Use spot instances")
    parser.add_argument("--max-runtime", type=int, default=86400)
    parser.add_argument("--dataset", type=str, default="zinc250k")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def create_training_job(args):
    """Create and launch SageMaker training job"""

    # Initialize managers
    s3_manager = S3Manager(args.bucket, args.region)
    sm_manager = SageMakerManager(args.region)

    # Create bucket if needed
    s3_manager.create_bucket()

    # Upload source code
    print("Uploading source code to S3...")
    source_dir = Path(__file__).parent.parent / "src"
    s3_manager.upload_directory(str(source_dir), "code/src")

    # Upload training script
    s3_manager.upload_file(
        str(Path(__file__).parent / "sagemaker_entry.py"),
        "code/sagemaker_entry.py"
    )

    # Generate job name
    import time
    timestamp = int(time.time())
    if args.mode == "finetune":
        job_name = f"molfm-ft-{args.dataset}-{timestamp}"
    else:
        job_name = f"molfm-{args.mode}-{timestamp}"

    # Hyperparameters
    hyperparameters = {
        "mode": args.mode,
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "learning_rate": str(args.learning_rate),
        "dataset": args.dataset,
        "hidden_dim": str(args.hidden_dim),
        "num_layers": str(args.num_layers),
        "patience": str(args.patience),
    }

    # Get PyTorch training image
    # Using AWS Deep Learning Container
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    image_uri = f"763104351884.dkr.ecr.{args.region}.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker"

    # Create training job
    print(f"Creating training job: {job_name}")
    sm_manager.create_training_job(
        job_name=job_name,
        image_uri=image_uri,
        instance_type=args.instance_type,
        instance_count=1,
        s3_input=s3_manager.get_s3_uri("data/"),  # All datasets are in the data folder
        s3_output=s3_manager.get_s3_uri(f"output/{job_name}"),
        hyperparameters=hyperparameters,
        max_runtime=args.max_runtime,
        spot_instances=args.spot,
    )

    print(f"Training job submitted: {job_name}")
    print(f"Monitor at: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}")

    return job_name


def main():
    args = parse_args()
    job_name = create_training_job(args)
    print(f"\nJob name: {job_name}")


if __name__ == "__main__":
    main()
