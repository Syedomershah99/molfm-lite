#!/usr/bin/env python
"""Download and prepare datasets for MolFM-Lite"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import download_zinc250k, download_moleculenet


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["zinc250k", "bbbp", "bace", "tox21", "lipophilicity"],
                        help="Datasets to download")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Downloading datasets...")
    print(f"Data directory: {args.data_dir}")
    print(f"Datasets: {args.datasets}")
    print()

    for dataset in args.datasets:
        print(f"Downloading {dataset}...")
        try:
            if dataset == "zinc250k":
                path = download_zinc250k(args.data_dir)
                if path:
                    print(f"  ✓ Downloaded to {path}")
                else:
                    print(f"  ✗ Failed to download {dataset}")
            else:
                path, info = download_moleculenet(dataset, args.data_dir)
                if path:
                    print(f"  ✓ Downloaded to {path}")
                    print(f"    Tasks: {info.get('num_tasks', 'N/A')}")
                    print(f"    Train/Val/Test: {info.get('train_size', 'N/A')}/{info.get('valid_size', 'N/A')}/{info.get('test_size', 'N/A')}")
                else:
                    print(f"  ✗ Failed to download {dataset}")
        except Exception as e:
            print(f"  ✗ Error downloading {dataset}: {e}")

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
