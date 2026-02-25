#!/usr/bin/env python
"""Evaluation script for MolFM-Lite on MoleculeNet benchmarks"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.molfm import MolFMLite
from src.evaluation.benchmarks import MoleculeNetBenchmark, AblationStudy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MolFM-Lite")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)

    # Evaluation
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["bbbp", "bace", "tox21", "lipophilicity"],
                        help="Datasets to evaluate on")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)

    # Ablation
    parser.add_argument("--run-ablations", action="store_true",
                        help="Run ablation studies")
    parser.add_argument("--ablation-dataset", type=str, default="bbbp")

    # Infrastructure
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("Creating model...")
    model = MolFMLite(
        hidden_dim=args.hidden_dim,
        num_layers_1d=args.num_layers,
        num_layers_2d=args.num_layers,
    )

    # Load pretrained weights if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained weights")
    else:
        print("No checkpoint provided, training from scratch")

    # Create benchmark
    benchmark = MoleculeNetBenchmark(
        model=model,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        device=device,
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Run evaluation
    print(f"\nEvaluating on datasets: {args.datasets}")
    results = {}

    for dataset in args.datasets:
        try:
            result = benchmark.evaluate_dataset(
                dataset,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                patience=args.patience,
            )
            results[dataset] = {
                "task_type": result.task_type,
                "metrics": result.metrics,
                "metrics_std": result.metrics_std,
            }
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    for dataset, result in results.items():
        print(f"\n{dataset.upper()}:")
        for metric, value in result["metrics"].items():
            std = result["metrics_std"].get(metric, 0)
            print(f"  {metric}: {value:.4f} ± {std:.4f}")

    # Run ablations if requested
    if args.run_ablations:
        print("\n" + "="*60)
        print("RUNNING ABLATION STUDIES")
        print("="*60)

        ablation_study = AblationStudy(model, benchmark)
        ablation_results = ablation_study.run_ablations(
            dataset_name=args.ablation_dataset,
        )

        # Save ablation results
        ablation_path = output_dir / "ablation_results.json"
        ablation_dict = {
            name: {
                "metrics": r.metrics,
                "metrics_std": r.metrics_std,
            }
            for name, r in ablation_results.items()
        }
        with open(ablation_path, "w") as f:
            json.dump(ablation_dict, f, indent=2)
        print(f"Ablation results saved to: {ablation_path}")


if __name__ == "__main__":
    main()
