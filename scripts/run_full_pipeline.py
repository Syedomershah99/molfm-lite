#!/usr/bin/env python
"""
Master script to run the full MolFM-Lite pipeline:
1. Download data
2. Pre-train on ZINC250K
3. Fine-tune on MoleculeNet benchmarks
4. Run ablation studies
5. Generate plots and results

Usage:
    python scripts/run_full_pipeline.py --mode local
    python scripts/run_full_pipeline.py --mode sagemaker --bucket your-bucket
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description, check=True):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    start_time = time.time()
    result = subprocess.run(cmd, check=check)
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full MolFM-Lite pipeline")
    parser.add_argument("--mode", choices=["local", "sagemaker"], default="local",
                        help="Run locally or on SageMaker")
    parser.add_argument("--bucket", type=str, default="molfm-lite-data",
                        help="S3 bucket for SageMaker mode")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pre-training (use existing checkpoint)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip fine-tuning")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation studies")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max molecules for pre-training")
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to existing checkpoint")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with minimal data/epochs")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    # Quick test mode
    if args.quick_test:
        args.max_samples = 1000
        args.pretrain_epochs = 2
        args.finetune_epochs = 5
        print("QUICK TEST MODE: Using minimal data and epochs")

    print("\n" + "="*60)
    print("MolFM-Lite Full Pipeline")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Pre-train samples: {args.max_samples}")
    print(f"Pre-train epochs: {args.pretrain_epochs}")
    print(f"Fine-tune epochs: {args.finetune_epochs}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {
        'start_time': datetime.now().isoformat(),
        'config': vars(args),
        'steps': {}
    }

    # Step 1: Download data
    print("\n[1/5] Downloading datasets...")
    success = run_command(
        [sys.executable, str(scripts_dir / "download_data.py")],
        "Download ZINC250K and MoleculeNet datasets"
    )
    results['steps']['download'] = success

    # Step 2: Pre-training
    if not args.skip_pretrain:
        print("\n[2/5] Pre-training on ZINC250K...")
        pretrain_cmd = [
            sys.executable, str(scripts_dir / "pretrain.py"),
            "--max-samples", str(args.max_samples),
            "--epochs", str(args.pretrain_epochs),
            "--batch-size", "64",
            "--checkpoint-dir", "checkpoints/pretrain",
        ]
        success = run_command(pretrain_cmd, "Pre-training")
        results['steps']['pretrain'] = success

        if success:
            args.checkpoint = "checkpoints/pretrain/best_model.pt"
    else:
        print("\n[2/5] Skipping pre-training (using existing checkpoint)")
        results['steps']['pretrain'] = 'skipped'

    # Step 3: Fine-tuning on benchmarks
    if not args.skip_finetune:
        print("\n[3/5] Fine-tuning on MoleculeNet benchmarks...")

        checkpoint_arg = ["--checkpoint", args.checkpoint] if args.checkpoint else []

        finetune_cmd = [
            sys.executable, str(scripts_dir / "evaluate.py"),
            "--datasets", "bbbp", "bace", "tox21", "lipophilicity",
            "--epochs", str(args.finetune_epochs),
            "--num-seeds", "3",
            "--output-dir", "results",
        ] + checkpoint_arg

        success = run_command(finetune_cmd, "Fine-tuning and evaluation")
        results['steps']['finetune'] = success
    else:
        print("\n[3/5] Skipping fine-tuning")
        results['steps']['finetune'] = 'skipped'

    # Step 4: Ablation studies
    if not args.skip_ablations:
        print("\n[4/5] Running ablation studies...")

        checkpoint_arg = ["--checkpoint", args.checkpoint] if args.checkpoint else []

        ablation_cmd = [
            sys.executable, str(scripts_dir / "evaluate.py"),
            "--run-ablations",
            "--ablation-dataset", "bbbp",
            "--epochs", str(min(args.finetune_epochs, 50)),
            "--output-dir", "results",
        ] + checkpoint_arg

        success = run_command(ablation_cmd, "Ablation studies")
        results['steps']['ablations'] = success
    else:
        print("\n[4/5] Skipping ablations")
        results['steps']['ablations'] = 'skipped'

    # Step 5: Generate plots
    print("\n[5/5] Generating plots and visualizations...")
    plot_cmd = [
        sys.executable, str(scripts_dir / "generate_plots.py"),
        "--results-dir", "results",
        "--output-dir", "plots",
        "--use-placeholder",  # Fill in missing data with placeholders
    ]
    success = run_command(plot_cmd, "Generate plots")
    results['steps']['plots'] = success

    # Save pipeline results
    results['end_time'] = datetime.now().isoformat()
    results_path = project_root / "results" / "pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nStep Results:")
    for step, status in results['steps'].items():
        icon = "✓" if status == True else ("⊘" if status == 'skipped' else "✗")
        print(f"  {icon} {step}")

    print("\nOutput locations:")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Results: results/")
    print(f"  - Plots: plots/")

    print("\nNext steps:")
    print("  1. Review plots in plots/")
    print("  2. Update paper with real results")
    print("  3. Run Streamlit demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
