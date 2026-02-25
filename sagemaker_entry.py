#!/usr/bin/env python
"""Entry point for SageMaker training"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# SageMaker paths
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# Add code path
sys.path.insert(0, "/opt/ml/code")
sys.path.insert(0, "/opt/ml/code/src")


def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")

    import torch
    torch_version = torch.__version__.split('+')[0]
    cuda_version = torch.version.cuda if torch.cuda.is_available() else 'cpu'
    print(f"  PyTorch: {torch_version}, CUDA: {cuda_version}")

    # Install torch_geometric with pre-built wheels for better compatibility
    print("  Installing torch_geometric...")
    try:
        # Try installing from PyG wheel index for better compatibility
        pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version.replace('.', '')[:3] if cuda_version != 'cpu' else 'cpu'}.html"
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "torch-scatter", "torch-sparse", "torch-cluster", "torch-geometric",
             "-f", pyg_url, "-q", "--no-cache-dir"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            print(f"  Warning: PyG wheel install failed, trying pip...")
            # Fallback to pip install
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch-geometric", "-q"],
                capture_output=True,
                text=True,
                timeout=300
            )
    except Exception as e:
        print(f"  Warning: torch_geometric installation failed: {e}")
        print("  Will use fallback GNN implementation")

    # Verify torch_geometric installation
    try:
        from torch_geometric.nn import GINConv, global_mean_pool
        print("  torch_geometric installed successfully!")
    except ImportError:
        print("  Warning: torch_geometric not available, using fallback GNN")

    # Install other packages
    packages = ["rdkit", "selfies"]
    for pkg in packages:
        print(f"  Installing {pkg}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                print(f"  Warning: {pkg} install issue: {result.stderr[:200]}")
        except Exception as e:
            print(f"  Warning: {pkg} install failed: {e}")

    print("Dependencies installed.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Training mode
    parser.add_argument("--mode", type=str, default="pretrain",
                        choices=["pretrain", "finetune"])

    # Data
    parser.add_argument("--dataset", type=str, default="zinc250k")
    parser.add_argument("--max_samples", type=int, default=50000)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    return parser.parse_args()


def pretrain(args):
    """Run pre-training"""
    import torch
    import numpy as np

    print("=" * 60)
    print("MOLFM-LITE PRE-TRAINING")
    print("=" * 60)

    # Import after dependencies installed
    from src.data.loaders import download_zinc250k, load_zinc250k, create_pretraining_dataloader
    from src.data.preprocessing import MoleculePreprocessor, ConformerGenerator
    from src.models.molfm import MolFMLite
    from src.training.trainer import PretrainingTrainer

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print(f"\nLoading data from {SM_CHANNEL_TRAINING}...")
    data_path = os.path.join(SM_CHANNEL_TRAINING, "zinc250k.csv")

    if not os.path.exists(data_path):
        print("Downloading ZINC250K...")
        data_path = download_zinc250k(SM_CHANNEL_TRAINING)

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find or download data at {data_path}")

    smiles, properties = load_zinc250k(data_path, max_samples=args.max_samples)
    print(f"Loaded {len(smiles)} molecules")

    # Create preprocessor
    preprocessor = MoleculePreprocessor(max_seq_len=256)
    conformer_gen = ConformerGenerator(num_conformers=5, optimize=True)

    # Create dataloader
    print("\nCreating dataloader...")
    cache_dir = os.path.join(SM_OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_loader = create_pretraining_dataloader(
        smiles,
        batch_size=args.batch_size,
        num_workers=2,
        cache_dir=cache_dir,
        preprocessor=preprocessor,
        conformer_generator=conformer_gen,
    )
    print(f"Created dataloader with {len(train_loader)} batches")

    # Create model
    print("\nCreating model...")
    model = MolFMLite(
        hidden_dim=args.hidden_dim,
        hidden_dim_3d=128,
        num_layers_1d=args.num_layers,
        num_layers_2d=args.num_layers,
        num_interactions_3d=3,
        use_energy_weights=True,
        fusion_type="attention",
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    trainer = PretrainingTrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_steps=500,
        contrastive_temp=0.07,
        device=device,
        checkpoint_dir=SM_MODEL_DIR,
        log_interval=50,
        save_interval=500,
    )

    history = trainer.train()

    # Save final model
    print("\nSaving model...")
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, os.path.join(SM_MODEL_DIR, "pretrained_model.pt"))

    # Save training history
    with open(os.path.join(SM_MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE!")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Model saved to: {SM_MODEL_DIR}")
    print("=" * 60)


def load_weights_safely(model, state_dict, verbose=True):
    """Load weights into model, skipping incompatible layers"""
    model_dict = model.state_dict()
    compatible_dict = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            else:
                skipped.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            skipped.append(f"{k}: not in model")

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    if verbose:
        print(f"  Loaded {len(compatible_dict)}/{len(state_dict)} weights")
        if skipped and len(skipped) <= 10:
            for s in skipped:
                print(f"  Skipped: {s}")
        elif skipped:
            print(f"  Skipped {len(skipped)} incompatible weights")

    return len(compatible_dict), len(skipped)


def finetune(args):
    """Run fine-tuning on a single MoleculeNet dataset"""
    import torch
    import numpy as np

    print("=" * 60)
    print(f"MOLFM-LITE FINE-TUNING: {args.dataset.upper()}")
    print("=" * 60)

    from src.models.molfm import MolFMLite
    from src.evaluation.benchmarks import MoleculeNetBenchmark

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model with correct atom_feat_dim
    print("\nCreating model...")
    model = MolFMLite(
        hidden_dim=args.hidden_dim,
        hidden_dim_3d=128,
        num_layers_1d=args.num_layers,
        num_layers_2d=args.num_layers,
        num_interactions_3d=3,
        use_energy_weights=True,
        fusion_type="attention",
        atom_feat_dim=38,  # Explicitly set to match MoleculePreprocessor output
    )

    # Try to load pretrained weights (safely handling size mismatches)
    pretrained_paths = [
        os.path.join(SM_CHANNEL_TRAINING, "pretrained_model.pt"),
        os.path.join(SM_CHANNEL_TRAINING, "model.pt"),
    ]

    for pretrained_path in pretrained_paths:
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}...")
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                load_weights_safely(model, state_dict)
                print("Pretrained weights loaded!")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Continuing with random initialization...")
            break
    else:
        print("No pretrained weights found, training from scratch")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Cache directory
    cache_dir = os.path.join(SM_OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Run benchmark on single dataset
    print(f"\nEvaluating on {args.dataset}...")
    benchmark = MoleculeNetBenchmark(
        model=model,
        data_dir=SM_CHANNEL_TRAINING,
        cache_dir=cache_dir,
        device=device,
        num_seeds=3,
        batch_size=args.batch_size,
        num_workers=2,
    )

    try:
        result = benchmark.evaluate_dataset(
            args.dataset,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )

        results = {
            args.dataset: {
                "task_type": result.task_type,
                "metrics": result.metrics,
                "metrics_std": result.metrics_std,
            }
        }

        # Print results
        print("\n" + "=" * 60)
        print(f"RESULTS: {args.dataset.upper()}")
        print("=" * 60)
        for metric, value in result.metrics.items():
            std = result.metrics_std.get(metric, 0)
            print(f"  {metric}: {value:.4f} +/- {std:.4f}")

    except Exception as e:
        print(f"Error evaluating {args.dataset}: {e}")
        import traceback
        traceback.print_exc()
        results = {args.dataset: {"error": str(e)}}

    # Save results
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    results_path = os.path.join(SM_MODEL_DIR, f"results_{args.dataset}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Save fine-tuned model
    model_path = os.path.join(SM_MODEL_DIR, f"model_{args.dataset}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'args': vars(args),
    }, model_path)

    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_path}")
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE!")
    print("=" * 60)


def main():
    print("=" * 60)
    print("MOLFM-LITE SAGEMAKER TRAINING")
    print("=" * 60)

    # Print environment info
    print(f"\nPython: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"SM_MODEL_DIR: {SM_MODEL_DIR}")
    print(f"SM_CHANNEL_TRAINING: {SM_CHANNEL_TRAINING}")
    print(f"SM_OUTPUT_DIR: {SM_OUTPUT_DIR}")

    # List training data
    print(f"\nTraining data directory contents:")
    if os.path.exists(SM_CHANNEL_TRAINING):
        for f in os.listdir(SM_CHANNEL_TRAINING)[:10]:
            print(f"  {f}")
    else:
        print("  (directory not found)")

    # Install dependencies
    install_dependencies()

    # Parse args
    args = parse_args()
    print(f"\nArguments: {vars(args)}")

    # Import torch after deps
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Run training
    if args.mode == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
