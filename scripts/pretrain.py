#!/usr/bin/env python
"""Local pre-training script for MolFM-Lite"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import download_zinc250k, load_zinc250k, create_pretraining_dataloader
from src.data.preprocessing import MoleculePreprocessor, ConformerGenerator
from src.models.molfm import MolFMLite
from src.training.trainer import PretrainingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train MolFM-Lite")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max molecules for pre-training (budget control)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-dim-3d", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-conformers", type=int, default=5)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)

    # Contrastive learning
    parser.add_argument("--temperature", type=float, default=0.07)

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)

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

    # Download and load data
    print("Loading ZINC250K dataset...")
    data_path = download_zinc250k(args.data_dir)
    if not data_path:
        print("Failed to download data. Please check your internet connection.")
        return

    smiles, properties = load_zinc250k(data_path, max_samples=args.max_samples)
    print(f"Loaded {len(smiles)} molecules")

    # Create preprocessor and conformer generator
    preprocessor = MoleculePreprocessor(max_seq_len=256)
    conformer_generator = ConformerGenerator(
        num_conformers=args.num_conformers,
        optimize=True,
    )

    # Create dataloader
    print("Creating dataloader...")
    train_loader = create_pretraining_dataloader(
        smiles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        preprocessor=preprocessor,
        conformer_generator=conformer_generator,
    )

    print(f"Created dataloader with {len(train_loader)} batches")

    # Create model
    print("Creating model...")
    model = MolFMLite(
        hidden_dim=args.hidden_dim,
        hidden_dim_3d=args.hidden_dim_3d,
        num_layers_1d=args.num_layers,
        num_layers_2d=args.num_layers,
        num_interactions_3d=3,
        use_energy_weights=True,
        fusion_type="attention",
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Create trainer
    trainer = PretrainingTrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        contrastive_temp=args.temperature,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    # Train
    print("\nStarting pre-training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print()

    history = trainer.train()

    print("\nPre-training complete!")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
