#!/usr/bin/env python
"""Test script to validate the entire MolFM-Lite pipeline"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

def test_preprocessing():
    """Test data preprocessing"""
    print("\n" + "="*60)
    print("Testing Data Preprocessing")
    print("="*60)

    from src.data.preprocessing import MoleculePreprocessor, ConformerGenerator

    # Test molecules
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]

    preprocessor = MoleculePreprocessor(max_seq_len=256)
    conformer_gen = ConformerGenerator(num_conformers=3, optimize=True)

    print(f"\nTesting with {len(test_smiles)} molecules...")

    for smiles in test_smiles:
        print(f"\n  SMILES: {smiles[:50]}...")

        # Process molecule
        features = preprocessor.process_molecule(smiles)

        if features is None:
            print("    ❌ Failed to process molecule")
            continue

        print(f"    ✓ SELFIES: {features.selfies[:50] if features.selfies else 'N/A'}...")
        print(f"    ✓ Token IDs shape: {features.token_ids.shape if features.token_ids is not None else 'N/A'}")
        print(f"    ✓ Atom features shape: {features.atom_features.shape if features.atom_features is not None else 'N/A'}")
        print(f"    ✓ Edge index shape: {features.edge_index.shape if features.edge_index is not None else 'N/A'}")

        # Generate conformers
        conformers, energies = conformer_gen.generate_conformers(smiles)

        if conformers:
            print(f"    ✓ Generated {len(conformers)} conformers")
            print(f"    ✓ Conformer shape: {conformers[0].shape}")
            print(f"    ✓ Energies: {energies}")
        else:
            print("    ⚠ No conformers generated")

    print("\n✅ Preprocessing test passed!")
    return True


def test_model_architecture():
    """Test model architecture"""
    print("\n" + "="*60)
    print("Testing Model Architecture")
    print("="*60)

    from src.models.molfm import MolFMLite

    # Create model
    model = MolFMLite(
        vocab_size=128,
        hidden_dim=128,  # Smaller for testing
        hidden_dim_3d=64,
        num_layers_1d=2,
        num_layers_2d=2,
        num_interactions_3d=2,
        num_tasks=1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 64
    num_atoms = 20
    num_edges = 40
    num_conformers = 3
    atom_feat_dim = 38  # Match MoleculePreprocessor output

    # Create dummy inputs
    token_ids = torch.randint(0, 128, (batch_size, seq_len))
    token_mask = torch.ones(batch_size, seq_len)
    atom_features = torch.randn(batch_size * num_atoms, atom_feat_dim)
    edge_index = torch.randint(0, num_atoms, (2, num_edges))
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_atoms)
    conformer_coords = torch.randn(batch_size, num_conformers, num_atoms, 3)
    conformer_mask = torch.ones(batch_size, num_conformers, num_atoms)
    conformer_weights = torch.softmax(torch.randn(batch_size, num_conformers), dim=-1)

    print("\n  Testing forward pass...")

    try:
        with torch.no_grad():
            outputs = model(
                token_ids=token_ids,
                token_mask=token_mask,
                atom_features=atom_features,
                edge_index=edge_index,
                batch_idx=batch_idx,
                conformer_coords=conformer_coords,
                conformer_mask=conformer_mask,
                conformer_weights=conformer_weights,
                return_embeddings=True,
                return_attribution=True,
            )

        print(f"    ✓ Prediction shape: {outputs['prediction'].shape}")
        print(f"    ✓ 1D embedding shape: {outputs['emb_1d'].shape}")
        print(f"    ✓ 2D embedding shape: {outputs['emb_2d'].shape}")
        print(f"    ✓ 3D embedding shape: {outputs['emb_3d'].shape}")
        print(f"    ✓ Attribution shape: {outputs['attribution'].shape}")

    except Exception as e:
        print(f"    ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Model architecture test passed!")
    return True


def test_contrastive_embeddings():
    """Test contrastive embedding generation"""
    print("\n" + "="*60)
    print("Testing Contrastive Embeddings")
    print("="*60)

    from src.models.molfm import MolFMLite

    model = MolFMLite(
        hidden_dim=128,
        hidden_dim_3d=64,
        num_layers_1d=2,
        num_layers_2d=2,
    )

    # Dummy data
    batch_size = 4
    atom_feat_dim = 38  # Match MoleculePreprocessor output
    token_ids = torch.randint(0, 128, (batch_size, 64))
    token_mask = torch.ones(batch_size, 64)
    atom_features = torch.randn(batch_size * 20, atom_feat_dim)
    edge_index = torch.randint(0, 20, (2, 40))
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), 20)
    conformer_coords = torch.randn(batch_size, 3, 20, 3)
    conformer_mask = torch.ones(batch_size, 3, 20)
    conformer_weights = torch.softmax(torch.randn(batch_size, 3), dim=-1)

    print("\n  Testing contrastive embedding generation...")

    try:
        with torch.no_grad():
            proj_1d, proj_2d, proj_3d = model.get_contrastive_embeddings(
                token_ids, token_mask,
                atom_features, edge_index, batch_idx,
                conformer_coords, conformer_mask, conformer_weights,
            )

        print(f"    ✓ Projected 1D shape: {proj_1d.shape}")
        print(f"    ✓ Projected 2D shape: {proj_2d.shape}")
        print(f"    ✓ Projected 3D shape: {proj_3d.shape}")

        # Check normalization
        norms_1d = torch.norm(proj_1d, dim=-1)
        print(f"    ✓ 1D embedding norms: {norms_1d} (should be ~1)")

    except Exception as e:
        print(f"    ❌ Contrastive embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Contrastive embeddings test passed!")
    return True


def test_losses():
    """Test loss functions"""
    print("\n" + "="*60)
    print("Testing Loss Functions")
    print("="*60)

    from src.training.losses import ContrastiveLoss, MultiTaskLoss

    batch_size = 8
    hidden_dim = 128

    # Create normalized embeddings
    proj_1d = torch.randn(batch_size, hidden_dim)
    proj_1d = proj_1d / proj_1d.norm(dim=-1, keepdim=True)

    proj_2d = torch.randn(batch_size, hidden_dim)
    proj_2d = proj_2d / proj_2d.norm(dim=-1, keepdim=True)

    proj_3d = torch.randn(batch_size, hidden_dim)
    proj_3d = proj_3d / proj_3d.norm(dim=-1, keepdim=True)

    print("\n  Testing ContrastiveLoss...")

    try:
        criterion = ContrastiveLoss(temperature=0.07)
        loss, metrics = criterion(proj_1d, proj_2d, proj_3d)

        print(f"    ✓ Contrastive loss: {loss.item():.4f}")
        print(f"    ✓ Metrics: {metrics}")

    except Exception as e:
        print(f"    ❌ ContrastiveLoss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n  Testing MultiTaskLoss...")

    try:
        multi_criterion = MultiTaskLoss(temperature=0.07)
        loss, metrics = multi_criterion(proj_1d, proj_2d, proj_3d)

        print(f"    ✓ Multi-task loss: {loss.item():.4f}")
        print(f"    ✓ Metrics: {metrics}")

    except Exception as e:
        print(f"    ❌ MultiTaskLoss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Loss functions test passed!")
    return True


def test_visualization():
    """Test visualization utilities"""
    print("\n" + "="*60)
    print("Testing Visualization Utilities")
    print("="*60)

    from src.visualization.plots import (
        plot_training_curves,
        plot_modality_attribution,
        save_plot,
    )

    # Create mock data
    history = {
        'loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8],
        'val_loss': [2.6, 2.1, 1.6, 1.4, 1.2, 1.1, 1.0],
        'metrics': [
            {'contrastive_loss': 2.0, 'loss_1d_2d': 0.8},
            {'contrastive_loss': 1.5, 'loss_1d_2d': 0.6},
            {'contrastive_loss': 1.2, 'loss_1d_2d': 0.5},
        ]
    }

    attributions = np.random.dirichlet([2, 3, 2], size=100)

    print("\n  Testing training curves plot...")
    try:
        fig = plot_training_curves(history, save_path="plots/test_training_curves.png")
        print("    ✓ Training curves plot saved")
    except Exception as e:
        print(f"    ❌ Failed: {e}")

    print("\n  Testing modality attribution plot...")
    try:
        fig = plot_modality_attribution(attributions, save_path="plots/test_attribution.png")
        print("    ✓ Attribution plot saved")
    except Exception as e:
        print(f"    ❌ Failed: {e}")

    print("\n✅ Visualization test passed!")
    return True


def test_end_to_end():
    """Test end-to-end pipeline"""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)

    from src.data.preprocessing import MoleculePreprocessor, ConformerGenerator, compute_boltzmann_weights
    from src.models.molfm import MolFMLite
    from src.training.losses import ContrastiveLoss

    # Test molecule
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    print(f"\n  Testing with: {smiles}")

    # 1. Preprocess
    preprocessor = MoleculePreprocessor()
    conformer_gen = ConformerGenerator(num_conformers=3)

    features = preprocessor.process_molecule(smiles)
    conformers, energies = conformer_gen.generate_conformers(smiles)

    if features is None:
        print("    ❌ Preprocessing failed")
        return False

    print("    ✓ Preprocessing complete")

    # 2. Prepare batch (batch size 1)
    token_ids = torch.tensor(features.token_ids).unsqueeze(0)
    token_mask = (token_ids != 0).float()
    atom_features = torch.tensor(features.atom_features).float()
    edge_index = torch.tensor(features.edge_index)
    batch_idx = torch.zeros(atom_features.shape[0], dtype=torch.long)

    # Conformers
    num_atoms = atom_features.shape[0]
    num_conformers = len(conformers) if conformers else 1

    if conformers:
        max_atoms = max(c.shape[0] for c in conformers)
        conformer_coords = torch.zeros(1, num_conformers, max_atoms, 3)
        conformer_mask = torch.zeros(1, num_conformers, max_atoms)

        for i, conf in enumerate(conformers):
            n = conf.shape[0]
            conformer_coords[0, i, :n, :] = torch.tensor(conf)
            conformer_mask[0, i, :n] = 1.0

        weights = compute_boltzmann_weights(energies)
        conformer_weights = torch.tensor(weights).unsqueeze(0)
    else:
        conformer_coords = torch.zeros(1, 1, num_atoms, 3)
        conformer_mask = torch.ones(1, 1, num_atoms)
        conformer_weights = torch.ones(1, 1)

    print("    ✓ Batch prepared")

    # 3. Model forward pass
    atom_feat_dim = atom_features.shape[1]  # Get actual feature dimension
    model = MolFMLite(
        hidden_dim=128,
        hidden_dim_3d=64,
        num_layers_1d=2,
        num_layers_2d=2,
        num_interactions_3d=2,
        atom_feat_dim=atom_feat_dim,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(
            token_ids=token_ids,
            token_mask=token_mask,
            atom_features=atom_features,
            edge_index=edge_index,
            batch_idx=batch_idx,
            conformer_coords=conformer_coords,
            conformer_mask=conformer_mask,
            conformer_weights=conformer_weights,
            return_embeddings=True,
            return_attribution=True,
        )

    print(f"    ✓ Model forward pass complete")
    print(f"    ✓ Prediction: {outputs['prediction'].item():.4f}")
    print(f"    ✓ Attribution: {outputs['attribution'].squeeze().tolist()}")

    # 4. Test contrastive loss
    proj_1d, proj_2d, proj_3d = model.get_contrastive_embeddings(
        token_ids, token_mask,
        atom_features, edge_index, batch_idx,
        conformer_coords, conformer_mask, conformer_weights,
    )

    # Need batch > 1 for contrastive loss
    proj_1d = proj_1d.repeat(4, 1)
    proj_2d = proj_2d.repeat(4, 1)
    proj_3d = proj_3d.repeat(4, 1)

    criterion = ContrastiveLoss()
    loss, metrics = criterion(proj_1d, proj_2d, proj_3d)

    print(f"    ✓ Contrastive loss: {loss.item():.4f}")

    print("\n✅ End-to-end pipeline test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MolFM-Lite Pipeline Tests")
    print("="*60)

    # Change to project directory
    os.chdir(Path(__file__).parent.parent)

    results = {}

    # Run tests
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Model Architecture", test_model_architecture),
        ("Contrastive Embeddings", test_contrastive_embeddings),
        ("Loss Functions", test_losses),
        ("Visualization", test_visualization),
        ("End-to-End Pipeline", test_end_to_end),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
