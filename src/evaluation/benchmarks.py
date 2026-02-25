"""Benchmark evaluation for MolFM-Lite"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .metrics import compute_metrics, compute_uncertainty
from ..data.dataset import MoleculeDataset, collate_molecules
from ..data.loaders import download_moleculenet, load_moleculenet
from ..models.molfm import MolFMLite
from ..training.trainer import Trainer


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    dataset_name: str
    task_type: str
    metrics: Dict[str, float]
    metrics_std: Dict[str, float]
    num_seeds: int
    predictions: Optional[np.ndarray] = None
    uncertainties: Optional[np.ndarray] = None


class MoleculeNetBenchmark:
    """MoleculeNet benchmark evaluation"""

    DATASETS = {
        "bbbp": {"task_type": "classification", "num_tasks": 1, "metric": "auc_mean"},
        "bace": {"task_type": "classification", "num_tasks": 1, "metric": "auc_mean"},
        "tox21": {"task_type": "classification", "num_tasks": 12, "metric": "auc_mean"},
        "sider": {"task_type": "classification", "num_tasks": 27, "metric": "auc_mean"},
        "clintox": {"task_type": "classification", "num_tasks": 2, "metric": "auc_mean"},
        "lipophilicity": {"task_type": "regression", "num_tasks": 1, "metric": "rmse_mean"},
        "freesolv": {"task_type": "regression", "num_tasks": 1, "metric": "rmse_mean"},
        "esol": {"task_type": "regression", "num_tasks": 1, "metric": "rmse_mean"},
    }

    def __init__(
        self,
        model: MolFMLite,
        data_dir: str = "data/raw",
        cache_dir: str = "data/cache",
        device: str = "cuda",
        num_seeds: int = 3,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.model = model
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.num_seeds = num_seeds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def evaluate_dataset(
        self,
        dataset_name: str,
        num_epochs: int = 100,
        learning_rate: float = 5e-5,
        patience: int = 15,
    ) -> BenchmarkResult:
        """Evaluate on a single MoleculeNet dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.DATASETS[dataset_name]
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name}")
        print(f"Task type: {config['task_type']}, Num tasks: {config['num_tasks']}")
        print(f"{'='*50}")

        # Download dataset
        file_path, info = download_moleculenet(dataset_name, str(self.data_dir))
        if not file_path:
            raise RuntimeError(f"Failed to download {dataset_name}")

        # Load dataset
        smiles, labels, splits = load_moleculenet(file_path)

        # Create datasets for each split
        train_mask = np.array([s == "train" for s in splits])
        val_mask = np.array([s == "valid" for s in splits])
        test_mask = np.array([s == "test" for s in splits])

        train_smiles = [s for s, m in zip(smiles, train_mask) if m]
        val_smiles = [s for s, m in zip(smiles, val_mask) if m]
        test_smiles = [s for s, m in zip(smiles, test_mask) if m]

        train_labels = labels[train_mask]
        val_labels = labels[val_mask]
        test_labels = labels[test_mask]

        # Run multiple seeds
        all_metrics = []

        for seed in range(self.num_seeds):
            print(f"\nSeed {seed + 1}/{self.num_seeds}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Clear GPU memory between seeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            try:
                # Create fresh model copy with correct num_tasks
                model = self._create_model_copy(num_tasks=config["num_tasks"])

                # Create datasets with error handling
                try:
                    train_dataset = MoleculeDataset(
                        train_smiles, train_labels,
                        cache_dir=str(self.cache_dir / dataset_name / "train"),
                    )
                    val_dataset = MoleculeDataset(
                        val_smiles, val_labels,
                        cache_dir=str(self.cache_dir / dataset_name / "val"),
                    )
                    test_dataset = MoleculeDataset(
                        test_smiles, test_labels,
                        cache_dir=str(self.cache_dir / dataset_name / "test"),
                    )
                except Exception as e:
                    print(f"Error creating datasets: {e}")
                    raise

                # Check we have enough samples
                if len(train_dataset) < self.batch_size:
                    print(f"Warning: train set ({len(train_dataset)}) smaller than batch size ({self.batch_size})")
                    effective_batch = min(self.batch_size, len(train_dataset))
                else:
                    effective_batch = self.batch_size

                # Create dataloaders (drop_last=True for train to avoid partial batch issues)
                train_loader = DataLoader(
                    train_dataset, batch_size=effective_batch, shuffle=True,
                    num_workers=0, collate_fn=collate_molecules, drop_last=True,
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=effective_batch,
                    num_workers=0, collate_fn=collate_molecules, drop_last=False,
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=effective_batch,
                    num_workers=0, collate_fn=collate_molecules, drop_last=False,
                )

                print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
                print(f"  Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")

                # Train
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    patience=patience,
                    task_type=config["task_type"],
                    num_tasks=config["num_tasks"],
                    device=self.device,
                    checkpoint_dir=f"checkpoints/{dataset_name}/seed_{seed}",
                )

                trainer.train()

                # Load best model (if available)
                trainer.load_checkpoint("best_model.pt")

                # Evaluate on test set
                metrics = self._evaluate(trainer.model, test_loader, config["task_type"])
                all_metrics.append(metrics)
                print(f"  Test metrics: {metrics}")

            except Exception as e:
                print(f"Error in seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next seed instead of failing completely
                continue
            finally:
                # Clean up to free memory
                if torch.cuda.is_available():
                    del model, trainer
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

        # Aggregate results
        mean_metrics = {}
        std_metrics = {}

        if not all_metrics:
            raise RuntimeError(f"All {self.num_seeds} seeds failed for {dataset_name}")

        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                mean_metrics[key] = np.mean(values)
                std_metrics[key] = np.std(values)

        print(f"\nFinal Results for {dataset_name} ({len(all_metrics)}/{self.num_seeds} seeds succeeded):")
        for key in mean_metrics:
            print(f"  {key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")

        return BenchmarkResult(
            dataset_name=dataset_name,
            task_type=config["task_type"],
            metrics=mean_metrics,
            metrics_std=std_metrics,
            num_seeds=self.num_seeds,
        )

    def evaluate_all(
        self,
        datasets: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, BenchmarkResult]:
        """Evaluate on multiple datasets"""
        if datasets is None:
            datasets = list(self.DATASETS.keys())

        results = {}
        for dataset_name in datasets:
            try:
                result = self.evaluate_dataset(dataset_name, **kwargs)
                results[dataset_name] = result
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")

        return results

    def _create_model_copy(self, num_tasks: int = 1) -> MolFMLite:
        """Create a fresh copy of the model with specified num_tasks"""
        # Get atom_feat_dim from the model's 2D encoder input projection
        atom_feat_dim = 38  # Default matching MoleculePreprocessor output
        if hasattr(self.model, 'encoder_2d'):
            if hasattr(self.model.encoder_2d, 'input_proj'):
                atom_feat_dim = self.model.encoder_2d.input_proj.in_features
            elif hasattr(self.model.encoder_2d, 'mlp'):
                # Fallback mode - get input dim from first layer
                atom_feat_dim = self.model.encoder_2d.mlp[0].in_features

        model = MolFMLite(
            vocab_size=self.model.encoder_1d.token_embedding.num_embeddings,
            hidden_dim=self.model.hidden_dim,
            hidden_dim_3d=self.model.hidden_dim_3d,
            num_tasks=num_tasks,
            atom_feat_dim=atom_feat_dim,
        )

        # Load pretrained weights, skipping layers with size mismatches
        # strict=False ignores missing/extra keys but NOT size mismatches
        pretrained_dict = self.model.state_dict()
        model_dict = model.state_dict()

        # Filter out weights that have different shapes
        compatible_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            else:
                print(f"  Skipping incompatible weight: {k}")

        # Load compatible weights
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)

        print(f"  Loaded {len(compatible_dict)}/{len(pretrained_dict)} pretrained weights")
        return model.to(self.device)

    @torch.no_grad()
    def _evaluate(
        self,
        model: MolFMLite,
        dataloader: DataLoader,
        task_type: str,
    ) -> Dict[str, float]:
        """Evaluate model on a dataloader"""
        model.eval()
        all_preds = []
        all_labels = []

        for batch in dataloader:
            # Move to device
            batch.token_ids = batch.token_ids.to(self.device)
            batch.token_mask = batch.token_mask.to(self.device)
            batch.atom_features = batch.atom_features.to(self.device)
            batch.edge_index = batch.edge_index.to(self.device)
            batch.batch_idx = batch.batch_idx.to(self.device)
            batch.conformer_coords = batch.conformer_coords.to(self.device)
            batch.conformer_mask = batch.conformer_mask.to(self.device)
            batch.conformer_weights = batch.conformer_weights.to(self.device)

            outputs = model(
                token_ids=batch.token_ids,
                token_mask=batch.token_mask,
                atom_features=batch.atom_features,
                edge_index=batch.edge_index,
                batch_idx=batch.batch_idx,
                conformer_coords=batch.conformer_coords,
                conformer_mask=batch.conformer_mask,
                conformer_weights=batch.conformer_weights,
            )

            preds = outputs["prediction"]
            if task_type == "classification":
                preds = torch.sigmoid(preds)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return compute_metrics(all_labels, all_preds, task_type)


class CrossContextBenchmark:
    """
    Novel benchmark for evaluating context-awareness.
    Tests generalization from biochemical to cell-based assays.
    """

    def __init__(
        self,
        model: MolFMLite,
        data_dir: str = "data/raw",
        device: str = "cuda",
    ):
        self.model = model
        self.data_dir = Path(data_dir)
        self.device = device

    def create_benchmark_data(
        self,
        chembl_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create cross-context benchmark from ChEMBL data.

        Splits data into:
        - Train: Biochemical assays
        - Test: Cell-based assays (same targets)
        """
        # This would load real ChEMBL data with assay type annotations
        # For now, we create a placeholder structure

        # In real implementation:
        # 1. Load ChEMBL data with assay_type column
        # 2. Filter to targets with both biochemical and cell-based assays
        # 3. Split by assay type

        print("Cross-context benchmark requires ChEMBL data with assay annotations.")
        print("This is a placeholder for the novel benchmark.")

        return None, None

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        **kwargs,
    ) -> BenchmarkResult:
        """Evaluate cross-context generalization"""
        # This would be similar to MoleculeNetBenchmark.evaluate_dataset
        # but with context conditioning enabled

        raise NotImplementedError("Requires ChEMBL data setup")


class AblationStudy:
    """Run ablation studies to understand model components"""

    def __init__(
        self,
        base_model: MolFMLite,
        benchmark: MoleculeNetBenchmark,
    ):
        self.base_model = base_model
        self.benchmark = benchmark

    def run_ablations(
        self,
        dataset_name: str = "bbbp",
        ablations: List[str] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run ablation studies.

        Ablations:
        - no_context: Disable context conditioning
        - single_conformer: Use single conformer instead of ensemble
        - 1d_only: Only use 1D encoder
        - 2d_only: Only use 2D encoder
        - 3d_only: Only use 3D encoder
        - no_pretraining: Train from scratch
        """
        if ablations is None:
            ablations = [
                "full_model",
                "no_context",
                "single_conformer",
                "1d_only",
                "2d_only",
                "3d_only",
            ]

        results = {}

        for ablation in ablations:
            print(f"\nRunning ablation: {ablation}")
            model = self._create_ablated_model(ablation)
            self.benchmark.model = model
            result = self.benchmark.evaluate_dataset(dataset_name, num_epochs=50)
            results[ablation] = result

        return results

    def _create_ablated_model(self, ablation: str) -> MolFMLite:
        """Create model with specific ablation"""
        # This would modify model architecture based on ablation type
        # For now, return base model
        return self.base_model
