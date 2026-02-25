"""Training utilities for MolFM-Lite"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json

from ..models.molfm import MolFMLite
from .losses import MultiTaskLoss, TaskLoss


class Trainer:
    """Base trainer class for fine-tuning"""

    def __init__(
        self,
        model: MolFMLite,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        patience: int = 15,
        task_type: str = "regression",
        num_tasks: int = 1,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = TaskLoss(task_type=task_type, num_tasks=num_tasks)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                token_ids=batch.token_ids,
                token_mask=batch.token_mask,
                atom_features=batch.atom_features,
                edge_index=batch.edge_index,
                batch_idx=batch.batch_idx,
                conformer_coords=batch.conformer_coords,
                conformer_mask=batch.conformer_mask,
                conformer_weights=batch.conformer_weights,
                context=batch.context,
            )

            # Compute loss
            predictions = outputs["prediction"]
            labels = batch.labels

            # Create mask for missing labels (labeled as -1)
            mask = (labels != -1).float()
            labels = labels.clamp(min=0)  # Replace -1 with 0 for loss computation

            loss = self.criterion(predictions, labels, mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss, {}

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            batch = self._to_device(batch)

            outputs = self.model(
                token_ids=batch.token_ids,
                token_mask=batch.token_mask,
                atom_features=batch.atom_features,
                edge_index=batch.edge_index,
                batch_idx=batch.batch_idx,
                conformer_coords=batch.conformer_coords,
                conformer_mask=batch.conformer_mask,
                conformer_weights=batch.conformer_weights,
                context=batch.context,
            )

            predictions = outputs["prediction"]
            labels = batch.labels

            mask = (labels != -1).float()
            labels_clean = labels.clamp(min=0)

            loss = self.criterion(predictions, labels_clean, mask)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / num_batches

        # Compute metrics
        metrics = self._compute_metrics(
            torch.cat(all_preds), torch.cat(all_labels)
        )

        return avg_loss, metrics

    def _compute_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        from sklearn.metrics import roc_auc_score, mean_squared_error
        import numpy as np

        metrics = {}

        # Handle multi-task
        num_tasks = predictions.shape[1] if predictions.dim() > 1 else 1

        for task_idx in range(num_tasks):
            if num_tasks > 1:
                preds = predictions[:, task_idx].numpy()
                labs = labels[:, task_idx].numpy()
            else:
                preds = predictions.squeeze().numpy()
                labs = labels.squeeze().numpy()

            # Filter out missing labels
            valid = labs != -1
            preds = preds[valid]
            labs = labs[valid]

            if len(labs) == 0:
                continue

            # Classification metrics
            if len(np.unique(labs)) == 2:
                try:
                    auc = roc_auc_score(labs, preds)
                    metrics[f"auc_task_{task_idx}"] = auc
                except Exception:
                    pass

            # Regression metrics
            rmse = np.sqrt(mean_squared_error(labs, preds))
            metrics[f"rmse_task_{task_idx}"] = rmse

        return metrics

    def train(self) -> Dict[str, Any]:
        """Full training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["metrics"].append(val_metrics)

            # Update scheduler
            self.scheduler.step()

            # Log
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            if val_metrics:
                print(f"  Metrics: {val_metrics}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save final model
        self.save_checkpoint("final_model.pt")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {filename} not found, skipping load")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model weights safely (handle potential size mismatches)
            if "model_state_dict" in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint["model_state_dict"]
                # Filter compatible weights
                compatible_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)

            # Load optimizer/scheduler (wrapped in try-except for robustness)
            try:
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")

            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "metrics": []})
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def _to_device(self, batch):
        """Move batch to device"""
        batch.token_ids = batch.token_ids.to(self.device)
        batch.token_mask = batch.token_mask.to(self.device)
        batch.atom_features = batch.atom_features.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        batch.edge_features = batch.edge_features.to(self.device)
        batch.batch_idx = batch.batch_idx.to(self.device)
        batch.conformer_coords = batch.conformer_coords.to(self.device)
        batch.conformer_mask = batch.conformer_mask.to(self.device)
        batch.conformer_weights = batch.conformer_weights.to(self.device)
        if batch.labels is not None:
            batch.labels = batch.labels.to(self.device)
        return batch


class PretrainingTrainer:
    """Trainer for contrastive pre-training"""

    def __init__(
        self,
        model: MolFMLite,
        train_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        warmup_steps: int = 1000,
        contrastive_temp: float = 0.07,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        save_interval: int = 1000,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = MultiTaskLoss(temperature=contrastive_temp)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=len(train_loader), T_mult=2
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        # Tracking
        self.global_step = 0
        self.history = {"loss": [], "metrics": []}

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            self.optimizer.zero_grad()

            # Get contrastive embeddings
            proj_1d, proj_2d, proj_3d = self.model.get_contrastive_embeddings(
                token_ids=batch.token_ids,
                token_mask=batch.token_mask,
                atom_features=batch.atom_features,
                edge_index=batch.edge_index,
                batch_idx=batch.batch_idx,
                conformer_coords=batch.conformer_coords,
                conformer_mask=batch.conformer_mask,
                conformer_weights=batch.conformer_weights,
            )

            # Compute loss
            loss, metrics = self.criterion(proj_1d, proj_2d, proj_3d)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Log
            if self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Step {self.global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        return avg_loss, avg_metrics

    def train(self) -> Dict[str, Any]:
        """Full pre-training loop"""
        print(f"Starting pre-training for {self.num_epochs} epochs...")
        print(f"Total steps: {self.num_epochs * len(self.train_loader)}")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            loss, metrics = self.train_epoch(epoch)
            self.history["loss"].append(loss)
            self.history["metrics"].append(metrics)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"  Metrics: {metrics}")

        # Save final model
        self.save_checkpoint("pretrained_model.pt")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "history": self.history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, filename: str) -> None:
        """Load checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {filename} not found, skipping load")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model weights safely
            if "model_state_dict" in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint["model_state_dict"]
                compatible_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)

            try:
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")

            self.global_step = checkpoint.get("global_step", 0)
            self.history = checkpoint.get("history", {"loss": [], "metrics": []})
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def _to_device(self, batch):
        """Move batch to device"""
        batch.token_ids = batch.token_ids.to(self.device)
        batch.token_mask = batch.token_mask.to(self.device)
        batch.atom_features = batch.atom_features.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        batch.edge_features = batch.edge_features.to(self.device)
        batch.batch_idx = batch.batch_idx.to(self.device)
        batch.conformer_coords = batch.conformer_coords.to(self.device)
        batch.conformer_mask = batch.conformer_mask.to(self.device)
        batch.conformer_weights = batch.conformer_weights.to(self.device)
        return batch
