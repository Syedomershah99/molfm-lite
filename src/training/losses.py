"""Loss functions for MolFM-Lite pre-training and fine-tuning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Cross-modal contrastive loss using InfoNCE.
    Aligns representations across 1D, 2D, and 3D modalities.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives

    def forward(
        self,
        proj_1d: torch.Tensor,
        proj_2d: torch.Tensor,
        proj_3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute cross-modal contrastive loss.

        Args:
            proj_1d: (batch, hidden_dim) - normalized 1D projections
            proj_2d: (batch, hidden_dim) - normalized 2D projections
            proj_3d: (batch, hidden_dim) - normalized 3D projections

        Returns:
            loss: scalar tensor
            metrics: dict with individual loss components
        """
        batch_size = proj_1d.shape[0]
        device = proj_1d.device

        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=device)

        # 1D-2D contrastive
        sim_1d_2d = torch.matmul(proj_1d, proj_2d.T) / self.temperature
        loss_1d_2d = F.cross_entropy(sim_1d_2d, labels)
        loss_2d_1d = F.cross_entropy(sim_1d_2d.T, labels)

        # 1D-3D contrastive
        sim_1d_3d = torch.matmul(proj_1d, proj_3d.T) / self.temperature
        loss_1d_3d = F.cross_entropy(sim_1d_3d, labels)
        loss_3d_1d = F.cross_entropy(sim_1d_3d.T, labels)

        # 2D-3D contrastive
        sim_2d_3d = torch.matmul(proj_2d, proj_3d.T) / self.temperature
        loss_2d_3d = F.cross_entropy(sim_2d_3d, labels)
        loss_3d_2d = F.cross_entropy(sim_2d_3d.T, labels)

        # Total loss
        loss = (loss_1d_2d + loss_2d_1d + loss_1d_3d + loss_3d_1d + loss_2d_3d + loss_3d_2d) / 6

        metrics = {
            "loss_1d_2d": (loss_1d_2d + loss_2d_1d).item() / 2,
            "loss_1d_3d": (loss_1d_3d + loss_3d_1d).item() / 2,
            "loss_2d_3d": (loss_2d_3d + loss_3d_2d).item() / 2,
        }

        return loss, metrics


class ConsistencyLoss(nn.Module):
    """
    Cross-modal consistency loss.
    Ensures predictions from different modalities are consistent.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        emb_1d: torch.Tensor,
        emb_2d: torch.Tensor,
        emb_3d: torch.Tensor,
        prediction_head: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute consistency loss between modality-specific predictions.

        Args:
            emb_1d, emb_2d, emb_3d: (batch, hidden_dim) - modality embeddings
            prediction_head: prediction module to get predictions from each embedding

        Returns:
            loss: scalar tensor
            metrics: dict with individual loss components
        """
        # Get predictions from each modality
        with torch.no_grad():
            # Use fused embedding prediction as target
            fused = (emb_1d + emb_2d + emb_3d) / 3
            target, _ = prediction_head(fused)

        pred_1d, _ = prediction_head(emb_1d)
        pred_2d, _ = prediction_head(emb_2d)
        pred_3d, _ = prediction_head(emb_3d)

        if self.loss_type == "mse":
            loss_1d = F.mse_loss(pred_1d, target)
            loss_2d = F.mse_loss(pred_2d, target)
            loss_3d = F.mse_loss(pred_3d, target)
        else:  # cosine
            loss_1d = 1 - F.cosine_similarity(pred_1d, target).mean()
            loss_2d = 1 - F.cosine_similarity(pred_2d, target).mean()
            loss_3d = 1 - F.cosine_similarity(pred_3d, target).mean()

        loss = (loss_1d + loss_2d + loss_3d) / 3

        metrics = {
            "consistency_1d": loss_1d.item(),
            "consistency_2d": loss_2d.item(),
            "consistency_3d": loss_3d.item(),
        }

        return loss, metrics


class MaskedAtomPredictionLoss(nn.Module):
    """
    Masked atom prediction loss for self-supervised learning.
    Similar to masked language modeling but for atoms.
    """

    def __init__(self, num_atom_types: int = 14, hidden_dim: int = 256):
        super().__init__()
        self.atom_classifier = nn.Linear(hidden_dim, num_atom_types)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute masked atom prediction loss.

        Args:
            node_embeddings: (num_atoms, hidden_dim)
            atom_types: (num_atoms,) - ground truth atom types
            mask: (num_atoms,) - True for masked atoms

        Returns:
            loss: scalar tensor
            metrics: dict with accuracy
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device), {"map_accuracy": 0.0}

        # Get masked embeddings
        masked_emb = node_embeddings[mask]
        masked_labels = atom_types[mask]

        # Predict
        logits = self.atom_classifier(masked_emb)
        loss = F.cross_entropy(logits, masked_labels)

        # Accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == masked_labels).float().mean()

        return loss, {"map_accuracy": accuracy.item()}


class MultiTaskLoss(nn.Module):
    """
    Combined loss for pre-training with multiple objectives.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        consistency_weight: float = 0.3,
        map_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.map_weight = map_weight

        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.consistency_loss = ConsistencyLoss()
        self.map_loss = MaskedAtomPredictionLoss()

    def forward(
        self,
        proj_1d: torch.Tensor,
        proj_2d: torch.Tensor,
        proj_3d: torch.Tensor,
        emb_1d: Optional[torch.Tensor] = None,
        emb_2d: Optional[torch.Tensor] = None,
        emb_3d: Optional[torch.Tensor] = None,
        prediction_head: Optional[nn.Module] = None,
        node_embeddings: Optional[torch.Tensor] = None,
        atom_types: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined pre-training loss.

        Returns:
            total_loss: scalar tensor
            metrics: dict with all loss components
        """
        total_loss = 0.0
        metrics = {}

        # Contrastive loss
        contrastive, c_metrics = self.contrastive_loss(proj_1d, proj_2d, proj_3d)
        total_loss = total_loss + self.contrastive_weight * contrastive
        metrics.update({f"contrastive_{k}": v for k, v in c_metrics.items()})
        metrics["contrastive_loss"] = contrastive.item()

        # Consistency loss (if embeddings and prediction head provided)
        if emb_1d is not None and prediction_head is not None:
            consistency, cons_metrics = self.consistency_loss(
                emb_1d, emb_2d, emb_3d, prediction_head
            )
            total_loss = total_loss + self.consistency_weight * consistency
            metrics.update(cons_metrics)
            metrics["consistency_loss"] = consistency.item()

        # Masked atom prediction (if provided)
        if node_embeddings is not None and atom_mask is not None:
            map_loss, map_metrics = self.map_loss(node_embeddings, atom_types, atom_mask)
            total_loss = total_loss + self.map_weight * map_loss
            metrics.update(map_metrics)
            metrics["map_loss"] = map_loss.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


class TaskLoss(nn.Module):
    """
    Loss function for fine-tuning on downstream tasks.
    Supports both regression and classification.
    """

    def __init__(
        self,
        task_type: str = "regression",
        num_tasks: int = 1,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks

        if task_type == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            self.loss_fn = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute task loss.

        Args:
            predictions: (batch, num_tasks)
            targets: (batch, num_tasks)
            mask: (batch, num_tasks) - optional mask for missing labels

        Returns:
            loss: scalar tensor
        """
        loss = self.loss_fn(predictions, targets)

        if mask is not None:
            # Only compute loss for non-missing labels
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1e-8)
        else:
            loss = loss.mean()

        return loss
