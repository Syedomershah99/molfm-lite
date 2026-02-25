"""MolFM-Lite: Multi-Modal Molecular Foundation Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .encoders import Encoder1D, Encoder2D, Encoder3D, ConformerEnsembleAttention
from .fusion import CrossModalFusion, ContextConditioning, ModalityAttribution


class PredictionHead(nn.Module):
    """Prediction head with uncertainty estimation"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128],
        output_dim: int = 1,
        dropout: float = 0.2,
        task_type: str = "regression",  # "regression" or "classification"
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False, num_samples: int = 10
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, input_dim)
            return_uncertainty: whether to compute MC Dropout uncertainty
            num_samples: number of forward passes for MC Dropout

        Returns:
            prediction: (batch, output_dim)
            uncertainty: (batch, output_dim) or None
        """
        if not return_uncertainty or not self.training:
            pred = self.mlp(x)
            return pred, None

        # MC Dropout for uncertainty
        self.train()  # Enable dropout
        preds = []
        for _ in range(num_samples):
            preds.append(self.mlp(x))
        preds = torch.stack(preds, dim=0)

        mean_pred = preds.mean(dim=0)
        uncertainty = preds.std(dim=0)

        return mean_pred, uncertainty


class MolFMLite(nn.Module):
    """MolFM-Lite: Context-Aware Multi-Modal Molecular Foundation Model"""

    def __init__(
        self,
        # 1D encoder config
        vocab_size: int = 128,
        max_seq_len: int = 256,
        # Shared dimensions
        hidden_dim: int = 256,
        hidden_dim_3d: int = 128,
        # 1D encoder
        num_layers_1d: int = 4,
        num_heads_1d: int = 8,
        # 2D encoder
        num_layers_2d: int = 4,
        atom_feat_dim: int = 38,  # Matches MoleculePreprocessor output
        # 3D encoder
        num_interactions_3d: int = 3,
        num_filters_3d: int = 128,
        cutoff_3d: float = 10.0,
        # Conformer attention
        use_energy_weights: bool = True,
        # Fusion
        fusion_type: str = "attention",
        num_heads_fusion: int = 8,
        # Context conditioning
        use_context: bool = True,
        num_assay_types: int = 32,
        # Prediction head
        head_hidden_dims: list = [256, 128],
        num_tasks: int = 1,
        task_type: str = "regression",
        # General
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_3d = hidden_dim_3d
        self.use_context = use_context

        # 1D Encoder (SELFIES)
        self.encoder_1d = Encoder1D(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers_1d,
            num_heads=num_heads_1d,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 2D Encoder (Graph)
        self.encoder_2d = Encoder2D(
            input_dim=atom_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_2d,
            dropout=dropout,
        )

        # 3D Encoder (Conformers)
        self.encoder_3d = Encoder3D(
            hidden_dim=hidden_dim_3d,
            num_interactions=num_interactions_3d,
            num_filters=num_filters_3d,
            cutoff=cutoff_3d,
            atom_feat_dim=atom_feat_dim,
        )

        # Project 3D to same dimension as 1D/2D
        self.proj_3d = nn.Linear(hidden_dim_3d, hidden_dim)

        # Conformer ensemble attention
        self.conformer_attention = ConformerEnsembleAttention(
            hidden_dim=hidden_dim_3d,
            use_energy_weights=use_energy_weights,
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads_fusion,
            dropout=dropout,
            fusion_type=fusion_type,
        )

        # Context conditioning
        if use_context:
            self.context_conditioning = ContextConditioning(
                hidden_dim=hidden_dim,
                num_assay_types=num_assay_types,
            )

        # Modality attribution
        self.modality_attribution = ModalityAttribution(hidden_dim=hidden_dim)

        # Prediction head
        self.prediction_head = PredictionHead(
            input_dim=hidden_dim,
            hidden_dims=head_hidden_dims,
            output_dim=num_tasks,
            dropout=dropout,
            task_type=task_type,
        )

        # For contrastive learning: projection heads
        self.proj_head_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj_head_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj_head_3d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_1d(
        self, token_ids: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode 1D (SELFIES) representation"""
        _, pooled = self.encoder_1d(token_ids, token_mask)
        return pooled

    def encode_2d(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 2D (graph) representation"""
        _, graph_emb = self.encoder_2d(atom_features, edge_index, batch_idx)
        return graph_emb

    def encode_3d(
        self,
        conformer_coords: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor,
        atom_features_batched: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 3D (conformer) representation with ensemble attention"""
        batch_size, num_conformers, max_atoms, _ = conformer_coords.shape

        # Encode each conformer
        conformer_embeddings = []
        for i in range(num_conformers):
            coords = conformer_coords[:, i, :, :]  # (batch, max_atoms, 3)
            mask = conformer_mask[:, i, :]  # (batch, max_atoms)

            _, conf_emb = self.encoder_3d(atom_features_batched, coords, mask)
            conformer_embeddings.append(conf_emb)

        conformer_embeddings = torch.stack(conformer_embeddings, dim=1)  # (batch, num_conf, hidden_3d)

        # Aggregate with attention
        ensemble_emb = self.conformer_attention(
            conformer_embeddings,
            conformer_weights,
            conformer_mask.any(dim=-1),  # (batch, num_conf)
        )

        # Project to common dimension
        ensemble_emb = self.proj_3d(ensemble_emb)

        return ensemble_emb

    def get_embeddings(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        conformer_coords: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get embeddings from all modalities"""
        # Determine batch size from token_ids (most reliable source)
        batch_size = token_ids.shape[0]

        # 1D encoding
        emb_1d = self.encode_1d(token_ids, token_mask)

        # 2D encoding - ensure batch_idx is properly bounded
        # Clamp batch_idx to valid range in case of any inconsistency
        batch_idx_clamped = batch_idx.clamp(0, batch_size - 1)
        emb_2d = self.encode_2d(atom_features, edge_index, batch_idx_clamped)

        # Ensure 2D embedding matches batch size
        if emb_2d.shape[0] != batch_size:
            # Pad or truncate to match batch size
            if emb_2d.shape[0] < batch_size:
                padding = torch.zeros(
                    batch_size - emb_2d.shape[0], emb_2d.shape[1],
                    device=emb_2d.device, dtype=emb_2d.dtype
                )
                emb_2d = torch.cat([emb_2d, padding], dim=0)
            else:
                emb_2d = emb_2d[:batch_size]

        # 3D encoding - need batched atom features
        max_atoms = conformer_coords.shape[2]
        atom_feat_dim = atom_features.shape[1]

        # Create batched atom features for 3D encoder
        atom_features_batched = torch.zeros(
            batch_size, max_atoms, atom_feat_dim, device=atom_features.device
        )
        for i in range(batch_size):
            mask = batch_idx_clamped == i
            num_atoms = mask.sum()
            if num_atoms > 0:
                # Ensure we don't exceed max_atoms
                actual_atoms = min(num_atoms.item(), max_atoms)
                atom_features_batched[i, :actual_atoms] = atom_features[mask][:actual_atoms]

        emb_3d = self.encode_3d(
            conformer_coords, conformer_mask, conformer_weights, atom_features_batched
        )

        return emb_1d, emb_2d, emb_3d

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        conformer_coords: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
        return_embeddings: bool = False,
        return_uncertainty: bool = False,
        return_attribution: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Returns:
            Dictionary with:
                - prediction: (batch, num_tasks)
                - uncertainty: (batch, num_tasks) if return_uncertainty
                - emb_1d, emb_2d, emb_3d: modality embeddings if return_embeddings
                - attribution: (batch, 3) modality contributions if return_attribution
        """
        # Get embeddings
        emb_1d, emb_2d, emb_3d = self.get_embeddings(
            token_ids, token_mask,
            atom_features, edge_index, batch_idx,
            conformer_coords, conformer_mask, conformer_weights,
        )

        # Fuse modalities
        fused = self.fusion(emb_1d, emb_2d, emb_3d)

        # Apply context conditioning
        if self.use_context and context is not None:
            fused = self.context_conditioning(fused, context)

        # Predict
        prediction, uncertainty = self.prediction_head(
            fused, return_uncertainty=return_uncertainty
        )

        # Build output
        output = {"prediction": prediction}

        if uncertainty is not None:
            output["uncertainty"] = uncertainty

        if return_embeddings:
            output["emb_1d"] = emb_1d
            output["emb_2d"] = emb_2d
            output["emb_3d"] = emb_3d
            output["emb_fused"] = fused

        if return_attribution:
            output["attribution"] = self.modality_attribution(emb_1d, emb_2d, emb_3d)

        return output

    def get_contrastive_embeddings(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        conformer_coords: torch.Tensor,
        conformer_mask: torch.Tensor,
        conformer_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get projected embeddings for contrastive learning"""
        emb_1d, emb_2d, emb_3d = self.get_embeddings(
            token_ids, token_mask,
            atom_features, edge_index, batch_idx,
            conformer_coords, conformer_mask, conformer_weights,
        )

        # Project for contrastive learning
        proj_1d = F.normalize(self.proj_head_1d(emb_1d), dim=-1)
        proj_2d = F.normalize(self.proj_head_2d(emb_2d), dim=-1)
        proj_3d = F.normalize(self.proj_head_3d(emb_3d), dim=-1)

        return proj_1d, proj_2d, proj_3d


def create_model(config: Dict[str, Any]) -> MolFMLite:
    """Create MolFM-Lite model from config"""
    model_config = config.get("model", {})

    return MolFMLite(
        vocab_size=model_config.get("encoder_1d", {}).get("vocab_size", 128),
        max_seq_len=model_config.get("encoder_1d", {}).get("max_seq_len", 256),
        hidden_dim=model_config.get("encoder_1d", {}).get("hidden_dim", 256),
        hidden_dim_3d=model_config.get("encoder_3d", {}).get("hidden_dim", 128),
        num_layers_1d=model_config.get("encoder_1d", {}).get("num_layers", 4),
        num_heads_1d=model_config.get("encoder_1d", {}).get("num_heads", 8),
        num_layers_2d=model_config.get("encoder_2d", {}).get("num_layers", 4),
        num_interactions_3d=model_config.get("encoder_3d", {}).get("num_interactions", 3),
        num_filters_3d=model_config.get("encoder_3d", {}).get("num_filters", 128),
        cutoff_3d=model_config.get("encoder_3d", {}).get("cutoff", 10.0),
        use_energy_weights=model_config.get("conformer_attention", {}).get("use_energy_weights", True),
        fusion_type=model_config.get("fusion", {}).get("type", "attention"),
        use_context=model_config.get("context", {}).get("use_film", True),
        dropout=model_config.get("encoder_1d", {}).get("dropout", 0.1),
    )
