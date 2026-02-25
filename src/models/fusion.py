"""Cross-modal fusion and context conditioning for MolFM-Lite"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            key_value: (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            mask: optional attention mask

        Returns:
            output: same shape as query
        """
        # Handle both 2D and 3D inputs
        is_2d = query.dim() == 2
        if is_2d:
            query = query.unsqueeze(1)
            key_value = key_value.unsqueeze(1)

        batch_size, seq_len, _ = query.shape
        _, kv_len, _ = key_value.shape

        # Project
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Combine
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        # Residual + layer norm
        out = self.layer_norm(query + out)

        if is_2d:
            out = out.squeeze(1)

        return out


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module for combining 1D, 2D, and 3D representations"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_type: str = "attention",  # "attention", "concat", "gated"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            # Cross-attention between modalities
            self.cross_attn_1d_2d = CrossModalAttention(hidden_dim, num_heads, dropout)
            self.cross_attn_1d_3d = CrossModalAttention(hidden_dim, num_heads, dropout)
            self.cross_attn_2d_3d = CrossModalAttention(hidden_dim, num_heads, dropout)

            # Final fusion layer
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        elif fusion_type == "gated":
            # Gated fusion
            self.gate_1d = nn.Linear(hidden_dim, hidden_dim)
            self.gate_2d = nn.Linear(hidden_dim, hidden_dim)
            self.gate_3d = nn.Linear(hidden_dim, hidden_dim)

            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        else:  # concat
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        emb_1d: torch.Tensor,
        emb_2d: torch.Tensor,
        emb_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb_1d: (batch, hidden_dim) - 1D encoder output
            emb_2d: (batch, hidden_dim) - 2D encoder output
            emb_3d: (batch, hidden_dim) - 3D encoder output

        Returns:
            fused: (batch, hidden_dim) - fused representation
        """
        if self.fusion_type == "attention":
            # Cross-attention fusion
            emb_1d_enhanced = self.cross_attn_1d_2d(emb_1d, emb_2d)
            emb_1d_enhanced = self.cross_attn_1d_3d(emb_1d_enhanced, emb_3d)

            emb_2d_enhanced = self.cross_attn_2d_3d(emb_2d, emb_3d)

            # Concatenate and project
            fused = torch.cat([emb_1d_enhanced, emb_2d_enhanced, emb_3d], dim=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_type == "gated":
            # Gated fusion
            gate_1d = torch.sigmoid(self.gate_1d(emb_1d))
            gate_2d = torch.sigmoid(self.gate_2d(emb_2d))
            gate_3d = torch.sigmoid(self.gate_3d(emb_3d))

            gated = torch.cat([
                gate_1d * emb_1d,
                gate_2d * emb_2d,
                gate_3d * emb_3d
            ], dim=-1)
            fused = self.fusion_proj(gated)

        else:  # concat
            fused = torch.cat([emb_1d, emb_2d, emb_3d], dim=-1)
            fused = self.fusion_proj(fused)

        fused = self.layer_norm(fused)
        return fused


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(context_dim, hidden_dim)
        self.beta_proj = nn.Linear(context_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim)
            context: (batch, context_dim)

        Returns:
            modulated: (batch, hidden_dim)
        """
        gamma = self.gamma_proj(context)
        beta = self.beta_proj(context)
        return gamma * x + beta


class ContextConditioning(nn.Module):
    """Context conditioning module for experimental conditions"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_assay_types: int = 32,
        target_dim: int = 128,
        cell_line_dim: int = 64,
        num_cell_lines: int = 100,
        use_film: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_film = use_film

        # Embeddings for categorical context features
        self.assay_embedding = nn.Embedding(num_assay_types, hidden_dim // 4)
        self.cell_line_embedding = nn.Embedding(num_cell_lines, cell_line_dim)

        # Projection for continuous context features
        self.continuous_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # concentration, temp, pH, time
            nn.ReLU(),
        )

        # Target protein embedding (could be pre-trained protein encoder)
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim // 4),
            nn.ReLU(),
        )

        # Context aggregation
        context_input_dim = hidden_dim // 4 * 3 + cell_line_dim
        self.context_aggregator = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM conditioning
        if use_film:
            self.film = FiLMLayer(hidden_dim, hidden_dim)

        # Alternative: additive conditioning
        self.additive_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mol_embedding: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            mol_embedding: (batch, hidden_dim)
            context: dict with keys:
                - assay_type: (batch,) int
                - cell_line: (batch,) int
                - target_embedding: (batch, target_dim)
                - continuous: (batch, 4) - [concentration, temp, pH, time]

        Returns:
            conditioned: (batch, hidden_dim)
        """
        if context is None:
            return mol_embedding

        batch_size = mol_embedding.shape[0]
        device = mol_embedding.device

        # Get context embeddings
        context_parts = []

        # Assay type
        if "assay_type" in context:
            assay_emb = self.assay_embedding(context["assay_type"])
        else:
            assay_emb = torch.zeros(batch_size, self.hidden_dim // 4, device=device)
        context_parts.append(assay_emb)

        # Cell line
        if "cell_line" in context:
            cell_emb = self.cell_line_embedding(context["cell_line"])
        else:
            cell_emb = torch.zeros(batch_size, 64, device=device)
        context_parts.append(cell_emb)

        # Target protein
        if "target_embedding" in context:
            target_emb = self.target_proj(context["target_embedding"])
        else:
            target_emb = torch.zeros(batch_size, self.hidden_dim // 4, device=device)
        context_parts.append(target_emb)

        # Continuous features
        if "continuous" in context:
            cont_emb = self.continuous_proj(context["continuous"])
        else:
            cont_emb = torch.zeros(batch_size, self.hidden_dim // 4, device=device)
        context_parts.append(cont_emb)

        # Aggregate context
        context_vec = torch.cat(context_parts, dim=-1)
        context_vec = self.context_aggregator(context_vec)

        # Apply conditioning
        if self.use_film:
            conditioned = self.film(mol_embedding, context_vec)
        else:
            conditioned = mol_embedding + self.additive_proj(context_vec)

        conditioned = self.layer_norm(conditioned)
        return conditioned


class ModalityAttribution(nn.Module):
    """Module for computing modality contributions to predictions"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.attribution = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        emb_1d: torch.Tensor,
        emb_2d: torch.Tensor,
        emb_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb_1d, emb_2d, emb_3d: (batch, hidden_dim)

        Returns:
            attributions: (batch, 3) - softmax weights for each modality
        """
        combined = torch.cat([emb_1d, emb_2d, emb_3d], dim=-1)
        return self.attribution(combined)
