"""Modality encoders for MolFM-Lite"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool
    from torch_geometric.utils import softmax as geometric_softmax
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Encoder1D(nn.Module):
    """1D Encoder for SELFIES/SMILES sequences using Transformer"""

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid tokens, 0 for padding

        Returns:
            sequence_output: (batch, seq_len, hidden_dim)
            pooled_output: (batch, hidden_dim)
        """
        # Embed tokens
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Transformer expects True for positions to mask
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Pool: use CLS token (first token) or mean pooling
        if attention_mask is not None:
            # Ensure mask matches sequence length
            seq_len = x.shape[1]
            if attention_mask.shape[1] != seq_len:
                # Pad or truncate mask to match
                if attention_mask.shape[1] < seq_len:
                    padding = torch.zeros(attention_mask.shape[0], seq_len - attention_mask.shape[1],
                                         device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
                else:
                    attention_mask = attention_mask[:, :seq_len]

            # Mean pooling over valid tokens
            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        return x, pooled


class Encoder2D(nn.Module):
    """2D Encoder for molecular graphs using GIN"""

    def __init__(
        self,
        input_dim: int = 38,  # Atom feature dimension (matches MoleculePreprocessor)
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        if not TORCH_GEOMETRIC_AVAILABLE:
            # Fallback to simple MLP if torch_geometric not available
            self.use_fallback = True
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return

        self.use_fallback = False

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_features: (num_atoms, atom_feat_dim)
            edge_index: (2, num_edges)
            batch_idx: (num_atoms,) - maps atoms to molecules

        Returns:
            node_embeddings: (num_atoms, hidden_dim)
            graph_embeddings: (batch_size, hidden_dim)
        """
        if self.use_fallback:
            x = self.mlp(atom_features)
            # Simple pooling by batch
            batch_size = batch_idx.max().item() + 1
            graph_emb = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(batch_size):
                mask = batch_idx == i
                graph_emb[i] = x[mask].mean(dim=0)
            return x, graph_emb

        # Input projection
        x = self.input_proj(atom_features)

        # Message passing
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection

        # Output projection
        node_emb = self.output_proj(x)

        # Graph-level pooling
        if self.pooling == "mean":
            graph_emb = global_mean_pool(node_emb, batch_idx)
        else:
            graph_emb = global_add_pool(node_emb, batch_idx)

        return node_emb, graph_emb


class SchNetInteraction(nn.Module):
    """SchNet interaction block"""

    def __init__(self, hidden_dim: int, num_filters: int, cutoff: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Filter network for continuous filter convolution
        self.filter_net = nn.Sequential(
            nn.Linear(1, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )

        # Continuous filter convolution
        self.conv = nn.Linear(num_filters, hidden_dim)

        # Update network
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_atoms, hidden_dim)
            pos: (batch, num_atoms, 3)
            batch_mask: (batch, num_atoms) - 1 for valid atoms

        Returns:
            x: (batch, num_atoms, hidden_dim)
        """
        batch_size, num_atoms, _ = x.shape

        # Compute pairwise distances
        # pos: (batch, num_atoms, 3)
        pos_i = pos.unsqueeze(2)  # (batch, num_atoms, 1, 3)
        pos_j = pos.unsqueeze(1)  # (batch, 1, num_atoms, 3)
        dist = torch.norm(pos_i - pos_j, dim=-1)  # (batch, num_atoms, num_atoms)

        # Apply cutoff - convert mask to bool for bitwise operations
        cutoff_mask = (dist < self.cutoff) & (dist > 0)
        batch_mask_bool = batch_mask.bool()
        cutoff_mask = cutoff_mask & batch_mask_bool.unsqueeze(1) & batch_mask_bool.unsqueeze(2)

        # Compute filter values
        dist_feat = dist.unsqueeze(-1)  # (batch, num_atoms, num_atoms, 1)
        filters = self.filter_net(dist_feat)  # (batch, num_atoms, num_atoms, num_filters)

        # Apply cutoff smoothly
        cutoff_vals = 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1)
        cutoff_vals = cutoff_vals * cutoff_mask.float()
        filters = filters * cutoff_vals.unsqueeze(-1)

        # Continuous filter convolution
        # x_j: (batch, 1, num_atoms, hidden_dim)
        x_j = x.unsqueeze(1)
        # filters after conv: (batch, num_atoms, num_atoms, hidden_dim)
        conv_filters = self.conv(filters)
        # Aggregate: sum over neighbors
        x_conv = (conv_filters * x_j).sum(dim=2)  # (batch, num_atoms, hidden_dim)

        # Update
        x = x + self.update(x_conv)

        return x


class Encoder3D(nn.Module):
    """3D Encoder for molecular conformers using SchNet-lite"""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_interactions: int = 3,
        num_filters: int = 128,
        cutoff: float = 10.0,
        atom_feat_dim: int = 38,  # Matches MoleculePreprocessor output
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Atom embedding (from atom features)
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_filters, cutoff)
            for _ in range(num_interactions)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_features: (batch, num_atoms, atom_feat_dim)
            positions: (batch, num_atoms, 3)
            mask: (batch, num_atoms) - 1 for valid atoms

        Returns:
            atom_embeddings: (batch, num_atoms, hidden_dim)
            mol_embedding: (batch, hidden_dim)
        """
        # Embed atoms
        x = self.atom_embedding(atom_features)

        # Apply interaction blocks
        for interaction in self.interactions:
            x = interaction(x, positions, mask)

        # Output projection
        x = self.output_proj(x)

        # Pool to get molecule-level representation
        mask_expanded = mask.unsqueeze(-1)
        mol_emb = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return x, mol_emb


class ConformerEnsembleAttention(nn.Module):
    """Attention mechanism to aggregate conformer representations"""

    def __init__(
        self,
        hidden_dim: int = 128,
        use_energy_weights: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_energy_weights = use_energy_weights
        self.temperature = temperature

        # Learnable attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Energy weight projection (optional)
        if use_energy_weights:
            self.energy_proj = nn.Linear(1, hidden_dim // 4)

    def forward(
        self,
        conformer_embeddings: torch.Tensor,
        boltzmann_weights: torch.Tensor,
        conformer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            conformer_embeddings: (batch, num_conformers, hidden_dim)
            boltzmann_weights: (batch, num_conformers) - pre-computed Boltzmann weights
            conformer_mask: (batch, num_conformers) - 1 for valid conformers

        Returns:
            ensemble_embedding: (batch, hidden_dim)
        """
        batch_size, num_conformers, _ = conformer_embeddings.shape

        # Compute attention scores
        attn_logits = self.attention(conformer_embeddings).squeeze(-1)  # (batch, num_conf)

        # Combine with Boltzmann weights if enabled
        if self.use_energy_weights:
            # Use Boltzmann weights as prior
            attn_logits = attn_logits + torch.log(boltzmann_weights + 1e-8)

        # Apply temperature
        attn_logits = attn_logits / self.temperature

        # Mask invalid conformers
        if conformer_mask is not None:
            attn_logits = attn_logits.masked_fill(conformer_mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Weighted sum
        ensemble_emb = (attn_weights.unsqueeze(-1) * conformer_embeddings).sum(dim=1)

        return ensemble_emb
