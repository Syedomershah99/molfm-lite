"""Dataset classes for MolFM-Lite"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    MoleculePreprocessor,
    ConformerGenerator,
    MolecularFeatures,
    compute_boltzmann_weights,
)


@dataclass
class MoleculeData:
    """Batched molecule data for model input"""
    # 1D data
    token_ids: torch.Tensor  # (batch, seq_len)
    token_mask: torch.Tensor  # (batch, seq_len)
    # 2D data
    atom_features: torch.Tensor  # (total_atoms, atom_feat_dim)
    edge_index: torch.Tensor  # (2, total_edges)
    edge_features: torch.Tensor  # (total_edges, edge_feat_dim)
    batch_idx: torch.Tensor  # (total_atoms,) - maps atoms to molecules
    # 3D data
    conformer_coords: torch.Tensor  # (batch, num_conf, max_atoms, 3)
    conformer_mask: torch.Tensor  # (batch, num_conf, max_atoms)
    conformer_weights: torch.Tensor  # (batch, num_conf) - Boltzmann weights
    # Context (optional)
    context: Optional[Dict[str, torch.Tensor]] = None
    # Labels (optional)
    labels: Optional[torch.Tensor] = None


def validate_smiles(smiles: str, preprocessor: MoleculePreprocessor) -> bool:
    """Check if a SMILES string can be processed"""
    try:
        features = preprocessor.process_molecule(smiles)
        return (
            features is not None
            and features.atom_features is not None
            and features.token_ids is not None
            and features.edge_index is not None
        )
    except (ValueError, ImportError, Exception):
        return False


def filter_valid_molecules(
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
    contexts: Optional[List[Dict[str, Any]]] = None,
    preprocessor: Optional[MoleculePreprocessor] = None,
    verbose: bool = True,
) -> Tuple[List[str], Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """Pre-filter molecules to only include valid ones"""
    if preprocessor is None:
        preprocessor = MoleculePreprocessor()

    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        if validate_smiles(smiles, preprocessor):
            valid_indices.append(i)

    if verbose:
        num_invalid = len(smiles_list) - len(valid_indices)
        if num_invalid > 0:
            print(f"Filtered out {num_invalid}/{len(smiles_list)} invalid molecules")

    valid_smiles = [smiles_list[i] for i in valid_indices]
    valid_labels = labels[valid_indices] if labels is not None else None
    valid_contexts = [contexts[i] for i in valid_indices] if contexts is not None else None

    return valid_smiles, valid_labels, valid_contexts


class MoleculeDataset(Dataset):
    """Dataset for molecular data with all modalities"""

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
        preprocessor: Optional[MoleculePreprocessor] = None,
        conformer_generator: Optional[ConformerGenerator] = None,
        cache_dir: Optional[str] = None,
        num_conformers: int = 5,
        use_cache: bool = True,
        pre_filter: bool = True,
    ):
        self.preprocessor = preprocessor or MoleculePreprocessor()
        self.conformer_generator = conformer_generator or ConformerGenerator(
            num_conformers=num_conformers
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.num_conformers = num_conformers

        # Pre-filter invalid molecules
        if pre_filter:
            self.smiles_list, self.labels, self.contexts = filter_valid_molecules(
                smiles_list, labels, contexts, self.preprocessor
            )
        else:
            self.smiles_list = smiles_list
            self.labels = labels
            self.contexts = contexts

        if self.cache_dir and use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.smiles_list)

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for a molecule"""
        return self.cache_dir / f"mol_{idx}.pkl"

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load processed molecule from cache"""
        if not self.use_cache or not self.cache_dir:
            return None
        cache_path = self._get_cache_path(idx)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, idx: int, data: Dict[str, Any]) -> None:
        """Save processed molecule to cache"""
        if not self.use_cache or not self.cache_dir:
            return
        cache_path = self._get_cache_path(idx)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Try loading from cache
        cached = self._load_from_cache(idx)
        if cached is not None:
            if self.labels is not None:
                cached["labels"] = self.labels[idx]
            return cached

        smiles = self.smiles_list[idx]
        context = self.contexts[idx] if self.contexts else None

        # Process molecule - should always succeed since we pre-filtered
        features = self.preprocessor.process_molecule(smiles, context)

        if features is None or features.atom_features is None:
            raise ValueError(f"Failed to process molecule at index {idx}: {smiles}")

        # Generate conformers
        conformers, energies = self.conformer_generator.generate_conformers(smiles)

        # If conformer generation fails, create a single conformer from 2D coords
        if conformers is None or len(conformers) == 0:
            num_atoms = features.atom_features.shape[0]
            conformers = [np.zeros((num_atoms, 3), dtype=np.float32)]
            energies = np.array([0.0], dtype=np.float32)

        # Prepare output with actual data
        data = {
            "smiles": smiles,
            "token_ids": features.token_ids,
            "atom_features": features.atom_features,
            "edge_index": features.edge_index,
            "edge_features": features.bond_features,
            "conformers": conformers,
            "conformer_energies": energies,
            "context": context,
        }

        # Cache processed data
        self._save_to_cache(idx, data)

        # Add labels
        if self.labels is not None:
            data["labels"] = self.labels[idx]

        return data


class PretrainingDataset(Dataset):
    """Dataset for contrastive pre-training"""

    def __init__(
        self,
        smiles_list: List[str],
        preprocessor: Optional[MoleculePreprocessor] = None,
        conformer_generator: Optional[ConformerGenerator] = None,
        cache_dir: Optional[str] = None,
        augment: bool = True,
        pre_filter: bool = True,
    ):
        self.preprocessor = preprocessor or MoleculePreprocessor()
        self.conformer_generator = conformer_generator or ConformerGenerator()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

        # Pre-filter invalid molecules
        if pre_filter:
            self.smiles_list, _, _ = filter_valid_molecules(
                smiles_list, None, None, self.preprocessor
            )
        else:
            self.smiles_list = smiles_list

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        smiles = self.smiles_list[idx]

        # Process molecule - should always succeed since we pre-filtered
        features = self.preprocessor.process_molecule(smiles)

        if features is None or features.atom_features is None:
            raise ValueError(f"Failed to process molecule at index {idx}: {smiles}")

        # Generate conformers
        conformers, energies = self.conformer_generator.generate_conformers(smiles)

        # If conformer generation fails, create a single conformer from 2D coords
        if conformers is None or len(conformers) == 0:
            num_atoms = features.atom_features.shape[0]
            conformers = [np.zeros((num_atoms, 3), dtype=np.float32)]
            energies = np.array([0.0], dtype=np.float32)

        # Prepare for contrastive learning with actual data
        data = {
            "smiles": smiles,
            "token_ids": features.token_ids,
            "atom_features": features.atom_features,
            "edge_index": features.edge_index,
            "edge_features": features.bond_features,
            "conformers": conformers,
            "conformer_energies": energies,
        }

        # Atom masking for self-supervised learning
        if self.augment:
            mask_ratio = 0.15
            num_atoms = features.atom_features.shape[0]
            mask = np.random.random(num_atoms) < mask_ratio
            data["atom_mask"] = mask

        return data


def collate_molecules(batch: List[Dict[str, Any]]) -> MoleculeData:
    """Collate function for batching molecular data"""
    batch_size = len(batch)

    # 1D: Token IDs
    token_ids = torch.tensor(
        np.stack([b["token_ids"] for b in batch]), dtype=torch.long
    )
    token_mask = (token_ids != 0).float()

    # 2D: Graph data (need to handle variable sizes)
    atom_features_list = []
    edge_index_list = []
    edge_features_list = []
    batch_idx_list = []
    atom_offset = 0

    for i, b in enumerate(batch):
        atom_feat = torch.tensor(b["atom_features"], dtype=torch.float32)
        atom_features_list.append(atom_feat)

        num_atoms = atom_feat.shape[0]
        batch_idx_list.extend([i] * num_atoms)

        edge_idx = torch.tensor(b["edge_index"], dtype=torch.long)
        if edge_idx.shape[1] > 0:
            edge_idx = edge_idx + atom_offset
        edge_index_list.append(edge_idx)

        edge_feat = torch.tensor(b["edge_features"], dtype=torch.float32)
        edge_features_list.append(edge_feat)

        atom_offset += num_atoms

    atom_features = torch.cat(atom_features_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
    edge_features = torch.cat(edge_features_list, dim=0) if edge_features_list else torch.zeros(0, 11)
    batch_idx = torch.tensor(batch_idx_list, dtype=torch.long)

    # 3D: Conformers (need padding)
    max_atoms = max(b["conformers"][0].shape[0] for b in batch)
    num_conformers = max(len(b["conformers"]) for b in batch)

    conformer_coords = torch.zeros(batch_size, num_conformers, max_atoms, 3)
    conformer_mask = torch.zeros(batch_size, num_conformers, max_atoms)
    conformer_weights = torch.zeros(batch_size, num_conformers)

    for i, b in enumerate(batch):
        confs = b["conformers"]
        energies = b["conformer_energies"]

        # Compute Boltzmann weights
        weights = compute_boltzmann_weights(energies)

        for j, conf in enumerate(confs):
            num_atoms = conf.shape[0]
            conformer_coords[i, j, :num_atoms, :] = torch.tensor(conf)
            conformer_mask[i, j, :num_atoms] = 1.0
            conformer_weights[i, j] = float(weights[j]) if j < len(weights) else 0.0

    # Normalize weights
    conformer_weights = conformer_weights / (conformer_weights.sum(dim=1, keepdim=True) + 1e-8)

    # Context (if available)
    context = None
    if batch[0].get("context") is not None:
        context = {}
        # TODO: Process context features

    # Labels (if available)
    labels = None
    if "labels" in batch[0]:
        labels = torch.tensor(
            np.stack([b["labels"] for b in batch]), dtype=torch.float32
        )

    return MoleculeData(
        token_ids=token_ids,
        token_mask=token_mask,
        atom_features=atom_features,
        edge_index=edge_index,
        edge_features=edge_features,
        batch_idx=batch_idx,
        conformer_coords=conformer_coords,
        conformer_mask=conformer_mask,
        conformer_weights=conformer_weights,
        context=context,
        labels=labels,
    )
