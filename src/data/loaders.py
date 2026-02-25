"""Data loaders and data downloading utilities for MolFM-Lite"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader, random_split

from .dataset import MoleculeDataset, PretrainingDataset, collate_molecules
from .preprocessing import MoleculePreprocessor, ConformerGenerator


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> bool:
    """Download a file from URL"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_zinc250k(data_dir: str = "data/raw") -> str:
    """Download ZINC250K dataset"""
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    save_path = Path(data_dir) / "zinc250k.csv"

    if save_path.exists():
        print(f"ZINC250K already exists: {save_path}")
        return str(save_path)

    if download_file(url, str(save_path)):
        return str(save_path)
    return ""


def download_moleculenet(
    dataset_name: str, data_dir: str = "data/raw"
) -> Tuple[str, Dict[str, Any]]:
    """Download MoleculeNet dataset - first checks for pre-existing CSV files"""
    data_dir = Path(data_dir)

    # Check for pre-existing CSV files in various locations
    possible_paths = [
        data_dir / "moleculenet" / f"{dataset_name}.csv",
        data_dir / f"{dataset_name}.csv",
        data_dir / f"moleculenet/{dataset_name}.csv",
    ]

    for csv_path in possible_paths:
        if csv_path.exists():
            print(f"Found existing dataset: {csv_path}")
            df = pd.read_csv(csv_path)
            return str(csv_path), {
                "tasks": [col for col in df.columns if col.startswith("task_")],
                "num_tasks": len([col for col in df.columns if col.startswith("task_")]),
                "total_size": len(df),
            }

    # Try DeepChem download as fallback
    try:
        import deepchem as dc
    except ImportError:
        print(f"DeepChem not installed and no pre-existing CSV found for {dataset_name}")
        return "", {}

    moleculenet_dir = data_dir / "moleculenet"
    moleculenet_dir.mkdir(parents=True, exist_ok=True)

    dataset_loaders = {
        "bbbp": dc.molnet.load_bbbp,
        "bace": dc.molnet.load_bace_classification,
        "tox21": dc.molnet.load_tox21,
        "toxcast": dc.molnet.load_toxcast,
        "sider": dc.molnet.load_sider,
        "clintox": dc.molnet.load_clintox,
        "hiv": dc.molnet.load_hiv,
        "muv": dc.molnet.load_muv,
        "lipophilicity": dc.molnet.load_lipo,
        "freesolv": dc.molnet.load_freesolv,
        "esol": dc.molnet.load_delaney,
    }

    if dataset_name not in dataset_loaders:
        print(f"Unknown dataset: {dataset_name}")
        return "", {}

    try:
        tasks, datasets, transformers = dataset_loaders[dataset_name](
            featurizer="Raw", data_dir=str(moleculenet_dir)
        )
        train, valid, test = datasets

        # Convert to DataFrames
        def dataset_to_df(ds, split_name):
            df = pd.DataFrame(
                {"smiles": ds.ids, **{f"task_{i}": ds.y[:, i] for i in range(ds.y.shape[1])}}
            )
            df["split"] = split_name
            return df

        df_train = dataset_to_df(train, "train")
        df_valid = dataset_to_df(valid, "valid")
        df_test = dataset_to_df(test, "test")

        df = pd.concat([df_train, df_valid, df_test], ignore_index=True)

        save_path = moleculenet_dir / f"{dataset_name}.csv"
        df.to_csv(save_path, index=False)

        return str(save_path), {
            "tasks": tasks,
            "num_tasks": len(tasks),
            "train_size": len(df_train),
            "valid_size": len(df_valid),
            "test_size": len(df_test),
        }
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return "", {}


def load_zinc250k(
    file_path: str, max_samples: Optional[int] = None
) -> Tuple[List[str], Optional[np.ndarray]]:
    """Load ZINC250K dataset"""
    df = pd.read_csv(file_path)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    smiles = df["smiles"].tolist()

    # Extract properties if available
    property_cols = ["logP", "qed", "SAS"]
    available_props = [col for col in property_cols if col in df.columns]

    if available_props:
        properties = df[available_props].values.astype(np.float32)
    else:
        properties = None

    return smiles, properties


def load_moleculenet(
    file_path: str, split: Optional[str] = None
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load MoleculeNet dataset - handles various CSV formats"""
    df = pd.read_csv(file_path)

    if split:
        df = df[df["split"] == split]

    # Find SMILES column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == 'smiles':
            smiles_col = col
            break
        elif col.lower() == 'mol':  # BACE uses 'mol'
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(f"Could not find SMILES column in {file_path}. Columns: {list(df.columns)}")

    smiles = df[smiles_col].tolist()

    # Get task columns - check for task_X format first, then look for label columns
    task_cols = [col for col in df.columns if col.startswith("task_")]

    if not task_cols:
        # Try to find label columns based on dataset
        # BBBP: p_np, BACE: Class, Tox21: NR-*, SR-*, Lipo: exp
        exclude_cols = {'smiles', 'mol', 'split', 'num', 'name', 'cid', 'model', 'cmpd_chemblid', 'pic50'}
        task_cols = [col for col in df.columns
                     if col.lower() not in exclude_cols
                     and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

    if not task_cols:
        # Last resort - look for specific known task columns
        known_tasks = {
            'p_np': 'bbbp',
            'Class': 'bace',
            'exp': 'lipophilicity',
        }
        for col in df.columns:
            if col in known_tasks:
                task_cols = [col]
                break

    if not task_cols:
        raise ValueError(f"Could not find task columns in {file_path}. Columns: {list(df.columns)}")

    labels = df[task_cols].values.astype(np.float32)

    # Handle NaN labels (common in multi-task datasets)
    labels = np.nan_to_num(labels, nan=-1)  # Use -1 for missing labels

    splits = df["split"].tolist() if "split" in df.columns else ["train"] * len(df)

    return smiles, labels, splits


def create_dataloaders(
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    preprocessor: Optional[MoleculePreprocessor] = None,
    conformer_generator: Optional[ConformerGenerator] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    dataset = MoleculeDataset(
        smiles_list=smiles_list,
        labels=labels,
        preprocessor=preprocessor,
        conformer_generator=conformer_generator,
        cache_dir=cache_dir,
    )

    # Split dataset
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def create_pretraining_dataloader(
    smiles_list: List[str],
    batch_size: int = 128,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    preprocessor: Optional[MoleculePreprocessor] = None,
    conformer_generator: Optional[ConformerGenerator] = None,
) -> DataLoader:
    """Create dataloader for pre-training"""
    dataset = PretrainingDataset(
        smiles_list=smiles_list,
        preprocessor=preprocessor,
        conformer_generator=conformer_generator,
        cache_dir=cache_dir,
        augment=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_molecules,
        pin_memory=True,
        drop_last=True,
    )
