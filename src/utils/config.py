"""Configuration management for MolFM-Lite"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    encoder_1d: Dict[str, Any] = field(default_factory=dict)
    encoder_2d: Dict[str, Any] = field(default_factory=dict)
    encoder_3d: Dict[str, Any] = field(default_factory=dict)
    conformer_attention: Dict[str, Any] = field(default_factory=dict)
    fusion: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    head: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration"""
    zinc250k: Dict[str, Any] = field(default_factory=dict)
    chembl: Dict[str, Any] = field(default_factory=dict)
    conformers: Dict[str, Any] = field(default_factory=dict)
    cache_dir: str = "data/cache"


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    contrastive: Dict[str, Any] = field(default_factory=dict)
    objectives: Dict[str, float] = field(default_factory=dict)


@dataclass
class AWSConfig:
    """AWS configuration"""
    region: str = "us-east-1"
    s3_bucket: str = "molfm-lite-data"
    sagemaker: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration"""
    project: Dict[str, Any] = field(default_factory=dict)
    aws: AWSConfig = field(default_factory=AWSConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretraining: TrainingConfig = field(default_factory=TrainingConfig)
    finetuning: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Load configuration from YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse nested configs
    aws_config = AWSConfig(**config_dict.get("aws", {}))

    data_dict = config_dict.get("data", {})
    data_config = DataConfig(
        zinc250k=data_dict.get("zinc250k", {}),
        chembl=data_dict.get("chembl", {}),
        conformers=data_dict.get("conformers", {}),
        cache_dir=data_dict.get("cache_dir", "data/cache"),
    )

    model_dict = config_dict.get("model", {})
    model_config = ModelConfig(
        encoder_1d=model_dict.get("encoder_1d", {}),
        encoder_2d=model_dict.get("encoder_2d", {}),
        encoder_3d=model_dict.get("encoder_3d", {}),
        conformer_attention=model_dict.get("conformer_attention", {}),
        fusion=model_dict.get("fusion", {}),
        context=model_dict.get("context", {}),
        head=model_dict.get("head", {}),
    )

    pretrain_dict = config_dict.get("pretraining", {})
    pretraining_config = TrainingConfig(
        batch_size=pretrain_dict.get("batch_size", 128),
        num_epochs=pretrain_dict.get("num_epochs", 50),
        learning_rate=pretrain_dict.get("learning_rate", 1e-4),
        weight_decay=pretrain_dict.get("weight_decay", 1e-5),
        warmup_steps=pretrain_dict.get("warmup_steps", 1000),
        contrastive=pretrain_dict.get("contrastive", {}),
        objectives=pretrain_dict.get("objectives", {}),
    )

    config = Config(
        project=config_dict.get("project", {}),
        aws=aws_config,
        data=data_config,
        model=model_config,
        pretraining=pretraining_config,
        finetuning=config_dict.get("finetuning", {}),
        evaluation=config_dict.get("evaluation", {}),
        logging=config_dict.get("logging", {}),
    )

    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
