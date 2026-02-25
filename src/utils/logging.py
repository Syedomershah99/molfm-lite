"""Logging utilities for MolFM-Lite"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "molfm",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Set up logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Logger for training metrics"""

    def __init__(self, use_wandb: bool = False, project_name: str = "molfm-lite"):
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.metrics_history = []

        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling wandb logging")
                self.use_wandb = False

    def init_run(self, run_name: str, config: dict) -> None:
        """Initialize a new run"""
        if self.use_wandb:
            self.wandb.init(project=self.project_name, name=run_name, config=config)

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics"""
        self.metrics_history.append({"step": step, **metrics})

        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish logging"""
        if self.use_wandb:
            self.wandb.finish()

    def get_history(self) -> list:
        """Get metrics history"""
        return self.metrics_history
