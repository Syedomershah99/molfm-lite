"""Evaluation metrics for MolFM-Lite"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: (n_samples,) or (n_samples, n_tasks) ground truth labels
        y_pred: (n_samples,) or (n_samples, n_tasks) predicted probabilities
        y_prob: same as y_pred if provided
        threshold: classification threshold

    Returns:
        Dictionary of metrics
    """
    if y_prob is None:
        y_prob = y_pred

    metrics = {}

    # Handle multi-task
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_prob = y_prob.reshape(-1, 1)

    num_tasks = y_true.shape[1]
    aucs = []
    aps = []

    for task_idx in range(num_tasks):
        task_true = y_true[:, task_idx]
        task_prob = y_prob[:, task_idx]

        # Filter out missing labels (-1)
        valid_mask = task_true != -1
        task_true = task_true[valid_mask]
        task_prob = task_prob[valid_mask]

        if len(task_true) == 0 or len(np.unique(task_true)) < 2:
            continue

        try:
            auc = roc_auc_score(task_true, task_prob)
            aucs.append(auc)
            metrics[f"auc_task_{task_idx}"] = auc
        except Exception:
            pass

        try:
            ap = average_precision_score(task_true, task_prob)
            aps.append(ap)
            metrics[f"ap_task_{task_idx}"] = ap
        except Exception:
            pass

    if aucs:
        metrics["auc_mean"] = np.mean(aucs)
    if aps:
        metrics["ap_mean"] = np.mean(aps)

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: (n_samples,) or (n_samples, n_tasks) ground truth values
        y_pred: (n_samples,) or (n_samples, n_tasks) predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Handle multi-task
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    num_tasks = y_true.shape[1]
    rmses = []
    maes = []
    r2s = []

    for task_idx in range(num_tasks):
        task_true = y_true[:, task_idx]
        task_pred = y_pred[:, task_idx]

        # Filter out missing labels
        valid_mask = ~np.isnan(task_true) & (task_true != -1)
        task_true = task_true[valid_mask]
        task_pred = task_pred[valid_mask]

        if len(task_true) == 0:
            continue

        rmse = np.sqrt(mean_squared_error(task_true, task_pred))
        mae = mean_absolute_error(task_true, task_pred)
        r2 = r2_score(task_true, task_pred)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        metrics[f"rmse_task_{task_idx}"] = rmse
        metrics[f"mae_task_{task_idx}"] = mae
        metrics[f"r2_task_{task_idx}"] = r2

    if rmses:
        metrics["rmse_mean"] = np.mean(rmses)
        metrics["mae_mean"] = np.mean(maes)
        metrics["r2_mean"] = np.mean(r2s)

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification",
) -> Dict[str, float]:
    """
    Compute metrics based on task type.

    Args:
        y_true: ground truth
        y_pred: predictions
        task_type: "classification" or "regression"

    Returns:
        Dictionary of metrics
    """
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)
    else:
        return compute_regression_metrics(y_true, y_pred)


def compute_uncertainty(
    predictions: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty from multiple forward passes (MC Dropout).

    Args:
        predictions: List of (n_samples, n_tasks) predictions from multiple passes

    Returns:
        mean: (n_samples, n_tasks) mean prediction
        epistemic: (n_samples, n_tasks) epistemic uncertainty (model uncertainty)
        aleatoric: (n_samples, n_tasks) placeholder for aleatoric uncertainty
    """
    preds = np.stack(predictions, axis=0)  # (num_passes, n_samples, n_tasks)

    mean = preds.mean(axis=0)
    epistemic = preds.std(axis=0)

    # Aleatoric uncertainty would require heteroscedastic model
    aleatoric = np.zeros_like(epistemic)

    return mean, epistemic, aleatoric


def compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics for uncertainty estimates.

    Args:
        y_true: ground truth
        y_pred: predictions
        uncertainties: uncertainty estimates
        num_bins: number of calibration bins

    Returns:
        Dictionary with calibration metrics
    """
    # Sort by uncertainty
    sorted_idx = np.argsort(uncertainties.flatten())
    y_true_sorted = y_true.flatten()[sorted_idx]
    y_pred_sorted = y_pred.flatten()[sorted_idx]

    # Compute error for each bin
    bin_size = len(sorted_idx) // num_bins
    bin_errors = []
    bin_uncertainties = []

    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size if i < num_bins - 1 else len(sorted_idx)

        bin_true = y_true_sorted[start:end]
        bin_pred = y_pred_sorted[start:end]
        bin_unc = uncertainties.flatten()[sorted_idx[start:end]]

        error = np.abs(bin_true - bin_pred).mean()
        unc = bin_unc.mean()

        bin_errors.append(error)
        bin_uncertainties.append(unc)

    # Compute calibration error (difference between error and uncertainty)
    calibration_error = np.mean(np.abs(np.array(bin_errors) - np.array(bin_uncertainties)))

    return {
        "calibration_error": calibration_error,
        "bin_errors": bin_errors,
        "bin_uncertainties": bin_uncertainties,
    }
