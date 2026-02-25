"""Plotting utilities for MolFM-Lite results visualization"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#C73E1D',
    'muted': '#6C757D',
    '1d': '#E63946',
    '2d': '#457B9D',
    '3d': '#2A9D8F',
}


def save_plot(fig, filename: str, plots_dir: str = "plots", dpi: int = 300):
    """Save plot to file"""
    # Handle case where filename already contains directory
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = Path(plots_dir) / filepath.name

    # Create parent directory
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")
    return str(filepath)


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss and metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1 = axes[0]
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], color=COLORS['primary'],
                linewidth=2, label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], color=COLORS['secondary'],
                linewidth=2, linestyle='--', label='Validation Loss')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Metrics curve (if available)
    ax2 = axes[1]
    if 'metrics' in history and history['metrics']:
        metrics_data = history['metrics']
        if isinstance(metrics_data[0], dict):
            # Extract specific metrics
            for metric_name in ['contrastive_loss', 'loss_1d_2d', 'loss_2d_3d']:
                values = [m.get(metric_name, 0) for m in metrics_data if metric_name in m]
                if values:
                    ax2.plot(range(1, len(values) + 1), values,
                            linewidth=2, label=metric_name.replace('_', ' ').title())

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Training Metrics', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_benchmark_results(
    results: Dict[str, Dict[str, float]],
    metric: str = "auc_mean",
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot benchmark results comparison"""
    datasets = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.25

    # MolFM-Lite results
    molfm_values = [results[d]['metrics'].get(metric, 0) for d in datasets]
    molfm_stds = [results[d].get('metrics_std', {}).get(metric, 0) for d in datasets]

    bars1 = ax.bar(x, molfm_values, width, yerr=molfm_stds,
                   label='MolFM-Lite', color=COLORS['primary'],
                   capsize=5, error_kw={'linewidth': 2})

    # Baselines
    if baselines:
        offset = 1
        colors = [COLORS['secondary'], COLORS['tertiary'], COLORS['muted']]
        for i, (baseline_name, baseline_results) in enumerate(baselines.items()):
            baseline_values = [baseline_results.get(d, {}).get(metric, 0) for d in datasets]
            ax.bar(x + width * offset, baseline_values, width,
                   label=baseline_name, color=colors[i % len(colors)], alpha=0.7)
            offset += 1

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('MoleculeNet Benchmark Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, molfm_values):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    metric: str = "auc_mean",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ablation study results"""
    variants = list(ablation_results.keys())
    values = [ablation_results[v]['metrics'].get(metric, 0) for v in variants]
    stds = [ablation_results[v].get('metrics_std', {}).get(metric, 0) for v in variants]

    # Sort by value for better visualization
    sorted_idx = np.argsort(values)[::-1]
    variants = [variants[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['primary'] if v == 'full_model' else COLORS['muted'] for v in variants]

    bars = ax.barh(variants, values, xerr=stds, color=colors,
                   capsize=5, error_kw={'linewidth': 2})

    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Model Variant', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Format variant names
    ax.set_yticklabels([v.replace('_', ' ').title() for v in variants])

    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}',
                   xy=(val, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=10, fontweight='bold')

    # Add baseline line
    if 'full_model' in ablation_results:
        baseline = ablation_results['full_model']['metrics'].get(metric, 0)
        ax.axvline(x=baseline, color=COLORS['primary'], linestyle='--',
                   linewidth=2, alpha=0.7, label='Full Model')

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_modality_attribution(
    attributions: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot modality contribution analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    modality_names = ['1D (SELFIES)', '2D (Graph)', '3D (Conformer)']
    colors = [COLORS['1d'], COLORS['2d'], COLORS['3d']]

    # Mean attribution pie chart
    ax1 = axes[0]
    mean_attr = attributions.mean(axis=0)
    wedges, texts, autotexts = ax1.pie(
        mean_attr, labels=modality_names, autopct='%1.1f%%',
        colors=colors, explode=(0.02, 0.02, 0.02),
        shadow=True, startangle=90
    )
    ax1.set_title('Mean Modality Contribution', fontsize=14, fontweight='bold')

    # Attribution distribution
    ax2 = axes[1]
    positions = np.arange(len(modality_names))

    bp = ax2.boxplot([attributions[:, i] for i in range(3)],
                     positions=positions, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticklabels(modality_names)
    ax2.set_ylabel('Attribution Weight', fontsize=12)
    ax2.set_title('Attribution Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Modality Attribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot uncertainty calibration diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by uncertainty
    sorted_idx = np.argsort(uncertainties.flatten())
    errors = np.abs(y_true.flatten() - y_pred.flatten())

    errors_sorted = errors[sorted_idx]
    unc_sorted = uncertainties.flatten()[sorted_idx]

    # Bin errors and uncertainties
    bin_size = len(sorted_idx) // num_bins
    bin_errors = []
    bin_uncertainties = []

    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size if i < num_bins - 1 else len(sorted_idx)
        bin_errors.append(errors_sorted[start:end].mean())
        bin_uncertainties.append(unc_sorted[start:end].mean())

    # Calibration plot
    ax1 = axes[0]
    ax1.scatter(bin_uncertainties, bin_errors, s=100, c=COLORS['primary'],
               edgecolors='white', linewidth=2, zorder=3)

    max_val = max(max(bin_uncertainties), max(bin_errors))
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=2,
            label='Perfect Calibration', alpha=0.7)

    ax1.set_xlabel('Mean Predicted Uncertainty', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error', fontsize=12)
    ax1.set_title('Uncertainty Calibration', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Error vs Uncertainty scatter
    ax2 = axes[1]
    scatter = ax2.scatter(uncertainties.flatten(), errors,
                         c=COLORS['primary'], alpha=0.3, s=20)
    ax2.set_xlabel('Predicted Uncertainty', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error vs Uncertainty', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(uncertainties.flatten(), errors)[0, 1]
    ax2.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95),
                xycoords='axes fraction', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Model Uncertainty Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_molecule_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2D visualization of molecule embeddings"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(10, 8))

    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title_method = "t-SNE"
    else:
        reducer = PCA(n_components=2)
        title_method = "PCA"

    coords = reducer.fit_transform(embeddings)

    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                           cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Property Value')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=COLORS['primary'],
                  alpha=0.6, s=30)

    ax.set_xlabel(f'{title_method} 1', fontsize=12)
    ax.set_ylabel(f'{title_method} 2', fontsize=12)
    ax.set_title(f'Molecule Embedding Space ({title_method})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_conformer_attention(
    attention_weights: np.ndarray,
    energies: np.ndarray,
    molecule_idx: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot conformer attention weights vs energies"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    num_conformers = len(attention_weights[molecule_idx])
    x = np.arange(num_conformers)

    # Attention weights bar plot
    ax1 = axes[0]
    bars = ax1.bar(x, attention_weights[molecule_idx], color=COLORS['primary'],
                   edgecolor='white', linewidth=2)
    ax1.set_xlabel('Conformer Index', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('Conformer Attention Weights', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')

    # Attention vs Energy scatter
    ax2 = axes[1]

    # Normalize energies (relative to minimum)
    rel_energies = energies[molecule_idx] - energies[molecule_idx].min()

    scatter = ax2.scatter(rel_energies, attention_weights[molecule_idx],
                         s=150, c=COLORS['secondary'], edgecolors='white',
                         linewidth=2, zorder=3)

    # Add conformer labels
    for i, (e, a) in enumerate(zip(rel_energies, attention_weights[molecule_idx])):
        ax2.annotate(f'C{i}', (e, a), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax2.set_xlabel('Relative Energy (kcal/mol)', fontsize=12)
    ax2.set_ylabel('Attention Weight', fontsize=12)
    ax2.set_title('Attention vs Conformer Energy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Conformer Ensemble Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def create_results_summary_figure(
    benchmark_results: Dict,
    ablation_results: Dict,
    training_history: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive results summary figure"""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Training curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'loss' in training_history:
        epochs = range(1, len(training_history['loss']) + 1)
        ax1.plot(epochs, training_history['loss'], color=COLORS['primary'],
                linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pre-training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Benchmark results (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if benchmark_results:
        datasets = list(benchmark_results.keys())
        values = [benchmark_results[d]['metrics'].get('auc_mean',
                  benchmark_results[d]['metrics'].get('rmse_mean', 0)) for d in datasets]
        bars = ax2.bar(datasets, values, color=COLORS['primary'])
        ax2.set_ylabel('Score')
        ax2.set_title('Benchmark Performance', fontweight='bold')
        ax2.set_xticklabels([d.upper() for d in datasets], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

    # Ablation study (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if ablation_results:
        variants = list(ablation_results.keys())
        values = [ablation_results[v]['metrics'].get('auc_mean', 0) for v in variants]
        colors = [COLORS['primary'] if v == 'full_model' else COLORS['muted'] for v in variants]
        ax3.barh(variants, values, color=colors)
        ax3.set_xlabel('AUC')
        ax3.set_title('Ablation Study', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

    # Modality contribution (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    modality_names = ['1D', '2D', '3D']
    mock_contributions = [0.25, 0.40, 0.35]  # Placeholder
    colors = [COLORS['1d'], COLORS['2d'], COLORS['3d']]
    ax4.pie(mock_contributions, labels=modality_names, autopct='%1.1f%%',
           colors=colors, shadow=True, startangle=90)
    ax4.set_title('Modality Contributions', fontweight='bold')

    fig.suptitle('MolFM-Lite Results Summary', fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        save_plot(fig, save_path)

    return fig
