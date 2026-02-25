#!/usr/bin/env python
"""Generate publication-quality plots with actual MolFM-Lite results"""

import os
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set publication quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'molfm': '#2ecc71',      # Green
    'chemberta': '#3498db',  # Blue
    'grover': '#9b59b6',     # Purple
    'schnet': '#e74c3c',     # Red
    'baseline': '#95a5a6',   # Gray
}

# Actual results from our experiments
MOLFM_RESULTS = {
    'bbbp': {'auc': 0.9558, 'auc_std': 0.0012},
    'bace': {'auc': 0.9022, 'auc_std': 0.0055},
    'tox21': {'auc': 0.8478, 'auc_std': 0.0021},
    'lipophilicity': {'rmse': 0.5696, 'rmse_std': 0.0015, 'r2': 0.7706},
}

# Baseline results from literature
BASELINES = {
    'bbbp': {'ChemBERTa': 0.872, 'GROVER': 0.894, 'SchNet': 0.847},
    'bace': {'ChemBERTa': 0.856, 'GROVER': 0.878, 'SchNet': 0.823},
    'tox21': {'ChemBERTa': 0.782, 'GROVER': 0.795, 'SchNet': 0.756},
    'lipophilicity': {'ChemBERTa': 0.654, 'GROVER': 0.631, 'SchNet': 0.692},
}


def plot_benchmark_classification(output_dir: Path):
    """Plot classification benchmark comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    datasets = ['bbbp', 'bace', 'tox21']
    titles = ['BBBP (BBB Penetration)', 'BACE (Inhibition)', 'Tox21 (Toxicity)']

    for ax, dataset, title in zip(axes, datasets, titles):
        methods = ['ChemBERTa', 'GROVER', 'SchNet', 'MolFM-Lite']
        values = [
            BASELINES[dataset]['ChemBERTa'],
            BASELINES[dataset]['GROVER'],
            BASELINES[dataset]['SchNet'],
            MOLFM_RESULTS[dataset]['auc'],
        ]
        errors = [0, 0, 0, MOLFM_RESULTS[dataset]['auc_std']]
        colors = [COLORS['chemberta'], COLORS['grover'], COLORS['schnet'], COLORS['molfm']]

        bars = ax.bar(methods, values, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1)

        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel('ROC-AUC')
        ax.set_title(title)
        ax.set_ylim(0.7, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('MoleculeNet Classification Benchmarks', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_classification.png')
    plt.savefig(output_dir / 'benchmark_classification.pdf')
    plt.close()
    print("Saved: benchmark_classification.png/pdf")


def plot_benchmark_regression(output_dir: Path):
    """Plot regression benchmark (RMSE - lower is better)"""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['ChemBERTa', 'GROVER', 'SchNet', 'MolFM-Lite']
    rmse_values = [
        BASELINES['lipophilicity']['ChemBERTa'],
        BASELINES['lipophilicity']['GROVER'],
        BASELINES['lipophilicity']['SchNet'],
        MOLFM_RESULTS['lipophilicity']['rmse'],
    ]
    errors = [0, 0, 0, MOLFM_RESULTS['lipophilicity']['rmse_std']]
    colors = [COLORS['chemberta'], COLORS['grover'], COLORS['schnet'], COLORS['molfm']]

    bars = ax.bar(methods, rmse_values, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1)

    # Highlight best (lowest)
    best_idx = np.argmin(rmse_values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Lipophilicity Prediction')
    ax.set_ylim(0, 0.9)

    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_regression.png')
    plt.savefig(output_dir / 'benchmark_regression.pdf')
    plt.close()
    print("Saved: benchmark_regression.png/pdf")


def plot_all_benchmarks_combined(output_dir: Path):
    """Create a combined figure with all benchmarks"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    datasets = [('bbbp', 'BBBP (BBB Penetration)'),
                ('bace', 'BACE (Inhibition)'),
                ('tox21', 'Tox21 (Toxicity)')]

    # Classification plots
    for idx, (dataset, title) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        methods = ['ChemBERTa', 'GROVER', 'SchNet', 'MolFM-Lite']
        values = [
            BASELINES[dataset]['ChemBERTa'],
            BASELINES[dataset]['GROVER'],
            BASELINES[dataset]['SchNet'],
            MOLFM_RESULTS[dataset]['auc'],
        ]
        errors = [0, 0, 0, MOLFM_RESULTS[dataset]['auc_std']]
        colors = [COLORS['chemberta'], COLORS['grover'], COLORS['schnet'], COLORS['molfm']]

        bars = ax.bar(methods, values, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1)
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel('ROC-AUC')
        ax.set_title(title)
        ax.set_ylim(0.7, 1.0)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Regression plot (bottom right)
    ax = axes[1, 1]
    methods = ['ChemBERTa', 'GROVER', 'SchNet', 'MolFM-Lite']
    rmse_values = [
        BASELINES['lipophilicity']['ChemBERTa'],
        BASELINES['lipophilicity']['GROVER'],
        BASELINES['lipophilicity']['SchNet'],
        MOLFM_RESULTS['lipophilicity']['rmse'],
    ]
    errors = [0, 0, 0, MOLFM_RESULTS['lipophilicity']['rmse_std']]
    colors = [COLORS['chemberta'], COLORS['grover'], COLORS['schnet'], COLORS['molfm']]

    bars = ax.bar(methods, rmse_values, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1)
    best_idx = np.argmin(rmse_values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Lipophilicity')
    ax.set_ylim(0, 0.9)

    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('MolFM-Lite: MoleculeNet Benchmark Results', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_all.png')
    plt.savefig(output_dir / 'benchmark_all.pdf')
    plt.close()
    print("Saved: benchmark_all.png/pdf")


def plot_improvement_bars(output_dir: Path):
    """Plot improvement over baselines"""
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = ['BBBP', 'BACE', 'Tox21', 'Lipophilicity']
    dataset_keys = ['bbbp', 'bace', 'tox21', 'lipophilicity']

    # Calculate improvements over best baseline
    improvements = []
    for dk in dataset_keys:
        if dk == 'lipophilicity':
            # For RMSE, lower is better
            best_baseline = min(BASELINES[dk].values())
            our_score = MOLFM_RESULTS[dk]['rmse']
            improvement = (best_baseline - our_score) / best_baseline * 100
        else:
            # For AUC, higher is better
            best_baseline = max(BASELINES[dk].values())
            our_score = MOLFM_RESULTS[dk]['auc']
            improvement = (our_score - best_baseline) / best_baseline * 100
        improvements.append(improvement)

    colors = [COLORS['molfm'] if imp > 0 else COLORS['schnet'] for imp in improvements]
    bars = ax.bar(datasets, improvements, color=colors, edgecolor='black', linewidth=1)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Improvement over Best Baseline (%)')
    ax.set_title('MolFM-Lite: Improvement over State-of-the-Art')

    for bar, imp in zip(bars, improvements):
        y_pos = bar.get_height() + 0.3 if imp > 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_bars.png')
    plt.savefig(output_dir / 'improvement_bars.pdf')
    plt.close()
    print("Saved: improvement_bars.png/pdf")


def plot_architecture_diagram(output_dir: Path):
    """Create architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Input representations
    draw_box(0.5, 6, 2, 1.2, '1D\nSELFIES', '#3498db', 11)
    draw_box(0.5, 4, 2, 1.2, '2D\nGraph', '#2ecc71', 11)
    draw_box(0.5, 2, 2, 1.2, '3D\nConformers', '#e74c3c', 11)

    # Encoders
    draw_box(3.5, 6, 2.2, 1.2, 'Transformer\nEncoder', '#85c1e9', 10)
    draw_box(3.5, 4, 2.2, 1.2, 'GIN\nEncoder', '#82e0aa', 10)
    draw_box(3.5, 2, 2.2, 1.2, 'SchNet\nEncoder', '#f1948a', 10)

    # Conformer attention
    draw_box(3.5, 0.5, 2.2, 1, 'Conformer\nAttention', '#f5b041', 9)

    # Cross-modal fusion
    draw_box(7, 3.5, 2.5, 2.5, 'Cross-Modal\nFusion\n(Attention)', '#bb8fce', 11)

    # Context
    draw_box(7, 0.5, 2.5, 1.5, 'Context\nConditioning\n(FiLM)', '#f9e79f', 10)

    # Prediction head
    draw_box(10.5, 3, 2, 3, 'Prediction\nHead\n(MLP)', '#aed6f1', 11)

    # Output
    draw_box(10.5, 0.5, 2, 1.5, 'Output\n+ Uncertainty', '#d5dbdb', 10)

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)

    for y in [6.6, 4.6, 2.6]:
        ax.annotate('', xy=(3.5, y), xytext=(2.5, y), arrowprops=arrow_style)

    ax.annotate('', xy=(7, 5), xytext=(5.7, 6.6), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 4.5), xytext=(5.7, 4.6), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 4), xytext=(5.7, 2.6), arrowprops=arrow_style)
    ax.annotate('', xy=(4.6, 2), xytext=(4.6, 1.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 3.5), xytext=(5.7, 1), arrowprops=arrow_style)
    ax.annotate('', xy=(10.5, 4.5), xytext=(9.5, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(10.5, 3), xytext=(9.5, 1.25), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 3), xytext=(11.5, 2), arrowprops=arrow_style)

    ax.set_title('MolFM-Lite Architecture', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_diagram.png')
    plt.savefig(output_dir / 'architecture_diagram.pdf')
    plt.close()
    print("Saved: architecture_diagram.png/pdf")


def main():
    output_dir = Path('/Users/Omer/Desktop/projects/molfm-lite/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots with actual MolFM-Lite results...")
    print("=" * 50)

    plot_benchmark_classification(output_dir)
    plot_benchmark_regression(output_dir)
    plot_all_benchmarks_combined(output_dir)
    plot_improvement_bars(output_dir)
    plot_architecture_diagram(output_dir)

    print("=" * 50)
    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
