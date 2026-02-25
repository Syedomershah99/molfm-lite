# MolFM-Lite: Multi-Modal Molecular Foundation Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)

A context-aware multi-modal molecular foundation model that jointly learns from 1D (SELFIES), 2D (graphs), and 3D (conformer ensembles) representations for molecular property prediction.

## Highlights

- **Multi-modal learning**: Combines 1D sequences, 2D graphs, and 3D structures
- **Conformer ensemble attention**: Captures molecular flexibility with Boltzmann-weighted attention
- **Context conditioning**: Accounts for experimental conditions (assay type, cell line)
- **Efficient training**: Achieves strong results with ~$150 AWS credits
- **Uncertainty quantification**: MC Dropout for reliable confidence estimation

## Architecture

```
┌─────────────┬─────────────┬─────────────────┐
│  1D Encoder │  2D Encoder │   3D Encoder    │
│ (Transformer)│    (GIN)    │  (SchNet-Lite)  │
└──────┬──────┴──────┬──────┴────────┬────────┘
       │             │               │
       │             │    ┌──────────┴──────────┐
       │             │    │ Conformer Ensemble  │
       │             │    │     Attention       │
       │             │    └──────────┬──────────┘
       │             │               │
       └─────────────┴───────────────┘
                     │
          ┌──────────┴──────────┐
          │  Cross-Modal Fusion │
          │   (Cross-Attention) │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │ Context Conditioning│
          │       (FiLM)        │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │   Prediction Head   │
          │  (with Uncertainty) │
          └─────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/Syedomershah99/molfm-lite.git
cd molfm-lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install RDKit (if not using conda)
pip install rdkit

# Optional: Install PyTorch Geometric
pip install torch-geometric
```

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py --datasets zinc250k bbbp bace tox21 lipophilicity
```

### 2. Pre-training

```bash
# Local training (CPU/small GPU)
python scripts/pretrain.py \
    --max-samples 50000 \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-4

# AWS SageMaker
python scripts/train_sagemaker.py \
    --mode pretrain \
    --bucket your-bucket \
    --spot
```

### 3. Fine-tuning & Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/pretrain/pretrained_model.pt \
    --datasets bbbp bace tox21 lipophilicity \
    --num-seeds 3
```

## Project Structure

```
molfm-lite/
├── configs/
│   └── config.yaml          # Configuration file
├── src/
│   ├── models/
│   │   ├── encoders.py      # 1D, 2D, 3D encoders
│   │   ├── fusion.py        # Cross-modal fusion, context conditioning
│   │   └── molfm.py         # Main model
│   ├── data/
│   │   ├── preprocessing.py # Molecular feature extraction
│   │   ├── dataset.py       # PyTorch datasets
│   │   └── loaders.py       # Data downloading and loading
│   ├── training/
│   │   ├── trainer.py       # Training loops
│   │   └── losses.py        # Loss functions
│   ├── evaluation/
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── benchmarks.py    # MoleculeNet benchmarks
│   └── utils/
│       ├── config.py        # Configuration management
│       ├── aws.py           # AWS utilities
│       └── logging.py       # Logging utilities
├── scripts/
│   ├── pretrain.py          # Local pre-training script
│   ├── evaluate.py          # Evaluation script
│   ├── download_data.py     # Data download script
│   └── train_sagemaker.py   # SageMaker training launcher
├── paper/
│   └── paper_draft.md       # Research paper draft
├── blog/
│   └── aws_blog_post.md     # AWS blog post
└── requirements.txt
```

## Results

### MoleculeNet Benchmarks (State-of-the-Art)

| Dataset | Task | Metric | MolFM-Lite | Previous SOTA | Improvement |
|---------|------|--------|------------|---------------|-------------|
| BBBP | Blood-Brain Barrier | AUC | **0.956 ± 0.001** | 0.894 | +6.9% |
| BACE | Beta-secretase Inhibition | AUC | **0.902 ± 0.006** | 0.878 | +2.7% |
| Tox21 | Toxicity (12 tasks) | AUC | **0.848 ± 0.002** | 0.795 | +6.7% |
| Lipophilicity | Solubility | RMSE | **0.570 ± 0.002** | 0.631 | -9.7% |

*Note: For RMSE, lower is better. For AUC, higher is better.*

### Ablation Studies

| Model Variant | BBBP AUC | Impact |
|--------------|----------|--------|
| Full model | 0.956 | Baseline |
| Single conformer | 0.938 | -1.9% |
| 2D only | 0.884 | -7.5% |
| 1D only | 0.872 | -8.8% |
| 3D only | 0.847 | -11.4% |

**Key insight**: Multi-modal fusion provides 7-11% improvement over any single modality.

## AWS Cost Breakdown

| Phase | GPU Hours | Cost |
|-------|-----------|------|
| Fine-tuning (4 datasets x 3 seeds) | ~6 | ~$5 |
| Debugging and iterations | ~15 | ~$11 |
| **Total** | **~21** | **~$16** |

**Under $20 for state-of-the-art results!** (Training from scratch, no pretraining required)

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  encoder_1d:
    hidden_dim: 256
    num_layers: 4
  encoder_2d:
    hidden_dim: 256
    num_layers: 4
  encoder_3d:
    hidden_dim: 128
    num_interactions: 3

pretraining:
  batch_size: 128
  num_epochs: 50
  learning_rate: 1e-4
```

## Citation

```bibtex
@article{shah2026molfm,
  title={MolFM-Lite: A Multi-Modal Molecular Foundation Model with Context-Aware Predictions},
  author={Shah, Syed Omer},
  journal={GitHub},
  year={2026},
  url={https://github.com/Syedomershah99/molfm-lite}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- AWS for compute resources
- RDKit for cheminformatics
- PyTorch and PyTorch Geometric for deep learning
- DeepChem for MoleculeNet benchmark data

## Contact

- **Author**: Syed Omer Shah
- **Email**: syedomer@buffalo.edu
- **GitHub**: [@Syedomershah99](https://github.com/Syedomershah99)
- **Hugging Face**: [OmerShah](https://huggingface.co/OmerShah)
- **Affiliation**: University at Buffalo
