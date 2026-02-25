# arXiv Submission Guide for MolFM-Lite

## Paper Information

**Title:** MolFM-Lite: Efficient Context-Aware Multi-Modal Molecular Foundation Model for Drug Discovery

**Authors:**
1. Syed Omer Shah (Primary/Corresponding Author)
   - Email: syedomer@buffalo.edu
   - Affiliation: University at Buffalo

2. Mohammed Maqsood Ahmed
   - Email: m58@buffalo.edu
   - Affiliation: University at Buffalo

3. Danish Mohiuddin Mohammed
   - Email: mohammed.dan@northeastern.edu
   - Affiliation: Northeastern University

## Files for Submission

### Required Files
- `arxiv_submission.tex` - Main LaTeX source file
- `references.bib` - BibTeX bibliography file

### Optional Supporting Files
- `paper_draft.md` - Markdown version for reference
- Architecture diagrams (when available)

## Compilation Instructions

```bash
# Compile the paper
pdflatex arxiv_submission.tex
bibtex arxiv_submission
pdflatex arxiv_submission.tex
pdflatex arxiv_submission.tex
```

Or use latexmk:
```bash
latexmk -pdf arxiv_submission.tex
```

## arXiv Submission Steps

1. **Create arXiv Account** (if you don't have one)
   - Go to https://arxiv.org/user/register

2. **Prepare Submission Package**
   - Ensure all `.tex` and `.bib` files are included
   - Add any figures in PDF/PNG/JPG format
   - Create a `.zip` file with all source files

3. **Submit to arXiv**
   - Go to https://arxiv.org/submit
   - Select category: **cs.LG** (Machine Learning) or **q-bio.QM** (Quantitative Methods)
   - Cross-list to: **cs.AI**, **physics.chem-ph**
   - Upload source files
   - Fill in metadata (title, authors, abstract)
   - Submit

4. **Recommended Categories**
   - Primary: `cs.LG` (Machine Learning)
   - Secondary: `q-bio.QM` (Quantitative Methods in Biology)
   - Cross-list: `cs.AI` (Artificial Intelligence)

## Abstract (for arXiv submission form)

Molecular property prediction is fundamental to drug discovery, yet current approaches typically rely on single molecular representations and ignore experimental context. We present MolFM-Lite, a multi-modal molecular foundation model that jointly learns from SELFIES sequences (1D), molecular graphs (2D), and conformer ensembles (3D), while conditioning predictions on experimental context such as assay type and cell line. Our key contributions include: (1) a novel conformer ensemble attention mechanism that captures molecular flexibility through Boltzmann-weighted aggregation; (2) context conditioning via Feature-wise Linear Modulation (FiLM) to account for experimental variability; (3) cross-modal contrastive pre-training that aligns representations across modalities; and (4) comprehensive ablation studies demonstrating the value of each component. Training from scratch with limited compute (under $150 AWS credits), MolFM-Lite achieves state-of-the-art performance on MoleculeNet benchmarks: 0.956 AUC on BBBP (+6.9%), 0.902 AUC on BACE (+2.7%), 0.848 AUC on Tox21 (+6.7%), and 0.570 RMSE on Lipophilicity (-9.7%). Our results demonstrate that thoughtful multi-modal fusion and architectural design can outperform computationally expensive approaches, democratizing advanced molecular modeling for academic laboratories. We release all code and trained models to facilitate reproducible research.

## Keywords

molecular property prediction, multi-modal learning, foundation models, drug discovery, contrastive learning, graph neural networks, conformer ensemble, deep learning

## Comments (for arXiv)

15 pages, 5 figures, 6 tables. Code available at https://github.com/Syedomershah99/molfm-lite

## License

We recommend using the arXiv.org perpetual, non-exclusive license (default).

## Notes

- The paper uses standard LaTeX packages available on arXiv
- No custom style files required
- Figures should be in vector format (PDF) when possible
- Ensure all references are properly formatted

## Checklist Before Submission

- [ ] All author names and emails correct
- [ ] Affiliations accurate
- [ ] Abstract within 1920 character limit for arXiv
- [ ] All figures included and referenced
- [ ] Bibliography compiles without errors
- [ ] No compilation warnings
- [ ] GitHub repository link is correct
- [ ] Code is publicly available
