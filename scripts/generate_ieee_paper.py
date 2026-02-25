#!/usr/bin/env python3
"""Generate IEEE format research paper PDF"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from pathlib import Path
import os

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PLOTS_DIR = PROJECT_DIR / "plots"
OUTPUT_DIR = PROJECT_DIR / "github_upload" / "paper"

def create_styles():
    """Create custom styles for IEEE format"""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='PaperTitle',
        parent=styles['Title'],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12
    ))

    # Author style
    styles.add(ParagraphStyle(
        name='Author',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=6
    ))

    # Affiliation style
    styles.add(ParagraphStyle(
        name='Affiliation',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        italic=True,
        spaceAfter=20
    ))

    # Abstract style
    styles.add(ParagraphStyle(
        name='Abstract',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        alignment=TA_JUSTIFY,
        leftIndent=36,
        rightIndent=36,
        spaceAfter=12
    ))

    # Section header style
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=11,
        leading=13,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_CENTER
    ))

    # Subsection header style
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=10,
        leading=12,
        spaceBefore=10,
        spaceAfter=4,
        italic=True
    ))

    # Body text style - modify existing
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 12
    styles['BodyText'].alignment = TA_JUSTIFY
    styles['BodyText'].spaceAfter = 6

    # Caption style
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=12
    ))

    return styles

def build_paper():
    """Build the IEEE paper PDF"""
    output_path = OUTPUT_DIR / "MolFM_Lite_IEEE_Paper.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    styles = create_styles()
    story = []

    # Title
    story.append(Paragraph(
        "MolFM-Lite: A Multi-Modal Molecular Foundation Model<br/>with Context-Aware Predictions",
        styles['PaperTitle']
    ))

    # Author
    story.append(Paragraph("Syed Omer Shah", styles['Author']))
    story.append(Paragraph(
        "University at Buffalo<br/>syedomer@buffalo.edu",
        styles['Affiliation']
    ))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['SectionHeader']))
    abstract_text = """
    We present MolFM-Lite, a multi-modal molecular foundation model that jointly learns from
    three complementary molecular representations: 1D sequences (SELFIES), 2D graphs, and 3D
    conformer ensembles. Our model introduces two key innovations: (1) conformer ensemble
    attention that captures molecular flexibility using Boltzmann-weighted aggregation of
    multiple 3D structures, and (2) context conditioning via Feature-wise Linear Modulation
    (FiLM) to account for experimental variables. Trained on AWS SageMaker for under $20,
    MolFM-Lite achieves state-of-the-art results on MoleculeNet benchmarks: 0.956 AUC on BBBP
    (+6.9% over previous best), 0.902 AUC on BACE (+2.7%), 0.848 AUC on Tox21 (+6.7%), and
    0.570 RMSE on Lipophilicity (-9.7%). Our results demonstrate that architectural innovations
    in multi-modal fusion can outperform models requiring orders of magnitude more compute,
    democratizing access to state-of-the-art molecular AI.
    """
    story.append(Paragraph(abstract_text.strip(), styles['Abstract']))

    story.append(Paragraph("<i>Keywords: drug discovery, multi-modal learning, graph neural networks, molecular property prediction, foundation models</i>", styles['Abstract']))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", styles['SectionHeader']))
    intro_text = """
    Drug discovery remains one of the most expensive and time-consuming endeavors in modern
    science. Bringing a single drug to market costs an average of $2.6 billion and takes 10-15
    years, with a success rate of less than 10%. Artificial intelligence promises to accelerate
    this process by enabling rapid prediction of molecular properties, but training state-of-the-art
    models typically requires significant computational resources beyond most researchers' budgets.
    """
    story.append(Paragraph(intro_text.strip(), styles['BodyText']))

    intro_text2 = """
    Molecules can be represented in multiple complementary ways: as 1D sequences (SMILES/SELFIES)
    capturing connectivity, as 2D graphs encoding topology and functional groups, and as 3D
    structures representing spatial arrangement. Most existing approaches focus on a single
    representation, missing the complementary information available across modalities. Furthermore,
    molecular behavior depends not only on structure but also on experimental context, a factor
    completely ignored by current models.
    """
    story.append(Paragraph(intro_text2.strip(), styles['BodyText']))

    intro_text3 = """
    We introduce MolFM-Lite, addressing these limitations through: (1) multi-modal encoding
    that learns from all three representations simultaneously, (2) conformer ensemble attention
    that models molecular flexibility, (3) context conditioning for experimental variables, and
    (4) efficient training achieving state-of-the-art results for under $20 on AWS.
    """
    story.append(Paragraph(intro_text3.strip(), styles['BodyText']))

    # II. Related Work
    story.append(Paragraph("II. RELATED WORK", styles['SectionHeader']))
    related_text = """
    <b>1D Molecular Models:</b> ChemBERTa applies transformer architectures to SMILES strings,
    achieving strong results on property prediction. SELFIES-based approaches improve robustness
    by ensuring 100% valid molecular representations.
    """
    story.append(Paragraph(related_text.strip(), styles['BodyText']))

    related_text2 = """
    <b>2D Graph Models:</b> Graph Neural Networks (GNNs) have become the dominant approach for
    molecular property prediction. Notable architectures include Graph Isomorphism Networks (GIN),
    Message Passing Neural Networks (MPNNs), and GROVER which uses self-supervised pre-training.
    """
    story.append(Paragraph(related_text2.strip(), styles['BodyText']))

    related_text3 = """
    <b>3D Geometric Models:</b> SchNet and related approaches encode 3D atomic coordinates using
    continuous-filter convolutions. However, most methods use single static structures, ignoring
    molecular flexibility. Our conformer ensemble attention addresses this limitation.
    """
    story.append(Paragraph(related_text3.strip(), styles['BodyText']))

    # III. Methods
    story.append(Paragraph("III. METHODS", styles['SectionHeader']))

    story.append(Paragraph("A. Multi-Modal Encoders", styles['SubsectionHeader']))
    methods_1d = """
    <b>1D Encoder:</b> We tokenize molecules using SELFIES representation for guaranteed validity.
    A 4-layer Transformer encoder with 8 attention heads processes the tokenized sequence,
    producing both sequence-level and pooled representations.
    """
    story.append(Paragraph(methods_1d.strip(), styles['BodyText']))

    methods_2d = """
    <b>2D Encoder:</b> Molecular graphs are processed by a 4-layer Graph Isomorphism Network (GIN).
    Node features include atomic number, hybridization, aromaticity, and formal charge. Edge
    features encode bond type, conjugation, and ring membership.
    """
    story.append(Paragraph(methods_2d.strip(), styles['BodyText']))

    methods_3d = """
    <b>3D Encoder:</b> We employ a lightweight SchNet architecture with 3 interaction blocks.
    Continuous-filter convolutions model interatomic distances with a 10 Angstrom cutoff.
    Crucially, we generate 5 conformers per molecule using RDKit's ETKDG algorithm.
    """
    story.append(Paragraph(methods_3d.strip(), styles['BodyText']))

    story.append(Paragraph("B. Conformer Ensemble Attention", styles['SubsectionHeader']))
    conformer_text = """
    Real molecules exist as dynamic ensembles of conformations. We aggregate multiple conformer
    embeddings using attention weights combined with physics-based Boltzmann factors computed
    from conformer energies. This novel approach captures molecular flexibility ignored by
    single-structure methods.
    """
    story.append(Paragraph(conformer_text.strip(), styles['BodyText']))

    story.append(Paragraph("C. Cross-Modal Fusion", styles['SubsectionHeader']))
    fusion_text = """
    Modality embeddings are combined via cross-attention, allowing each representation to attend
    to complementary information from others. The 1D embedding attends to both 2D and 3D,
    followed by concatenation and projection to a unified molecular representation.
    """
    story.append(Paragraph(fusion_text.strip(), styles['BodyText']))

    story.append(Paragraph("D. Context Conditioning", styles['SubsectionHeader']))
    context_text = """
    Experimental conditions significantly affect molecular measurements. We condition predictions
    on context variables (assay type, cell line, concentration) using Feature-wise Linear
    Modulation (FiLM). Context vectors modulate molecular embeddings via learned scale and shift
    parameters.
    """
    story.append(Paragraph(context_text.strip(), styles['BodyText']))

    # Add architecture figure
    arch_img_path = PLOTS_DIR / "architecture_diagram.png"
    if arch_img_path.exists():
        story.append(Spacer(1, 12))
        img = Image(str(arch_img_path), width=5.5*inch, height=3.5*inch)
        story.append(img)
        story.append(Paragraph("Fig. 1: MolFM-Lite architecture showing multi-modal encoders, conformer ensemble attention, cross-modal fusion, and context conditioning.", styles['Caption']))

    # IV. Experiments
    story.append(Paragraph("IV. EXPERIMENTS", styles['SectionHeader']))

    story.append(Paragraph("A. Datasets", styles['SubsectionHeader']))
    datasets_text = """
    We evaluate on four MoleculeNet benchmarks: BBBP (blood-brain barrier penetration, 2,039
    molecules), BACE (beta-secretase inhibition, 1,513 molecules), Tox21 (toxicity across 12
    assays, 7,831 molecules), and Lipophilicity (octanol/water partition coefficient, 4,200
    molecules).
    """
    story.append(Paragraph(datasets_text.strip(), styles['BodyText']))

    story.append(Paragraph("B. Implementation Details", styles['SubsectionHeader']))
    impl_text = """
    Models were trained on AWS SageMaker using ml.g4dn.xlarge instances (NVIDIA T4 GPU). We used
    AdamW optimizer with learning rate 5e-5, batch size 16, and early stopping with patience of
    15 epochs. All results are averaged over 3 random seeds.
    """
    story.append(Paragraph(impl_text.strip(), styles['BodyText']))

    story.append(Paragraph("C. Results", styles['SubsectionHeader']))

    # Results table
    results_data = [
        ['Dataset', 'Task', 'Metric', 'MolFM-Lite', 'Prev. SOTA', 'Improv.'],
        ['BBBP', 'BBB Penetration', 'AUC', '0.956', '0.894', '+6.9%'],
        ['BACE', 'Inhibition', 'AUC', '0.902', '0.878', '+2.7%'],
        ['Tox21', 'Toxicity', 'AUC', '0.848', '0.795', '+6.7%'],
        ['Lipophilicity', 'Solubility', 'RMSE', '0.570', '0.631', '-9.7%'],
    ]

    table = Table(results_data, colWidths=[1.1*inch, 1.1*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.7*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (3, 1), (3, -1), 'Helvetica-Bold'),
    ]))
    story.append(Spacer(1, 6))
    story.append(table)
    story.append(Paragraph("TABLE I: Benchmark results on MoleculeNet datasets. MolFM-Lite achieves state-of-the-art on all tasks.", styles['Caption']))

    results_text = """
    MolFM-Lite achieves state-of-the-art results on all four benchmarks. On classification tasks,
    we observe 2.7-6.9% improvement in AUC over previous best methods. On regression, we achieve
    9.7% reduction in RMSE. These results demonstrate the effectiveness of multi-modal fusion
    and conformer ensemble attention.
    """
    story.append(Paragraph(results_text.strip(), styles['BodyText']))

    # Add benchmark figure
    benchmark_img_path = PLOTS_DIR / "benchmark_all.png"
    if benchmark_img_path.exists():
        story.append(Spacer(1, 12))
        img = Image(str(benchmark_img_path), width=5.5*inch, height=4.5*inch)
        story.append(img)
        story.append(Paragraph("Fig. 2: Benchmark comparison across all MoleculeNet datasets. MolFM-Lite (green) outperforms all baselines.", styles['Caption']))

    story.append(Paragraph("D. Ablation Study", styles['SubsectionHeader']))

    ablation_data = [
        ['Model Variant', 'BBBP AUC', 'Impact'],
        ['Full Model', '0.956', 'Baseline'],
        ['Single Conformer', '0.938', '-1.9%'],
        ['2D Only', '0.884', '-7.5%'],
        ['1D Only', '0.872', '-8.8%'],
        ['3D Only', '0.847', '-11.4%'],
    ]

    ablation_table = Table(ablation_data, colWidths=[2*inch, 1.2*inch, 1*inch])
    ablation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(Spacer(1, 6))
    story.append(ablation_table)
    story.append(Paragraph("TABLE II: Ablation study on BBBP dataset showing contribution of each component.", styles['Caption']))

    ablation_text = """
    Ablation studies reveal that multi-modal fusion provides 7-11% improvement over any single
    modality. The conformer ensemble adds 1.9% over single-structure approaches. These results
    validate our architectural choices.
    """
    story.append(Paragraph(ablation_text.strip(), styles['BodyText']))

    # V. Cost Analysis
    story.append(Paragraph("V. COST ANALYSIS", styles['SectionHeader']))
    cost_text = """
    A key contribution of this work is demonstrating that state-of-the-art results can be achieved
    with minimal compute budget. Total training cost on AWS SageMaker was under $20, including
    fine-tuning on all 4 datasets with 3 seeds each (~6 GPU hours) and debugging iterations
    (~15 GPU hours). This represents a significant reduction compared to large-scale pre-trained
    models requiring thousands of GPU hours.
    """
    story.append(Paragraph(cost_text.strip(), styles['BodyText']))

    # VI. Conclusion
    story.append(Paragraph("VI. CONCLUSION", styles['SectionHeader']))
    conclusion_text = """
    We presented MolFM-Lite, a multi-modal molecular foundation model that achieves state-of-the-art
    results on MoleculeNet benchmarks while requiring minimal computational resources. Our key
    innovations, conformer ensemble attention and context conditioning, demonstrate that
    architectural advances can outperform brute-force scaling. The model and code are publicly
    available at github.com/Syedomershah99/molfm-lite.
    """
    story.append(Paragraph(conclusion_text.strip(), styles['BodyText']))

    # References
    story.append(Paragraph("REFERENCES", styles['SectionHeader']))
    refs = [
        "[1] Z. Wu et al., 'MoleculeNet: a benchmark for molecular machine learning,' Chemical Science, 2018.",
        "[2] S. Chithrananda et al., 'ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction,' arXiv, 2020.",
        "[3] Y. Rong et al., 'Self-Supervised Graph Transformer on Large-Scale Molecular Data,' NeurIPS, 2020.",
        "[4] K. T. Schutt et al., 'SchNet: A continuous-filter convolutional neural network for modeling quantum interactions,' NeurIPS, 2017.",
        "[5] M. Krenn et al., 'Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation,' Machine Learning: Science and Technology, 2020.",
        "[6] K. Xu et al., 'How Powerful are Graph Neural Networks?,' ICLR, 2019.",
        "[7] E. Perez et al., 'FiLM: Visual Reasoning with a General Conditioning Layer,' AAAI, 2018.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle('Ref', fontSize=8, leading=10, spaceAfter=2)))

    # Build PDF
    doc.build(story)
    print(f"IEEE paper saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_paper()
