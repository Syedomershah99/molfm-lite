# MolFM-Lite: Rejection Analysis & Revision Strategy

## Why the Submission Was Held/Rejected

After a thorough analysis of every file—code, data splits, results, and the full LaTeX—here are the specific, substantiated reasons the submission failed arXiv moderation and would fail peer review.

---

## Critical Issue 1: Benchmark Numbers Are Not Credible

**Problem:** The reported results are dramatically higher than the published literature for these exact benchmarks.

| Dataset | MolFM-Lite (claimed) | Uni-Mol (published, scaffold) | ChemBERTa (published, scaffold) |
|---------|----------------------|-------------------------------|----------------------------------|
| BBBP AUC | **0.956** | 0.729 | ~0.737 |
| BACE AUC | **0.902** | 0.858 | ~0.856 |
| Tox21 AUC | **0.848** | 0.831 | ~0.782 |
| Lipo RMSE | **0.570** | 0.603 | ~0.654 |

A model trained from scratch on a single T4 GPU in under $15 outperforming Uni-Mol — pretrained on 209 million conformers — by 4 AUC points on BBBP is extraordinary. arXiv moderators routinely flag results that deviate this far from community consensus.

**Root cause:** The CSV data splits (verified by scaffold-overlap analysis: ~16% overlap, consistent with scaffold splitting) are your own splits. The **baseline numbers in the table do not come from re-running those baselines on your splits** — they don't match what those papers report either on their own splits. The table mixes your method's results on your splits with published numbers from different papers using different splits. This is an apples-to-oranges comparison that makes your model look far better than a fair comparison would show.

**Fix:** Either (a) re-run all baselines on your exact splits, or (b) report published numbers and explicitly note different splits with a clear caveat. Option (a) is far stronger.

---

## Critical Issue 2: "Training from Scratch" Contradicts Pre-training Section

**Problem:** The abstract and conclusion repeatedly state "training from scratch without pre-training." But Section 3.6 describes a full cross-modal contrastive pre-training pipeline on ZINC250K with InfoNCE loss, masked atom prediction, and consistency loss over 30 epochs.

These two statements cannot both be true. The code in `src/training/` confirms pre-training is implemented. The config shows `pretraining.num_epochs: 50`.

**Fix:** Remove the "training from scratch" claim entirely. The correct framing is: "pre-trained on ZINC250K with $\sim$250K molecules (vs Uni-Mol's 209M conformers), demonstrating that thoughtful architecture can compensate for smaller-scale pre-training."

---

## Critical Issue 3: Context Conditioning Has No Experimental Evidence

**Problem:** FiLM-based context conditioning is listed as a key contribution. But:
1. MoleculeNet datasets have **no experimental context metadata** (no assay type, cell line, etc.)
2. The ablation shows only **0.5% gain** from context conditioning on BBBP — but this is with dummy/empty context since there's no metadata to condition on
3. The `CrossContextBenchmark` in `src/evaluation/benchmarks.py` is entirely a stub: `raise NotImplementedError("Requires ChEMBL data setup")`
4. The "cross-context generalization benchmark" mentioned in earlier drafts never appears with results

**Fix:** Either (a) implement the benchmark with real ChEMBL assay data, or (b) clearly reposition context conditioning as a design capability rather than an evaluated contribution, removing it from the main claims.

---

## Critical Issue 4: Unverifiable Quantitative Claims

**Problem:** Several specific quantitative claims appear without any described experimental protocol:
- "Individual modalities provide 2.2–2.8 bits. Joint information reaches 5.2 bits." — How was MINE run? On which dataset? What architecture?
- "approximately 1.7 bits appear synergistic" — No protocol for this estimate
- "learned conformer attention weights correlate with Boltzmann factors (ρ ≈ 0.73)" — Which dataset? Which molecules? How was this measured?
- "Predictions with high uncertainty (σ > 0.15) have 2.3× higher error rate" — Based on how many samples?
- "Mean calibration error: 0.034" — No comparison baseline; is this good?

These read as numbers inserted to sound rigorous without actual experiments. arXiv moderators are trained to spot this pattern.

**Fix:** Either implement these analyses properly with described protocols, or remove them entirely. The information-theoretic section in particular should either have a full empirical validation appendix or be substantially toned down.

---

## Critical Issue 5: Ablation Studies Are Incomplete

**Problem:**
- Ablation is only on BBBP; no results for BACE, Tox21, or Lipophilicity
- The ablation table contains combinations (1D+3D, 2D+3D) that weren't in the original draft but appear in the LaTeX — these should be verified results, not interpolated estimates
- The `AblationStudy._create_ablated_model()` in `benchmarks.py` returns `base_model` without modification — meaning the ablation infrastructure is a placeholder and ablation results may not have been computed correctly

**Fix:** Run ablations across all four datasets. Report mean ± std over 3 seeds. The ablation is the most important scientific content of the paper — it needs to be comprehensive.

---

## Issue 6: Compute Budget Inconsistency

**Problem:** Three different numbers appear:
- Abstract: "under $150 AWS credits"
- Appendix Table: ~$15 total (fine-tuning only)
- Acknowledgments: "AWS Credits program"

Pre-training on ZINC250K is conspicuously absent from the cost table. 30-50 epochs on ZINC250K (250K molecules, conformer generation) would take considerably more than 15 GPU-hours total.

**Fix:** Create a complete and honest compute table that includes pre-training cost. If pre-training was done with a free tier or AWS credits, say so explicitly.

---

## Issue 7: Code Availability Is Unverified

The paper cites `https://github.com/Syedomershah99/molfm-lite` but the git status shows the repo is not yet pushed to remote. arXiv moderators and reviewers expect linked code to be accessible. Papers citing unavailable code repositories are a red flag.

**Fix:** Push the code before submitting. Ensure all results are reproducible with the provided code.

---

## Issue 8: Information-Theoretic Section Needs Grounding

**Problem:** Section 3 (Theoretical Motivation) presents a "Theorem" and "Proposition" but:
- The Theorem contains no proof (just "follows from the data processing inequality and the observation that aggregation introduces no information loss" — this is incorrect; aggregation CAN lose information)
- The Proposition is a tautology, not a theorem
- The MINE analysis uses vague language: "We estimated mutual information using MINE on held-out data"

**Fix:** Either (a) properly prove the theorem or remove it, (b) describe the MINE experiment with full details, or (c) remove this section and fold the motivation into the introduction.

---

## Issue 9: Baseline Attribution Problems

The paper lists numbers for GPS++ and Graphormer on these benchmarks but:
- The cited GPS++ reference is `masters2023gps` — GPS++ was primarily evaluated on OGB benchmarks, not MoleculeNet; the numbers are unclear
- Graphormer's MoleculeNet results are also non-standard

**Fix:** For baselines that don't have published MoleculeNet results, either re-run them yourself on your splits or remove them.

---

## Summary of Rejection-Risk Rating

| Issue | Severity | Fix Difficulty |
|-------|----------|----------------|
| Incredible benchmark results | Critical | Re-run baselines |
| Training-from-scratch contradiction | Critical | Easy (word change) |
| Context conditioning unverified | High | Medium (ChEMBL experiment) |
| Unverifiable quantitative claims | High | Medium (run experiments or remove) |
| Incomplete ablations | High | Medium (run experiments) |
| Compute budget inconsistency | Medium | Easy |
| Code unavailability | Medium | Easy (push repo) |
| Weak theoretical section | Medium | Medium |
| Baseline attribution | Medium | Medium |
