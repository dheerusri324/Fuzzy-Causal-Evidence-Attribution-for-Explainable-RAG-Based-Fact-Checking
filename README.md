# Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking

[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Frozen-success.svg)](PHASE_13_FINAL_RESEARCH_SYNTHESIS.md)
[![Phase](https://img.shields.io/badge/Phase-13%20Research%20Synthesis-blue.svg)](PHASE_13_FINAL_RESEARCH_SYNTHESIS.md)
[![Audit](https://img.shields.io/badge/Audit-Pre--Phase--14%20Errata%20V4-orange.svg)](PRE_PHASE_14_AUDIT_ERRATA_V4.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reliability-aware evidence attribution and explanation framework for automated fact-checking. This project combines post-hoc feature attribution (SHAP, LIME) with counterfactual leave-one-out intervention testing and a Mamdani fuzzy evidence reasoning engine to evaluate and explain claim verdicts in a Retrieval-Augmented Generation (RAG) setting.

---

## Table of Contents
1. [Research Motivation & Objectives](#1-research-motivation--objectives)
2. [Base Paper & Theoretical Foundation](#2-base-paper--theoretical-foundation)
3. [System Architecture & Implemented Pipeline](#3-system-architecture--implemented-pipeline)
4. [Methodological Components](#4-methodological-components)
   - [Hybrid Retrieval](#41-hybrid-retrieval)
   - [Post-Hoc Evidence Attribution](#42-post-hoc-evidence-attribution)
   - [Intervention-Based Influence Testing](#43-intervention-based-influence-testing)
   - [Conflict Detection & Stability Analysis](#44-conflict-detection--stability-analysis)
   - [Mamdani Fuzzy Evidence Reasoning Engine](#45-mamdani-fuzzy-evidence-reasoning-engine)
5. [Repository Structure & Phase Identifiers](#5-repository-structure--phase-identifiers)
6. [Dataset Sources, Sampling & Provenance](#6-dataset-sources-sampling--provenance)
7. [Summary of Formal Experimental Findings](#7-summary-of-formal-experimental-findings)
   - [Experiment 1: Attribution vs. Counterfactual Intervention](#experiment-1-attribution-vs-counterfactual-intervention-500-claims-up-to-1500-evidence-level-attributioninfluence-values)
   - [Experiment 2: Fuzzy Reasoning vs. Linear Baseline](#experiment-2-fuzzy-reasoning-vs-linear-baseline-500-claims-1500-claim-evidence-rows)
   - [NEI Evidence Label Sensitivity Audit](#nei-evidence-label-sensitivity-audit-protocols-a--b)
8. [Installation & Environment Setup](#8-installation--environment-setup)
9. [Running the Application, Tests, and Verification Scripts](#9-running-the-application-tests-and-verification-scripts)
10. [Audit Trail & Technical Errata Index](#10-audit-trail--technical-errata-index)
11. [Known Methodological Limitations & Boundaries](#11-known-methodological-limitations--boundaries)
12. [Citation & Licensing](#12-citation--licensing)

---

## 1. Research Motivation & Objectives

Modern automated fact-checking systems verify natural language claims against external knowledge bases using dense retrieval and neural cross-encoders. While post-hoc explainable AI (XAI) methods such as SHAP and LIME assign importance weights to input tokens or evidence sentences, they suffer from two critical shortcomings in fact-checking pipelines:
1. **Attribution Correlates, But Does Not Cause**: Post-hoc attribution identifies which features a model attended to locally, but it does not measure whether the discrete verifier verdict is *counterfactually dependent* on that evidence (i.e., whether removing or perturbing the evidence actually changes the verdict).
2. **Conflation of Disparate Signals**: Existing systems often conflate retrieval relevance, model sensitivity, mutual evidence contradiction, perturbation stability, and source credibility into uncalibrated linear combinations or single scalar scores.

### Research Questions
- **RQ1 (Attribution vs. Intervention Alignment)**: *To what extent does post-hoc feature attribution reflect the empirical influence of evidence under counterfactual deletion intervention in a fact-checking cross-encoder?*
- **RQ2 (Fuzzy Evidence Reasoning vs. Linear Baseline)**: *Does fuzzy evidence-reliability scoring distinguish human-annotated gold evidence from unannotated distractor passages more effectively than a linear weighted baseline in a closed RAG setting, and what are the effects of dynamic contradiction and stability signals?*

---

## 2. Base Paper & Theoretical Foundation

This project directly investigates and extends the research questions and empirical gaps identified in:

> **Reference Study:**  
> Jean-Flavien Bussotti, Andrea Baraldi, Francesco Guerra, and Paolo Papotti.  
> *"Explaining the Role of Evidence in Data-Driven Fact-Checking."*  
> *IEEE Access*, vol. 12, pp. 37190–37204, 2026.  
> DOI: [10.1109/ACCESS.2026.3719082](https://doi.org/10.1109/ACCESS.2026.3719082)

### Confirmed Gaps Addressed
1. **Attribution $\neq$ Causal Influence**: Demonstrating empirical cases where high SHAP/LIME importance does not prevent verdict collapse, and conversely, where removing passages ranked 2nd or 3rd flips the prediction.
2. **Beyond Binary Useful/Noise**: Replacing rigid threshold binarization with continuous fuzzy truth and reliability representations.
3. **Handling Disagreement & Mixed Evidence**: Incorporating an explicit Natural Language Inference (NLI) contradiction detector rather than discarding conflicting claims.
4. **Quantifying Explanation Robustness**: Integrating rank stability analysis under embedding noise perturbations.

---

## 3. System Architecture & Implemented Pipeline

The complete pipeline operates across eight integrated stages:

```
[Claim] ──► [Hybrid Retrieval: BM25 + all-MiniLM-L6-v2] ──► Top-k Evidence Passages (k=3)
                 │
                 ▼
     [Cross-Encoder Verifier: DeBERTa-v3-base] ──► Verdict & P(y|E)
                 │
  ┌──────────────┼──────────────────────────────┐
  ▼              ▼                              ▼
[SHAP / LIME]  [Leave-One-Out Intervention]   [Conflict & Stability Engines]
Attribution    Influence ΔP & Verdict Flips    Contradiction (NLI) & Stability (Spearman)
  │              │                              │
  └──────────────┼──────────────────────────────┘
                 ▼
  [Mamdani Fuzzy Evidence Reasoning Engine]
   (11 Rules across 5 Input Signals, Centroid Defuzzification)
                 │
                 ▼
  [Reliability-Aware Output & Interactive Streamlit Dashboard]
```

---

## 4. Methodological Components

### 4.1 Hybrid Retrieval
- **Lexical Component**: BM25 algorithm scoring token overlap between claim and corpus passages.
- **Dense Component**: Dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` indexed with FAISS (`IndexFlatIP` over L2-normalized vectors).
- **Hybrid Fusion**: Min-max normalized reciprocal or convex combination of BM25 and dense cosine scores (`hybrid_alpha = 0.5`).

### 4.2 Post-Hoc Evidence Attribution
- **SHAP (KernelSHAP)**: Treats passage inclusion as binary coalition features, evaluating marginal contributions over evidence subsets.
- **LIME**: Fits a sparse linear surrogate on perturbed evidence masks weighted by exponential kernel distances.
- **Attribution Normalization**: Produces relative importance distributions across the retrieved passage set.

### 4.3 Intervention-Based Influence Testing
- **Leave-One-Out Deletion**: Measures empirical influence $\Delta P_i = P(y^* \mid E) - P(y^* \mid E \setminus \{e_i\})$, where $y^*$ is the predicted verdict given all retrieved evidence.
- **Verdict Flip Detection**: Identifies whether $\arg\max_y P(y \mid E \setminus \{e_i\}) \neq y^*$.
- *Note on Terminology*: Intervention influence quantifies the sensitivity of the model's classification head to evidence removal; it does not constitute proof of real-world physical causality.

### 4.4 Conflict Detection & Stability Analysis
- **Contradiction Degree ($c_{\text{deg}}$)**: Pairwise cross-encoder NLI between retrieved evidence passages, aggregated via soft maximum contradiction penalties.
- **Explanation Stability ($s_{\text{stab}}$)**: Evaluates ranking stability by adding Gaussian noise ($\sigma = 0.05$) to dense passage embeddings and calculating the mean Spearman rank correlation across 5 perturbed runs.

### 4.5 Mamdani Fuzzy Evidence Reasoning Engine
- **Inputs**: Relevance ($r_i$), Post-hoc Attribution ($a_i$), Intervention Influence ($\Delta P_i$), Contradiction Degree ($c_{\text{deg}}$), Explanation Stability ($s_{\text{stab}}$).
- **Fuzzy Sets**: Low, Medium, High triangular and trapezoidal membership functions.
- **Rule Base**: 11 Mamdani inference rules modeling non-linear interactions (e.g., dampening relevance when contradiction is high unless intervention influence is decisive).
- **Defuzzification**: Centroid (Center of Gravity) defuzzification yielding continuous evidence reliability scores in $[0, 1]$.
- *Note on Terminology*: The resulting score is a calibrated operational proxy for evidence utility; it is not a formal Bayesian probability of truth.

---

## 5. Repository Structure & Phase Identifiers

All formal experiment directories and documentation retain their explicit phase identifiers to ensure auditable provenance:

```
├── app/                                 # Streamlit web application & end-to-end pipeline
│   ├── main.py                          # Streamlit application entry point
│   ├── pipeline.py                      # FactCheckingPipeline orchestrator
│   └── schema.py                        # Pipeline data models
├── attribution/                         # Feature attribution modules (SHAP & LIME wrappers)
├── conflict/                            # Evidence contradiction detection & consensus metrics
├── data/                                # Data loaders, sampled FEVER splits & evidence corpus
│   ├── fever_loader.py                  # Provenance loader & sampling script
│   ├── fever_dev.jsonl                  # Sampled 750-record evaluation split
│   ├── fever_full.jsonl                 # Streamed raw sample (5,000 records)
│   ├── evidence_corpus.json             # 808-passage closed evidence corpus
│   └── dataset_meta.json                # Lineage & provenance metadata
├── config/                              # Configuration schemas and default YAML settings
├── experiments/                         # Formal experiment execution scripts
│   ├── experiment1_formal.py            # Experiment 1 formal driver (RQ1)
│   ├── experiment2_formal.py            # Experiment 2 formal driver (RQ2)
│   └── run_all.py                       # Master experiment execution wrapper
├── fuzzy/                              # Mamdani fuzzy logic engine, membership & rules
├── intervention/                        # Counterfactual intervention & leave-one-out tests
├── outputs/                             # Frozen formal experimental outputs & plots
│   ├── phase10_formal/                  # Experiment 1 frozen results (results.csv, config.json)
│   ├── phase11_formal/                  # Experiment 2 frozen results & .signals_cache.json
│   ├── phase12_analysis/                # Cross-experiment comparative tables & audit JSON
│   ├── phase10_sanity/                  # Phase 10 preliminary sanity check artifacts
│   └── sensitivity_nei_label_audit/     # NEI label semantics sensitivity audit outputs
├── retrieval/                           # BM25 lexical, FAISS dense, and hybrid retriever
├── stability/                           # Rank stability and perturbation analysis
├── tests/                               # 11 unit & regression test suites (pytest)
├── utils/                               # Reproducibility, logging, and mathematical utilities
├── verification/                        # Cross-encoder claim-evidence verifier (NLI)
├── audit_bundle/                        # Historical pre-Phase-14 audit assets & manifests
├── supplemental_audit/                  # Supplemental audit bundle
├── sensitivity_audit_sourcepack/        # NEI label semantics sourcepack
├── scratch/                             # Auxiliary validation and verification scripts
├── cache/                               # FAISS index and MiniLM dense embeddings (2.5 MB)
├── docs/                                # Technical specifications and architectural docs
├── PRE_PHASE_14_AUDIT_ERRATA.md         # Errata V1
├── PRE_PHASE_14_AUDIT_ERRATA_V2.md      # Errata V2 (Holm correction & SHAP mean rho)
├── PRE_PHASE_14_AUDIT_ERRATA_V3.md      # Errata V3 (Hybrid BM25+dense retrieval correction)
├── PRE_PHASE_14_AUDIT_ERRATA_V4.md      # Errata V4 (Final frozen report & label semantics)
├── PRE_PHASE_14_ERRATA_V2_VALIDATION.md # Independent validation report for V2
├── PRE_PHASE_14_ERRATA_V3_VALIDATION.md # Independent validation report for V3
├── PRE_PHASE_14_ERRATA_V4_VALIDATION.md # Independent validation report for V4
├── DATASET_OVERLAP_AUDIT.md             # Dataset overlap audit report
├── DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md # NEI label semantics sensitivity audit
├── EXPERIMENT_1_FORMAL_REPORT.md        # Formal report for Experiment 1
├── EXPERIMENT_2_FORMAL_REPORT.md        # Formal report for Experiment 2
├── EXPERIMENT_2_POSTRUN_VALIDATION.md   # Post-run statistical validation report
└── Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx # 6-slide proposal presentation
```

---

## 6. Dataset Sources, Sampling & Provenance

The evaluation corpus is constructed via [`data/fever_loader.py`](data/fever_loader.py) using the Hugging Face `copenlu/fever_gold_evidence` repository:
- **Sampling Protocol**: Streamed up to the first 5,000 records from `split="train"`. Balanced random sampling selected exactly 250 records per label (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) using random seed `42` ($N = 750$).
- **Local Split Labeling**: The resulting 750-record sample is stored as [`data/fever_dev.jsonl`](data/fever_dev.jsonl). The first 500 records form the frozen evaluation cohort.
- **Closed Candidate Pool**: All human-annotated gold evidence passages from the sampled claims were extracted to form a closed candidate pool of **808 unique Wikipedia passages** stored in [`data/evidence_corpus.json`](data/evidence_corpus.json).
- **Claim Retrieval Coverage**: Under hybrid retrieval with $k = 3$, **86.00%** ($430 / 500$) of claims have at least one positive label in the retrieved set. This coverage is specific to the closed pool and cannot be extrapolated to open-domain retrieval.

---

## 7. Summary of Formal Experimental Findings

### Experiment 1: Attribution vs. Counterfactual Intervention (500 claims; up to 1,500 evidence-level attribution/influence values)
Evaluates alignment between post-hoc feature attribution (SHAP/LIME) and leave-one-out empirical influence across 500 evaluated claims (with up to $k=3$ retrieved passages per claim, corresponding to up to 1,500 evidence-level attribution and influence values when all three passages are evaluated):
- **Top-1 Agreement**: Evaluated per claim; LIME achieved **84.00%** ($420 / 500$) and SHAP achieved **72.20%** ($361 / 500$) agreement with the most influential passage under leave-one-out removal.
- **Rank Correlation ($\rho$)**: Evaluated across claim-evidence rankings; LIME mean Spearman $\rho = \mathbf{0.5877}$ (median $\mathbf{0.5000}$); SHAP mean Spearman $\rho = \mathbf{0.4517}$ (median $\mathbf{0.5000}$).
- **Attribution Blind Spots**: In **21.47%** of evidence removals ($322 / 1,500$ evaluation instances across the 500 claims), the discrete verifier verdict flipped. In **16.15% (LIME)** and **16.77% (SHAP)** of these flips, the removed passage was ranked 2nd or 3rd by attribution, demonstrating that feature attribution alone fails to catch critical decision vulnerabilities.

### Experiment 2: Fuzzy Reasoning vs. Linear Baseline (500 claims; 1,500 claim-evidence rows)
Evaluates evidence reliability ranking under dynamic conflict and stability signals across 500 claims ($k=3$ passages per claim, producing 1,500 claim-evidence rows with 443 positive proxy labels in the official frozen run):
- **Hypothesis $H_1$ (AUROC)**: Fuzzy reasoning achieved a statistically significant **$+0.0365$ higher AUROC** than the baseline under dynamic conflict ($\text{AUROC}_{\text{fuzzy}} = \mathbf{0.7630}$ vs. $\text{AUROC}_{\text{base}} = \mathbf{0.7265}$, $p_{\text{Holm}} = 0.0240$, FWER $\alpha = 0.05$).
- **Hypothesis $H_2$ (Average Precision)**: The linear baseline achieved a statistically significant **$+0.1396$ higher AP** ($\text{AP}_{\text{base}} = \mathbf{0.6280}$ vs. $\text{AP}_{\text{fuzzy}} = \mathbf{0.4884}$, $p_{\text{Holm}} = 0.0020$).
- **Within-Claim Top-1 Proxy-Label Hit Rate**: Under the original frozen proxy-label scheme, Fuzzy reasoning ranked a proxy-positive passage first in **73.00%** (365/500) of claims versus **69.00%** (345/500) for the baseline (Δ = +4.00 percentage points, +20 claims). These are proxy-label ranking results, not verified-gold evidence selection rates. Among the 430 claims with proxy-positive evidence in the retrieved top three, Fuzzy achieved **84.88%** (365/430) versus **80.23%** (345/430) for the baseline (+4.65 percentage points). This is a conditional proxy-label ranking result, not a verified-gold evidence selection rate.
- **Secondary Ablation**: Transitioning from static defaults ($c_{\text{deg}}=0.0, s_{\text{stab}}=1.0$) to dynamic inputs caused the baseline's uniform multiplicative penalty to force **20.80% ($312 / 1,500$)** of passage scores identically to 0.0, collapsing baseline AUROC by $-0.1624$. The fuzzy engine's rule synergies mitigated this loss, dropping by only $-0.0316$.

### NEI Evidence Label Sensitivity Audit (Protocols A & B)
In `copenlu/fever_gold_evidence`, `SUPPORTS` and `REFUTES` claims contain human-annotated gold evidence, while `NOT_ENOUGH_INFO` (NEI) claims contain retrieved Papelo candidate passages. Of the 443 positive proxy instances in the official frozen Experiment 2, exactly 147 belong to NEI claims. Two sensitivity protocols in `DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md` evaluate the impact of correcting these label semantics:
- **Protocol A (500 claims; 1,500 claim-evidence rows, NEI reclassified as distractors):**
  All 500 claims are evaluated, with the 147 NEI passages reclassified as negative distractors ($y = 0$), leaving 296 verified gold instances and 1,204 distractors across 1,500 rows.
  - **AUROC**: The estimated AUROC difference (Fuzzy − Baseline) is **$+0.0898$** under Protocol A ($\text{AUROC}_{\text{fuzzy}} = \mathbf{0.7386}$ vs. $\text{AUROC}_{\text{base}} = \mathbf{0.6488}$; Holm-adjusted $p = \mathbf{0.0020}$, statistically significant at $\alpha = 0.05$). No separate statistical test compared this difference against the corresponding estimates from the other evaluation protocols.
  - **Average Precision**: Baseline AP advantage magnitude is **$0.0680$** ($\text{AP}_{\text{base}} = \mathbf{0.4217}$ vs. $\text{AP}_{\text{fuzzy}} = \mathbf{0.3537}$; Holm-adjusted $p = \mathbf{0.0020}$, statistically significant at $\alpha = 0.05$).
  - **Top-1 Gold Selection**: On the 283 claims with verified gold evidence in the top 3, Fuzzy selects gold evidence as Rank 1 in **87.63%** ($248 / 283$) vs. Baseline **83.39%** ($236 / 283$) ($+4.24$ percentage points). Across all 500 claims, Fuzzy achieves **49.60%** ($248 / 500$) vs. Baseline **47.20%** ($236 / 500$) ($+2.40$ percentage points).
- **Protocol B (334 verifiable claims; 1,002 claim-evidence rows, NEI claims excluded):**
  Excludes all 166 NEI claims completely, evaluating only the 334 verifiable `SUPPORTS` and `REFUTES` claims ($k=3$ passages each, yielding 1,002 claim-evidence rows with 296 verified gold instances and 706 distractors).
  - **AUROC**: The estimated AUROC difference (Fuzzy − Baseline) is **$+0.0827$** under Protocol B ($\text{AUROC}_{\text{fuzzy}} = \mathbf{0.7777}$ vs. $\text{AUROC}_{\text{base}} = \mathbf{0.6949}$; Holm-adjusted $p = \mathbf{0.0020}$, statistically significant at $\alpha = 0.05$). As with Protocol A, protocol-specific AUROC differences were evaluated within-protocol and not separately compared statistically against other protocols.
  - **Average Precision**: Baseline AP advantage magnitude is **$0.1098$** ($\text{AP}_{\text{base}} = \mathbf{0.6189}$ vs. $\text{AP}_{\text{fuzzy}} = \mathbf{0.5091}$; Holm-adjusted $p = \mathbf{0.0020}$, statistically significant at $\alpha = 0.05$).
  - **Top-1 Gold Selection**: On the 283 claims with verified gold evidence in the top 3, Fuzzy selects gold evidence as Rank 1 in **87.63%** ($248 / 283$) vs. Baseline **83.39%** ($236 / 283$) ($+4.24$ percentage points). Across all 334 verifiable claims, Fuzzy achieves **74.25%** ($248 / 334$) vs. Baseline **70.66%** ($236 / 334$) ($+3.59$ percentage points).

---

## 8. Installation & Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/dheerusri324/Fuzzy-Causal-Evidence-Attribution-for-Explainable-RAG-Based-Fact-Checking.git
cd Fuzzy-Causal-Evidence-Attribution-for-Explainable-RAG-Based-Fact-Checking

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Running the Application, Tests, and Verification Scripts

The commands below document how to run the application, test suites, and verification scripts in a local environment. Their inclusion here does not imply that every command was executed during the latest repository-publication process (during which scripts and dependencies were statically inspected to preserve frozen assets). Consult the available phase reports and project documentation for previously documented development and testing checks.

### 9.1 Run Unit & Regression Tests
Command to execute available unit tests across all 11 test suites:
```bash
pytest tests/
```

### 9.2 Launch the Interactive Streamlit Dashboard
Command to launch the Streamlit web application:
```bash
streamlit run app/main.py
```
Provides an interactive web interface to enter custom claims, retrieve evidence, view cross-encoder verdicts, inspect SHAP/LIME attributions, trigger counterfactual interventions, and review fuzzy reliability breakdowns.

### 9.3 Instant Tabular Verification (< 30 seconds)
Command to verify primary tabular metrics, confidence intervals, p-values, and ablations directly from frozen evaluation CSV outputs without re-running neural models:
```bash
python scratch/verify_exp2.py
```

### 9.4 Re-running Formal Experiments
Commands to re-execute formal evaluation runs from scratch:
```bash
# Experiment 1 (Attribution vs. Intervention)
python experiments/experiment1_formal.py

# Experiment 2 (Fuzzy Reasoning vs. Linear Baseline)
python experiments/experiment2_formal.py
```
*(Note: Re-running models performs full neural inference and requires sufficient GPU/CPU compute).*

---

## 10. Audit Trail & Technical Errata Index

Prior to publishing, this project underwent an extensive multi-stage technical audit. All errata documents and independent validation reports are preserved in full:

| Document | Key Corrections & Verified Findings |
| :--- | :--- |
| [`PRE_PHASE_14_AUDIT_ERRATA_V4.md`](PRE_PHASE_14_AUDIT_ERRATA_V4.md) | **Canonical Technical Errata**: Final errata documenting dataset provenance, label semantics in official results, hybrid retrieval configuration, Holm FWER adjustments, and exact statistical point estimates. |
| [`PRE_PHASE_14_ERRATA_V4_VALIDATION.md`](PRE_PHASE_14_ERRATA_V4_VALIDATION.md) | Independent mathematical and textual validation report confirming zero discrepancies in Errata V4. |
| [`DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md`](DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md) | Comprehensive sensitivity audit isolating the impact of NEI label semantics on AUROC, AP, and Top-1 hit rates. |
| [`DATASET_OVERLAP_AUDIT.md`](DATASET_OVERLAP_AUDIT.md) | Formal audit of claim and evidence overlap across evaluation splits. |
| [`EXPERIMENT_2_POSTRUN_VALIDATION.md`](EXPERIMENT_2_POSTRUN_VALIDATION.md) | Independent verification of all Experiment 2 point estimates, bootstrap inferences, and ablations. |
| [`PRE_PHASE_14_AUDIT_ERRATA_V3.md`](PRE_PHASE_14_AUDIT_ERRATA_V3.md) | Earlier errata correcting retrieval documentation to hybrid BM25+dense. |
| [`PRE_PHASE_14_AUDIT_ERRATA_V2.md`](PRE_PHASE_14_AUDIT_ERRATA_V2.md) | Earlier errata correcting SHAP mean rho (0.4517) and Holm inference. |
| [`PRE_PHASE_14_AUDIT_ERRATA.md`](PRE_PHASE_14_AUDIT_ERRATA.md) | Errata V1 establishing initial audit findings. |

---

## 11. Known Methodological Limitations & Boundaries

1. **Closed Evidence Pool**: The retriever operates over a closed candidate pool of 808 passages pre-seeded with gold evidence. The 86.00% retrieval coverage reflects this closed setting and should not be cited as open-domain performance.
2. **Label Semantics Asymmetry**: In the evaluated 500-claim cohort (1,500 claim-evidence rows), 147 positive evidence labels belong to NEI claims where evidence represents retrieved candidate passages rather than annotator-verified gold evidence. Under the corrected NEI-label sensitivity protocols, the estimated AUROC difference (Fuzzy − Baseline) is $+0.0898$ under Protocol A (all 500 claims, 1,500 rows; Holm-adjusted $p = 0.0020$, statistically significant at $\alpha = 0.05$) and $+0.0827$ under Protocol B (334 verifiable claims, 1,002 rows; Holm-adjusted $p = 0.0020$, statistically significant at $\alpha = 0.05$). These sensitivity analyses evaluate evidence-level proxy labels; evidence passages not annotated as gold are not necessarily false or irrelevant in an open-world sense.
3. **Model Sensitivity vs. Real Causality**: Leave-one-out intervention influence ($\Delta P$) measures the sensitivity of the DeBERTa cross-encoder classification head, not real-world factual causality.
4. **Context Window Constraint**: Fixed at top-$k = 3$ retrieved passages per claim.
5. **Linear Baseline Zero-Collapse**: The linear baseline uses a uniform multiplicative penalty $(1 - c_{\text{deg}}) \cdot s_{\text{stab}}$, which forces scores to zero when contradiction is complete ($c_{\text{deg}} = 1.0$), heavily penalizing baseline AUROC under dynamic conflict.

---

## 12. Citation & Licensing

### Academic Citation
If you use this codebase or build upon this research, please cite:

```bibtex
@article{bussotti2026explaining,
  author    = {Bussotti, Jean-Flavien and Baraldi, Andrea and Guerra, Francesco and Papotti, Paolo},
  title     = {Explaining the Role of Evidence in Data-Driven Fact-Checking},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {37190--37204},
  year      = {2026},
  doi       = {10.1109/ACCESS.2026.3719082}
}

@misc{chennavaram2026fuzzycausal,
  author    = {Chennavaram, Bala Dheeraj and Saketh Vamshee, A. and Rohit Reddy, V.},
  title     = {Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dheerusri324/Fuzzy-Causal-Evidence-Attribution-for-Explainable-RAG-Based-Fact-Checking}}
}
```

### Licensing
- **Code & Repository**: Licensed under the [MIT License](LICENSE).
- **Datasets**: The FEVER dataset is distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **Reference Paper**: The reference IEEE paper is preserved locally and cited under its original publication details (DOI: [10.1109/ACCESS.2026.3719082](https://doi.org/10.1109/ACCESS.2026.3719082)).
