# Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking

[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Frozen-success.svg)](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/PHASE_13_FINAL_RESEARCH_SYNTHESIS.md)
[![Phase](https://img.shields.io/badge/Phase-13%20Research%20Synthesis-blue.svg)](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/PHASE_13_FINAL_RESEARCH_SYNTHESIS.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A reliability-aware evidence attribution framework combining post-hoc feature attribution (SHAP/LIME) with counterfactual intervention-based influence and Mamdani fuzzy evidence reasoning for explainable Retrieval-Augmented Generation (RAG) fact-checking.

---

## 1. Project Overview & Research Questions

Modern automated fact-checking pipelines use dense retrieval and neural cross-encoders to verify claims against knowledge bases. While post-hoc attribution methods (SHAP, LIME) identify which tokens or passages a model attended to, they do not measure whether a verdict is counterfactually dependent on that evidence. Moreover, existing systems conflate retrieval relevance, model sensitivity, mutual contradiction, stability, and source reliability into single scalars or uncalibrated linear combinations.

This project investigates two core research questions:
1. **RQ1 (Attribution vs. Intervention Alignment)**: *To what extent does post-hoc feature attribution reflect the empirical influence of evidence under counterfactual deletion intervention in a fact-checking cross-encoder?*
2. **RQ2 (Fuzzy Evidence Reasoning vs. Linear Baseline)**: *Does fuzzy evidence-reliability scoring distinguish human-annotated gold evidence from unannotated distractor passages more effectively than a linear weighted baseline in a closed RAG setting, and what are the effects of dynamic contradiction and stability signals?*

---

## 2. System Architecture

The pipeline processes natural language claims through an eight-stage architecture:

```
[Claim] ──► [Hybrid RAG (BM25 + all-MiniLM-L6-v2)] ──► Top-3 Passages
                 │
                 ▼
     [Cross-Encoder Verifier (DeBERTa-v3-base)] ──► Predicted Verdict & P(y|E)
                 │
  ┌──────────────┼──────────────────────────────┐
  ▼              ▼                              ▼
[SHAP/LIME]   [Leave-One-Out Intervention]   [Conflict & Stability Engines]
Attribution   Influence ΔP & Verdict Flips    c_deg (NLI) & s_stab (Spearman)
  │              │                              │
  └──────────────┼──────────────────────────────┘
                 ▼
  [Mamdani Fuzzy Evidence Reasoning Engine]
   (11 Rules across 5 Outputs, Centroid Defuzz.)
                 │
                 ▼
  [Reliability-Aware Output & Interactive Streamlit UI]
```

Detailed technical specifications and module definitions are documented in [`docs/ARCHITECTURE.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/ARCHITECTURE.md).

---

## 3. Summary of Formal Experimental Findings

### Experiment 1: Attribution vs. Counterfactual Intervention ($N = 1,500$ evaluations across 500 claims)
- **Top-1 Agreement**: LIME achieved **84.00%** ($420 / 500$) and SHAP achieved **72.20%** ($361 / 500$) agreement with the most influential evidence piece under leave-one-out removal.
- **Rank Correlation ($\rho$)**: LIME mean Spearman $\rho = \mathbf{0.5877}$ (median $\mathbf{0.5000}$); SHAP mean Spearman $\rho = \mathbf{0.4517}$ (median $\mathbf{0.5000}$).
- **Attribution Blind Spots**: In **$21.47\%$** of evidence removals ($322 / 1,500$), the discrete verifier verdict flipped. In **$16.15\%$ (LIME) to $16.77\%$ (SHAP)** of these flips, the removed passage was ranked 2nd or 3rd by attribution, proving that feature attribution alone fails to identify all counterfactual vulnerabilities.
- *Full Report*: [`EXPERIMENT_1_FORMAL_REPORT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_1_FORMAL_REPORT.md).

### Experiment 2: Fuzzy Reasoning vs. Linear Baseline ($N = 1,500$ instances across 500 claims)
- **$H_1$ (AUROC)**: Fuzzy reasoning achieved a statistically significant **$+0.0365$ higher AUROC** than the baseline under dynamic conflict ($\text{AUROC}_{\text{fuzzy}} = \mathbf{0.7630}$ vs. $\text{AUROC}_{\text{base}} = \mathbf{0.7265}$, $p_{\text{Holm}} = 0.0240$, FWER $\alpha = 0.05$).
- **$H_2$ (Average Precision)**: The linear baseline achieved a statistically significant **$+0.1396$ higher Average Precision** ($\text{AP}_{\text{base}} = \mathbf{0.6280}$ vs. $\text{AP}_{\text{fuzzy}} = \mathbf{0.4884}$, $p_{\text{Holm}} = 0.0020$).
- **Within-Claim Top-1 Hit Rate**: Fuzzy reasoning selected an annotated gold passage as Rank 1 in **$73.00\%$** ($365 / 500$) of claims vs. **$69.00\%$** ($345 / 500$) for the baseline ($\Delta = +4.00\%$, $+20$ claims).
- **Secondary Ablation**: Evaluating the joint transition from static defaults ($c_{\text{deg}}=0.0, s_{\text{stab}}=1.0$) to dynamic inputs showed that the baseline's uniform multiplicative penalty forced **$20.80\%$ ($312 / 1,500$)** of scores identically to 0.0, collapsing baseline AUROC by $-0.1624$. The fuzzy engine's rule synergies mitigated this loss, dropping by only $-0.0316$.
- *Full Report*: [`EXPERIMENT_2_FORMAL_REPORT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_2_FORMAL_REPORT.md).

---

## 4. Key Documentation Index

| Document | Description |
| :--- | :--- |
| [`PHASE_13_FINAL_RESEARCH_SYNTHESIS.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/PHASE_13_FINAL_RESEARCH_SYNTHESIS.md) | **Primary Research Synthesis**: Complete consolidated research report, statistical tables, threats to validity, and paper-ready text. |
| [`docs/ARCHITECTURE.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/ARCHITECTURE.md) | Detailed architectural breakdown, component specifications, and data flow. |
| [`docs/DATASETS_AND_PROVENANCE.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/DATASETS_AND_PROVENANCE.md) | Provenance of FEVER dev split, construction of the 808-passage corpus, and annotation proxy caveats. |
| [`docs/EXPERIMENTS.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/EXPERIMENTS.md) | Experimental protocols, mathematical formulations, hypotheses, and formal verification results. |
| [`docs/REPRODUCIBILITY.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/REPRODUCIBILITY.md) | Instructions for reproducing tabular metrics (< 30 seconds) and end-to-end neural re-execution. |
| [`docs/LIMITATIONS_AND_FUTURE_WORK.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/docs/LIMITATIONS_AND_FUTURE_WORK.md) | Explicit threats to validity, methodological boundaries, and future research directions. |
| [`EXPERIMENT_2_POSTRUN_VALIDATION.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_2_POSTRUN_VALIDATION.md) | Independent audit report confirming all Experiment 2 point estimates, bootstrap inferences, and ablations. |

---

## 5. Quickstart & Verification

### 5.1 Environment Setup
```bash
# Clone and enter workspace
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Instant Verification (< 30 seconds)
To independently verify all primary metrics, bootstrap confidence intervals, p-values, Top-1 hit rates, and condition isolation directly from frozen tabular artifacts:
```bash
python scratch/verify_exp2.py
```

### 5.3 Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/main.py
```
Explore live claim verification, hybrid retrieval, SHAP/LIME attribution, counterfactual intervention, and reliability-weighted explanations interactively.

---

## 6. Dataset Caveats & Known Limitations

1. **Closed Candidate Pool**: The hybrid retriever operates over a pre-seeded, closed candidate pool of 808 Wikipedia passages containing all gold evidence for evaluated claims. The **86.00%** claim retrieval coverage reflects this closed setting and cannot be cited as open-domain retrieval performance.
2. **Annotation Proxy**: FEVER gold labels represent human annotators' minimal sufficient sets. Unannotated distractor passages are not necessarily false or irrelevant.
3. **Model Sensitivity vs. Real Causality**: Intervention influence ($\Delta P$) quantifies the cross-encoder's classification head sensitivity, not real-world factual causality.
4. **Context Window**: Fixed at $k = 3$ retrieved passages per claim.
