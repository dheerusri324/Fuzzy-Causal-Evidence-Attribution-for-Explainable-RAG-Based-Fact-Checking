# Corrected Formal Evaluation Report: Fuzzy Reasoning vs. Linear Baseline (Phase 11 V2)

**Execution Date:** 2026-10-22 19:49:03<br>
**Pipeline Revision:** V2 Corrected (33-rule Mamdani engine, harmonic mean conflict, explicit ID alignment)<br>
**Execution Mode:** Controlled Follow-Up Evaluation with Frozen Attribution and Live Conflict Sensing<br>
**Run Manifest:** [`run_manifest.json`](file:///C:/Users/dheer/OneDrive/Documents/Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking/outputs/phase11_v2_corrected/run_manifest.json)

## 1. Executive Summary & Evaluation Protocols

This report presents the corrected evaluation of the V2 Fuzzy Evidence Reasoning pipeline against the Linear Weighted Baseline. To resolve historical ambiguities regarding dataset labeling, results are evaluated under two explicitly named, distinct protocols:

- **Protocol A (All 500 Claims; NEI as Negative Distractors)**:
  Evaluates all 500 claims (1,500 evidence rows). For `SUPPORTS` and `REFUTES`, positive labels correspond strictly to genuine human-annotated gold evidence (`rec.verified_gold_evidence`). For `NOT_ENOUGH_INFO` (NEI) claims (166 claims, 498 rows), all retrieved candidate passages are marked as negative distractors ($y=0$). Total: **296 positive instances** and **1,204 negative instances**.

- **Protocol B (Verifiable Claims Only)**:
  Restricts evaluation to verifiable claims (`rec.is_verifiable`), completely excluding the 166 NEI claims. Total: **334 claims, 1,002 evidence rows (296 positive instances, 706 negative instances)**.

### Technical Integrity & Remediation Impact

| Dimension | Historical V1 | Remediated V2 | Status |
| :--- | :--- | :--- | :--- |
| **Fallback Rate** | 59.07% (886/1,500) | **0.00% (0/1,500)** | Full defuzzification coverage across all rows |
| **Conflict Scoring** | Min-max scaled (saturated) | **Continuous Harmonic Mean** | Smooth variation; 0 rows saturated at 1.0 |
| **ID Alignment** | Positional indexing (`c_rows[i]`) | **Explicit `(claim_id, evidence_id)` matching** | 100% verified alignment |
| **Bootstrap P-Values** | Uncorrected empirical ($0.0$) | **Finite-simulation corrected `(k+1)/(B+1)`** | Statistically valid resolution |

## 2. Quantitative Results: Protocol A (Primary Evaluation)

| Endpoint | Fuzzy V2 | Baseline | Difference (Fuzzy - Base) | 95% Bootstrap CI | Raw $p$ | Holm Adjusted $p$ | Result Favors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | 0.6906 | 0.6879 | **+0.0026** | [-0.0219, +0.0279] | 0.8282 | 0.8282 | **Fuzzy** |
| **Average Precision** | 0.4495 | 0.5294 | **-0.0799** | [-0.1136, -0.0390] | 0.0020 | 0.0040 | **Baseline** |

- **Spearman Rank Correlation $\rho$ with Gold**: Fuzzy $\rho = 0.2796$ vs. Baseline $\rho = 0.2591$
- **Top-1 Gold Hit Rate (All 500 Claims)**: Fuzzy = 209/500 (41.80%) vs. Baseline = 173/500 (34.60%)
- **Direct Top-1 Agreement**: 298/500 claims (59.60%) select the exact same Top-1 evidence passage.

## 3. Quantitative Results: Protocol B (Sensitivity Analysis on Verifiable Claims)

| Endpoint | Fuzzy V2 | Baseline | Difference (Fuzzy - Base) | 95% Bootstrap CI | Raw $p$ | Holm Adjusted $p$ | Result Favors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | 0.7163 | 0.7121 | **+0.0042** | [-0.0222, +0.0300] | 0.7493 | 0.7493 | **Fuzzy** |
| **Average Precision** | 0.6110 | 0.6552 | **-0.0442** | [-0.0775, -0.0120] | 0.0120 | 0.0240 | **Baseline** |

- **Spearman Rank Correlation $\rho$ with Gold**: Fuzzy $\rho = 0.3669$ vs. Baseline $\rho = 0.3352$
- **Top-1 Gold Hit Rate (334 Verifiable Claims)**: Fuzzy = 209/334 (62.57%) vs. Baseline = 173/334 (51.80%)
- **Direct Top-1 Agreement**: 201/334 claims (60.18%) select the exact same Top-1 evidence passage.

## 4. Honest Scientific Interpretation

1. **Discrimination vs. Precision Divergence**:
   Under Protocol A, AUROC difference is +0.0026 (favoring Fuzzy) and AP difference is -0.0799 (favoring Baseline).
   Under Protocol B, AUROC difference is +0.0042 (favoring Fuzzy) and AP difference is -0.0442 (favoring Baseline).

2. **Epistemic Limitations**:
- **Attribution is Not Causal Truth**: Attribution scores (SHAP/LIME) and intervention influences measure verifier sensitivity to evidence perturbation; they do not establish that evidence causally produces real-world truth.
- **Fuzzy Reliability is an Utility Index**: Fuzzy evidence reliability reflects designed operational heuristics (relevance, attribution, absence of contradiction, stability), not calibrated posterior probabilities of factual truth.
- **Closed Candidate Pool Scope**: All evaluations are conditioned on a curated 808-passage candidate pool, not unrestricted open-domain Wikipedia retrieval.
