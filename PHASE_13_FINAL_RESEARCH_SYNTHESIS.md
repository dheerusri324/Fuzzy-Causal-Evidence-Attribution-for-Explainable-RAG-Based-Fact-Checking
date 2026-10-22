# Final Research Synthesis: Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking

**Project:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Phase:** 13 — Research Synthesis & Final Documentation  
**Status:** **COMPLETED & FROZEN**  
**Authoritative References:**
- Phase 10 Formal Report: [`EXPERIMENT_1_FORMAL_REPORT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_1_FORMAL_REPORT.md)
- Phase 11 Formal Report: [`EXPERIMENT_2_FORMAL_REPORT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_2_FORMAL_REPORT.md)
- Independent Validation: [`EXPERIMENT_2_POSTRUN_VALIDATION.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/EXPERIMENT_2_POSTRUN_VALIDATION.md)
- Dataset Audit: [`DATASET_OVERLAP_AUDIT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/DATASET_OVERLAP_AUDIT.md)
- Research Analysis: [`PHASE_12_RESEARCH_ANALYSIS.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/PHASE_12_RESEARCH_ANALYSIS.md)
- Errata V4: [`PRE_PHASE_14_AUDIT_ERRATA_V4.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/PRE_PHASE_14_AUDIT_ERRATA_V4.md)
- Sensitivity Audit: [`DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md)

> [!IMPORTANT]
> **Errata & Label-Semantics Notice (Post-Phase-13 Review)**:
> - **Proxy-Label Scheme Demarcation**: The 443 positive instances evaluated in the official frozen Experiment 2 run represent a legacy proxy-label scheme consisting of **296 genuine human-annotated gold evidence passages** (154 for `SUPPORTS` and 142 for `REFUTES`) and **147 unannotated candidate proxy passages** associated with `NOT_ENOUGH_INFO` (NEI) claims from `copenlu/fever_gold_evidence`. In the FEVER benchmark, NEI claims have no human-annotated gold evidence.
> - **Sensitivity Protocol Findings**: Full sensitivity audits isolating this distinction are documented in [`DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md`](DATASET_LABEL_SEMANTICS_SENSITIVITY_AUDIT.md) and [`PRE_PHASE_14_AUDIT_ERRATA_V4.md`](PRE_PHASE_14_AUDIT_ERRATA_V4.md):
>   - *Protocol A (500 claims; NEI candidate passages treated as negative distractors)*: $\text{AUROC}_{\text{fuzzy}} = 0.7386$ vs. $\text{AUROC}_{\text{base}} = 0.6488$ ($\Delta = +0.0898$, $p_{\text{Holm}} = 0.0020$).
>   - *Protocol B (334 verifiable claims; NEI claims excluded entirely)*: $\text{AUROC}_{\text{fuzzy}} = 0.7777$ vs. $\text{AUROC}_{\text{base}} = 0.6949$ ($\Delta = +0.0827$, $p_{\text{Holm}} = 0.0020$).
> - **Corrected Foundation Reference Citation**: The base paper citation is: Jean-Flavien Bussotti, Andrea Baraldi, Francesco Guerra, and Paolo Papotti, *"Explaining the Role of Evidence in Data-Driven Fact-Checking,"* *IEEE Access*, vol. 14, pp. 127752–127765, 2026, DOI: [10.1109/ACCESS.2026.3719082](https://doi.org/10.1109/ACCESS.2026.3719082).

---

## A. Research Motivation and Problem Statement

Automated fact-checking systems increasingly rely on **Retrieval-Augmented Generation (RAG)** and cross-encoder architectures to verify natural language claims against large-scale document collections. While these systems demonstrate strong classification performance, explaining *why* a specific verdict was reached remains an urgent challenge.

Current explainability frameworks predominantly rely on post-hoc feature attribution techniques, such as **SHAP** (Shapley Additive Explanations) and **LIME** (Local Interpretable Model-agnostic Explanations). However, deploying standard post-hoc attribution in multi-document fact-checking reveals critical vulnerabilities:
1. **Attribution vs. Counterfactual Influence**: Feature attribution highlights which input tokens or passages the neural network attended to, but does not establish whether the model's prediction counterfactually depends on that evidence. A piece of evidence may receive high attribution due to keyword matching while having negligible influence on the actual verdict.
2. **Conflation of Distinct Epistemic Dimensions**: In real-world verification, evidence quality is multidimensional. Existing systems frequently conflate **retrieval relevance** (query matching), **verifier sensitivity** (counterfactual influence), **mutual conflict** (cross-evidence contradiction), **explanation stability** (robustness to noise), and **source reliability** (external credibility). Treating these distinct concepts as a single scalar or simple linear sum fails to capture complex interactions, such as evidence that is highly relevant yet internally contradictory.

---

## B. Research Gap Addressed

This research program addresses two foundational gaps:

1. **The Attribution–Intervention Gap**:
   Feature attribution methods produce continuous importance weights over inputs, whereas decision-makers require knowing whether a verdict is counterfactually fragile. We formalize and empirically evaluate the degree of alignment—and discordance—between post-hoc attribution (SHAP/LIME) and leave-one-out counterfactual intervention influence ($\Delta P = |P(\hat{y} \mid E) - P(\hat{y} \mid E \setminus \{e_i\})|$).
2. **Non-Linear Evidence Reasoning Under Conflict**:
   When retrieved evidence exhibits internal contradictions, linear combination rules (e.g., multiplying scores by a uniform discount factor) uniformly collapse discrimination across all candidates. We investigate whether a **Mamdani Fuzzy Inference System** with synergistic linguistic rules can maintain graded evidence reliability and achieve superior ranking discrimination in the presence of contradictory evidence.

---

## C. Proposed System Architecture

The project implements a complete, modular, end-to-end fact-checking and explainability pipeline:

$$\text{Claim} \xrightarrow{\text{Hybrid RAG}} \text{Top-}k \text{ Evidence} \xrightarrow{\text{Cross-Encoder}} \text{Verdict } \hat{y} \xrightarrow{\text{SHAP/LIME}} \text{Attribution } a$$
$$\xrightarrow{\text{LOO Intervention}} \text{Influence } i \xrightarrow{\text{NLI Contradiction}} \text{Conflict } c_{\text{deg}} \xrightarrow{\text{Perturbation}} \text{Stability } s_{\text{stab}} \xrightarrow{\text{Mamdani Fuzzy Engine}} \text{Reliability-Aware Output}$$

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE DATA FLOW                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
 [Claim: c] ──► [Hybrid Retriever (BM25 + all-MiniLM-L6-v2)] ──► Top-3 Passages (E)
                       │
                       ▼
         [Cross-Encoder Verifier (DeBERTa-v3-base)] ──► Predicted Verdict & P(y|E)
                       │
        ┌──────────────┴──────────────┬──────────────────────────────┐
        ▼                             ▼                              ▼
 [SHAP & LIME Engine]    [LOO Intervention Engine]     [Conflict & Stability Engines]
  Token/Piece Weights     Influence ΔP & Verdict Flips  c_deg (NLI) & s_stab (Spearman)
        │                             │                              │
        └──────────────┬──────────────┴──────────────────────────────┘
                       ▼
        [Mamdani Fuzzy Evidence Reasoning Engine]
         (11 Rules across 5 Outputs, Centroid Defuzzification)
                       │
                       ▼
         [Reliability-Aware Explanation & UI Presentation]
```

---

## D. Experiment 1: Attribution vs. Counterfactual Intervention Alignment

### 1. Research Question
**RQ1:** *To what extent does post-hoc feature attribution reflect the empirical influence of evidence under counterfactual deletion intervention in a fact-checking cross-encoder?*

### 2. Evaluation Setup
- **Dataset**: First 500 claims from `data/fever_dev.jsonl`.
- **Candidate Pool**: 808-passage Wikipedia corpus (containing all 534 gold passages for evaluated claims).
- **Evaluated Units**: Exactly 1,500 evidence-removal evaluations ($k=3$ retrieved passages per claim).
- **Target Models**: Cross-encoder `cross-encoder/nli-deberta-v3-base`; SHAP partition explainer; LIME linear surrogate.

### 3. Verified Formal Findings

| Metric | Target / Description | SHAP | LIME |
| :--- | :--- | :---: | :---: |
| **Evaluated Claims** | Total claim count | 500 | 500 |
| **Evidence Removals** | Total intervention evaluations | 1,500 | 1,500 |
| **Top-1 Agreement** | Fraction where top attribution matches top $\Delta P$ | **72.20%** ($361 / 500$) | **84.00%** ($420 / 500$) |
| **Mean Per-Claim Spearman $\rho$** | Average rank correlation ($k=3$) | **0.4517** | **0.5877** |
| **Median Per-Claim Spearman $\rho$** | Median rank correlation ($k=3$) | **0.5000** | **0.5000** |
| **Verdict-Flip Rate** | Percentage of removals altering discrete verdict | **21.47%** ($322 / 1,500$) | **21.47%** ($322 / 1,500$) |
| **Flips on Top-1 Attribution** | Verdict flips on removing Top-1 attributed piece | $268 / 322$ ($83.23\%$) | $270 / 322$ ($83.85\%$) |
| **Flips on Non-Top Attribution** | Verdict flips on removing Rank 2 or 3 piece | $54 / 322$ ($16.77\%$) | $52 / 322$ ($16.15\%$) |

### 4. Interpretation
1. **Moderate-to-Strong Directional Correspondence**: Attribution correlates positively with counterfactual influence; LIME exhibits higher Top-1 agreement ($84.0\%$) and higher mean Spearman correlation ($0.5877$) than SHAP ($72.2\%$ and $0.4517$).
2. **The Attribution–Intervention Gap Exists**: In $16.0\%$ (LIME) and $27.8\%$ (SHAP) of claims, the top-attributed passage is not the most influential passage under intervention.
3. **Counterfactual Blind Spots**: In $16.15\%$ (LIME) and $16.77\%$ (SHAP) of verdict-altering interventions, the removed piece was ranked 2nd or 3rd by attribution. Explanations relying solely on Top-1 attribution fail to reveal these critical points of fragility.

### 5. Limitations
- With $k=3$, per-claim Spearman correlation is restricted to a limited, discrete set of attainable values.
- Leave-one-out removal measures single-piece sensitivity, ignoring multi-piece synergistic interactions.
- Measured influence reflects verifier model behavior, not real-world causality.

---

## E. Experiment 2: Fuzzy Evidence Reasoning vs. Linear Baseline

### 1. Research Question
**RQ2:** *Does fuzzy evidence-reliability scoring distinguish human-annotated gold evidence from unannotated distractor passages more effectively than a linear weighted baseline in a closed RAG setting, and what are the effects of dynamic contradiction and stability signals?*

### 2. Evaluated Scoring Methods
- **Fuzzy Primary Score**: `evidence_reliability` (Mamdani 11-rule inference system across 5 output variables with centroid defuzzification over $[0, 1]$).
- **Linear Baseline Primary Score**: `adjusted_baseline_score` computed in two stages:
  1. Additive weighted positive score incorporating stability:
     $$S_{\text{raw}} = 0.30 \cdot r + 0.25 \cdot a + 0.25 \cdot i + 0.10 \cdot s + 0.10 \cdot s_{\text{stab}}$$
  2. Multiplicative contradiction discounting:
     $$S_{\text{adj}} = S_{\text{raw}} \cdot (1 - c_{\text{deg}})$$
- **Secondary Diagnostic Scores**: `explanation_confidence` (fuzzy) and `baseline_raw_score` (baseline without contradiction discounting).

### 3. Sample Size & Annotation Proxy
- Exactly 500 claims, 1,500 evidence instances (443 annotated gold, 1,057 distractors).
- FEVER gold annotations represent extrinsic human annotators' minimal sufficient sets, functioning as a relevance proxy rather than an exhaustive real-world truth label.

### 4. Co-Primary Discrimination Results (Dynamic Condition)
Evaluated via a **Null-Centered Paired Claim-Level Cluster Bootstrap Test** ($B = 1,000$, seed 42) with **Holm–Bonferroni step-down correction**:

| Hypothesis | Metric | Fuzzy Estimate | Baseline Estimate | Difference ($\Delta$) | 95% Marginal CI | 97.5% Bonferroni CI | Raw $p$ | Holm Step Threshold | Holm Adjusted $p$ | Statistically Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_1$** | **AUROC** | **0.7630** | 0.7265 | **+0.0365** | $[+0.0053, +0.0686]$ | $[+0.0001, +0.0710]$ | 0.0240 | $\alpha / 1 = 0.050$ | **0.0240** | **YES** ($p_{\text{Holm}} < 0.05$, FWER $\alpha = 0.05$) |
| **$H_2$** | **Average Precision** | 0.4884 | **0.6280** | **-0.1396** | $[-0.1777, -0.1025]$ | $[-0.1856, -0.0988]$ | 0.0010 | $\alpha / 2 = 0.025$ | **0.0020** | **YES** ($p_{\text{raw}} \le 0.025$, FWER $\alpha = 0.05$) |

### 5. Within-Claim Top-1 Gold Hit Rates

| Metric | Target Population | Fuzzy Hits (Rate) | Baseline Hits (Rate) | Difference ($\Delta$) | Direct Agreement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Metric A** | All 500 Claims | **365 ($73.00\%$)** | 345 ($69.00\%$) | **+4.00%** (+20 claims) | **$437 / 500$ ($87.40\%$)** |
| **Metric B** | Gold-Present Claims ($N=430$) | **365 ($84.88\%$)** | 345 ($80.23\%$) | **+4.65%** (+20 claims) | — |

In $87.40\%$ of claims, both systems select the exact same top evidence piece. In the $12.60\%$ of claims where they disagree ($63$ claims), fuzzy reasoning selects the gold passage in 20 additional claims.

### 6. Dynamic vs. Static Ablation Findings
The secondary ablation evaluates the combined transition from schema default inputs ($c_{\text{deg}} = 0.0, s_{\text{stab}} = 1.0$) to dynamically computed signals.

| Condition | Metric | Fuzzy Score | Baseline Score | Difference ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Primary (Dynamic Full Pipeline)** | **AUROC** | **0.7630** | 0.7265 | **+0.0365** |
| Primary (Dynamic Full Pipeline) | **AP** | 0.4884 | **0.6280** | **-0.1396** |
| **Secondary Ablation (Static Defaults)** | **AUROC** | 0.7946 | **0.8889** | **-0.0943** |
| **Secondary Ablation (Static Defaults)** | **AP** | 0.5550 | **0.7600** | **-0.2051** |

- **Baseline Vulnerability Under Dynamic Contradiction**: The linear baseline's multiplicative penalty $(1 - c_{\text{deg}})$ forced **312 out of 1,500 scores ($20.80\%$) identically to 0.0** under dynamic conflict. This mass saturation collapsed baseline AUROC by **$-0.1624$** (from 0.8889 down to 0.7265).
- **Fuzzy Resilience**: The Mamdani rule synergies avoid catastrophic collapse, dropping by only **$-0.0316$** (from 0.7946 to 0.7630), surpassing the baseline under conflict.
- **Centroid Compression Trade-Off**: Centroid defuzzification over overlapping sets naturally compresses fuzzy scores to $[0.1341, 0.8083]$, which dampens precision at extreme thresholds (lower Average Precision) while preserving superior global ranking (higher AUROC and higher Top-1 hit rates).

### 7. Exploratory Attribution-Gap Subgroup Findings
- **Pre-Registered Membership Subgroups ($S_{\text{gap}}^{\text{std}}, S_{\text{gap}}^{\text{ext}}$)**: Contain **$N = 0$ instances**. In this FEVER dataset, high attribution strength ($a \ge 0.60$) was always accompanied by high intervention influence ($i \ge 0.98$). No empirical effect could be estimated for the membership criteria.
- **Exploratory Rank-Divergence Subgroup ($S_{\text{gap}}^{\text{rank}}$, $N = 87$ instances, $5.80\%$)**:
  - `explanation_confidence` median: 0.5000 (IQR 0.0000).
  - `baseline_raw_score` median: 0.3352 (IQR 0.2081).
  - Mean within-claim ranks: Fuzzy 2.23 vs. Baseline 2.30.
  - Relative ranks tied in **$88.51\%$** of instances.
  - Labeled strictly as **exploratory descriptive only**.

---

## F. Combined Research Discussion

When viewed together, Experiments 1 and 2 provide complementary evidence:
1. **Experiment 1 establishes the necessity of multi-faceted evaluation**: Post-hoc attribution alone cannot guarantee causal necessity under intervention. Relying exclusively on SHAP or LIME leaves up to $27.8\%$ of claims with misidentified top evidence and misses $16.5\%$ of fatal verdict flips.
2. **Experiment 2 demonstrates the trade-off in synthesis**: Combining attribution, intervention, contradiction, and stability into a single scalar score is fundamentally sensitive to the aggregation algebra. A linear baseline with multiplicative discounting performs well in tranquil, uncontradicted settings, but collapses when real-world evidence exhibits contradiction. The non-linear fuzzy reasoning engine provides superior ranking resilience across thresholds and higher Top-1 gold selection, at the cost of compressing extreme scores.

---

## G. Threats to Validity & Methodological Boundaries

The scientific conclusions of this research are bounded by the following threats to validity:

1. **Gold-Conditioned Closed Candidate Pool**:
   Retrieval operated over an 808-passage corpus constructed from the gold sentences of the 750 FEVER dev claims. The 86.00% Top-3 retrieval coverage reflects this closed setting and cannot be extrapolated to open-domain Wikipedia search.
2. **Cluster Dependence**:
   Each claim contributed exactly $k=3$ retrieved passages. Observations within a claim share common verifier logits and claim semantics. This dependence was controlled via cluster-level bootstrap resampling, but remains an inherent property of multi-passage evaluation.
3. **Limited Context ($k = 3$)**:
   The small context window limits the exploration of complex multi-document reasoning chains and makes per-claim rank metrics discrete.
4. **FEVER Gold Annotations as Relevance Proxy**:
   Gold labels indicate human annotators' minimal sufficient evidence sets, not an objective real-world truth or broad source credibility label. Distractor passages are not necessarily false.
5. **Leave-One-Out Intervention Scope**:
   LOO ablation evaluates single-evidence removal in isolation and cannot detect combinatorial synergies or redundancies between multiple passages.
6. **Verifier Sensitivity vs. Real-World Causality**:
   Intervention measures the sensitivity of the DeBERTa classification head. It does not prove that the real-world factual claim causally depends on that evidence piece.
7. **Exploratory Status of Subgroups**:
   Membership-based gap criteria yielded zero instances; rank-divergence findings are strictly exploratory.

---

## H. Contributions: Engineering vs. Empirical Findings

To preserve strict scientific discipline, we explicitly separate engineering contributions from empirically validated research claims:

### Engineering Contributions
- Developed a modular, reproducible, end-to-end fact-checking pipeline integrating hybrid RAG, cross-encoder NLI, SHAP, LIME, LOO intervention, pairwise conflict detection, perturbation stability, and Mamdani fuzzy inference.
- Implemented an interactive Streamlit UI ([`app/main.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/app/main.py)) providing live verification, ranked attribution, and reliability breakdowns.
- Created an automated, zero-model tabular verification suite ([`scratch/verify_exp2.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/scratch/verify_exp2.py)) reproducing all formal statistical results in $< 30$ seconds.

### Empirically Supported Research Findings
- **Attribution–Intervention Divergence**: Post-hoc attribution (SHAP/LIME) diverges from counterfactual intervention influence in $16.0\%$ to $27.8\%$ of claims, and fails to identify the removed piece in $16.5\%$ of verdict flips.
- **AUROC vs. AP Trade-Off**: Under dynamic conflict, fuzzy reasoning achieves statistically significant superior AUROC ($\Delta = +0.0365, p = 0.0240$), whereas the linear baseline achieves statistically significant superior Average Precision ($\Delta = -0.1396, p = 0.0020$).
- **Top-1 Gold Hit Rate Advantage**: Fuzzy reasoning yields a $+4.00\%$ net improvement in Top-1 gold passage selection across 500 claims ($73.00\%$ vs. $69.00\%$).
- **Multiplicative Penalty Fragility**: Linear baselines applying uniform multiplicative penalties suffer catastrophic saturation ($20.80\%$ zero-scores) under evidence contradiction, dropping AUROC by $-0.1624$, whereas Mamdani fuzzy rules mitigate this degradation.

*(We do **not** claim that every pipeline component independently improves performance, nor do we claim universal superiority of fuzzy logic or LIME).*

---

## I. Conclusion and Future Work

### Conclusion
This project establishes that post-hoc feature attribution is an incomplete proxy for evidence necessity in automated fact-checking, and that non-linear fuzzy reasoning offers a principled, robust mechanism to aggregate multi-attribute evidence signals under conflict without catastrophic score collapse.

### Future Work
1. **Open-Domain Scaling**: Testing the pipeline on unconstrained, multi-million document corpora (e.g., KILT, full Wikipedia).
2. **Multi-Hop Benchmarks**: Evaluating combinatorial multi-piece interventions on multi-hop benchmarks (HoVer, FEVEROUS).
3. **Real-World Source Metadata**: Replacing uniform source reliability with live publisher credibility ratings, domain authority, and cryptographic content credentials.
4. **Alternative Defuzzification**: Investigating Sugeno fuzzy systems or calibrated non-linear scaling to improve precision-recall concentration without losing global AUROC gains.

---

## J. Paper-Ready Findings (Draft Abstracts & Paragraphs)

### Abstract Paragraph:
> Post-hoc feature attribution methods such as SHAP and LIME are widely employed to explain neural fact-checking models, yet they quantify input attention rather than counterfactual necessity. In a formal evaluation of 500 FEVER claims ($1,500$ evidence-removal evaluations) using a DeBERTa-v3 cross-encoder, we show that attribution diverges from leave-one-out intervention influence in $16.0\%$ (LIME) to $27.8\%$ (SHAP) of cases, with $16.5\%$ of verdict-altering interventions occurring on non-top-attributed evidence. To synthesize relevance, attribution, intervention influence, stability, and mutual conflict, we evaluate a Mamdani Fuzzy Evidence Reasoning Engine against a linear weighted baseline. Under dynamic conflict, the fuzzy engine achieves a statistically significant improvement in overall ranking discrimination ($\text{AUROC} = 0.7630$ vs. $0.7265$, $\Delta = +0.0365$, $p = 0.0240$) and increases within-claim Top-1 gold-evidence hit rate from $69.00\%$ to $73.00\%$ ($+20$ claims), while the baseline achieves higher precision concentration ($\text{AP} = 0.6280$ vs. $0.4884$, $\Delta = -0.1396$, $p = 0.0020$). An ablation study reveals that linear multiplicative discounting causes $20.80\%$ of baseline scores to saturate at zero under contradiction, whereas fuzzy rule synergies preserve graded discrimination.

### Results Excerpt (AUROC vs. AP Trade-Off):
> Across 1,500 evaluated passages, the choice of aggregation algebra reveals a fundamental structural trade-off. Evaluated via a null-centered paired claim-level cluster bootstrap ($B = 1,000$, seed 42) with Holm–Bonferroni correction (FWER $\alpha = 0.05$), the fuzzy engine's primary score (`evidence_reliability`) attained significantly higher AUROC than the baseline (`adjusted_baseline_score`): $\Delta = +0.0365$ ($95\%\text{ CI } [+0.0053, +0.0686]$, raw $p = 0.0240$, Holm $p = 0.0240$). Conversely, the baseline achieved significantly higher Average Precision: $\Delta = -0.1396$ ($95\%\text{ CI } [-0.1777, -0.1025]$, raw $p = 0.0010$, Holm $p = 0.0020$). In within-claim Top-1 evidence selection, the two systems agreed in $87.40\%$ of claims ($437 / 500$); in the remaining $12.60\%$ of claims, fuzzy reasoning selected an annotated gold passage in 20 additional claims, demonstrating practical utility for top-ranked evidence presentation.
