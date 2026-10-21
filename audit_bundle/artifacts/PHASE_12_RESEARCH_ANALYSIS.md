# Phase 12 — Research Analysis: Evidence Attribution vs. Intervention Influence

## Executive Summary

This research analysis report provides an empirical evaluation of **Experiment 1 (Formal)**, conducted on 500 claims from the FEVER development dataset. The experiment investigates the central research question:

> **"When post-hoc feature attribution methods (SHAP and LIME) identify an evidence piece as influential in a claim-verification decision, does removing that same evidence piece produce a measurable change in the verifier's predicted decision?"**

All underlying models, configurations, and raw data are mathematically frozen from Phase 10. This phase conducts purely analytical, statistical, and interpretive verification on the generated results without altering the experimental pipeline.

---

## 1. Formal Result Verification & Retrieval Corpus Audit

### 1.1 Result Counts Verification

Every metric reported in `EXPERIMENT_1_FORMAL_REPORT.md` was programmatically recomputed directly from the 1,500 raw intervention rows in `outputs/phase10_formal/results.csv`. All counts and rates were validated with exact precision:

| Metric | Recorded in Formal Report | Recomputed from Raw CSV | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Claims Processed** | 500 | 500 | **Verified (100% Match)** |
| **Total Evidence Evaluations (Interventions)** | 1,500 | 1,500 | **Verified (100% Match)** |
| **Evidence per Claim ($k$)** | 3 | 3 | **Verified (100% Match)** |
| **Total Verdict Flips** | 322 | 322 | **Verified (100% Match)** |
| **Verdict-Flip Rate** | 21.5% | 21.47% (322/1500) | **Verified (100% Match)** |
| **Valid SHAP Correlations ($\rho$)** | 500 | 500 | **Verified (100% Match)** |
| **Undefined SHAP Correlations** | 0 | 0 | **Verified (100% Match)** |
| **Valid LIME Correlations ($\rho$)** | 500 | 500 | **Verified (100% Match)** |
| **Undefined LIME Correlations** | 0 | 0 | **Verified (100% Match)** |
| **SHAP Top-1 Agreement Count** | 361 | 361 (72.20%) | **Verified (100% Match)** |
| **LIME Top-1 Agreement Count** | 420 | 420 (84.00%) | **Verified (100% Match)** |

### 1.2 Retrieval Corpus Size and Source Verification

An audit of the retrieval subsystem, corpus files, and data preparation pipeline (`retrieval/retriever.py`, `data/fever_loader.py`, `data/dataset_meta.json`, and `data/evidence_corpus.json`) establishes the empirical grounding of the retrieval stage:

- **Evaluated Claims**: 500 claims, taken as the first 500 records from `data/fever_dev.jsonl` (which contains 750 stratified claims).
- **Available Evidence Corpus Size**: Exactly **808 unique evidence passages** in `data/evidence_corpus.json`.
- **Corpus Source**: Extracted via `extract_evidence_corpus(dev_subset)` in `data/fever_loader.py` from the Hugging Face dataset `copenlu/fever_gold_evidence`. It comprises the deduplicated evidence sentences associated with the 750 dev claims.
- **Retrieved Evidence Instances Evaluated**: **1,500 evidence instances** ($500 \text{ claims} \times 3 \text{ retrieved pieces}$).
- **Corpus Scope (Restricted Subset vs. Full Corpus)**:
  Retrieval searched a **curated, restricted evaluation corpus of 808 unique evidence passages**, NOT the full multi-million-article Wikipedia dump or full open-domain FEVER database. 
  
> [!IMPORTANT]
> **Corpus Scope Limitation**: The retrieval module operated over a closed evaluation candidate pool of 808 passages rather than an open-domain web or full-Wikipedia corpus. Consequently, the experiment evaluates evidence ranking, attribution, and intervention sensitivity within a focused candidate set rather than large-scale document discovery.

### 1.3 Data Integrity and Anomaly Investigation

A complete audit of all 1,500 evidence rows was executed to detect any data corruption, boundary violations, or structural defects:

1. **Probability Bounds**:
   - `baseline_target_probability`: Min = $0.3444$, Max = $0.9999$. All values lie strictly in $[0.0, 1.0]$.
   - `intervened_target_probability`: Min = $0.0001$, Max = $0.9999$. All values lie strictly in $[0.0, 1.0]$.
   - No NaN, null, or out-of-bounds probabilities exist.

2. **Rank Integrity**:
   - Every claim contains exactly three evidence pieces ($k=3$).
   - Ranks for `SHAP_magnitude_rank`, `LIME_magnitude_rank`, and `intervention_rank` are strictly in $\{1, 2, 3\}$.
   - Handled with `scipy.stats.rankdata(..., method='min')`. Ties were assigned the minimum rank of the tied group. Exactly 1 claim exhibited a tied magnitude for Rank 1, producing Rank 1 ties and skipping Rank 2. No unexpected ranks occurred.

3. **Evidence Identifier Uniqueness**:
   - In all 500 claims, the three retrieved evidence IDs within each claim are distinct. There are 0 duplicate evidence IDs per claim.

4. **Missing Values**:
   - Zero missing, null, or empty string values across all 17 schema columns.

5. **Numerical Ranges**:
   - Raw SHAP scores: $[-0.9934, +0.9980]$
   - Raw LIME scores: $[-0.3547, +0.2682]$
   - SHAP magnitude $| \text{SHAP} |$: $[0.0000, 0.9990]$
   - LIME magnitude $| \text{LIME} |$: $[0.0000, 0.2682]$
   - Intervention influence $\Delta P$: $[0.0000, 0.9994]$
   - Verdict flip flag is strictly boolean (`True` or `False`).

---

## 2. Primary Summary Table

Below is the consolidated summary of Experiment 1 evaluating post-hoc evidence attribution against empirical leave-one-out intervention influence:

| Metric Category | Metric Name | Value | Unit / Format | Methodological Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Sample Size & Corpus** | Evaluated Claims | **500** | Claims | First 500 claims of FEVER dev split |
| | Evidence Corpus Size | **808** | Passages | Closed candidate pool (`copenlu/fever_gold_evidence`) |
| | Retrieval Budget ($k$) | **3** | Evidence / claim | Top-3 hybrid BM25 + dense retrieval |
| | Evidence Evaluations | **1,500** | Removals | Total leave-one-out (LOO) interventions ($500 \times 3$) |
| **Retrieval Coverage (Top-3)**| All Annotated Gold IDs Retrieved | **81.80%** | 409 / 500 claims | Full gold coverage ($N=500$ non-empty gold claims) |
| | Partial Gold Coverage | **4.20%** | 21 / 500 claims | At least 1, but not all, gold IDs retrieved in Top-3 |
| | Zero Gold Coverage | **14.00%** | 70 / 500 claims | Annotated gold exists, but 0 matches in Top-3 |
| | At Least One Gold ID Retrieved | **86.00%** | 430 / 500 claims | Cumulative coverage ($409 \text{ full} + 21 \text{ partial} = 430$) |
| **Direct Attribution Agreement**| Direct SHAP–LIME Top-1 Agreement | **82.40%** | 412 / 500 claims | Both methods select the identical evidence ID for rank #1 |
| | Direct SHAP–LIME Top-1 Disagreement| **17.60%** | 88 / 500 claims | Methods select different evidence IDs for rank #1 |
| **Top-1 Agreement (Intervention)**| SHAP Top-1 Agreement | **72.20%** | 361 / 500 claims | $| \text{SHAP} |$ top rank matches max $\Delta P$ |
| | LIME Top-1 Agreement | **84.00%** | 420 / 500 claims | $| \text{LIME} |$ top rank matches max $\Delta P$ |
| | Consensus (Both Agree with Infl) | **70.80%** | 354 / 500 claims | Both attribution methods match max $\Delta P$ |
| | LIME-Only Agreement with Infl | **13.20%** | 66 / 500 claims | LIME matches max $\Delta P$, SHAP does not |
| | SHAP-Only Agreement with Infl | **1.40%** | 7 / 500 claims | SHAP matches max $\Delta P$, LIME does not |
| | Total Disagreement with Infl | **14.60%** | 73 / 500 claims | Neither method identifies max $\Delta P$ (58 agree, 15 disagree) |
| **Per-Claim Correlation** | SHAP Mean Spearman $\rho$ | **0.4517** | Correlation $[-1, 1]$ | Arithmetic mean across 500 valid claims |
| | SHAP Median Spearman $\rho$ | **0.5000** | Correlation $[-1, 1]$ | 50th percentile across claims |
| | LIME Mean Spearman $\rho$ | **0.5877** | Correlation $[-1, 1]$ | Arithmetic mean across 500 valid claims |
| | LIME Median Spearman $\rho$ | **0.5000** | Correlation $[-1, 1]$ | 50th percentile across claims |
| **Intervention Dynamics**| Overall Verdict-Flip Rate | **21.47%** | 322 / 1,500 removals | Individual removals that flipped predicted verdict |
| | Affected Claims ($\ge 1$ Flip) | **50.60%** | 253 / 500 claims | Claims with at least one verdict-changing evidence piece |

---

## 3. Statistical Distribution of Per-Claim Spearman Correlation ($\rho$)

### 3.1 Parametric and Non-Parametric Descriptives

Because intervention influence is evaluated per claim across retrieved evidence, Spearman's rank correlation coefficient $\rho$ was calculated independently for each claim between absolute attribution magnitude ($| \text{SHAP} |$, $| \text{LIME} |$) and intervention influence $\Delta P = | P_{\text{base}} - P_{\text{intervene}} |$.

| Statistic | SHAP vs. Influence ($\rho_{\text{SHAP}}$) | LIME vs. Influence ($\rho_{\text{LIME}}$) | Notes |
| :--- | :---: | :---: | :--- |
| **Valid Observations ($N$)** | 500 | 500 | Complete evaluation across all claims |
| **Undefined / Excluded ($NaN$)** | 0 | 0 | Non-zero variance in all claims |
| **Mean** | **0.4517** | **0.5877** | Sample arithmetic average |
| **Standard Deviation ($s$, ddof=1)** | **0.6397** | **0.5565** | Sample dispersion |
| **Standard Deviation ($\sigma$, ddof=0)**| 0.6391 | 0.5559 | Population dispersion |
| **Minimum** | **-1.0000** | **-1.0000** | Complete inverse ranking |
| **First Quartile ($Q_1$, 25%)** | **+0.5000** | **+0.5000** | Lower quartile |
| **Median ($Q_2$, 50%)** | **+0.5000** | **+0.5000** | Central tendency |
| **Third Quartile ($Q_3$, 75%)** | **+1.0000** | **+1.0000** | Upper quartile |
| **Maximum** | **+1.0000** | **+1.0000** | Perfect rank alignment |
| **Interquartile Range (IQR)** | **0.5000** | **0.5000** | Spread of central 50% |
| **Positive Correlations ($\rho > 0$)** | **78.80%** (394 / 500) | **87.00%** (435 / 500) | Proportion with positive rank alignment |

### 3.2 Discrete Nature of Spearman Correlation with Small Evidence Sets

> [!NOTE]
> **Mathematical Interpretation of Within-Claim Correlation**:
> With only three evidence pieces per claim, per-claim Spearman correlation takes values from a limited, discrete set. The attainable values depend on the rank patterns and the presence of ties. Therefore, these correlations should be interpreted as coarse measures of within-claim rank agreement.

For three evidence items without ties, there are $3! = 6$ possible permutations, producing values such as $+1.0000$ (identical ordering), $+0.5000$ (single transposition), $-0.5000$ (cyclic shift), and $-1.0000$ (complete inverse ordering). When ties in magnitude occur, intermediate values such as $+0.8660$ ($\frac{\sqrt{3}}{2}$) arise.

The empirical distribution of observed correlation values across the 500 claims was:

```
Observed Per-Claim Spearman Correlation Breakdown (N = 500):

Observed ρ Value        SHAP Claims (%)        LIME Claims (%)        Observed Rank Configuration
-------------------------------------------------------------------------------------------------
+1.0000                 202  (40.4%)           243  (48.6%)           Identical rank ordering
+0.8660                   1   (0.2%)             1   (0.2%)           Tied ranking configuration
+0.5000                 191  (38.2%)           191  (38.2%)           Single transposition / partial swap
-0.5000                  67  (13.4%)            39   (7.8%)           Cyclic shift / primary swap
-1.0000                  39   (7.8%)            26   (5.2%)           Complete inverse ordering
-------------------------------------------------------------------------------------------------
Positive Total (ρ > 0)  394  (78.8%)           435  (87.0%)           Positive rank concordance
Negative Total (ρ < 0)  106  (21.2%)            65  (13.0%)           Negative rank concordance
```

![Distribution of Per-Claim Spearman Correlation](file:///outputs/phase12_analysis/spearman_rho_distribution.png)

---

## 4. Top-1 Agreement and Attribution Consensus

While Spearman correlation measures the full ordering across all three evidence pieces, **Top-1 Agreement** assesses whether the single most important evidence piece identified by an attribution method corresponds to the piece whose removal causes the largest shift in verifier confidence.

### 4.1 Agreement Breakdown

- **SHAP Top-1 Agreement**: **72.20%** ($361 / 500$ claims)
- **LIME Top-1 Agreement**: **84.00%** ($420 / 500$ claims)

### 4.2 Cross-Method Overlap

Analyzing the intersection of SHAP and LIME with intervention influence isolates where the methods agree or diverge:

```
                          Intervention Top-1 Match
                           /                   \
                 SHAP Matches (361)       SHAP Fails (139)
                  /             \           /             \
        LIME Matches     LIME Fails    LIME Matches    LIME Fails
           (354)            (7)           (66)            (73)
         [70.8%]           [1.4%]        [13.2%]         [14.6%]
```

1. **High-Confidence Consensus ($70.8\%$, 354 claims)**: Both SHAP and LIME identify the evidence piece that has the highest intervention influence.
2. **LIME-Specific Alignment ($13.2\%$, 66 claims)**: LIME correctly identifies the top intervention-influence piece, whereas SHAP ranks a secondary piece highest.
3. **SHAP-Specific Alignment ($1.4\%$, 7 claims)**: SHAP matches the top intervention-influence piece, whereas LIME selects a secondary piece.
4. **Attribution Blind Spot ($14.6\%$, 73 claims)**: Neither SHAP nor LIME places the highest attribution on the evidence piece whose removal produces the greatest impact on the verifier (in 58 of these claims, SHAP and LIME agree with each other on the wrong piece; in 15 claims, they also disagree with each other).

### 4.3 Direct Pairwise SHAP–LIME Top-1 Comparison

When directly comparing the Top-1 evidence piece selected by $| \text{SHAP} |$ against that selected by $| \text{LIME} |$ across the 500 claims (with zero ties at Rank 1 for either method):
- **Direct Top-1 Agreement**: **412 claims (82.40%)**
  Both $| \text{SHAP} |$ and $| \text{LIME} |$ assigned Rank 1 to the exact same evidence piece.
- **Direct Top-1 Disagreement**: **88 claims (17.60%)**
  $| \text{SHAP} |$ and $| \text{LIME} |$ selected different evidence pieces for Rank 1.

> [!NOTE]
> **Reconciliation**: The **13.20%** figure ($66 / 500$ claims) describes the specific subset of claims where **LIME alone agrees with the empirical intervention while SHAP does not**. The **direct pairwise Top-1 disagreement between SHAP and LIME** across all evaluated claims is **17.60%** ($88 / 500$ claims).

![Top-1 Agreement Breakdown](file:///outputs/phase12_analysis/top1_agreement_breakdown.png)

---

## 5. Verdict-Flip Dynamics and Intervention Sensitivity

A verdict flip occurs when removing an individual evidence piece changes the verifier's predicted discrete classification (among `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO`).

### 5.1 Removal Sensitivity and Clarified Denominators

> [!IMPORTANT]
> **Clarification of Denominators**:
> Class-specific flip rates are calculated over **individual leave-one-out evidence-removal evaluations grouped by baseline verifier verdict**, not over distinct claims. 
> Because each claim has $k=3$ evidence pieces, the number of removal evaluations in any class is exactly $3 \times \text{number of claims}$ in that class.

- **Total Interventions Tested**: 1,500 leave-one-out evidence removals across 500 claims
- **Total Verdict Flips**: 322 removals (**21.47%** of all 1,500 removals)
- **Total Non-Flips**: 1,178 removals (**78.53%** of all 1,500 removals)
- **Distinct Claims with $\ge 1$ Verdict Flip**: 253 claims (**50.60%** of all 500 claims)
- **Distinct Claims with 0 Verdict Flips**: 247 claims (**49.40%** of all 500 claims)

### 5.2 Verdict Flips per Claim Distribution

| Number of Flips in Claim | Number of Claims | Percentage of Claims | Interpretation |
| :---: | :---: | :---: | :--- |
| **0 flips** | 247 | 49.40% | Robust claim; verdict preserved under any single removal |
| **1 flip** | 195 | 39.00% | Single critical evidence piece governs the verdict |
| **2 flips** | 47 | 9.40% | Dual-critical evidence dependency |
| **3 flips** | 11 | 2.20% | Highly sensitive decision; removing any piece flips the verdict |

### 5.3 Class-Specific Sensitivity (Claims vs. Removal Evaluations)

To eliminate any reporting ambiguity, both the claim-level distribution and the removal-level evaluations are documented:

| Baseline Predicted Class | Claims in Group ($N$) | Claim Group Share (%) | Claims with $\ge 1$ Flip ($N$) | Claim Flip Prevalence (%) | Removal Evaluations ($N \times 3$) | Verdict Flips ($N$) | Removal-Level Flip Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SUPPORTS** | 154 | 30.80% | 135 | **87.66%** | 462 | 172 | **37.23%** (172 / 462) |
| **NOT_ENOUGH_INFO** | 79 | 15.80% | 48 | **60.76%** | 237 | 69 | **29.11%** (69 / 237) |
| **REFUTES** | 267 | 53.40% | 70 | **26.22%** | 801 | 81 | **10.11%** (81 / 801) |
| **Total / Overall** | **500** | **100.00%** | **253** | **50.60%** | **1,500** | **322** | **21.47%** (322 / 1,500) |

```
Full Intervention Transition Matrix (1,500 Removals):

Baseline Class        Intervened Class       Count    Percent of Total
----------------------------------------------------------------------
REFUTES            -> REFUTES (Stable)        720          48.00%
SUPPORTS           -> SUPPORTS (Stable)       290          19.33%
NOT_ENOUGH_INFO    -> NOT_ENOUGH_INFO (Stable)168          11.20%
SUPPORTS           -> REFUTES (Flip)          114           7.60%
REFUTES            -> NOT_ENOUGH_INFO (Flip)   68           4.53%
NOT_ENOUGH_INFO    -> REFUTES (Flip)           63           4.20%
SUPPORTS           -> NOT_ENOUGH_INFO (Flip)   58           3.87%
REFUTES            -> SUPPORTS (Flip)          13           0.87%
NOT_ENOUGH_INFO    -> SUPPORTS (Flip)           6           0.40%
----------------------------------------------------------------------
Total Removals                               1,500         100.00%
```

![Verdict Flip Dynamics](file:///outputs/phase12_analysis/verdict_flip_dynamics.png)

---

## 6. Stratified Intervention Influence by Attribution Rank

To evaluate whether higher attribution ranks correspond to higher empirical influence, we measure the intervention influence $\Delta P$ stratified by attribution rank:

| Attribution Rank | SHAP Mean $\Delta P$ | SHAP Median $\Delta P$ | LIME Mean $\Delta P$ | LIME Median $\Delta P$ | Intervention Rank Mean $\Delta P$ (Reference) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Rank 1 (Highest)** | **0.450** | **0.241** | **0.487** | **0.443** | 0.487 |
| **Rank 2 (Middle)** | **0.120** | **0.002** | **0.109** | **0.001** | 0.113 |
| **Rank 3 (Lowest)** | **0.064** | **0.001** | **0.039** | **0.000** | 0.034 |

![Intervention Influence by Attribution Rank](file:///outputs/phase12_analysis/influence_by_attribution_rank.png)

Both SHAP and LIME concentrate the vast majority of intervention influence in their Rank 1 evidence pieces. When evidence is ranked first by SHAP or LIME, its removal induces an average probability shift of $0.450$ to $0.487$, compared to $< 0.120$ for Rank 2 and $< 0.065$ for Rank 3.

---

## 7. Methodological Limitations

1. **Restricted Evidence Corpus Scope & Candidate Pre-Seeding**:
   As established in Section 1.2 and the dedicated Dataset Overlap Audit (`DATASET_OVERLAP_AUDIT.md`), the 808-passage retrieval corpus was constructed by pooling evidence annotations from the 750 dev split records (`copenlu/fever_gold_evidence`). Consequently, all 534 gold evidence passages for the 500 evaluated claims are pre-seeded within this 808-passage candidate pool alongside 274 distractors. This evaluates evidence ranking, attribution, and intervention sensitivity in a focused candidate setting, but does not benchmark open-domain discovery across full Wikipedia.
2. **Fixed Evidence Context ($k=3$)**:
   The formal experiment retrieved exactly Top-3 evidence pieces per claim. While standard for concise fact-checking pipelines, this constrained context restricts the combinatorial space of evidence interactions.
3. **Discrete Small-$n$ Rank Correlation**:
   With only three evidence pieces per claim, per-claim Spearman correlation takes values from a limited, discrete set. The attainable values depend on the rank patterns and the presence of ties. Therefore, these correlations should be interpreted as coarse measures of within-claim rank agreement rather than fine-grained continuous statistics.
4. **Single-Omission (Leave-One-Out) Interventions**:
   The intervention mechanism removes evidence pieces individually. It does not measure higher-order multi-evidence interactions (e.g., cases where two evidence pieces are jointly necessary to support a claim).
5. **Model-Internal Behavior vs. Factual Ground Truth**:
   The intervention influence $\Delta P$ measures the sensitivity of the specific verifier model (`cross-encoder/nli-deberta-v3-base`). It does not establish real-world causality, factual truth, or real-world source reliability.

---

## 8. Tripartite Scientific Demarcation

To maintain rigorous scientific standards, findings are categorized into three distinct levels of certainty:

### 8.1 Descriptive Results (Empirical Observations)
- LIME achieved an 84.00% Top-1 agreement with intervention influence, while SHAP achieved 72.20%.
- Both methods achieved a median per-claim Spearman correlation of $\rho = 0.5000$, with mean correlations of $0.5877$ (LIME) and $0.4517$ (SHAP).
- Positive rank correlation ($\rho > 0$) was observed in 87.00% of claims for LIME and 78.80% for SHAP.
- In 14.60% of claims (73 / 500), neither SHAP nor LIME placed their highest attribution on the evidence piece whose removal caused the largest probability shift.
- Leave-one-out evidence removal resulted in a discrete verdict flip in 21.47% of all removals (322 / 1,500), impacting 50.60% of evaluated claims (253 / 500).
- Removal evaluations on claims with baseline verdict `SUPPORTS` flipped at a rate of 37.23% (172 / 462 removals), while those on `REFUTES` flipped at 10.11% (81 / 801 removals).

### 8.2 Interpretations (Plausible Inferences)
- The substantial Top-1 agreement ($72\%$--$84\%$) suggests that post-hoc perturbation and linear surrogate approximations generally identify the primary decision-bearing evidence in cross-encoder verifiers.
- The higher Top-1 agreement of LIME ($84.0\%$) compared to SHAP ($72.2\%$) in this setup may stem from LIME's binary masking scheme aligning more closely with the discrete binary leave-one-out intervention removal than SHAP's multi-coalition game-theoretic averaging.
- The heightened sensitivity of `SUPPORTS` verdicts implies that supporting evidence is often concentrated in a single crucial sentence, whereas `REFUTES` verdicts may rely on broader inconsistencies that are more resilient to the omission of any single evidence piece.

### 8.3 Limitations (What the Experiment Does Not Establish)
- These results do **not** prove that LIME is universally superior to SHAP; performance differences reflect the specific leave-one-out evaluation criterion in a $k=3$ regime.
- High intervention influence does **not** imply that the evidence piece is factually true in reality—only that the verifier model relies heavily upon it.
- A verdict flip under evidence removal does **not** demonstrate mathematical or physical causality in the external world.

---

## 9. How Experiment 1 Supports the Research Gap

The original motivation of this research stems from a key gap identified in the literature (including the base IEEE paper):

> **Research Gap**: Existing explainable fact-checking architectures rely almost exclusively on post-hoc feature attribution (e.g., attention weights, saliency maps, or SHAP/LIME surrogates) to generate "explanations." However, post-hoc attribution explains the local behavior of a model surrogate; it does not measure how the system's actual decision changes when the evidence is actively manipulated or omitted.

Experiment 1 directly addresses this gap with empirical evidence:

1. **Attribution is Not Equivalent to Intervention Influence**:
   While attribution and intervention influence positively correlate in the majority of cases ($\rho > 0$ in $78.8\%$--$87.0\%$ of claims), there is a persistent divergence. In **14.60%** of claims, relying on attribution alone would lead a user or auditing system to misidentify which evidence piece actually drives the verifier's decision.
2. **Attribution Disagreement**:
   In **17.60%** of claims (88 / 500), LIME and SHAP directly disagree on which evidence piece is most important (with LIME matching the intervention top-1 in 66 claims, SHAP matching in 7 claims, and both failing in 15 claims). When post-hoc methods conflict, an auditor cannot determine which attribution reflects the model's actual functional reliance without intervention-based testing.
3. **Justification for Fuzzy Integration (Phases 7 & 8)**:
   The empirical evidence shows that attribution provides a useful, but imperfect, indicator of model behavior. Treating evidence quality as binary ("useful" vs. "useless") ignores this inherent uncertainty. The gradual continuous membership functions developed in our fuzzy evidence-reasoning layer are directly justified by these empirical discrepancies.

---

## 10. Paper-Ready Findings

The following statements are drafted for direct inclusion in the experimental evaluation section of the research paper:

1. **On Attribution–Intervention Concordance**:
   > *"Across 500 claims from the FEVER benchmark evaluated over a restricted candidate corpus of 808 evidence passages, evidence-level attribution magnitudes from SHAP and LIME exhibited positive rank concordance with empirical leave-one-out intervention influence in 78.80% and 87.00% of claims, respectively (median per-claim Spearman $\rho = 0.5000$ for both methods; mean $\rho = 0.4517$ for SHAP, mean $\rho = 0.5877$ for LIME)."*

2. **On Primary Evidence Identification (Top-1 Agreement)**:
   > *"When evaluating the primary decision-bearing evidence piece, binary-surrogate LIME correctly identified the highest-impact evidence under leave-one-out omission in 84.00% of claims (420/500), while kernel SHAP achieved 72.20% agreement (361/500), with both methods reaching consensus on 70.80% of claims."*

3. **On Attribution Divergence and the Need for Multi-Signal Auditing**:
   > *"Importantly, post-hoc attribution alone failed to identify the most decision-sensitive evidence piece in 14.60% of claims (73/500). Furthermore, SHAP and LIME directly disagreed on the primary evidence in 17.60% of cases (88/500 claims), demonstrating that feature attribution cannot serve as an unquestioned proxy for empirical evidence influence in automated fact-checking pipelines."*

4. **On Decision Fragility and Verdict-Flip Dynamics**:
   > *"Leave-one-out evidence omission induced a discrete classification flip in 21.47% of all intervention evaluations (322/1,500), altering the verdict in 50.60% of evaluated claims (253/500). Removal evaluations on claims initially verified as SUPPORTS exhibited the highest sensitivity to evidence removal (37.23% flip rate across 462 evaluations), whereas REFUTES classifications were substantially more resilient (10.11% flip rate across 801 evaluations)."*
