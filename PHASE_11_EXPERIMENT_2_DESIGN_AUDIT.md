# Phase 11 — Experiment 2 Design and Methodology Audit

## Executive Summary

This report establishes the finalized, audited research design, mathematical formalization, and statistical inference protocol for **Experiment 2**, which evaluates:
1. **The Fuzzy Evidence-Reliability Reasoning System** (`fuzzy/engine.py`), versus
2. **The Non-Fuzzy Linear Weighted Baseline** (`fuzzy/baseline.py`).

Following the final methodological and statistical review, all elements of the experimental protocol have been audited, clarified, and frozen:
- **Decision D1 (Option C — Dual Evaluation)**: Execution evaluates both the **Primary Condition** (dynamically computed conflict and order-permutation stability) and the **Secondary Ablation Condition** (static schema defaults $c_{\text{deg}} = 0.0$, $s_{\text{stab}} = 1.0$), reported side by side.
- **Decision D2 (Option A — Primary Comparison)**: The primary comparative pairing is **`evidence_reliability`** (Fuzzy) versus **`adjusted_baseline_score`** (Baseline). `explanation_confidence` and unadjusted `baseline_score` serve as secondary diagnostic metrics.
- **Statistical Inference Procedure Finalized**: Rather than an uncentered tail-count heuristic, inference for co-primary metrics ($\Delta \text{AUROC}$ and $\Delta \text{AP}$) employs an exact **Null-Centered Paired Claim-Level Cluster Bootstrap Test** ($B=1,000$, seed=42) with finite-replicate corrections, degenerate replicate logging, and **Holm-Bonferroni correction** across the two co-primary hypotheses. Multiplicity-adjusted (97.5% Bonferroni) confidence intervals are reported alongside marginal 95% CIs.
- **Attribution-Gap Analysis Categorization Finalized**:
  - *Membership-based subgroups ($S_{\text{gap}}^{\text{std}}, S_{\text{gap}}^{\text{ext}}$)*: Preserved with frozen linguistic thresholds. Because observed counts are $N = 0$, the report will transparently state that no empirical effect can be estimated.
  - *Rank-based subgroup ($S_{\text{gap}}^{\text{rank}}$)*: Formally categorized as **EXPLORATORY / POST-HOC MOTIVATED**. It will report descriptive sample counts, medians, and distributions only; no claims of formal statistical significance will be made in the current evaluation. Any confirmatory claim is explicitly deferred to an independent validation set.
  - *Output-specific interpretation*: Rule 2 directly targets **`explanation_confidence`**; `evidence_reliability` is evaluated strictly as a secondary diagnostic without assuming universal suppression across all fuzzy outputs.
- **Gold-Evidence Proxy Demarcated**: FEVER gold annotations are explicitly characterized as an extrinsic human relevance/sufficiency proxy, rather than ground-truth evidence reliability. Unannotated retrieved passages are recognized as not necessarily irrelevant or false.

### Strict Governance Compliance:
- **This document represents an audit and frozen research protocol only. Experiment 2 has NOT been executed.**
- Experiment 1 results remain strictly frozen and unaltered in Phase 10 (`outputs/phase10_formal/`) and Phase 12 (`PHASE_12_RESEARCH_ANALYSIS.md`, `DATASET_OVERLAP_AUDIT.md`).
- All underlying implementations (`fuzzy/`, `retrieval/`, `verification/`, `attribution/`, `intervention/`, `conflict/`, `stability/`), model weights, retrieval indices, and evaluation corpora remain completely unmodified.

---

## 1. Verified Input Signal Construction

An exhaustive audit of the actual codebase verifies the exact mathematical pipeline used to construct the six normalized inputs in `FuzzyInput`:

### 1.1 Attribution Strength ($a \in [0, 1]$)
- **Raw Fields Used**: `SHAP_score_raw` and `LIME_score_raw` (or `SHAP_magnitude` and `LIME_magnitude` from `results.csv`).
- **Absolute Magnitudes**: Absolute values are strictly applied via `normalize_attribution`:
  $$\text{normalize\_attribution}(v) = \max\left(0.0, \min\left(1.0, \frac{|v|}{\text{max\_expected}}\right)\right) \quad (\text{with } \text{max\_expected} = 1.0)$$
- **Aggregation Formula**: In `fuzzy/normalization.py` (lines 22–32):
  $$a = \text{compute\_attribution\_strength}(\text{shap\_val}, \text{lime\_val}, w_{\text{shap}}=0.5, w_{\text{lime}}=0.5) = \frac{0.5 \cdot \text{normalize\_attribution}(\text{shap\_val}) + 0.5 \cdot \text{normalize\_attribution}(\text{lime\_val})}{0.5 + 0.5}$$
- **Clipping & Range**: Because normalized SHAP and LIME are bounded in $[0.0, 1.0]$ and the weights sum to $1.0$, $a$ is strictly bounded in $[0.0, 1.0]$. In `fuzzy/engine.py` (line 123), it is further guarded with `max(0.0, min(1.0, attr_strength))`.
- **Codebase Consistency**: The implementation in `fuzzy/engine.py` (line 114) and `fuzzy/baseline.py` (line 16) calls `compute_attribution_strength` identically. The protocol matches the frozen codebase exactly.

### 1.2 Intervention Influence ($i \in [0, 1]$)
- **Source Definition**: Empirical leave-one-out probability shift on the baseline predicted verdict:
  $$i = \Delta P = | P_{\text{baseline}}(\text{target}) - P_{\text{intervened}}(\text{target}) |$$
- **Units & Range**: Absolute probability difference, strictly in $[0.0, 1.0]$.
- **Normalization**: `normalize_intervention(delta_p) = max(0.0, min(1.0, abs(delta_p)))` in `fuzzy/normalization.py` (lines 34–39).

### 1.3 Retrieval Relevance ($r \in [0, 1]$)
- **Source Definition**: Hybrid BM25 + dense retrieval score from `retrieval/retriever.py` (lines 146–163):
  $$r = \alpha \cdot \text{BM25}_{\text{norm}} + (1.0 - \alpha) \cdot \text{Dense}_{\text{norm}} \quad (\text{with } \alpha = 0.5)$$
  where both BM25 and dense scores are min-max normalized across retrieval candidates.
- **Units & Range**: Normalized retrieval score in $[0.0, 1.0]$.

### 1.4 Contradiction Degree ($c_{\text{deg}} \in [0, 1]$)
- **Primary Condition (Dynamic)**: Computed by `conflict.detector.ConflictDetector.detect` (lines 80–83):
  $$c_{\text{deg}} = \begin{cases} \min(1.0, 2.0 \cdot \min(\max(P_{\text{sup}}), \max(P_{\text{ref}}))) & \text{if both supporting and refuting evidence exist} \\ 0.0 & \text{otherwise} \end{cases}$$
- **Secondary Ablation**: Static schema default $c_{\text{deg}} = 0.0$.
- **Units & Range**: Conflict index in $[0.0, 1.0]$.

### 1.5 Explanation Stability ($s_{\text{stab}} \in [0, 1]$)
- **Primary Condition (Dynamic)**: Computed by `stability.analyzer.StabilityAnalyzer.analyze` (lines 71–80):
  $$s_{\text{stab}} = \max\left(0.0, \min\left(1.0, \frac{\rho_{\text{perm}} + 1.0}{2.0}\right)\right)$$
  where $\rho_{\text{perm}}$ is the Spearman rank correlation of feature attributions under evidence-order permutation (reversal).
- **Secondary Ablation**: Static schema default $s_{\text{stab}} = 1.0$.
- **Units & Range**: Normalized stability score in $[0.0, 1.0]$.

### 1.6 Source Reliability ($s_{\text{rel}} \in [0, 1]$)
- **Value**: Fixed static prior $s_{\text{rel}} = 0.75$ (Wikipedia default in `fuzzy/schema.py`, line 21).

---

## 2. Frozen Comparison Methods

### 2.1 Method A: Fuzzy Evidence Reasoning Engine (`fuzzy/engine.py`)
- **Antecedents**: $r, a, i, c_{\text{deg}}, s_{\text{rel}}, s_{\text{stab}}$ operating over $[0, 1]$.
- **Membership Functions**:
  - **LOW**: $\text{trimf}(x; [0.0, 0.0, 0.4])$ — non-zero for $x < 0.40$, peak at $0.0$.
  - **MEDIUM**: $\text{trimf}(x; [0.2, 0.5, 0.8])$ — non-zero for $0.20 < x < 0.80$, peak at $0.50$.
  - **HIGH**: $\text{trapmf}(x; [0.6, 0.8, 1.0, 1.0])$ — non-zero for $x > 0.60$, full membership ($1.0$) for $x \ge 0.80$.
- **Inference & Defuzzification**: Mamdani min-max composition with centroid defuzzification over 11 active control rules.
- **Primary Comparative Score**: **`evidence_reliability`** (Decision D2: Option A).
- **Secondary Diagnostic Score**: **`explanation_confidence`**.

### 2.2 Method B: Non-Fuzzy Linear Weighted Baseline (`fuzzy/baseline.py`)
- **Unadjusted Baseline Score (`baseline_score`)**:
  $$S_{\text{raw}} = 0.30 \cdot r + 0.25 \cdot a + 0.25 \cdot i + 0.10 \cdot s_{\text{rel}} + 0.10 \cdot s_{\text{stab}}$$
- **Adjusted Baseline Score (`adjusted_baseline_score` — Primary Comparative Score)**:
  $$S_{\text{adj}} = S_{\text{raw}} \cdot (1.0 - c_{\text{deg}})$$
- **Structural Properties**: Strictly additive and compensatory; contradiction acts as a uniform multiplicative discount across all passages in a claim.

---

## 3. Demarcation of Ground Truth Proxy & Candidate Pool

A rigorous scientific framing requires explicit conceptual separation of the evaluation target:

### 3.1 Four Distinct Epistemic Categories
1. **FEVER-Annotated Gold Evidence ($y_i = 1$)**:
   Passages explicitly labeled by human annotators in FEVER as forming a minimal sufficient set to verify or refute a claim.
2. **Retrieved Distractors / Non-Annotated Evidence ($y_i = 0$)**:
   Passages retrieved by BM25/dense search that do not carry a FEVER gold annotation for that claim. **Crucially, unannotated passages must NOT be equated with "definitely irrelevant" or "factually incorrect."** They may represent unannotated true statements, partially relevant background context, or valid alternative proofs not selected by the annotators.
3. **Intrinsic Factual Relevance**:
   The genuine semantic and factual relationship between a passage and a claim, which exists independently of whether annotators flagged that specific sentence.
4. **Composite Evidence Reliability**:
   A multi-attribute construct that incorporates factual relevance, causal necessity (intervention influence), explainer attribution, source credibility, and cross-evidence conflict/stability.

### 3.2 Evaluation Target Framing
- **Discrimination Proxy**: AUROC and Average Precision quantify the model's ability to **distinguish human-annotated gold evidence from other retrieved passages**, NOT real-world evidence reliability or factual truth.
- **Candidate Pool Scope**: Closed candidate pool of 808 Wikipedia passages, pre-seeded with all 534 gold passages from evaluated claims, alongside 274 distractors.
- **Evaluated Population**: Exactly 1,500 retrieved instances ($k=3$ passages across 500 claims): 443 gold instances ($29.53\%$) and 1,057 distractors ($70.47\%$). All 500 evaluated claims have non-empty gold annotations.
- **Symmetrical Expectations**: Results reflect this gold-conditioned setting rather than full-Wikipedia open retrieval. Results may favor neither method, and both methods may demonstrate low discrimination.

---

## 4. Primary Evaluation Metrics & Finalized Statistical Inference

### 4.1 Primary Research Question
> **"Does fuzzy evidence-reliability scoring distinguish FEVER-annotated gold evidence from other retrieved passages differently from the existing non-fuzzy weighted baseline?"**

### 4.2 Co-Primary Metrics
The evaluation pre-specifies two co-primary discrimination metrics across the 1,500 pooled evidence evaluations:
1. **$\text{AUROC}$**: $\text{AUROC}_{\text{fuzzy}}$ (`evidence_reliability`) vs. $\text{AUROC}_{\text{base}}$ (`adjusted_baseline_score`), and their paired difference $\Delta \text{AUROC} = \text{AUROC}_{\text{fuzzy}} - \text{AUROC}_{\text{base}}$.
2. **$\text{Average Precision}$ (PR-AUC)**: $\text{AP}_{\text{fuzzy}}$ vs. $\text{AP}_{\text{base}}$, and their paired difference $\Delta \text{AP} = \text{AP}_{\text{fuzzy}} - \text{AP}_{\text{base}}$.

#### Pooled Discrimination Scope Note:
Pooled AUROC and Average Precision evaluate discrimination across all 1,500 evidence instances. The paired claim-level bootstrap handles intra-cluster dependence, but pooled metrics assume that score thresholds have a consistent meaning across claims. Within-claim Top-1 hit rates serve as an essential calibration-independent complement.

---

### 4.3 Finalized Statistical Inference: Null-Centered Paired Cluster Bootstrap Test

To evaluate the statistical significance of $\Delta \text{AUROC}$ and $\Delta \text{AP}$, an exact **Null-Centered Paired Claim-Level Cluster Bootstrap Test** is specified:

1. **Observed Difference**:
   Compute the point estimates $\widehat{\theta}_{\text{fuzzy}}$ and $\widehat{\theta}_{\text{base}}$ on the full observed sample of 1,500 instances across $M = 500$ claims. The observed paired difference is:
   $$\widehat{\Delta} = \widehat{\theta}_{\text{fuzzy}} - \widehat{\theta}_{\text{base}}$$
   Under the null hypothesis $H_0$, the true difference is $\Delta_0 = 0$.

2. **Cluster Resampling with Multiplicity Preservation**:
   - For replicate $b = 1, \dots, B$ ($B = 1,000$, seed = $42$ using `numpy.random.Generator(np.random.PCG64(42))`):
   - Draw $M = 500$ claim indices with replacement: $j_1^{(b)}, \dots, j_M^{(b)}$.
   - Each selected claim cluster contributes all three of its associated evidence records together:
     $$C_j = \{(e_{j,1}, y_{j,1}), (e_{j,2}, y_{j,2}), (e_{j,3}, y_{j,3})\}$$
   - When claim $j$ is drawn with multiplicity $m \ge 1$, all 3 associated evidence records appear exactly $m$ times in the resampled evaluation pool. The resampled pool size is strictly $3 \times 500 = 1,500$.
   - Compute the resampled metric estimates $\theta_{\text{fuzzy}}^{*(b)}$ and $\theta_{\text{base}}^{*(b)}$, and their resampled difference:
     $$\Delta^{*(b)} = \theta_{\text{fuzzy}}^{*(b)} - \theta_{\text{base}}^{*(b)}$$

3. **Centering under the Null Hypothesis**:
   The raw bootstrap distribution of $\Delta^{*(b)}$ is centered near the sample estimate $\widehat{\Delta}$. To test $H_0: \Delta = 0$, the bootstrap distribution is shifted to have mean zero under the null:
   $$\widetilde{\Delta}^{*(b)} = \Delta^{*(b)} - \widehat{\Delta}$$
   This transformation aligns the bootstrap replicates with the null hypothesis while preserving the empirical variance, skewness, and dependency structure of the estimator.

4. **Exact Two-Sided P-Value Formula**:
   The two-sided p-value is computed as the proportion of null-centered bootstrap replicates that are as extreme as or more extreme than the observed effect size $|\widehat{\Delta}|$:
   $$p_{\text{boot}} = \frac{1 + \sum_{b=1}^{B_{\text{valid}}} \mathbb{I}\left( |\widetilde{\Delta}^{*(b)}| \ge |\widehat{\Delta}| \right)}{B_{\text{valid}} + 1}$$
   - **Finite-Replicate Correction**: The $+1$ correction in numerator and denominator (Davison & Hinkley 1997, MacKinnon 2009) ensures that $p_{\text{boot}}$ cannot be strictly zero and provides a conservative, strictly valid probability.
   - **Treatment of Ties and Zeros**: The weak inequality $|\widetilde{\Delta}^{*(b)}| \ge |\widehat{\Delta}|$ includes equality. If $\widehat{\Delta} = 0$, then $|\widetilde{\Delta}^{*(b)}| \ge 0$ holds for all replicates, yielding $p_{\text{boot}} = 1.0$.

5. **Confidence Intervals**:
   - **Marginal 95% Percentile Confidence Intervals**: Reported for each metric individually:
     $$\text{CI}_{95\%}(\Delta) = \left[ \Delta^*_{(0.025)}, \Delta^*_{(0.975)} \right]$$
     reflecting unadjusted marginal parameter uncertainty.
   - **Multiplicity-Adjusted CIs**: To provide family-wise 95% coverage across the co-primary pair, **Bonferroni-adjusted 97.5% confidence intervals** $[\Delta^*_{(0.0125)}, \Delta^*_{(0.9875)}]$ will be reported alongside the 95% CIs.

6. **Multiple Comparisons for Co-Primary Tests**:
   - Family-Wise Error Rate (FWER) across the two co-primary tests ($H_1: \Delta \text{AUROC} \neq 0$ and $H_2: \Delta \text{AP} \neq 0$) is controlled at $\alpha = 0.05$ via the **Holm-Bonferroni step-down procedure**:
     1. Sort two-sided p-values: $p_{(1)} \le p_{(2)}$.
     2. Step 1: Compare $p_{(1)}$ to $\alpha / 2 = 0.025$. If $p_{(1)} \ge 0.025$, fail to reject both null hypotheses.
     3. Step 2: If $p_{(1)} < 0.025$, reject $H_{(1)}$ and compare $p_{(2)}$ to $\alpha / 1 = 0.05$. If $p_{(2)} < 0.05$, reject $H_{(2)}$.
     4. Adjusted p-values: $p_{\text{adj},(1)} = \min(1.0, 2 \cdot p_{(1)})$, $p_{\text{adj},(2)} = \min(1.0, \max(p_{\text{adj},(1)}, p_{(2)}))$.
   - Both unadjusted and Holm-adjusted p-values will be documented in reports.

7. **Handling Degenerate / Undefined Replicates**:
   - If a bootstrap replicate contains zero positive instances ($\sum y = 0$) or zero negative instances ($\sum y = N$), AUROC or Average Precision is mathematically undefined.
   - Contingency protocol: Such replicates are flagged as `UNDEFINED_DEGENERATE_SAMPLE` and will **NOT** be silently discarded or replaced. The failure reason and count $B_{\text{fail}}$ will be explicitly logged.
   - Effective valid replicates $B_{\text{valid}} = B - B_{\text{fail}}$ will be documented and used in the p-value denominator. If $B_{\text{fail}} / B > 0.01$, a formal methodological alert is triggered.

---

## 5. Pre-Specified Top-1 Evidence Selection Metrics

To evaluate practical fact-checking utility, three distinct Top-1 metrics are pre-specified with explicit numerators and denominators:

| Metric ID | Metric Name | Denominator | Numerator Definition | Evaluation Scope |
| :--- | :--- | :---: | :--- | :--- |
| **Metric A** | **Top-1 Gold Hit Rate (All Claims)** | $N = 500$ | Number of claims where the top-ranked evidence piece (Rank 1 by score) is annotated gold ($y_i = 1$). | All 500 evaluated claims |
| **Metric B** | **Top-1 Gold Hit Rate (Gold-Present Claims)** | $N = 430$ | Number of claims with $\ge 1$ gold piece in Top-3 where the top-ranked evidence piece (Rank 1 by score) is annotated gold ($y_i = 1$). | Subset of 430 claims containing at least one gold passage in retrieved Top-3 |
| **Metric C** | **Direct Top-1 Agreement Rate** | $N = 500$ | Number of claims where fuzzy primary (`evidence_reliability`) and baseline primary (`adjusted_baseline_score`) rank the exact same `evidence_id` as Rank 1. | All 500 evaluated claims |

### 5.1 Deterministic Tie-Breaking Order
To guarantee 100% reproducibility in the presence of tied scores:
1. **Primary Key**: Score metric value descending (`evidence_reliability` or `adjusted_baseline_score`).
2. **Secondary Key**: Normalized retrieval score descending (`ev.retrieval_score` from hybrid fusion).
3. **Tertiary Key**: Original retrieval rank ascending (`ev.retrieval_rank`: $1, 2, 3$).

*Pipeline Verification*: All three tie-breaking inputs are deterministically available from `retrieval/retriever.py` and have been verified to match the 1,500 records evaluated in Experiment 1.

---

## 6. Targeted Attribution-Gap Analysis Aligned with Fuzzy Rules

### 6.1 Theoretical Grounding: Rule 2 Causal Gating
The attribution-gap analysis is a targeted evaluation of **Rule 2** in `fuzzy/engine.py`:
$$\text{IF relevance is HIGH} \land \text{attribution is HIGH} \land \text{intervention is LOW} \implies \text{explanation\_confidence is LOW}$$

- **Target Fuzzy Output Evaluated**: **`explanation_confidence`** (the explicit consequent of Rule 2).
- **Baseline Diagnostic Comparison**: **`baseline_score`** (the unadjusted additive baseline $0.30 \cdot r + 0.25 \cdot a + 0.25 \cdot i + 0.10 \cdot s_{\text{rel}} + 0.10 \cdot s_{\text{stab}}$), which compensatorily rewards high attribution even when intervention is low.
- **Status of Evidence Reliability**: `evidence_reliability` behavior within the attribution-gap subgroup is evaluated strictly as a **secondary diagnostic**. Rule 2 does not specify `evidence_reliability` in its consequent; suppression of `evidence_reliability` is **NOT** assumed a priori.
- **Epistemic Wording Standard**: Disagreement between attribution magnitude and intervention influence is characterized as **"attribution-intervention divergence"** or **"causally unsupported post-hoc attribution."** It must **NOT** be equated with "proof of causal invalidity."

---

### 6.2 Subgroup Categorization: Confirmatory vs. Exploratory

To prevent conflating confirmatory hypothesis tests with post-hoc exploratory analyses, the subgroups are formally bifurcated:

#### Category 1: Membership-Based Subgroups (Confirmatory Specification)
Derived strictly from the membership functions defined in `fuzzy/engine.py`:
- `attribution_strength`: $\text{trapmf}(x; [0.6, 0.8, 1.0, 1.0])$ ($\mu_{\text{HIGH}} > 0$ for $a \ge 0.60$; $\mu_{\text{HIGH}} = 1.0$ for $a \ge 0.80$).
- `intervention_influence`: $\text{trimf}(x; [0.0, 0.0, 0.4])$ ($\mu_{\text{LOW}} > 0$ for $i \le 0.40$; peak at $i = 0.00$).

1. **Standard Attribution-Gap Subgroup ($S_{\text{gap}}^{\text{std}}$)**:
   $$S_{\text{gap}}^{\text{std}} = \{e_i \mid a_i \ge 0.60 \land i_i \le 0.40\}$$
2. **Extreme Attribution-Gap Subgroup ($S_{\text{gap}}^{\text{ext}}$)**:
   $$S_{\text{gap}}^{\text{ext}} = \{e_i \mid a_i \ge 0.80 \land i_i \le 0.10\}$$

*Empirical Finding & Handling*: In the existing 1,500 evaluations, exactly **$N = 0$ instances** satisfy $a \ge 0.60 \land i \le 0.40$. The original thresholds are preserved. The report will transparently state that because these subgroups are empty on the evaluated set, the corresponding analysis cannot estimate an empirical effect.

#### Category 2: Rank-Divergence Gap Subgroup (Exploratory Specification)
3. **Rank-Divergence Gap Subgroup ($S_{\text{gap}}^{\text{rank}}$)**:
   $$S_{\text{gap}}^{\text{rank}} = \{e_i \mid \text{Attribution Rank}(e_i) = 1 \land \text{Intervention Rank}(e_i) = 3\}$$

*Classification & Status*:
- **EXPLORATORY ONLY**: This definition was introduced after inspecting the distribution of Experiment 1 results (where $N = 76$ instances exist for SHAP and $N = 87$ for composite attribution).
- **No Confirmatory Claims**: It will **NOT** be presented as a confirmatory hypothesis test, and no claims of formal statistical significance will be made in the current evaluation.
- **Reporting Scope**: Descriptive sample counts, means, medians, IQR, and score distributions only.
- **Independent Validation Requirement**: Any confirmatory claim regarding rank-divergent evidence behavior requires pre-specification on an **independent validation set** (e.g. FEVER test split or an out-of-domain benchmark).

---

## 7. Direct Comparative Hypotheses

Replacing exploratory thresholds ($> 0.50$), all hypotheses are classified into confirmatory co-primary tests, confirmatory secondary tests, and exploratory descriptive evaluations:

### Hypothesis $H_1$ (Co-Primary: Discrimination Difference — AUROC)
- **Null Hypothesis ($H_{1,0}$)**: $\Delta \text{AUROC} = \text{AUROC}(\text{fuzzy}) - \text{AUROC}(\text{baseline}) = 0$.
- **Alternative Hypothesis ($H_{1,A}$)**: $\Delta \text{AUROC} \neq 0$.
- **Status**: **CONFIRMATORY**.
- **Inference**: Null-centered paired claim-level cluster bootstrap test ($B=1,000$, seed=42); tested at Holm-Bonferroni adjusted $\alpha = 0.025$ or $0.05$.

### Hypothesis $H_2$ (Co-Primary: Precision Difference — Average Precision)
- **Null Hypothesis ($H_{2,0}$)**: $\Delta \text{AP} = \text{AP}(\text{fuzzy}) - \text{AP}(\text{baseline}) = 0$.
- **Alternative Hypothesis ($H_{2,A}$)**: $\Delta \text{AP} \neq 0$.
- **Status**: **CONFIRMATORY**.
- **Inference**: Null-centered paired claim-level cluster bootstrap test ($B=1,000$, seed=42); tested at Holm-Bonferroni adjusted $\alpha = 0.025$ or $0.05$.

### Hypothesis $H_3$ (Secondary: Top-1 Gold Alignment Difference)
- **Null Hypothesis ($H_{3,0}$)**: $\text{HitRate}_A(\text{fuzzy}) = \text{HitRate}_A(\text{baseline})$ and $\text{HitRate}_B(\text{fuzzy}) = \text{HitRate}_B(\text{baseline})$.
- **Alternative Hypothesis ($H_{3,A}$)**: The Top-1 gold hit rates differ between fuzzy and baseline primary scoring.
- **Status**: **SECONDARY CONFIRMATORY**.
- **Inference**: Paired claim-level bootstrap 95% confidence interval on difference in hit rates for Metric A and Metric B.

### Hypothesis $H_4$ (Secondary Diagnostic: Attribution-Gap Rule 2 Behavior)
- **Status**: **EXPLORATORY DESCRIPTIVE**.
- **Inference**: Descriptive summary of `explanation_confidence` vs. `baseline_score` on $S_{\text{gap}}^{\text{rank}}$ without formal significance claims; confirmatory testing deferred to an independent validation set.

---

## 8. Distributional Diagnostics & Score Saturation

To evaluate score dispersion, compression, and calibration across all 1,500 evaluations:
- Mean, standard deviation, median, interquartile range (IQR), min, max, skewness.
- Boundary Saturation Frequency: Count and percentage of scores equal to $0.0$ or $1.0$.
- Centroid compression documentation: Quantifies the expected effect of Mamdani centroid defuzzification over overlapping sets.

---

## 9. Deliverables and Execution Governance

When formal approval to execute Experiment 2 is granted, execution will proceed with the following artifacts:
- **Runner Script**: `experiments/experiment2_formal.py`
- **Output Artifacts**:
  - `outputs/phase11/experiment2_results.csv` (complete 1,500-instance evaluation data)
  - `outputs/phase11/experiment2_summary.csv` (metrics table for Primary Condition and Secondary Ablation)
  - `outputs/phase11/gold_discrimination_roc.png` (side-by-side ROC and PR curves)
  - `outputs/phase11/score_distribution_comparison.png` (histograms and density plots)
  - `outputs/phase11/attribution_gap_behavior.png` (scatter and rank shifts on gap subsets)
  - `EXPERIMENT_2_FORMAL_REPORT.md` (comprehensive formal research report)

---

## 10. Methodological Audit & Sign-off Checklist

| Audit Item | Protocol Specification | Scientific Status |
| :--- | :--- | :---: |
| **Statistical Inference Method** | Null-Centered Paired Claim-Level Cluster Bootstrap ($B=1,000$, seed=42) | **FINALIZED** |
| **P-Value Exact Formula** | $p = (1 + \sum \mathbb{I}(\|\widetilde{\Delta}^*\| \ge \|\widehat{\Delta}\|)) / (B_{\text{valid}} + 1)$ with finite-sample correction | **FINALIZED** |
| **Multiple Testing** | Holm-Bonferroni correction over $H_1$ and $H_2$; 97.5% Bonferroni CIs reported | **FINALIZED** |
| **Cluster Resampling** | Claim clusters ($M=500$), preserves claim multiplicity; pool size = 1,500 | **FINALIZED** |
| **Degenerate Replicates** | Logged as `UNDEFINED_DEGENERATE_SAMPLE`, never silently dropped | **FINALIZED** |
| **Attribution-Strength Construction** | Verified in `fuzzy/normalization.py`; matches frozen code | **VERIFIED** |
| **Membership Gap Subgroups** | $S_{\text{gap}}^{\text{std}}, S_{\text{gap}}^{\text{ext}}$ preserved; $N=0$ reported transparently | **CLASSIFIED** |
| **Rank-Based Gap Subgroup** | $S_{\text{gap}}^{\text{rank}}$ labeled EXPLORATORY; descriptive only, no significance claims | **CLASSIFIED** |
| **Rule 2 Output Target** | Directly evaluates `explanation_confidence`; `evidence_reliability` is secondary | **ALIGNED** |
| **Gold-Evidence Proxy** | Demarcates FEVER gold, unannotated retrieved, factual relevance, and reliability | **CLARIFIED** |
| **Top-1 Tie-Breaking** | Score $\rightarrow$ Retrieval Score $\rightarrow$ Retrieval Rank; all inputs verified | **VERIFIED** |
| **Candidate Pool Demarcation** | Closed 808-passage pool, 534 gold, 1,500 evaluations; non-inferiority possible | **CLARIFIED** |
| **Execution State** | **AUDITED & FROZEN — AWAITING USER APPROVAL BEFORE EXECUTION** | **PHASE GATE MAINTAINED** |
