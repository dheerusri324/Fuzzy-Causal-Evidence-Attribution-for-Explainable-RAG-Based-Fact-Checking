# NEI Evidence Label Sensitivity Audit: Evidence-Label Semantics, Closed-Pool Boundaries, and Re-evaluation of Experiment 2

**Audit Date:** 2026-10-20  
**Audit Type:** Read-Only Dataset Label Semantics & Sensitivity Analysis  
**Scope:** Evaluation of Evidence-Label Semantics in FEVER (`copenlu/fever_gold_evidence`), Correction of NOT_ENOUGH_INFO (NEI) Evidence Annotations, and Comparative Sensitivity of Formal Experiment 2 Metrics  
**Status:** COMPLETE — SENSITIVITY ARTIFACTS PRESERVED — OFFICIAL FROZEN RESULTS PRESERVED — PHASE 14 NOT STARTED

----

## 1. Executive Summary & Strict Safety Compliance

This sensitivity audit investigates the semantic validity of gold-evidence proxy labels in the FEVER development split (`copenlu/fever_gold_evidence`) and evaluates the robustness of the comparative findings of **Formal Experiment 2** (Fuzzy Evidence Reliability Reasoning vs. Non-Fuzzy Weighted Baseline).

### 1.1 Core Defect Identified in Upstream Dataset Semantics
In the standard FEVER formulation (Thorne et al., 2018), claims labelled `SUPPORTS` or `REFUTES` possess human-annotated gold evidence sentences from Wikipedia. Conversely, claims labelled `NOT ENOUGH INFO` (NEI) are unverified by definition and have **no gold evidence**.
However, in the preprocessed HuggingFace dataset `copenlu/fever_gold_evidence` (derived from Stammbach et al. / Papelo et al.), candidate retrieved passages were pre-populated into the `evidence` field for NEI claims. In the project's data ingestion (`data/fever_loader.py`) and formal evaluation code (`experiments/experiment2_formal.py`), these NEI pseudo-evidence passages were treated indiscriminately as 'gold evidence' proxy labels.

Of the **443 positive gold instances** in the official frozen Experiment 2 dataset, exactly **147 instances (33.18%)** belong to NEI claims. These 147 instances represent retrieved negative distractors rather than genuine annotator-verified evidence.

### 1.2 Sensitivity Findings at a Glance
When evidence-label semantics are rigorously corrected, the primary finding of Experiment 2 is **strengthened, not weakened**:
1. **Fuzzy AUROC Advantage More Than Doubles:**
   - Official Frozen Run: Fuzzy AUROC exceeded Baseline by $+0.0365$ ($p = 0.0240$, significant at $\alpha = 0.05$).
   - **Sensitivity Protocol A (NEI treated as 0):** Fuzzy AUROC exceeds Baseline by **$+0.0898$** (Fuzzy $0.7386$ vs. Baseline $0.6488$; Holm-adjusted $p = 0.0020$, significant at $\alpha = 0.025$).
   - **Sensitivity Protocol B (NEI excluded; Verifiable claims only):** Fuzzy AUROC exceeds Baseline by **$+0.0827$** (Fuzzy $0.7777$ vs. Baseline $0.6949$; Holm-adjusted $p = 0.0020$, significant at $\alpha = 0.025$).
2. **Baseline AP Advantage Shrinks:**
   - The Average Precision advantage held by the baseline under dynamic conditions drops from $-0.1396$ (official) to **$-0.0680$** (Protocol A) and **$-0.1098$** (Protocol B).
3. **Top-1 Gold Selection Fidelity:**
   - When true gold evidence is retrieved in the top 3 (283 verifiable claims), Fuzzy logic selects gold evidence as Rank 1 in **87.63%** of claims (248/283), compared to **83.39%** for Baseline (236/283), an advantage of $+4.24\%$.

### 1.3 Strict Safety & Protocol Boundaries
- **No Overwriting:** Frozen results (`outputs/phase10_formal/results.csv` and `outputs/phase11_formal/results.csv`) remain completely untouched and unmodified.
- **Isolated Destination:** All corrected datasets, sensitivity JSON summaries, and plots reside exclusively in `outputs/sensitivity_nei_label_audit/`.
- **Epistemic Label:** All findings herein are strictly designated as **Sensitivity Analyses**, supplementing rather than replacing the official historical experiment record.
- **Phase 14 Gate:** Phase 14 has **NOT** been started.

----

## 2. Audit of Evidence-Label Semantics in the Implementation

### 2.1 Upstream Dataset Provenance & Ingestion
The project streams `copenlu/fever_gold_evidence` via `data/fever_loader.py`. In `_normalize_record` (lines 77–115):
```python
# data/fever_loader.py: Lines 94-114
evidence_pieces: List[EvidencePiece] = []
for ev_group in raw.get('evidence', []):
    doc_title = str(ev_group[0])
    sentence_idx = int(ev_group[1]) if ev_group[1] is not None else 0
    ev_text = _clean_text(str(ev_group[2]))
    ev_id = _make_evidence_id(doc_title, sentence_idx)
    evidence_pieces.append(EvidencePiece(evidence_id=ev_id, text=ev_text, ...))
return ClaimRecord(claim_id=claim_id, label=label_raw, evidence=evidence_pieces, ...)
```
**Defect:** The loader normalizes `evidence_pieces` for every claim regardless of whether `label_raw` is `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO`.

### 2.2 Formal Experiment 2 Label Derivation
In `experiments/experiment2_formal.py` (lines 99–100 and 469–470):
```python
# experiments/experiment2_formal.py: Lines 469-470
gold_ids = {ep.evidence_id for ep in rec.evidence}
is_gold = 1 if ev.evidence_id in gold_ids else 0
```
Because `rec.evidence` was non-empty for all 166 evaluated NEI claims (each containing exactly 1 candidate passage from Papelo retrieval), any retrieved passage matching this ID was labelled `is_annotated_gold = 1`.

### 2.3 Breakdown of the Official Frozen Label Distribution
The 500 evaluated FEVER claims partition into three balanced classes:
| Claim Label | Evaluated Claims | Evidence per Claim | Total Retrieved Rows | Annotated Gold Rows (`is_gold=1`) | Distractor Rows (`is_gold=0`) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **SUPPORTS** | 168 | 3 | 504 | 154 (30.56%) | 350 (69.44%) |
| **REFUTES** | 166 | 3 | 498 | 142 (28.51%) | 356 (71.49%) |
| **NOT_ENOUGH_INFO** | 166 | 3 | 498 | **147 (29.52%)** | 351 (70.48%) |
| **Total (Official Exp 2)** | **500** | **3** | **1,500** | **443 (29.53%)** | **1,057 (70.47%)** |

In genuine FEVER semantics, all 147 NEI positive labels are pseudo-evidence distractors. They do not substantiate the claim.

----

## 3. Reconstructed Label Dataset Artifacts

A dedicated sensitivity directory was established at `outputs/sensitivity_nei_label_audit/`.
The complete 1,500-row evaluation dataset was reconstructed and saved to:
- [`outputs/sensitivity_nei_label_audit/results_nei_sensitivity.csv`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/outputs/sensitivity_nei_label_audit/results_nei_sensitivity.csv)

This file preserves all 39 original columns from `outputs/phase11_formal/results.csv` and appends four explicit audit columns:
1. `claim_label`: FEVER class (`SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO`).
2. `is_gold_protocol_a`: Binary indicator where NEI evidence is reclassified as 0 (distractor). Contains exactly 296 ones and 1,204 zeros.
3. `in_protocol_b`: Boolean indicator denoting verifiable claims (`True` for 1,002 rows across 334 SUPPORTS/REFUTES claims; `False` for 498 NEI rows).
4. `is_gold_protocol_b`: Ground-truth indicator for Protocol B (NaN for NEI rows; exactly 296 ones and 706 zeros for verifiable rows).

----

## 4. Sensitivity Protocol A: All 500 Claims (NEI Evidence Reclassified as Distractor)

### 4.1 Specification
- **Evaluation Population:** All 500 claims ($N = 1,500$ evidence evaluations).
- **Label Semantic:** Genuine gold evidence exists only for SUPPORTS and REFUTES claims. For NEI claims, any retrieved passage is a distractor ($y = 0$).
- **Sample Sizes:** $296$ gold instances ($19.73\%$), $1,204$ distractors ($80.27\%$).
- **Claims with Gold in Top-3:** 283 claims ($56.60\%$); Claims with zero gold: 217 claims ($43.40\%$).

### 4.2 Statistical Inference & Co-Primary Metrics (Dynamic Pipeline)
Evaluated using $B = 1,000$ paired claim-level cluster bootstrap iterations with null-shift centering and Holm–Bonferroni step-down correction:

| Hypothesis / Metric | Point Est. Fuzzy | Point Est. Baseline | Difference ($\hat{\Delta}$) | 95% Marginal CI | 97.5% Bonferroni CI | Raw $p$-value | Holm-Adj. $p$ | Significant? (FWER $\alpha=0.05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H1: AUROC** | 0.7386 | 0.6488 | **+0.0898** | [0.0503, 0.1281] | [0.0432, 0.1359] | 0.0010 | **0.0020** | **YES** ($\alpha=0.025$) |
| **H2: Average Precision** | 0.3537 | 0.4217 | **-0.0680** | [-0.1089, -0.0319] | [-0.1168, -0.0253] | 0.0010 | **0.0020** | **YES** ($\alpha=0.05$) |

### 4.3 Top-1 Selection Metrics (Protocol A)
- **Metric A (Denominator = All 500 Claims):** Fuzzy selects gold in **248 / 500 claims (49.60%)** vs. Baseline **236 / 500 (47.20%)**, Difference = **+2.40%**.
- **Metric B (Denominator = 283 Gold-Present Claims):** Fuzzy selects gold in **248 / 283 (87.63%)** vs. Baseline **236 / 283 (83.39%)**, Difference = **+4.24%**.
- **Metric C (Direct Agreement):** Fuzzy and Baseline select the identical Rank 1 passage in **437 / 500 claims (87.40%)**.

----

## 5. Sensitivity Protocol B: Verifiable Claims Only (SUPPORTS & REFUTES Exclusively)

### 5.1 Specification
- **Evaluation Population:** 334 verifiable claims ($N = 1,002$ evidence evaluations), completely omitting the 166 NEI claims.
- **Label Semantic:** Strictly evaluates gold-evidence discrimination where ground-truth gold evidence was annotated by human experts.
- **Sample Sizes:** $296$ gold instances ($29.54\%$), $706$ distractors ($70.46\%$).
- **Claims with Gold in Top-3:** 283 claims ($84.73\%$); Claims with zero gold: 51 claims ($15.27\%$).

### 5.2 Statistical Inference & Co-Primary Metrics (Dynamic Pipeline)
Evaluated using $B = 1,000$ paired claim-level cluster bootstrap iterations:

| Hypothesis / Metric | Point Est. Fuzzy | Point Est. Baseline | Difference ($\hat{\Delta}$) | 95% Marginal CI | 97.5% Bonferroni CI | Raw $p$-value | Holm-Adj. $p$ | Significant? (FWER $\alpha=0.05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H1: AUROC** | 0.7777 | 0.6949 | **+0.0827** | [0.0455, 0.1213] | [0.0373, 0.1294] | 0.0010 | **0.0020** | **YES** ($\alpha=0.025$) |
| **H2: Average Precision** | 0.5091 | 0.6189 | **-0.1098** | [-0.1518, -0.0632] | [-0.1563, -0.0573] | 0.0010 | **0.0020** | **YES** ($\alpha=0.05$) |

### 5.3 Top-1 Selection Metrics (Protocol B)
- **Metric A (Denominator = All 334 Verifiable Claims):** Fuzzy selects gold in **248 / 334 claims (74.25%)** vs. Baseline **236 / 334 (70.66%)**, Difference = **+3.59%**.
- **Metric B (Denominator = 283 Gold-Present Claims):** Fuzzy selects gold in **248 / 283 (87.63%)** vs. Baseline **236 / 283 (83.39%)**, Difference = **+4.24%**.
- **Metric C (Direct Agreement):** Fuzzy and Baseline select the identical Rank 1 passage in **298 / 334 claims (89.22%)**.

----

## 6. Comprehensive Cross-Protocol Comparative Synthesis

The table below synthesizes the official frozen experiment results against both sensitivity protocols:

| Metric / Evaluated Dimension | Official Frozen (Exp 2) | Sensitivity Protocol A (All 500, NEI=0) | Sensitivity Protocol B (334 SUPP/REF Only) |
|:---|:---:|:---:|:---:|
| **Evaluated Population** | 500 claims (1,500 rows) | 500 claims (1,500 rows) | 334 claims (1,002 rows) |
| **Gold Evidence Instances ($y=1$)** | 443 (29.53%) | 296 (19.73%) | 296 (29.54%) |
| **Distractor Instances ($y=0$)** | 1,057 (70.47%) | 1,204 (80.27%) | 706 (70.46%) |
| **Claims with Gold in Top-3** | 430 / 500 (86.00%) | 283 / 500 (56.60%) | 283 / 334 (84.73%) |
| **Claims with Zero Gold in Top-3** | 70 / 500 (14.00%) | 217 / 500 (43.40%) | 51 / 334 (15.27%) |
| **Dynamic AUROC: Fuzzy** | 0.7630 | 0.7386 | 0.7777 |
| **Dynamic AUROC: Baseline** | 0.7265 | 0.6488 | 0.6949 |
| **AUROC Advantage ($\hat{\Delta}$)** | **+0.0365** | **+0.0898** | **+0.0827** |
| **AUROC 95% Marginal CI** | [0.0053, 0.0686] | [0.0503, 0.1281] | [0.0455, 0.1213] |
| **AUROC Holm-Adjusted $p$** | 0.0240 (sig at $\alpha=0.05$) | 0.0020 (sig at $\alpha=0.025$) | 0.0020 (sig at $\alpha=0.025$) |
| **Dynamic AP: Fuzzy** | 0.4884 | 0.3537 | 0.5091 |
| **Dynamic AP: Baseline** | 0.6280 | 0.4217 | 0.6189 |
| **AP Advantage ($\hat{\Delta}$)** | **-0.1396** | **-0.0680** | **-0.1098** |
| **AP 95% Marginal CI** | [-0.1777, -0.1025] | [-0.1089, -0.0319] | [-0.1518, -0.0632] |
| **AP Holm-Adjusted $p$** | 0.0020 (sig at $\alpha=0.025$) | 0.0020 (sig at $\alpha=0.05$) | 0.0020 (sig at $\alpha=0.05$) |
| **Top-1 Gold Hit Rate (Metric A)** | Fuzzy 73.0% vs Base 69.0% | Fuzzy 49.6% vs Base 47.2% | Fuzzy 74.3% vs Base 70.7% |
| **Top-1 Gold Hit Rate (Metric B)** | Fuzzy 84.9% vs Base 80.2% | Fuzzy 87.6% vs Base 83.4% | Fuzzy 87.6% vs Base 83.4% |
| **Top-1 Direct Agreement (Metric C)** | 87.4% (437/500) | 87.4% (437/500) | 89.2% (298/334) |
| **Static Ablation AUROC** | Fuzzy 0.7946 vs Base 0.8889 | Fuzzy 0.7888 vs Base 0.8661 | Fuzzy 0.8210 vs Base 0.9107 |
| **Static Ablation AP** | Fuzzy 0.5550 vs Base 0.7600 | Fuzzy 0.4757 vs Base 0.6360 | Fuzzy 0.6038 vs Base 0.8086 |

### 6.1 Key Methodological Insights
1. **The Fuzzy AUROC Advantage is Robust to Label Semantics:**
   In the official experiment, critics might have questioned whether the $+0.0365$ AUROC advantage was driven by arbitrary pseudo-evidence attached to NEI claims. The sensitivity audit proves the exact opposite: **reclassifying or removing NEI pseudo-evidence substantially increases the fuzzy advantage to $+0.0898$ (Protocol A) and $+0.0827$ (Protocol B)**. Both lower bounds of the 95% marginal and 97.5% Bonferroni CIs are strictly greater than zero.
2. **Reason for Baseline Degradation on Clean Gold Labels:**
   Under dynamic conditions, the baseline scoring formula multiplies by $(1 - \text{contradiction})$. For verifiable claims (`SUPPORTS` and `REFUTES`), when the cross-encoder verifier detects contradiction among retrieved distractors, the baseline adjusted score collapses to 0.0 for all 3 evidence items. In contrast, the fuzzy logic engine's centroid defuzzification smoothly downweights evidence without zeroing out valid items, preserving discriminative ranking power.
3. **Average Precision Trade-off Persists but Narrows:**
   Under both sensitivity protocols, the baseline maintains higher Average Precision under dynamic conditions, but the margin shrinks from $-0.1396$ to $-0.0680$ in Protocol A and $-0.1098$ in Protocol B. This confirms the nuanced scientific conclusion: fuzzy logic optimizes global ranking discrimination (AUROC), while the weighted baseline produces a steeper precision curve on highly ranked subsets.

----

## 7. Limitations of the 808-Passage Closed Candidate Pool vs. Open-Domain Wikipedia Retrieval

A critical threat to validity in scientific dissemination is the conflation of candidate-pool retrieval with open-domain information retrieval.

### 7.1 Structure of the Closed Candidate Pool
- The corpus `data/evidence_corpus.json` contains exactly **808 passage records**.
- It was constructed by pooling the evidence annotations of the 750 claims in `data/fever_dev.jsonl`.
- For the 334 verifiable claims in Experiment 2, all 374 distinct gold passages were placed directly into this 808-passage pool (**100% corpus inclusion rate**).
- The remaining passages serve as local distractors.

### 7.2 Distinguishing Closed-Pool Retrieval from Full-Wikipedia Search
| Dimension | Project Evaluation Setup | True Open-Domain Wikipedia Retrieval |
|:---|:---|:---|
| **Candidate Pool Size** | **808 passages** | **~5.4 million Wikipedia articles (>30 million sentences)** |
| **Gold Availability** | **100.0% guaranteed present** | Search space requires document discovery + sentence extraction |
| **Top-3 Retrieval Recall** | **84.73% (verifiable claims)** | Standard dense retrievers achieve ~50%–65% Top-3 sentence recall on FEVER |
| **Evaluation Focus** | **Within-context evidence attribution & ranking** | End-to-end open-domain information retrieval |

> [!IMPORTANT]
> **Formal Boundary Statement:** All retrieval recall metrics (84.7% for verifiable claims; 86.0% in official pool) and all downstream evidence discrimination metrics (AUROC 0.7630 / 0.7777) must be presented with the explicit qualifier: *'Conditioned on a closed candidate pool of 808 passages'*. They must never be claimed as full open-domain Wikipedia retrieval recall.

----

## 8. Summary of Generated Sensitivity Artifacts

All sensitivity analysis outputs have been generated cleanly and preserved in `outputs/sensitivity_nei_label_audit/`:
```
outputs/sensitivity_nei_label_audit/
├── results_nei_sensitivity.csv       # 1,500 rows with original columns + 4 audit columns
├── summary_sensitivity.json          # Machine-readable metrics, CIs, p-values, Holm corrections
└── plots/
    ├── sensitivity_roc_curves.png    # High-resolution ROC curves for Protocol A and Protocol B
    └── sensitivity_pr_curves.png     # High-resolution PR curves for Protocol A and Protocol B
```

## 9. Mandatory Phase Gate Freeze

In strict compliance with user instructions:
- No existing project source code was modified.
- No frozen experimental outputs under `outputs/phase10_formal/` or `outputs/phase11_formal/` were modified or overwritten.
- No models, retrievers, or explainers were rerun.
- Phase 14 has **NOT** been started.
- All sensitivity data and findings are preserved for user review.
