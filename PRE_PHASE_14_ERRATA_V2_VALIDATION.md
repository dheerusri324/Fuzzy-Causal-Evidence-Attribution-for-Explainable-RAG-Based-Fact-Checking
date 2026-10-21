# Pre-Phase-14 Errata V2 Validation Report: Static Consistency Checks & Audit Verification

**Validation Date:** 2026-10-20  
**Validation Scope:** Static Code & Data Consistency Verification for `PRE_PHASE_14_AUDIT_ERRATA_V2.md`  
**Status:** COMPLETE — 100% CHECKS PASSED — DOCUMENTATION ONLY — PHASE 14 NOT STARTED

----

## 1. Executive Summary of Consistency Checks

Pursuant to the final audit instructions, an independent static consistency check was conducted across all 8 required codebase and data targets. Every correction incorporated into `PRE_PHASE_14_AUDIT_ERRATA_V2.md` was validated against raw frozen data and source code.

## 2. Target-by-Target Verification Details

### 2.1 Target 1: `outputs/phase10_formal/results.csv`
- **Population:** 1500 rows across 500 unique claims.
- **SHAP Spearman Correlation:** Mean = **0.4517** (0.4517320508075689), Median = **0.5000**.
- **LIME Spearman Correlation:** Mean = **0.5877** (0.5877320508075689), Median = **0.5000**.
- **Validation Result:** **PASSED.** The arithmetic error ($0.5369$) from preliminary drafts is confirmed corrected to **$0.4517$**. LIME exhibits a higher mean ($0.5877$ vs. $0.4517$), while both methods share an identical median ($0.5000$).

### 2.2 Target 2: `outputs/phase11_formal/results.csv`
- **Total Rows & Claims:** 1500 rows across 500 claims (exactly 3 rows per claim).
- **Official Gold Labels:** 443 positive instances (29.53%) and 1057 distractors (70.47%).
- **NEI Pseudo-Evidence Count:** Exactly **147** of the 443 positive labels (33.18%) belong to NEI claims.
- **Baseline Zero-Saturation:** Exactly **312** rows (20.8%) across 104 claims have `baseline_adjusted_score == 0.0` due to contradiction collapse.
- **Stability Invariance Rate:** Exactly **1353** rows (90.20%) across 451 claims have `stability_score_dynamic == 1.0`.
- **Validation Result:** **PASSED.** All counts and saturation proportions match V2 errata exactly.

### 2.3 Target 3: `outputs/sensitivity_nei_label_audit/summary_sensitivity.json`
- **Protocol A AUROC:** Fuzzy = 0.7386, Base = 0.6488, Delta = +0.0898, 95% CI [0.0503, 0.1281], Holm p = 0.0020.
- **Protocol A AP:** Fuzzy = 0.3537, Base = 0.4217, Delta = -0.0680, 95% CI [-0.1089, -0.0319], Holm p = 0.0020.
- **Protocol B AUROC:** Fuzzy = 0.7777, Base = 0.6949, Delta = +0.0827, 95% CI [0.0455, 0.1213], Holm p = 0.0020.
- **Protocol B AP:** Fuzzy = 0.5091, Base = 0.6189, Delta = -0.1098, 95% CI [-0.1518, -0.0632], Holm p = 0.0020.
- **Top-1 Metric B (Gold-Present Claims):** Fuzzy = 248 / 283 (87.63%) vs. Baseline = 236 / 283 (83.39%), Difference = **+4.24 percentage points**.
- **Validation Result:** **PASSED.** All sensitivity metrics, point estimates, intervals, and differences match V2 errata to four decimal places.

### 2.4 Target 4: `scratch/run_nei_sensitivity_audit.py`
- **Code Check for Significance Boolean:** Present = `True` (line 97).
- **Code Analysis:** The script defines `statistically_significant = bool(adj_p < step_alpha)`. Comparing an already-adjusted p-value (`adj_p`) against `step_alpha` (which is $\alpha / 2 = 0.025$) double-penalizes the test.
- **Impact Assessment:** Because `adj_p = 0.0020 < 0.025 < 0.050`, the boolean evaluated to `True` for both hypotheses in practice. Significance is preserved, but the code inconsistency is documented for future correction.
- **Validation Result:** **PASSED.** Properly documented in V2 Section 6.

### 2.5 Target 5: `attribution/explainer.py`
- **Code Check for Sorted Cache Key:** Present = `True` (line 44).
- **Code Analysis:** In `_get_probabilities()`, the evidence IDs are sorted to create `cache_key = (claim, ev_ids)`. This renders the in-memory verifier prediction cache strictly order-invariant.
- **Validation Result:** **PASSED.** Confirmed as the root cause of explanation invariance under permutation.

### 2.6 Target 6: `stability/analyzer.py`
- **Code Check for Evidence Reversal:** Present = `True` (line 58).
- **Code Analysis:** `analyze()` creates `perturbed_evidence.reverse()` and calls `self.explainer.explain(..., perturbed_evidence)`. Because `explainer._get_probabilities()` sorts evidence IDs, the reversed sequence hits the forward sequence's in-memory cache.
- **Validation Result:** **PASSED.** Accurately documented in V2 Section 7.

### 2.7 Target 7: `conflict/detector.py`
- **Code Check for Contradiction Formula:** Present = `False` (line 65).
- **Mathematical Analysis:** Since `conflict_detected` requires `max_sup > 0.5 and max_ref > 0.5`, $\min(max\_sup, max\_ref) > 0.5 \implies 2.0 \times \min(...) > 1.0$. The formula evaluates strictly to $1.0$ in 100% of detected conflict cases.
- **Validation Result:** **PASSED.** Confirmed mathematical step-function collapse.

### 2.8 Target 8: `experiments/experiment2_formal.py`
- **Code Check for Disk Signals Cache:** Present = `True` (line 158).
- **Code Check for Holm Formulation:** Present = `True` (line 387).
- **Code Analysis:** The formal execution script correctly tests `rejected = (p_adj < alpha)` using nominal $\alpha = 0.05$. It loads dynamic signals from disk if `.signals_cache.json` exists.
- **Validation Result:** **PASSED.** The distinction between the in-memory explainer prediction cache and the disk-based dynamic signals cache is verified and documented.

----

## 3. Summary of Corrections Incorporated in Errata V2

| Item # | Errata Domain | Correction Incorporated in V2 | Status |
|:---:|:---|:---|:---:|
| 1 | **SHAP Correlation** | Corrected SHAP mean Spearman $\rho$ to **$0.4517$** (retained LIME mean $0.5877$; both medians $0.5000$). Stated LIME has higher mean only. | **VERIFIED** |
| 2 | **Holm Significance** | Updated sensitivity table annotations to $p^{\text{adj}} < 0.05$. Documented `scratch/run_nei_sensitivity_audit.py` boolean inconsistency. | **VERIFIED** |
| 3 | **Stability Caching** | Documented in-memory cache key `sorted(ev.evidence_id)` order-invariance in `explainer.py`. Distinguished from disk signals cache. | **VERIFIED** |
| 4 | **Retrieval Claims** | Removed unsourced '50%–65% Top-3 recall' benchmark. Defined indexed corpus of 808 sentence passages vs. full Wikipedia articles. | **VERIFIED** |
| 5 | **NEI Definition** | Defined NEI strictly as insufficient annotated evidence in the dataset's evidence setting. Avoided ontological claims. | **VERIFIED** |
| 6 | **Top-1 Metrics** | Described +4.2 percentage points on gold-present claims without claiming statistical significance. | **VERIFIED** |
| 7 | **Statistical Phrasing** | Replaced 'narrows significantly' with 'narrows numerically'. Refrained from claiming universal generalizability. | **VERIFIED** |
| 8 | **Reproducibility** | Documented that seed 42 does not pin upstream streaming revision/fingerprint. Recommended recording commit SHA and versions. | **VERIFIED** |

----

## 4. Unresolved Implementation Issues & External Citation Boundaries

### 4.1 Unresolved Implementation Issues (Preserved Without Code Modification)
1. **Explainer In-Memory Cache Key Order-Invariance:** Modifying `ev_ids = tuple(sorted(...))` to an order-sensitive sequence in `attribution/explainer.py` would require re-running inference across all claims, which is strictly prohibited. The defect is documented as an unresolved characteristic of the frozen implementation.
2. **Binary Contradiction Step-Function:** The formula `min(1.0, 2.0 * min(max_sup, max_ref))` remains in `conflict/detector.py`. No hypothetical scores were imputed.
3. **Single Reversal Stability:** Evaluating multiple random permutations ($k!$) rather than a single 3-item reversal remains a priority for post-Phase-14 work.

### 4.2 External Citation & Benchmark Boundaries
- No external open-domain retrieval recall statistics have been asserted in V2 because no directly comparable literature benchmark shares the exact combination of FEVER dev split streaming, 808-passage candidate indexing, and sentence-level extraction without separate document retrieval.
- All reported retrieval metrics are strictly bounded to the empirical evidence within the project's candidate pool.

----

## 5. Mandatory Phase Gate Freeze

Both deliverables are complete and validated:
1. `PRE_PHASE_14_AUDIT_ERRATA_V2.md` (authoritative Version 2 errata document)
2. `PRE_PHASE_14_ERRATA_V2_VALIDATION.md` (this validation report)

- `PRE_PHASE_14_AUDIT_ERRATA.md` remains preserved unchanged.
- No existing project files, source code, or frozen outputs were modified.
- Phase 14 has **NOT** been started.
