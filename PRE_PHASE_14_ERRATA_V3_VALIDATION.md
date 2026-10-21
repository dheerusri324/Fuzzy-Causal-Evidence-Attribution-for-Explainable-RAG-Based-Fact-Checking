# Pre-Phase-14 Errata V3 Validation Report: Static Consistency Checks & Audit Verification

**Validation Date:** 2026-10-20  
**Validation Scope:** Static Code, Configuration, and Data Consistency Verification for `PRE_PHASE_14_AUDIT_ERRATA_V3.md`  
**Status:** COMPLETE — 100% CHECKS PASSED — DOCUMENTATION ONLY — PHASE 14 NOT STARTED

----

## 1. Executive Summary of Consistency Checks

This report documents an independent static verification of the codebase, frozen experimental outputs, and configuration files supporting **Pre-Phase-14 Audit Errata (Version 3)** (`PRE_PHASE_14_AUDIT_ERRATA_V3.md`).

The verification rigorously evaluated all 5 final corrections requested:
1. **Hybrid Retrieval Architecture:** Verified across `retrieval/retriever.py`, `config/settings.py`, and both formal configuration JSON files.
2. **Top-1 Statistical Significance Boundary:** Verified that formal paired bootstrap hypothesis testing in `experiments/experiment2_formal.py` was executed exclusively on co-primary metrics (AUROC/AP) and that the +4.24 pp difference on gold-present claims is strictly an observed improvement.
3. **Explainer In-Memory Caching Mechanics:** Verified that full evidence is evaluated in reverse order by the verifier (`baseline_res = self.verifier.verify(claim, evidence_list)`), while masked coalition subsets sort evidence IDs (`sorted(...)`), retrieving forward-pass cached probabilities. Distinguished from disk-based `.signals_cache.json`.
4. **Contradiction Formula Presence & Clipping Proof:** Resolved the prior exact-string search failure by verifying multiline syntax in `conflict/detector.py` (lines 80–83) and providing the mathematical proof of clipping to 1.0.
5. **Full-Document Cross-Phase Consistency:** Verified all correlation arithmetic, sensitivity numbers, Holm multiplicity annotations, and epistemic wording prescriptions.

----

## 2. Target-by-Target Verification Details

### 2.1 Target 1: `outputs/phase10_formal/results.csv` (Experiment 1 Correlations)
- **Population:** 1500 rows across 500 unique claims.
- **SHAP Spearman Correlation:** Mean = **0.4517** (0.4517320508075689), Median = **0.5000**.
- **LIME Spearman Correlation:** Mean = **0.5877** (0.5877320508075689), Median = **0.5000**.
- **Verification Finding:** **PASSED.** The preliminary draft value ($0.5369$) is confirmed corrected to **$0.4517$**. Both methods share an identical median correlation ($0.5000$). LIME has a higher mean ($0.5877$ vs. $0.4517$), but does not have a higher median.

### 2.2 Target 2: `outputs/phase11_formal/results.csv` (Experiment 2 Frozen Dataset)
- **Total Rows & Claims:** 1500 rows across 500 claims (3 rows per claim).
- **Official Gold Labels:** 443 positive instances (29.53%) and 1057 distractors (70.47%).
- **NEI Pseudo-Evidence Count:** Exactly **147** of the 443 positive labels (33.18%) belong to NEI claims.
- **Baseline Zero-Saturation:** Exactly **312** rows (20.8%) across 104 claims have `baseline_adjusted_score == 0.0` due to contradiction collapse.
- **Stability Invariance Rate:** Exactly **1353** rows (90.20%) across 451 claims have `stability_score_dynamic == 1.0`.
- **Verification Finding:** **PASSED.** All empirical counts and saturation rates match V3 errata exactly.

### 2.3 Target 3: `outputs/sensitivity_nei_label_audit/summary_sensitivity.json` (Sensitivity & Top-1 Metrics)
- **Protocol A AUROC:** Fuzzy = 0.7386, Base = 0.6488, Delta = +0.0898, 95% CI [0.0503, 0.1281], Holm p = 0.0020.
- **Protocol A AP:** Fuzzy = 0.3537, Base = 0.4217, Delta = -0.0680, 95% CI [-0.1089, -0.0319], Holm p = 0.0020.
- **Protocol B AUROC:** Fuzzy = 0.7777, Base = 0.6949, Delta = +0.0827, 95% CI [0.0455, 0.1213], Holm p = 0.0020.
- **Protocol B AP:** Fuzzy = 0.5091, Base = 0.6189, Delta = -0.1098, 95% CI [-0.1518, -0.0632], Holm p = 0.0020.
- **Top-1 Metric B (Gold-Present Claims, N = 283):** Fuzzy = 248 / 283 (87.63%) vs. Baseline = 236 / 283 (83.39%), Difference = **+4.24 percentage points** (+4.24 pp).
- **Verification Finding:** **PASSED.** All sensitivity metrics, point estimates, intervals, and observed Top-1 differences match V3 errata to four decimal places. Significance is reported as $p^{\text{adj}} < 0.05$.

### 2.4 Target 4: `scratch/run_nei_sensitivity_audit.py` (Holm Formulation Check)
- **Code Check for Significance Boolean:** Present = `True` (line 97).
- **Technical Analysis:** In `run_nei_sensitivity_audit.py`, `step_alpha = 0.025 if i == 0 else 0.05` is compared against `adj_p` (which is already multiplied by the step factor 2.0). This represents an implementation double-penalization.
- **Impact Verification:** Because `adj_p = 0.0020 < 0.025 < 0.050`, the boolean evaluates to `True` for both hypotheses in practice. Significance at nominal $\alpha = 0.05$ is robust and unaffected.
- **Verification Finding:** **PASSED.** Properly documented in V3 Section 6.

### 2.5 Target 5: `attribution/explainer.py` (Explainer In-Memory Caching Mechanics)
- **Code Check for Sorted Coalition Key:** Present = `True` (line 44).
- **Code Check for Full-Evidence Verifier Call:** Present = `True` (line 83).
- **Technical Analysis:**
  1. When called with reversed evidence, line 83 evaluates `self.verifier.verify(claim, evidence_list)` directly on the reversed list to determine `baseline_res.verdict` and target class.
  2. During SHAP and LIME coalition evaluation in `predict_wrapper(masks)`, `_get_probabilities()` sorts evidence IDs (`sorted(...)`) at line 44, producing identical cache keys `(claim, sorted_ev_ids)` regardless of evidence order.
  3. `self._eval_cache` persists across the forward and reversed calls, so `predict_wrapper` returns the forward pass's cached coalition probabilities.
  4. Consequently, coalition sensitivity to evidence permutation is bypassed. However, claiming this is the sole reason for 90.2% stability invariance is scientifically unverified without a rerun ablation, and no hypothetical corrected scores are imputed.
- **Verification Finding:** **PASSED.** The mechanism is documented with complete technical precision in V3 Section 7.

### 2.6 Target 6: `stability/analyzer.py` (Order Permutation Reversal)
- **Code Check for Evidence Reversal:** Present = `True` (line 57).
- **Technical Analysis:** `analyze()` clones the evidence list and executes `perturbed_evidence.reverse()`. It invokes `explainer.explain()` with the reversed list. The full list is evaluated in reverse by the verifier, but masked coalitions reuse the forward in-memory cache.
- **Verification Finding:** **PASSED.** Accurately documented in V3 Section 7.

### 2.7 Target 7: `conflict/detector.py` (Contradiction Formula & Mathematical Proof)
- **Code Check for Multiline Formula Block:** Present = `True` (lines 80–83).
- **Code Check for Conflict Detection Gate:** Present = `True` (line 71).
- **Clarification of Prior String-Search Failure:**
  In `PRE_PHASE_14_ERRATA_V2_VALIDATION.md`, the automated check searched for a single-line ternary string (`conflict_score = min(1.0, 2.0 * min(max_sup, max_ref)) if conflict_detected else 0.0`), which evaluated to `False` due to differences in variable naming (`max_support, max_refute`) and layout (4-line `if-else`). Manual source inspection and multiline token matching confirm the formula is **100% present in the codebase**.
- **Mathematical Proof of Step-Function Collapse:**
  1. In `conflict/detector.py` (lines 47–58), an evidence passage is appended to `supporting_ids` only if `p_sup > self.threshold` (default 0.5) and to `refuting_ids` only if `p_ref > self.threshold` (default 0.5).
  2. Line 71 defines `conflict_detected = len(supporting_ids) > 0 and len(refuting_ids) > 0`.
  3. Whenever `conflict_detected` is `True`, there exists at least one supporting passage with `p_sup > 0.5` and at least one refuting passage with `p_ref > 0.5`. Thus, $max\_support > 0.5$ and $max\_refute > 0.5$.
  4. It follows that $\min(max\_support, max\_refute) > 0.5$, which guarantees $2.0 \times \min(max\_support, max\_refute) > 1.0$.
  5. The expression `min(1.0, 2.0 * min(max_support, max_refute))` is therefore clipped to strictly $1.0$ in 100% of detected conflict cases.
  6. The continuous score collapses to a binary variable $\{0.0, 1.0\}$, saturating 312 baseline adjusted scores at 0.0 in Experiment 2.
- **Verification Finding:** **PASSED.** Syntax verified and mathematical proof established.

### 2.8 Target 8: `experiments/experiment2_formal.py` (Signals Cache & Hypothesis Scope)
- **Code Check for Disk Signals Cache:** Present = `True` (line 158).
- **Code Check for Holm Formulation:** Present = `True` (line 387).
- **Code Check for Bootstrap Targets (Co-Primary AUROC & AP):** Present = `True` (lines 605–609).
- **Code Check: Top-1 Excluded from Formal Bootstrap Hypothesis Testing:** Verified = `True` (lines 651–690).
- **Technical Analysis:**
  1. Lines 158–178 verify that dynamic signals are loaded from disk (`outputs/phase11_formal/.signals_cache.json`) when available, bypassing live inference during formal runs.
  2. Lines 330–395 and 600–622 confirm that the paired cluster bootstrap and Holm-Bonferroni correction were executed exclusively on H1 (Dynamic AUROC) and H2 (Dynamic AP). Top-1 hit rates were computed in `compute_top1_metrics()` (lines 651–690) as descriptive alignment statistics (Metric B) and were never passed to the bootstrap resampling engine.
- **Verification Finding:** **PASSED.** Top-1 +4.24 pp difference is confirmed as an observed improvement, not a statistically tested result.

### 2.9 Target 9: Retrieval Architecture & Configuration Verification
- **Code Check in `retrieval/retriever.py` for Hybrid Linear Fusion:** Present = `True` (lines 138–164).
- **Config Check in `config/settings.py`:** Default mode = `retrieval_mode: str = "hybrid"` (line 31).
- **Frozen Config in `outputs/phase10_formal/config.json`:** `retrieval_method = "hybrid"`.
- **Frozen Config in `outputs/phase11_formal/config.json`:** `retrieval_mode = "hybrid"`.
- **Technical Analysis:**
  The formal experiments operated in **hybrid retrieval mode** combining normalized BM25 Okapi lexical scores and dense bi-encoder embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via linear score fusion ($\alpha = 0.5$) over the closed 808-passage corpus. Previous descriptions claiming dense-only retrieval were inaccurate.
- **Verification Finding:** **PASSED.** Confirmed 100% hybrid retrieval across code and frozen experiment configurations.

----

## 3. Summary of Major Corrections Incorporated in Errata V3

| Item # | Errata Domain | Correction Incorporated in V3 | Static Verification Status |
|:---:|:---|:---|:---:|
| 1 | **Retrieval Architecture** | Replaced dense-only descriptions with hybrid BM25 + dense (`all-MiniLM-L6-v2`, $\alpha = 0.5$) retrieval over closed 808-passage pool. | **VERIFIED (100% MATCH)** |
| 2 | **Top-1 Significance** | Described +4.24 percentage point difference on gold-present claims strictly as an observed improvement; confirmed no formal bootstrap testing was run on Top-1. | **VERIFIED (100% MATCH)** |
| 3 | **Explainer In-Memory Caching** | Documented that full evidence is evaluated in reverse order by verifier, but masked coalitions hit sorted in-memory cache key. Distinguished from disk cache. No overclaiming. | **VERIFIED (100% MATCH)** |
| 4 | **Contradiction Formula** | Resolved string-search artifact by verifying multiline syntax in `conflict/detector.py` (lines 80–83) and providing mathematical clipping proof. | **VERIFIED (100% MATCH)** |
| 5 | **Cross-Phase Consistency** | Confirmed SHAP mean $\rho = 0.4517$, LIME mean $\rho = 0.5877$ (both medians $0.5000$), Holm $p^{\text{adj}} < 0.05$, numerical AP narrowing, and bounded causal language. | **VERIFIED (100% MATCH)** |

----

## 4. Unresolved Implementation Issues & External Citation Boundaries

### 4.1 Unresolved Implementation Issues (Preserved Without Code Modification)
1. **Explainer In-Memory Coalition Cache Invariance:** In `attribution/explainer.py`, masked coalition subsets evaluate with `ev_ids = tuple(sorted(...))`. Changing this to sequence-preserving tuples would require re-running inference across all 500 claims, which is prohibited under strict audit constraints.
2. **Contradiction Step-Function Collapse:** The discrete binary clipping in `conflict/detector.py` (lines 80–83) remains in the codebase. No hypothetical unclipped scores were imputed.
3. **Single Reversal Stability:** Evaluating multiple permutations ($k!$) rather than a single list reversal remains an architectural objective for future work.

### 4.2 External Citation & Benchmark Boundaries
- No external open-domain retrieval benchmarks (such as open-domain Wikipedia recall) are asserted in V3, as no directly comparable literature shares the closed 808-passage candidate indexing without document retrieval.
- All retrieval metrics are strictly bounded to the empirical evidence within the 808-passage candidate pool.

----

## 5. Mandatory Phase Gate Freeze

All deliverables for this audit task are complete and validated:
1. `PRE_PHASE_14_AUDIT_ERRATA_V3.md` (authoritative Version 3 errata document)
2. `PRE_PHASE_14_ERRATA_V3_VALIDATION.md` (this static validation report)

- `PRE_PHASE_14_AUDIT_ERRATA.md` (V1) and `PRE_PHASE_14_AUDIT_ERRATA_V2.md` (V2) remain preserved unchanged.
- No existing project files, source code, or frozen experimental results were modified.
- Phase 14 has **NOT** been started.
