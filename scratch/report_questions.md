
## 6. Task 4 — Detailed Technical Audit of the 12 Core Questions

This section provides exhaustive, evidence-backed evaluations for each of the 12 technical, methodological, and implementation questions formulated for this independent audit.

### Question 1: Signed Attribution Values vs. Absolute Attribution Magnitudes in Experiment 1

**Audit Prompt:** *Does Experiment 1 calculate signed SHAP/LIME values and absolute attribution magnitudes consistently? Which exact columns are used for ranking and correlation?*

- **Severity:** Low / Informational (Code is mathematically sound; early documentation lacked explicit notation)
- **Exact File & Line Reference:**
  - `experiments/experiment1_formal.py`, lines 145–188
  - `fuzzy/normalization.py`, lines 18–34
  - `outputs/phase10_formal/results.csv`, columns: `SHAP_score_raw`, `LIME_score_raw`, `SHAP_magnitude`, `LIME_magnitude`, `SHAP_magnitude_rank`, `LIME_magnitude_rank`, `intervention_rank`
- **Relevant Code & Data Evidence:**
  In `experiments/experiment1_formal.py`:
  ```python
  # Line 152: Raw signed model explanations
  shap_scores = shap_explainer.explain(rec.claim, retrieved_texts)
  lime_scores = lime_explainer.explain(rec.claim, retrieved_texts)
  ...
  # Lines 170-175: Absolute magnitude transformation
  shap_mags = [abs(s) for s in shap_scores]
  lime_mags = [abs(s) for s in lime_scores]
  inf_mags  = [abs(s) for s in influences] # influences are >= 0 by definition
  ...
  # Lines 184-188: Magnitude-based ranking and Spearman correlation
  shap_ranks = rankdata([-x for x in shap_mags], method='min')
  lime_ranks = rankdata([-x for x in lime_mags], method='min')
  inf_ranks  = rankdata([-x for x in inf_mags],  method='min')
  rho_shap, _ = spearmanr(shap_mags, inf_mags)
  rho_lime, _ = spearmanr(lime_mags, inf_mags)
  ```
  Empirical validation across all 1,500 rows in `outputs/phase10_formal/results.csv` confirms:
  - `SHAP_score_raw`: signed range $[-0.8286, +0.9990]$ with 329 negative values.
  - `LIME_score_raw`: signed range $[-0.1851, +0.2682]$ with 304 negative values.
  - `SHAP_magnitude`: strictly positive range $[3.92 \times 10^{-6}, 0.9990]$.
  - `LIME_magnitude`: strictly positive range $[3.19 \times 10^{-7}, 0.2682]$.
  - In `fuzzy/normalization.py` (lines 20–28), `normalize_attribution(val)` explicitly calls `abs(float(val))` prior to range clipping.
- **Why it is a Problem (if verified):**
  Leave-one-out counterfactual intervention influence $\Delta p = |p_{\text{base}} - p_{\text{abl}}|$ measures overall importance magnitude, agnostic to class polarity. Had ranking or Spearman correlation used signed attribution values, strong negative attributions (evidence strongly driving a 'REFUTES' verdict) would have been ranked as 'lowest importance', completely corrupting attribution fidelity metrics. The code implemented the transformation correctly. However, early Phase 0–5 informal documentation referred loosely to 'SHAP score' without prefixing 'absolute magnitude'.
- **Status:** Verified Fact (Implementation is mathematically consistent; documentation ambiguity resolved in Phase 12/13).
- **Minimal Proposed Remedy:** Retain frozen calculations. Consistently use $|A_{\text{SHAP}}|$ and $|A_{\text{LIME}}|$ notation in all Phase 14 tables and presentations.

### Question 2: Per-Claim Rank Statistics, Ties, Undefined Cases, and Top-1 Agreement

**Audit Prompt:** *Are per-claim rank statistics, ties, undefined cases, and top-1 agreement handled consistently with the documented protocol?*

- **Severity:** Low / Pass (Empirically verified zero rank-1 ties and zero undefined correlations)
- **Exact File & Line Reference:**
  - `experiments/experiment1_formal.py`, lines 180–210
  - `experiments/experiment2_formal.py`, lines 210–245
  - `outputs/phase12_analysis/dataset_overlap_audit.json`, section `attribution_and_disagreement_verification`
- **Relevant Code & Data Evidence:**
  In `experiments/experiment1_formal.py`:
  - Ranks are assigned using `scipy.stats.rankdata([-x for x in mags], method='min')`.
  - Top-1 set agreement evaluates `bool(shap_top1 & inf_top1)`.
  - Empirical tie check across all 500 claims in `outputs/phase10_formal/results.csv` reveals:
    - `shap_ties_at_rank1 == 0` (0.0%)
    - `lime_ties_at_rank1 == 0` (0.0%)
    - `inf_ties_at_rank1 == 0` (0.0%)
  - Undefined Spearman $\rho$: If all 3 items have identical values (variance = 0), `scipy.stats.spearmanr` returns `NaN`. In the formal 500-claim dataset, all 500 claims exhibited distinct non-zero variance; exactly 500/500 correlations were valid finite numbers.
  In `experiments/experiment2_formal.py`:
  - Within-claim ranking enforces a strict multi-key tie-breaker: `df.sort_values(by=[score_col, 'retrieval_score', 'retrieval_rank'], ascending=[False, False, True])`.
  - This guarantees strictly unique ordinal ranks $\{1, 2, 3\}$ for every claim, preventing any multi-item top-1 ties.
- **Why it is a Problem (if verified):**
  If rank-1 ties had occurred in Experiment 1 under `method='min'`, set intersection could have artificially inflated top-1 agreement rates by allowing multiple items to count as top-1. Because zero rank-1 ties occurred in practice, the reported top-1 agreement rates (82.4% direct SHAP–LIME agreement; 72.2% SHAP–Intervention; 84.0% LIME–Intervention) are strictly exact and uninflated.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** Explicitly document the deterministic three-key sorting cascade in all methodological descriptions.

### Question 3: Preservation of Claim-Level Clusters in Experiment 2 Bootstrap Resampling

**Audit Prompt:** *Does Experiment 2 correctly preserve claim-level clusters when resampling the 1,500 evidence evaluations?*

- **Severity:** Low / Pass (Cluster structure strictly preserved)
- **Exact File & Line Reference:**
  - `experiments/experiment2_formal.py`, lines 280–320
- **Relevant Code & Data Evidence:**
  In `experiments/experiment2_formal.py`:
  ```python
  # Group evidence rows by claim cluster
  claim_ids = df['claim_id'].unique() # M = 500 clusters
  claim_dict = {cid: df[df['claim_id'] == cid] for cid in claim_ids}
  ...
  for b in range(n_bootstraps):
      # Resample clusters with replacement
      sampled_cids = np.random.choice(claim_ids, size=len(claim_ids), replace=True)
      # Concatenate entire 3-row cluster for each sampled claim
      boot_df = pd.concat([claim_dict[cid] for cid in sampled_cids], ignore_index=True)
  ```
  At every replicate $b$, exactly 500 claims are sampled with replacement, and each sampled claim brings its complete triplet of 3 retrieved evidence passages (totaling exactly 1,500 rows per replicate). Intra-cluster correlation and within-claim ranking structures are 100% preserved.
- **Why it is a Problem (if verified):**
  Naively resampling 1,500 individual evidence rows would violate independence assumptions, destroy within-claim ranking constraints, and generate invalid standard error estimates. Cluster bootstrap at the claim level is the mathematically required procedure (Cameron et al., 2008).
- **Status:** Verified Fact (Correctly implemented).
- **Minimal Proposed Remedy:** Retain implementation; cite cluster bootstrap methodology in Phase 14 documentation.

### Question 4: Statistical Inference, Bootstrap P-Values, CIs, and Holm–Bonferroni Correction

**Audit Prompt:** *Are bootstrap p-values, confidence intervals, degenerate replicates, and Holm–Bonferroni corrections implemented exactly as specified?*

- **Severity:** Low / Pass (Correctly implemented according to approved protocol)
- **Exact File & Line Reference:**
  - `experiments/experiment2_formal.py`, lines 325–398
  - `outputs/phase11_formal/summary.json`, section `co_primary_results_dynamic_pipeline`
  - `outputs/phase11_formal/config.json`, line `"n_bootstraps": 1000`
- **Relevant Code & Data Evidence:**
  In `experiments/experiment2_formal.py`:
  - Null-Shift Centering: $\tilde{\Delta}^*_b = \Delta^*_b - \hat{\Delta}$. Centering the bootstrap distribution around 0 enforces the null hypothesis $H_0: \Delta = 0$.
  - Two-Sided Empirical P-Value:
    $$p = \frac{1 + \sum_{b=1}^{B_{\text{valid}}} \mathbb{I}(|\tilde{\Delta}^*_b| \ge |\hat{\Delta}|)}{B_{\text{valid}} + 1}$$
    With $B_{\text{valid}} = 1,000$ and $B_{\text{degenerate}} = 0$:
    - H1 (AUROC): $\hat{\Delta} = +0.0365$, raw $p = 0.0240$.
    - H2 (Average Precision): $\hat{\Delta} = -0.1396$, raw $p = 0.0010$.
  - Confidence Intervals: Computed via empirical percentiles:
    - 95% Marginal CI: $[q_{0.025}, q_{0.975}]$. For AUROC: $[0.0053, 0.0686]$. For AP: $[-0.1777, -0.1025]$.
    - 97.5% Bonferroni CI: $[q_{0.0125}, q_{0.9875}]$. For AUROC: $[0.00006, 0.0710]$. For AP: $[-0.1856, -0.0988]$.
  - Holm–Bonferroni Step-Down Procedure:
    Hypotheses ordered by raw p-value: $p_{(1)} = 0.0010$ (H2), $p_{(2)} = 0.0240$ (H1).
    - Step 1 (H2): Target $\alpha / 2 = 0.025$. Adjusted $p = 2 \times 0.0010 = 0.0020 < 0.025 \implies$ Significant.
    - Step 2 (H1): Target $\alpha / 1 = 0.050$. Adjusted $p = \max(0.0020, 1 \times 0.0240) = 0.0240 < 0.050 \implies$ Significant.
    Family-wise error rate (FWER) is strictly controlled at $\alpha = 0.05$.
- **Why it is a Problem (if verified):**
  Uncentered bootstrap p-values $2 \min(P(\Delta^* \le 0), P(\Delta^* \ge 0))$ can be severely biased when the bootstrap distribution is skewed. The null-shifted pivot test correctly centers the distribution, ensuring exact Type I error control.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** Clarify in reporting text that while the initial design proposal considered $B=2,000$, the formal frozen execution script used $B=1,000$ valid replicates as configured in `outputs/phase11_formal/config.json`.

### Question 5: Instance and Label Parity Between Fuzzy and Baseline Systems

**Audit Prompt:** *Are fuzzy scores and baseline scores calculated from the same evidence instances and labels under each comparison condition?*

- **Severity:** Low / Pass (Strict mathematical and instance parity maintained)
- **Exact File & Line Reference:**
  - `experiments/experiment2_formal.py`, lines 160–230
  - `outputs/phase11_formal/results.csv`, columns: `claim_id`, `evidence_id`, `is_annotated_gold`, `fuzzy_evidence_reliability`, `baseline_adjusted_score`
- **Relevant Code & Data Evidence:**
  In `experiments/experiment2_formal.py`:
  - For every claim and every retrieved passage, a single `FuzzyInput` instance is instantiated.
  - Both `fuzzy_engine.evaluate(fuzzy_input)` and `baseline_engine.evaluate(fuzzy_input)` are invoked sequentially on the exact same instance.
  - Binary gold annotation labels (`is_annotated_gold`) are extracted directly from the ground truth FEVER annotations in `data/fever_dev.jsonl`.
  - Total evaluation rows: exactly 1,500. Exactly 443 gold instances and 1,057 distractors for both methods.
- **Why it is a Problem (if verified):**
  If fuzzy scores and baseline scores were evaluated on divergent subsets, or if label definitions differed, the comparative AUROC and AP results would be invalid. Here, identical inputs and labels are guaranteed.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** None required.

### Question 6: Static-Default Ablation Isolation

**Audit Prompt:** *Is the static-default ablation isolated correctly? Could any other pipeline input differ between conditions?*

- **Severity:** Medium (Dual-input ablation; not a single-factor ablation)
- **Exact File & Line Reference:**
  - `experiments/experiment2_formal.py`, lines 220–235
  - `outputs/phase11_formal/results.csv`, columns: `conflict_score_static`, `stability_score_static`, `fuzzy_evidence_reliability_static`, `baseline_adjusted_score_static`
- **Relevant Code & Data Evidence:**
  In `experiments/experiment2_formal.py`:
  ```python
  static_input = copy.deepcopy(primary_input)
  static_input.contradiction_degree = 0.0
  static_input.stability_score = 1.0
  ```
  All other four inputs (`relevance_score`, `attribution_strength`, `intervention_influence`, `source_reliability`) remain bit-for-bit identical to the primary condition.
- **Why it is a Problem (if verified):**
  While no extraneous pipeline inputs differed, the static ablation modifies **two variables simultaneously** ($c = 0.0$ AND $s = 1.0$) rather than isolating contradiction and stability independently. As a result, the massive jump in baseline performance under static defaults (AUROC jumping from 0.7265 to 0.8889; AP jumping from 0.6280 to 0.7600) cannot be strictly attributed to contradiction alone or stability alone through a factorial design. (However, because stability was already 1.0 in 90.2% of claims, the shift was mathematically driven almost entirely by the removal of the contradiction penalty).
- **Status:** Verified Fact (Dual ablation confirmed).
- **Minimal Proposed Remedy:** In all reporting, describe this strictly as the 'Dual Static-Defaults Ablation (Conflict=0.0 & Stability=1.0)'. Recommend a factorial $2 \times 2$ ablation in future work.

### Question 7: Dynamic Signals Computation, Contradiction Binary Collapse, and Baseline Saturation

**Audit Prompt:** *Are contradiction and stability signals actually computed dynamically, or are any values silently hard-coded or cached incorrectly?*

- **Severity:** High (Mathematical formula defect causes contradiction to collapse to a binary variable, driving 312 baseline scores to zero)
- **Exact File & Line Reference:**
  - `conflict/detector.py`, lines 50–70
  - `stability/analyzer.py`, lines 40–65
  - `fuzzy/baseline.py`, lines 25–35
  - `outputs/phase11_formal/results.csv`, columns: `conflict_score_dynamic`, `stability_score_dynamic`, `baseline_adjusted_score`
- **Relevant Code & Data Evidence:**
  **1. Dynamic Contradiction Binary Collapse:**
  In `conflict/detector.py`:
  ```python
  conflict_detected = (max_sup > 0.5 and max_ref > 0.5)
  conflict_score = min(1.0, 2.0 * min(max_sup, max_ref)) if conflict_detected else 0.0
  ```
  *Mathematical Proof of Degeneracy:*
  Whenever `conflict_detected` is True, $max\_sup > 0.5$ and $max\_ref > 0.5$.
  Therefore, $\min(max\_sup, max\_ref) > 0.5$.
  Multiplying by 2.0 yields $2.0 \times \min(max\_sup, max\_ref) > 1.0$.
  Taking $\min(1.0, 2.0 \times \min(...))$ **strictly evaluates to 1.0 in 100% of detected conflict cases**!
  Empirical proof from `outputs/phase11_formal/results.csv`:
  - `conflict_score_dynamic == 0.0`: 1,188 rows (396 claims)
  - `conflict_score_dynamic == 1.0`: 312 rows (104 claims)
  - Value count for any value between 0.0 and 1.0: **EXACTLY ZERO**.

  **2. Consequence on Baseline Saturation:**
  In `fuzzy/baseline.py`:
  ```python
  adjusted_score = baseline_score * (1.0 - contradiction_degree) * stability_score
  ```
  When $c = 1.0$, $(1.0 - 1.0) = 0.0$. Thus, **all 3 evidence passages in all 104 conflict claims have `baseline_adjusted_score == 0.0`** (exactly 312 rows, or 20.8% of the entire 1,500-evidence dataset!). Any gold evidence present in those 104 claims is wiped out to 0.0, severely depressing baseline dynamic AUROC.

  **3. Stability Discreteness:**
  In `stability/analyzer.py`, stability is computed by reversing the 3 retrieved passages ($k=3$) and calculating normalized Spearman rank correlation $s = (\rho + 1)/2$.
  For $k=3$, Spearman $\rho$ can only assume values in $\{-1.0, -0.5, 0.5, 1.0\}$, mapping stability strictly into $\{0.0, 0.25, 0.75, 1.0\}$.
  Empirical counts: $s=1.0$: 1,353 rows (90.2%); $s=0.0$: 105 rows (7.0%); $s=0.25$: 24 rows (1.6%); $s=0.75$: 18 rows (1.2%).
- **Why it is a Problem (if verified):**
  The dynamic conflict detector was specified to provide a continuous fuzzy degree of contradiction in $[0, 1]$, but mathematical clipping collapsed it into a binary indicator. This binary indicator caused extreme zero-saturation in the baseline scoring formula (20.8% zero scores), which explains why the baseline AUROC dropped from 0.8889 (static) to 0.7265 (dynamic).
- **Status:** Verified Fact (Mathematical and empirical defect confirmed).
- **Minimal Proposed Remedy:** In Phase 13/14 documentation, explicitly disclose this mathematical collapse and baseline zero-saturation artifact. Propose an unclipped continuous scaling formula for future iterations (e.g. `2.0 * (min(max_sup, max_ref) - 0.5)` or softmax-based NLI divergence).

### Question 8: Gold-Evidence Corpus Construction and Closed-Pool Evaluation Scope

**Audit Prompt:** *Does the gold-evidence corpus construction create a closed, gold-conditioned evaluation pool? Are retrieval coverage and gold-discrimination metrics labelled with the appropriate scope?*

- **Severity:** Medium (Evaluation scope boundary; must be rigorously hedged)
- **Exact File & Line Reference:**
  - `data/evidence_corpus.json` (808 passages)
  - `data/fever_loader.py`
  - `DATASET_OVERLAP_AUDIT.md`, Section 2 & 4
  - `docs/DATASETS_AND_PROVENANCE.md`
- **Relevant Code & Data Evidence:**
  - The 808-passage corpus in `data/evidence_corpus.json` was constructed by deduplicating all annotated evidence passages across 750 claims in `data/fever_dev.jsonl`.
  - For the 500 evaluated claims, all 534 distinct annotated gold evidence passages were included in the 808-passage corpus (100.0% corpus inclusion rate).
  - The remaining 274 passages in the corpus served as distractors.
  - The resulting Top-3 retrieval coverage was:
    - Claims retrieving all annotated gold evidence: 409 / 500 (81.8%)
    - Claims retrieving partial gold evidence: 21 / 500 (4.2%)
    - Claims retrieving at least one gold evidence: 430 / 500 (86.0%)
    - Claims retrieving zero gold evidence: 70 / 500 (14.0%)
- **Why it is a Problem (if verified):**
  If the 86.0% retrieval recall or the 0.7630 AUROC is presented without the qualifier 'closed candidate pool (808 passages)', readers could mistakenly conclude that the dense retriever achieved 86% recall over open-domain Wikipedia (5.4 million documents). While within-context attribution and ranking remain 100% mathematically valid, the retrieval setup is gold-conditioned.
- **Status:** Verified Fact (Closed-pool construction confirmed; correctly scoped in Phase 12/13 documents).
- **Minimal Proposed Remedy:** Ensure every abstract, executive summary, and slide states: 'Evaluated under a closed candidate pool of 808 FEVER passages'.

### Question 9: Quantitative Reproducibility from Frozen CSV Data

**Audit Prompt:** *Are the reported counts, denominators, percentages, and statistical conclusions reproducible from the frozen raw CSV files?*

- **Severity:** Low / Pass (100% Mathematically and Empirically Reproducible)
- **Exact File & Line Reference:**
  - `outputs/phase10_formal/results.csv`
  - `outputs/phase11_formal/results.csv`
  - `outputs/phase12_analysis/analysis_summary.csv`
  - `outputs/phase11_formal/summary.json`
- **Relevant Code & Data Evidence:**
  Independent recomputation directly from the raw frozen CSV files reproduces all formal metrics to machine precision:
  1. **Population Integrity:** Exactly 500 unique claim IDs, exactly 3 evidence rows per claim, exactly 1,500 total rows in both `results.csv` files.
  2. **Gold Label Counts:** Exactly 443 gold instances (29.53%) and 1,057 distractors (70.47%).
  3. **Retrieval Coverage:** Exactly 430 claims retrieve $\ge 1$ gold piece (86.0%); exactly 70 retrieve 0 gold (14.0%). Exactly 409 retrieve all gold (81.8%).
  4. **Attribution Rank Fidelity (Exp 1):**
     - SHAP Mean Spearman $\rho = 0.5369$, Median $= 0.5000$, Positive $\rho = 78.8\%$.
     - LIME Mean Spearman $\rho = 0.5877$, Median $= 0.5000$, Positive $\rho = 87.0\%$.
     - Direct Top-1 SHAP–LIME agreement: 412 / 500 (82.4%).
     - Contingency: Both agree with intervention = 354 (70.8%); LIME only = 66 (13.2%); SHAP only = 7 (1.4%); Neither = 73 (14.6%).
  5. **Comparative Performance (Exp 2):**
     - Dynamic AUROC: Fuzzy 0.7630 vs Baseline 0.7265 ($\hat{\Delta} = +0.0365$, Holm $p = 0.0240$).
     - Dynamic AP: Fuzzy 0.4884 vs Baseline 0.6280 ($\hat{\Delta} = -0.1396$, Holm $p = 0.0020$).
     - Static AUROC: Fuzzy 0.7946 vs Baseline 0.8889 ($\hat{\Delta} = -0.0943$).
- **Why it is a Problem (if verified):**
  In preliminary Phase 12 working drafts, minor typographical discrepancies appeared (e.g. unrounded LIME median rho 0.4998 vs 0.5000). The final consolidated `analysis_summary.csv` and `summary.json` are 100% verified and reproducible.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** Cite `outputs/phase12_analysis/analysis_summary.csv` and `outputs/phase11_formal/summary.json` as the definitive numerical sources.

### Question 10: Scientific Overclaiming Boundaries (Causality, Ground Truth, General Superiority)

**Audit Prompt:** *Does any report claim causality, factual correctness, open-domain retrieval performance, or general superiority beyond what the experiments establish?*

- **Severity:** Medium (Terminology in early project title and proposal; rigorously bounded in Phase 13 synthesis)
- **Exact File & Line Reference:**
  - Project Title: 'Fuzzy-Causal Evidence Attribution...'
  - `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx`, Slide 2 & 4
  - `PHASE_13_FINAL_RESEARCH_SYNTHESIS.md`, Section 5
  - `docs/LIMITATIONS_AND_FUTURE_WORK.md`, Sections 1–4
- **Relevant Code & Data Evidence:**
  - **Causality Boundary:** The counterfactual intervention implemented in `intervention/intervener.py` evaluates model probability shifts under leave-one-out passage erasure: $\Delta p_i = p(y|E) - p(y|E \setminus \{e_i\})$. This measures post-hoc model sensitivity / empirical feature erasure, not structural causal model (SCM) DAG mediation analysis or do-calculus interventions across latent confounders.
  - **Ground Truth Boundary:** Gold evidence labels represent FEVER annotator consensus, serving as an annotation-fidelity proxy rather than infallible real-world factual veracity.
  - **Superiority Boundary:** Fuzzy scoring does NOT dominate baseline scoring across the board. While Fuzzy achieved statistically significant higher AUROC under dynamic conditions (+0.0365, $p=0.0240$), Baseline achieved significantly higher Average Precision under dynamic conditions (+0.1396, $p=0.0020$) and higher AUROC under static conditions (+0.0943).
- **Why it is a Problem (if verified):**
  Claiming unhedged 'causal attribution' or 'superior fact-checking performance' would misrepresent the empirical findings. Phase 13 synthesis carefully hedged these boundaries, but early proposal documents and the project title retain the unhedged phrasing.
- **Status:** Verified Fact (Boundaries identified; early assets require synchronization).
- **Minimal Proposed Remedy:** Clarify in the final presentation and report that 'causal intervention' denotes leave-one-out feature ablation, and present the AUROC vs. AP trade-off transparently.

### Question 11: Specification vs. Implementation Alignment and Module Verification

**Audit Prompt:** *Are the implementation claims in the project specification and presentation supported by actual code?*

- **Severity:** Medium (Path naming divergence and missing presentation)
- **Exact File & Line Reference:**
  - `PROJECT_SPECIFICATION.md`, Section 3
  - `final_ppt.pptx` (Missing)
  - `pipeline/pipeline.py` vs `app/pipeline.py`
- **Relevant Code & Data Evidence:**
  - Verification of the 10 planned architectural modules from `PROJECT_SPECIFICATION.md` confirms that all 10 exist and function in code:
    1. FEVER Data Loader: `data/fever_loader.py` — Present.
    2. Evidence Retriever: `retrieval/retriever.py` — Present.
    3. Fact-Checking Cross-Encoder: `app/pipeline.py` — Present.
    4. Counterfactual Intervener: `intervention/intervener.py` — Present.
    5. Attribution Explainer: `attribution/explainer.py` — Present.
    6. Contradiction Detector: `conflict/detector.py` — Present.
    7. Stability Analyzer: `stability/analyzer.py` — Present.
    8. Fuzzy Engine: `fuzzy/engine.py` — Present.
    9. Weighted Baseline: `fuzzy/baseline.py` — Present.
    10. Web Interface: `app/main.py` — Present.
  - **Discrepancy 1:** The specification references `pipeline/pipeline.py`. The actual codebase houses the orchestrator in `app/pipeline.py`.
  - **Discrepancy 2:** The specification references `final_ppt.pptx`, which does not exist in the repository.
- **Why it is a Problem (if verified):**
  Discrepancies in file paths confuse external auditors, while a missing final presentation means project deliverables are incomplete for Phase 14 review.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** Note the actual path `app/pipeline.py` in audit documentation, and flag `final_ppt.pptx` as a required deliverable for Phase 14.

### Question 12: Cross-Asset Discrepancies Across Project Lifecycle

**Audit Prompt:** *Are there discrepancies between the frozen results, formal reports, research synthesis, README, and presentation?*

- **Severity:** High (Presentation missing/outdated; README out of sync with final empirical synthesis)
- **Exact File & Line Reference:**
  - `final_ppt.pptx` (Missing)
  - `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx`
  - `README.md`
  - `PHASE_13_FINAL_RESEARCH_SYNTHESIS.md`
- **Relevant Code & Data Evidence:**
  - **Discrepancy A (Presentation):** The only presentation file in the workspace is `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx`, dated August 2026. It contains 6 proposal slides created prior to running any experiments. The formal presentation `final_ppt.pptx` is missing.
  - **Discrepancy B (README):** `README.md` reflects early phase descriptions and quickstart instructions, but lacks the formal empirical results of Experiment 1 (Spearman rho = 0.5877 LIME vs 0.5369 SHAP) and Experiment 2 (AUROC +0.0365 Fuzzy vs AP +0.1396 Baseline).
  - **Discrepancy C (Baseline Saturation Documentation):** Post-run validation revealed that 312 baseline adjusted scores zeroed out due to contradiction collapse ($c=1.0$). This empirical finding is thoroughly documented in Phase 13 synthesis, but is absent from early design audits.
- **Why it is a Problem (if verified):**
  Reviewers consulting the repository presentation or README will encounter obsolete Phase 0 proposals rather than the verified Phase 10–13 scientific findings.
- **Status:** Verified Fact.
- **Minimal Proposed Remedy:** In Phase 14, author a complete `final_ppt.pptx` that synthesizes the frozen results, and update `README.md` to reflect the final Phase 13 research conclusions.

## 7. Categorized Summary of Audit Findings

The technical audit findings are categorized by severity below:

### 7.1 High Severity Defects
1. **Contradiction Degree Formula Collapse & Baseline Zero-Saturation:** In `conflict/detector.py`, the formula `min(1.0, 2.0 * min(max_sup, max_ref))` under the condition $max\_sup > 0.5 \land max\_ref > 0.5$ mathematically collapses to strictly $1.0$ for all detected conflicts. This binary step-function causes 312 baseline adjusted scores (20.8%) to saturate at 0.0, artificially depressing baseline dynamic AUROC.
2. **Missing Final Presentation (`final_ppt.pptx`):** The repository lacks `final_ppt.pptx`. The only slide deck is a 6-slide Phase 0 proposal (`Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx`) created before any experiments were run.

### 7.2 Medium Severity Issues
1. **Dual Static Ablation Design:** The static ablation simultaneously sets $c = 0.0$ and $s = 1.0$, precluding a factorial disentanglement of contradiction vs. stability effects.
2. **Closed Candidate Pool Boundary:** Top-3 retrieval recall (86.0%) and discrimination metrics are conditioned on a closed 808-passage candidate pool and must not be presented as open-domain Wikipedia performance.
3. **Causal Terminology Scoping:** The intervention measures post-hoc model sensitivity to leave-one-out feature erasure, not Pearl structural causal model mediation.
4. **Path Divergence:** The pipeline orchestrator resides at `app/pipeline.py` rather than `pipeline/pipeline.py`.

### 7.3 Low Severity / Verified Passing Items
1. **Attribution Magnitudes:** Signed vs. absolute values are handled consistently via $|A|$ across ranking, correlations, and fuzzy inputs.
2. **Rank Statistics & Ties:** Zero rank-1 ties and zero undefined correlations occurred in Experiment 1; Exp 2 enforced a deterministic 3-key tie-breaker.
3. **Cluster Bootstrap:** Resampling over claims correctly preserves the 3-evidence cluster structure and intra-cluster correlation.
4. **Statistical Inference:** Null-shifted bootstrap p-values, empirical percentile CIs, and Holm–Bonferroni step-down corrections are correctly implemented.
5. **Instance & Label Parity:** Identical evidence instances and binary gold labels are evaluated across fuzzy and baseline conditions.
6. **Quantitative Reproducibility:** All formal numbers in `analysis_summary.csv` and `summary.json` are 100% reproducible from the raw CSV data.

## 8. Final Directive Compliance & Execution Freeze

**AUDIT CONCLUSION:** The read-only audit bundle and comprehensive technical evaluation are complete. All 29 source files and 34 artifacts are securely preserved in `audit_bundle/` alongside `integrity_manifest.json` and this report.

**MANDATORY STOPPING POINT:** In accordance with the final prompt instructions:
- No existing project files were modified.
- No experiments were rerun.
- Phase 14 has **NOT** been started.
- Execution is frozen pending user review of this audit report.
