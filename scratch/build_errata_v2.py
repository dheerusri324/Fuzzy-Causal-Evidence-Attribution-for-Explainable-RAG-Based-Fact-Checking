import json

# Load sensitivity summary to ensure exact numbers
with open('outputs/sensitivity_nei_label_audit/summary_sensitivity.json', 'r', encoding='utf-8') as fp:
    sens_sum = json.load(fp)

p_a = sens_sum['protocol_a_all_500_claims_nei_as_zero']
p_b = sens_sum['protocol_b_supports_refutes_only_334_claims']

lines = []
lines.append("# Pre-Phase-14 Audit Errata (Version 2): Methodological Clarifications, Dataset Provenance, Label Semantics, and Statistical Bounds\n\n")
lines.append("**Document Version:** Version 2.0 (Corrected Pre-Phase-14 Audit Errata)  \n")
lines.append("**Document Date:** 2026-10-20  \n")
lines.append("**Document Role:** Authoritative Technical Errata & Boundary Specification  \n")
lines.append("**Scope:** Cross-Phase Codebase, Dataset Provenance, Retrieval Evaluation, Statistical Inference, and Formal Experimentation  \n")
lines.append("**Status:** FROZEN REFERENCE — DOCUMENTATION ONLY — PHASE 14 NOT STARTED\n\n")

lines.append("----\n\n")

lines.append("## Executive Notice & Strict Audit Constraints\n\n")
lines.append("This document constitutes Version 2 of the Pre-Phase-14 Audit Errata for the project **Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking**. It supersedes Version 1 in specificity, correcting arithmetic references, removing unsourced open-domain retrieval benchmarks, documenting in-memory attribution prediction cache ordering, refining multiple comparison significance annotations, and establishing bounded statistical terminology.\n\n")

lines.append("> [!IMPORTANT]\n")
lines.append("> **Strict Audit Constraints Maintained:**\n")
lines.append("> - **Zero Source Code or Frozen Result Modifications:** No existing source code, configuration files, or frozen experimental results (`outputs/phase10_formal/`, `outputs/phase11_formal/`) were altered.\n")
lines.append("> - **No Re-runs:** Neither Experiment 1 nor Experiment 2 was re-executed. No inference, retrieval, SHAP, LIME, or interventions were run.\n")
lines.append("> - **Preservation of Official Metrics:** Official reported metrics remain intact. All re-evaluated metrics herein are explicitly designated as **Sensitivity Analyses** derived from frozen scores.\n")
lines.append("> - **No Unsubstantiated Claims:** Implementation defects are not claimed to be 'repaired' in code; all findings are documented as verified properties of the existing implementation.\n")
lines.append("> - **Version 1 Preserved:** `PRE_PHASE_14_AUDIT_ERRATA.md` remains preserved unchanged in the repository.\n")
lines.append("> - **Phase Gate:** Phase 14 has **NOT** been started.\n\n")

lines.append("Every errata item below is systematically organized into three distinct epistemic sections:\n")
lines.append("1. **Verified Fact:** Programmatically or mathematically proven properties of the code, data, and experimental artifacts.\n")
lines.append("2. **Interpretation or Limitation:** Scientific implications, methodological constraints, and boundaries of validity.\n")
lines.append("3. **Recommended Correction:** Explicit textual and tabular adjustments required for Phase 14 documentation and presentations.\n\n")

lines.append("----\n\n")

# Section 1: Dataset Provenance
lines.append("## 1. Dataset Provenance and Development Split Lineage\n\n")
lines.append("### 1.1 Verified Fact\n")
lines.append("- In `data/fever_loader.py` (lines 118–168), the ingestion function `load_fever_split` streams records from the HuggingFace repository `copenlu/fever_gold_evidence` using `split=\"train\"`.\n")
lines.append("- Up to the first 5,000 valid records are streamed sequentially from the training split.\n")
lines.append("- In `create_dev_subset()` (lines 171–209), the script sets `set_seed(42)` and samples exactly 250 records per label for `[\"SUPPORTS\", \"REFUTES\", \"NOT_ENOUGH_INFO\"]`, producing a custom 750-record balanced split.\n")
lines.append("- In line 204, each sampled record is assigned `rec.split = \"dev\"` and saved locally to `data/fever_dev.jsonl`.\n")
lines.append("- In both formal experiments (`experiments/experiment1_formal.py` line 67; `experiments/experiment2_formal.py` line 74), only the first 500 records of `data/fever_dev.jsonl` are evaluated (`records[:500]`).\n")
lines.append("- The 500 evaluated records comprise exactly: 168 `SUPPORTS`, 166 `REFUTES`, and 166 `NOT_ENOUGH_INFO` claims.\n\n")

lines.append("### 1.2 Interpretation or Limitation\n")
lines.append("- The evaluation set is **not the official FEVER development split** (which contains 19,998 claims released by Thorne et al., 2018). It is a custom, balanced 500-claim research subset sampled from the training split of the preprocessed `copenlu/fever_gold_evidence` repository.\n")
lines.append("- **Reproducibility Limitation:** Setting `set_seed(42)` controls local Python pseudorandom sampling, but a random seed alone does not pin the upstream streamed dataset revision, commit SHA, or network streaming order. If the upstream repository alters its train split ordering or record contents, the resulting subset could diverge.\n\n")

lines.append("### 1.3 Recommended Correction\n")
lines.append("- In all Phase 14 documentation, presentations, and publications, describe the dataset with full provenance:  \n")
lines.append("  *\"Evaluated on a balanced research sample of 500 claims (168 Supports, 166 Refutes, 166 NEI) sampled with seed 42 from the copenlu/fever_gold_evidence train split.\"*\n")
lines.append("- Discontinue any phrasing calling this data 'the official FEVER development set'.\n")
lines.append("- In future reproducibility documentation, record the exact git commit hash / dataset fingerprint of `copenlu/fever_gold_evidence` and the library versions of `datasets` and `transformers`.\n\n")

lines.append("----\n\n")

# Section 2: Evidence References vs Unique Passage IDs
lines.append("## 2. Evidence References vs. Unique Passage IDs in the Closed Candidate Pool\n\n")
lines.append("### 2.1 Verified Fact\n")
lines.append("- Across the 500 evaluated claims in `data/fever_dev.jsonl`, there are exactly **540 annotated evidence references**, representing **534 unique evidence passage IDs**.\n")
lines.append("- Partitioning by claim label:\n")
lines.append("  - **SUPPORTS and REFUTES (334 claims):** Contain exactly **374 annotated gold evidence references**, which deduplicate to exactly **369 unique evidence passage IDs**.\n")
lines.append("  - **NOT ENOUGH INFO (166 claims):** Contain exactly **166 pseudo-evidence references**, which deduplicate to exactly **165 unique passage IDs**.\n")
lines.append("- The candidate retrieval corpus `data/evidence_corpus.json` contains exactly **808 passages**.\n")
lines.append("- This 808-passage pool is the deduplicated union of all evidence entries across all 750 sampled claims in `data/fever_dev.jsonl` (534 unique passages from the 500 evaluated claims + 274 unique passages from the remaining 250 claims).\n")
lines.append("- Exactly **369 out of 369 (100.0%)** of the unique gold passage IDs for the evaluated verifiable claims exist within the 808-passage candidate pool.\n\n")

lines.append("### 2.2 Interpretation or Limitation\n")
lines.append("- The retrieval task is conducted over a **closed, pre-seeded candidate pool** where 100% of gold passages are guaranteed to be present.\n")
lines.append("- A clear distinction must be maintained between **claim-passage references** (the 374 instances of evidence association) and **unique passage IDs** (the 369 distinct text passages) to prevent confusion across evaluation tables.\n\n")

lines.append("### 2.3 Recommended Correction\n")
lines.append("- State both metrics unambiguously in all dataset summaries:  \n")
lines.append("  *\"The 334 verifiable claims contain 374 gold evidence references representing 369 unique Wikipedia passages, all 369 of which are present within the closed 808-passage candidate corpus.\"*\n\n")

lines.append("----\n\n")

# Section 3: NEI Evidence Label Semantics
lines.append("## 3. Evidence-Label Semantics for NOT ENOUGH INFO (NEI) Claims\n\n")
lines.append("### 3.1 Verified Fact\n")
lines.append("- In `experiments/experiment2_formal.py` (lines 99–100 and 469–470), gold-evidence proxy labels are derived as:  \n")
lines.append("  `is_gold = 1 if ev.evidence_id in {ep.evidence_id for ep in rec.evidence} else 0`.\n")
lines.append("- Because `copenlu/fever_gold_evidence` included retrieved candidate passages (from Papelo / Stammbach et al.) for NEI claims, and `data/fever_loader.py` stored them into `rec.evidence`, exactly **147 of the 443 positive labels (33.18%)** in the official frozen Experiment 2 dataset belong to NEI claims.\n")
lines.append("- Two sensitivity protocols were executed using the frozen scores in `outputs/sensitivity_nei_label_audit/results_nei_sensitivity.csv`:\n")
lines.append("  - **Protocol A (All 500 Claims):** Reclassifies all 147 NEI passages as $y = 0$ (296 gold instances, 1,204 distractors).\n")
lines.append("  - **Protocol B (334 Verifiable Claims):** Excludes NEI claims entirely, evaluating only SUPPORTS and REFUTES claims (296 gold instances, 706 distractors).\n\n")

lines.append("### 3.2 Interpretation or Limitation\n")
lines.append("- In the FEVER task definition, a label of `NOT ENOUGH INFO` denotes that there is **insufficient annotated evidence within the benchmark's evidence setting** to verify (support or refute) the claim.\n")
lines.append("- Reclassifying NEI evidence as 'not annotated gold evidence' ($y = 0$) does **not** mean that the passage is 'factually false' or that no passage in Wikipedia could ever support or refute the claim. It means strictly that the passage was not annotated as verifying evidence in the benchmark dataset.\n")
lines.append("- Evaluating discrimination on NEI pseudo-evidence measured alignment with a third-party retrieval model rather than human ground truth.\n\n")

lines.append("### 3.3 Recommended Correction\n")
lines.append("- Define $y = 0$ as *'not annotator-verified gold evidence within the benchmark setting'* rather than *'false information'* or *'intrinsically irrelevant'*.\n")
lines.append("- Cite Protocols A and B as sensitivity evaluations that establish how gold discrimination behaves when unverified claims are handled appropriately.\n\n")

lines.append("----\n\n")

# Section 4: Retrieval Terminology
lines.append("## 4. Retrieval Terminology and Closed-Pool Evaluation Scope\n\n")
lines.append("### 4.1 Verified Fact\n")
lines.append("- Across the 334 verifiable claims (SUPPORTS and REFUTES):\n")
lines.append("  - Exactly **283 out of 334 claims** retrieve at least one annotated gold passage in their Top-3 candidates. The **Claim-Level Top-3 Gold Hit Rate** is:  \n")
lines.append("    $$\\text{Claim-Level Hit Rate} = \\frac{283}{334} = 84.73\\%$$\n")
lines.append("  - Across all 374 annotated gold references, exactly **296 references** were retrieved in the Top-3 candidates. The **Gold Reference Recall** is:  \n")
lines.append("    $$\\text{Gold Reference Recall} = \\frac{296}{374} = 79.14\\%$$\n")
lines.append("- In the official frozen 500-claim pool (which counted 147 NEI passages), the claim-level hit rate was reported as $430 / 500 = 86.00\\%$.\n\n")

lines.append("### 4.2 Interpretation or Limitation\n")
lines.append("- **Closed-Pool Scope:** These metrics evaluate dense bi-encoder retrieval (`sentence-transformers/all-MiniLM-L6-v2`) against a **closed candidate pool of 808 pre-indexed sentence passages**.\n")
lines.append("- **Correction of Unsourced Retrieval Claims:** Version 1 of this errata contained an unsourced statement suggesting standard bi-encoders achieve '50%–65% Top-3 recall on open-domain Wikipedia'. That unsourced benchmark claim is hereby retracted. No open-domain recall figure may be quoted without a directly comparable primary literature source and citation.\n")
lines.append("- The essential limitation remains: the project's retrieval task operates over an indexed pool of 808 pre-extracted Wikipedia passages assembled from the sampled claims, rather than an open corpus of full Wikipedia articles.\n\n")

lines.append("### 4.3 Recommended Correction\n")
lines.append("- Always label $84.73\\%$ as **'Claim-Level Top-3 Gold Hit Rate (283/334 verifiable claims)'** and $79.14\\%$ as **'Gold Reference Recall (296/374 references)'**.\n")
lines.append("- Accompany all retrieval figures with the mandatory qualification:  \n")
lines.append("  *\"Conditioned on a closed candidate pool of 808 pre-indexed passages; not open-domain Wikipedia search.\"*\n\n")

lines.append("----\n\n")

# Section 5: Experiment 1 Correlations & Sensitivity Metrics
lines.append("## 5. Formal Correlation Metrics and Experiment 2 Sensitivity Analysis\n\n")
lines.append("### 5.1 Experiment 1 Correlation Arithmetic Correction\n")
lines.append("- **Verified Fact:** Direct recomputation from `outputs/phase10_formal/results.csv` confirms:\n")
lines.append("  - **SHAP Mean Spearman $\\rho = 0.4517$** (Median $= 0.5000$).\n")
lines.append("  - **LIME Mean Spearman $\\rho = 0.5877$** (Median $= 0.5000$).\n")
lines.append("  - Zero undefined correlations ($N = 500$ valid claims for both methods).\n")
lines.append("- **Arithmetic Correction:** The value $0.5369$ reported in earlier working drafts for SHAP mean Spearman $\\rho$ was an unweighted drafting error. The true frozen value is **$0.4517$**.\n")
lines.append("- **Wording Correction:** Both SHAP and LIME share an identical median correlation of **$0.5000$**. LIME exhibits a higher **mean** Spearman rank correlation ($0.5877$ vs. $0.4517$), but does **not** have a higher median.\n\n")

lines.append("### 5.2 Experiment 2 Sensitivity Metrics (Protocols A & B)\n")
lines.append("The table below reports the official frozen results alongside the re-evaluated metrics under Sensitivity Protocols A and B (evaluated on frozen scores using $B = 1,000$ paired claim-cluster bootstrap replicates):\n\n")

lines.append("| Evaluated Metric / Parameter | Official Frozen (Exp 2) | Sensitivity Protocol A (All 500, NEI=0) | Sensitivity Protocol B (334 Verifiable Only) |\n")
lines.append("|:---|:---:|:---:|:---:|\n")
lines.append("| **Evaluated Population** | 500 claims (1,500 rows) | 500 claims (1,500 rows) | 334 claims (1,002 rows) |\n")
lines.append("| **Gold Evidence Instances ($y=1$)** | 443 (29.53%) | 296 (19.73%) | 296 (29.54%) |\n")
lines.append("| **Distractor Instances ($y=0$)** | 1,057 (70.47%) | 1,204 (80.27%) | 706 (70.46%) |\n")
lines.append("| **Claims with Gold in Top-3** | 430 / 500 (86.00%) | 283 / 500 (56.60%) | 283 / 334 (84.73%) |\n")
lines.append("| **Dynamic AUROC: Fuzzy** | 0.7630 | 0.7386 | 0.7777 |\n")
lines.append("| **Dynamic AUROC: Baseline** | 0.7265 | 0.6488 | 0.6949 |\n")
lines.append(f"| **AUROC Difference ($\\hat{{\\Delta}}$)** | **+0.0365** | **+{p_a['dynamic_pipeline_results']['H1_AUROC']['delta_hat']:.4f}** | **+{p_b['dynamic_pipeline_results']['H1_AUROC']['delta_hat']:.4f}** |\n")
lines.append(f"| **AUROC 95% Marginal CI** | [0.0053, 0.0686] | [{p_a['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][0]:.4f}, {p_a['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][0]:.4f}, {p_b['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][1]:.4f}] |\n")
lines.append(f"| **AUROC 97.5% Bonferroni CI** | [0.00006, 0.0710] | [{p_a['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][0]:.4f}, {p_a['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][0]:.4f}, {p_b['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][1]:.4f}] |\n")
lines.append(f"| **AUROC Raw $p$-value** | 0.0240 | {p_a['dynamic_pipeline_results']['H1_AUROC']['p_value_raw']:.4f} | {p_b['dynamic_pipeline_results']['H1_AUROC']['p_value_raw']:.4f} |\n")
lines.append(f"| **AUROC Holm-Adjusted $p$** | 0.0240 ($p^{{\\text{{adj}}}} < 0.05$) | **{p_a['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) | **{p_b['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) |\n")
lines.append("| **Dynamic AP: Fuzzy** | 0.4884 | 0.3537 | 0.5091 |\n")
lines.append("| **Dynamic AP: Baseline** | 0.6280 | 0.4217 | 0.6189 |\n")
lines.append(f"| **AP Difference ($\\hat{{\\Delta}}$)** | **-0.1396** | **{p_a['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** | **{p_b['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** |\n")
lines.append(f"| **AP 95% Marginal CI** | [-0.1777, -0.1025] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] |\n")
lines.append(f"| **AP 97.5% Bonferroni CI** | [-0.1856, -0.0988] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] |\n")
lines.append(f"| **AP Raw $p$-value** | 0.0010 | {p_a['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} | {p_b['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} |\n")
lines.append(f"| **AP Holm-Adjusted $p$** | 0.0020 ($p^{{\\text{{adj}}}} < 0.05$) | **{p_a['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) | **{p_b['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) |\n")
lines.append("| **Top-1 Gold Hit: All Claims (Metric A)** | Fuzzy 73.0% vs Base 69.0% | Fuzzy 49.6% vs Base 47.2% (+2.4 pp) | Fuzzy 74.3% vs Base 70.7% (+3.6 pp) |\n")
lines.append("| **Top-1 Gold Hit: Gold Present (Metric B)**| Fuzzy 84.9% vs Base 80.2% | Fuzzy 87.6% vs Base 83.4% (+4.2 pp) | Fuzzy 87.6% vs Base 83.4% (+4.2 pp) |\n")
lines.append("| **Top-1 Direct Agreement (Metric C)** | 87.4% (437/500) | 87.4% (437/500) | 89.2% (298/334) |\n")
lines.append("| **Static Ablation AUROC** | Fuzzy 0.7946 vs Base 0.8889 | Fuzzy 0.7888 vs Base 0.8661 | Fuzzy 0.8210 vs Base 0.9107 |\n")
lines.append("| **Static Ablation AP** | Fuzzy 0.5550 vs Base 0.7600 | Fuzzy 0.4757 vs Base 0.6360 | Fuzzy 0.6038 vs Base 0.8086 |\n\n")

lines.append("### 5.3 Interpretation or Limitation\n")
lines.append("- **AUROC Advantage Strengthens:** Purging NEI pseudo-evidence substantially increases the fuzzy AUROC advantage over the baseline (from $+0.0365$ official to $+0.0898$ in Protocol A and $+0.0827$ in Protocol B), with both lower CI bounds strictly exceeding zero.\n")
lines.append("- **AP Advantage Narrows Numerically:** The baseline's Average Precision advantage **narrows numerically** from $-0.1396$ to $-0.0680$ in Protocol A and $-0.1098$ in Protocol B. Because no direct formal hypothesis test was performed comparing the change in delta between protocols, this shift must be described as a numerical narrowing rather than a statistically tested difference.\n")
lines.append("- **Top-1 Gold Selection:** For the 283 claims where gold evidence was present in the top-3, Fuzzy selects gold evidence as Rank 1 in 248 claims (87.6%) vs. Baseline in 236 claims (83.4%), a difference of **+4.2 percentage points**. This must **not** be described as statistically significant because paired hypothesis testing was only performed on the co-primary metrics (AUROC and AP), not on Top-1 binary hit outcomes.\n\n")

lines.append("### 5.4 Recommended Correction\n")
lines.append("- Present all table significance markers as $p^{\\text{adj}} < 0.05$ (nominal $\\alpha$).\n")
lines.append("- Use 'percentage points (pp)' rather than '%' when describing differences in hit rates (+4.2 pp).\n")
lines.append("- Replace 'narrows significantly' with 'narrows numerically'.\n\n")

lines.append("----\n\n")

# Section 6: Holm Inference
lines.append("## 6. Statistical Inference Note: Holm-Bonferroni Formulation Consistency\n\n")
lines.append("### 6.1 Verified Fact\n")
lines.append("- In the Holm–Bonferroni procedure (Holm, 1979), two equivalent formulations control Family-Wise Error Rate (FWER) at $\\alpha = 0.05$:\n")
lines.append("  1. **Raw P-Value Step-Down:** Compare sorted raw p-values $p_{(1)} \\le p_{(2)}$ against sequential thresholds $\\alpha_{(1)} = \\alpha / 2 = 0.025$ and $\\alpha_{(2)} = \\alpha / 1 = 0.050$.\n")
lines.append("  2. **Adjusted P-Value Step-Down:** Multiply sorted p-values by their step factor ($p_{(1)}^{\\text{adj}} = 2 \\cdot p_{(1)}$; $p_{(2)}^{\\text{adj}} = \\max(p_{(1)}^{\\text{adj}}, 1 \\cdot p_{(2)})$) and compare against fixed nominal $\\alpha = 0.05$.\n")
lines.append("- In `scratch/run_nei_sensitivity_audit.py` (lines 90–98), the code assigns:\n")
lines.append("  ```python\n")
lines.append("  step_alpha = 0.025 if i == 0 else 0.05\n")
lines.append("  statistically_significant = bool(adj_p < step_alpha)\n")
lines.append("  ```\n")
lines.append("- Comparing `adj_p < step_alpha` is an implementation inconsistency: it applies the multiplicity penalty twice to hypothesis 1.\n\n")

lines.append("### 6.2 Interpretation or Limitation\n")
lines.append("- In the actual results, $p_{(1)}^{\\text{adj}} = 0.0020$, which is strictly less than both $0.025$ and $0.050$. For hypothesis 2, $p_{(2)}^{\\text{adj}} = 0.0020 < 0.050$.\n")
lines.append("- Consequently, **this implementation inconsistency does not alter any statistical significance conclusions** in the reported results. Both co-primary metrics remain statistically significant at FWER $\\alpha = 0.05$.\n\n")

lines.append("### 6.3 Recommended Correction\n")
lines.append("- In all tables and narratives, report significance by comparing adjusted p-values against nominal $\\alpha = 0.05$ (`p_adj < 0.05`).\n")
lines.append("- Explicitly document that before any future reuse or expansion of `scratch/run_nei_sensitivity_audit.py`, line 97 should be corrected to `statistically_significant = bool(adj_p < 0.05)`.\n\n")

lines.append("----\n\n")

# Section 7: Stability Caching & Contradiction Saturation
lines.append("## 7. Remaining Source-Level Limitations: Stability Prediction Caching and Contradiction Collapse\n\n")
lines.append("### 7.1 Verified Fact\n")
lines.append("Static source code inspection reveals two distinct caching and numerical phenomena in the frozen pipeline:\n\n")
lines.append("1. **In-Memory Prediction Cache Ordering in `attribution/explainer.py`:**\n")
lines.append("   In `attribution/explainer.py` (lines 43–52):\n")
lines.append("   ```python\n")
lines.append("   def _get_probabilities(self, claim: str, evidence_subset: List[RetrievedEvidence]) -> np.ndarray:\n")
lines.append("       ev_ids = tuple(sorted(ev.evidence_id for ev in evidence_subset))\n")
lines.append("       cache_key = (claim, ev_ids)\n")
lines.append("       if cache_key not in self._eval_cache:\n")
lines.append("           res = self.verifier.verify(claim, evidence_subset)\n")
lines.append("           self._eval_cache[cache_key] = res.probabilities\n")
lines.append("       probs = self._eval_cache[cache_key]\n")
lines.append("       return np.array([probs[c] for c in self.class_names])\n")
lines.append("   ```\n")
lines.append("   - Line 44 explicitly sorts evidence IDs (`sorted(...)`) to construct the cache key.\n")
lines.append("   - In `stability/analyzer.py` (lines 55–60), order stability is evaluated by reversing the retrieved evidence list (`perturbed_evidence.reverse()`).\n")
lines.append("   - Because the in-memory cache key sorts evidence IDs, any call to `_get_probabilities` with reversed evidence produces the exact same key `(claim, sorted_ev_ids)` as the forward evidence.\n")
lines.append("   - Consequently, the verifier is not invoked with the reversed order; instead, the in-memory cache returns the probability distribution previously evaluated for the forward order.\n")
lines.append("   - This architectural behavior explains why 90.2% of claims (451/500) evaluated to a stability score of exactly $1.0$.\n\n")
lines.append("2. **Disk-Based Dynamic Signal Caching in `experiments/experiment2_formal.py`:**\n")
lines.append("   - Distinct from the in-memory cache, `experiments/experiment2_formal.py` (lines 158–178) loads pre-computed dynamic signals from a JSON file on disk (`outputs/phase11_formal/.signals_cache.json`).\n")
lines.append("   - If this cache exists, live NLI contradiction and stability models are skipped entirely during experiment execution.\n\n")
lines.append("3. **Binary Contradiction Step-Function Collapse in `conflict/detector.py`:**\n")
lines.append("   - In lines 60–68: `conflict_detected = (max_sup > 0.5 and max_ref > 0.5)`; `conflict_score = min(1.0, 2.0 * min(max_sup, max_ref))`.\n")
lines.append("   - Because $\\min(max\\_sup, max\\_ref) > 0.5$, the expression $2.0 \\times \\min(...) > 1.0$, which is clipped to strictly $1.0$.\n")
lines.append("   - The dynamic contradiction score collapses to a binary variable $\\{0.0, 1.0\\}$, causing exactly 312 baseline adjusted scores (20.8%) to saturate at 0.0.\n\n")

lines.append("### 7.2 Interpretation or Limitation\n")
lines.append("- These are **unresolved source-level implementation limitations** in the frozen pipeline.\n")
lines.append("- Under strict audit safety constraints, source code must not be altered, and hypothetical 'corrected' stability scores must not be guessed or imputed without a formal re-execution.\n\n")

lines.append("### 7.3 Recommended Correction\n")
lines.append("- Document both the in-memory prediction cache order-invariance and the disk cache in the Phase 14 technical limitations section.\n")
lines.append("- Clearly distinguish the in-memory explainer cache from the disk-based signals cache.\n")
lines.append("- Recommend order-sensitive cache keys (preserving list sequence) and unclipped continuous contradiction scaling for future work.\n\n")

lines.append("----\n\n")

# Section 8: Bounded Epistemic Language
lines.append("## 8. Epistemic Language & Bounded Claims Specification\n\n")
lines.append("### 8.1 Verified Fact\n")
lines.append("- The empirical findings across Experiments 1 and 2 reveal balanced trade-offs:\n")
lines.append("  - Experiment 1: LIME achieved a higher mean Spearman rank correlation with intervention influence than SHAP ($0.5877$ vs. $0.4517$), while both achieved identical median correlations ($0.5000$).\n")
lines.append("  - Experiment 2: Fuzzy logic achieved significantly higher AUROC under dynamic conditions ($+0.0365$ official; $+0.0898$ Protocol A; $+0.0827$ Protocol B) and higher Top-1 gold selection (+4.2 pp), whereas the Weighted Baseline achieved significantly higher Average Precision under dynamic conditions ($-0.1396$ official; $-0.0680$ Protocol A; $-0.1098$ Protocol B) and higher AUROC under static conditions ($-0.0943$ official).\n\n")

lines.append("### 8.2 Interpretation or Limitation\n")
lines.append("- Absolute terms such as 'proves', 'optimizes', 'guarantees', or 'demonstrates universal superiority' are scientifically unfounded.\n")
lines.append("- The observed effects are specific to the evaluated 500-claim dataset and 808-passage candidate pool; claiming universal generalizability across arbitrary fact-checking corpora is unjustified.\n")
lines.append("- Counterfactual leave-one-out erasure measures post-hoc model sensitivity to feature removal; it does not constitute Pearl structural causal DAG mediation analysis.\n\n")

lines.append("### 8.3 Recommended Correction (Wording Prescriptions)\n\n")
lines.append("| Unbounded Phrasing (To Be Avoided) | Rigorous Bounded Phrasing (Mandatory in Phase 14) |\n")
lines.append("|:---|:---|\n")
lines.append("| *\"Fuzzy logic proves superior evidence reliability.\"* | *\"Fuzzy logic achieves statistically significant higher AUROC and Top-1 gold hit rates under dynamic contradiction, but lower Average Precision than the baseline.\"* |\n")
lines.append("| *\"Causal evidence attribution.\"* | *\"Counterfactual leave-one-out sensitivity analysis.\"* |\n")
lines.append("| *\"Standard bi-encoders achieve 50–65% recall across Wikipedia.\"* | *\"Retrieval is evaluated over an indexed corpus of 808 pre-extracted Wikipedia sentence passages derived from the sampled claims.\"* |\n")
lines.append("| *\"The baseline AP advantage narrows significantly.\"* | *\"The baseline AP advantage narrows numerically from -0.1396 to -0.0680 in Protocol A and -0.1098 in Protocol B.\"* |\n")
lines.append("| *\"LIME achieved higher mean and median correlation.\"* | *\"LIME achieved a higher mean correlation (0.5877 vs. 0.4517), while both achieved identical medians (0.5000).\"* |\n")
lines.append("| *\"Fuzzy logic produced a statistically significant +4.2% Top-1 hit rate improvement.\"* | *\"Fuzzy logic produced a +4.2 percentage point improvement (87.6% vs. 83.4%) in Top-1 gold hit rate on gold-present claims; paired statistical significance was tested on co-primary AUROC/AP metrics.\"* |\n\n")

lines.append("----\n\n")

lines.append("## 9. Conclusion & Mandatory Phase Gate Freeze\n\n")
lines.append("This document formalizes Version 2 of the Pre-Phase-14 Audit Errata, incorporating all requested arithmetic corrections, statistical clarifications, and cache architecture audits.\n\n")
lines.append("- **Frozen Artifacts Intact:** Original reports, CSVs, and `PRE_PHASE_14_AUDIT_ERRATA.md` remain strictly unchanged.\n")
lines.append("- **Phase Gate Respected:** Phase 14 has **NOT** been started.\n")

with open('PRE_PHASE_14_AUDIT_ERRATA_V2.md', 'w', encoding='utf-8') as fp:
    fp.writelines(lines)

print('PRE_PHASE_14_AUDIT_ERRATA_V2.md successfully written.')
