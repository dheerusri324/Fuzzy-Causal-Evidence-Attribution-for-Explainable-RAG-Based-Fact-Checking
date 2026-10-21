import json

# Load sensitivity summary to ensure exact numbers
with open('outputs/sensitivity_nei_label_audit/summary_sensitivity.json', 'r', encoding='utf-8') as fp:
    sens_sum = json.load(fp)

p_a = sens_sum['protocol_a_all_500_claims_nei_as_zero']
p_b = sens_sum['protocol_b_supports_refutes_only_334_claims']

lines = []
lines.append("# Pre-Phase-14 Audit Errata: Methodological Clarifications, Dataset Provenance, Label Semantics, and Statistical Bounds\n\n")
lines.append("**Document Date:** 2026-10-20  \n")
lines.append("**Document Role:** Authoritative Technical Errata & Boundary Specification  \n")
lines.append("**Scope:** Cross-Phase Codebase, Dataset Provenance, Retrieval Evaluation, Statistical Inference, and Formal Experimentation  \n")
lines.append("**Status:** FROZEN REFERENCE — DOCUMENTATION ONLY — PHASE 14 NOT STARTED\n\n")

lines.append("----\n\n")

lines.append("## Executive Notice & Strict Audit Constraints\n\n")
lines.append("This errata document records the independently verified technical, dataset, statistical, and semantic findings established during the Pre-Phase-14 technical audit of the project **Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking**.\n\n")
lines.append("> [!IMPORTANT]\n")
lines.append("> **Strict Audit Constraints:**\n")
lines.append("> - **Zero Overwrites:** No existing project source code, formal experiment outputs (`outputs/phase10_formal/`, `outputs/phase11_formal/`), earlier audit reports, or manifests have been modified or replaced.\n")
lines.append("> - **Zero Re-executions:** Neither formal experiment was re-run; no inference, retrieval, models, SHAP, LIME, or interventions were executed.\n")
lines.append("> - **Preservation of Official Metrics:** Official reported metrics remain intact. All corrected or reclassified metrics herein are explicitly designated as **Sensitivity Analyses** derived from frozen scores.\n")
lines.append("> - **Epistemic Discipline:** In accordance with audit standards, no issue is declared 'repaired' in code; all findings are documented with explicit boundaries.\n")
lines.append("> - **Phase Gate:** Phase 14 has **NOT** been started.\n\n")

lines.append("Every errata item below is strictly structured into three distinct epistemic categories:\n")
lines.append("1. **Verified Fact:** Quantitatively or programmatically proven characteristics of the current code and data.\n")
lines.append("2. **Interpretation or Limitation:** Scientific context, methodological scope, and potential sources of bias or conflation.\n")
lines.append("3. **Recommended Correction:** Concrete guidance for Phase 14 documentation, presentations, and publications.\n\n")

lines.append("----\n\n")

# Item 1: Dataset Provenance
lines.append("## 1. Dataset Provenance and Development Split Lineage\n\n")
lines.append("### 1.1 Verified Fact\n")
lines.append("- In `data/fever_loader.py` (lines 118–168), the data ingestion pipeline connects to the HuggingFace dataset `copenlu/fever_gold_evidence` using `split=\"train\"` in streaming mode.\n")
lines.append("- Up to the first 5,000 valid records are streamed from the training split.\n")
lines.append("- In `create_dev_subset()` (lines 171–209), exactly 250 records per label (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) are sampled using random seed 42 (`set_seed(42)`), creating a custom 750-record balanced split.\n")
lines.append("- In line 204, each sampled record is assigned `rec.split = \"dev\"` and written locally to `data/fever_dev.jsonl`.\n")
lines.append("- In both formal experiments (`experiments/experiment1_formal.py` line 67 and `experiments/experiment2_formal.py` line 74), only the first 500 records of `data/fever_dev.jsonl` are loaded and evaluated (`records[:500]`).\n")
lines.append("- The 500 evaluated records comprise exactly: 168 `SUPPORTS`, 166 `REFUTES`, and 166 `NOT_ENOUGH_INFO` claims.\n\n")

lines.append("### 1.2 Interpretation or Limitation\n")
lines.append("- The evaluated dataset is **not the official FEVER development split** (which contains 19,998 claims released by Thorne et al., 2018). It is a custom, balanced 500-claim research subset sampled from the training split of the preprocessed `copenlu/fever_gold_evidence` repository.\n")
lines.append("- Referring to this data simply as 'the FEVER development split' without qualification misrepresents its lineage and scale.\n\n")

lines.append("### 1.3 Recommended Correction\n")
lines.append("- All project documentation, presentations, and paper manuscripts must explicitly state:  \n")
lines.append("  *\"Evaluated on a balanced research sample of 500 claims (168 Supports, 166 Refutes, 166 NEI) sampled with seed 42 from the copenlu/fever_gold_evidence train split.\"*\n")
lines.append("- Discontinue any unqualified phrasing describing the evaluation pool as 'the official FEVER dev set'.\n\n")

lines.append("----\n\n")

# Item 2: Evidence Counts
lines.append("## 2. Evidence References vs. Unique Passage IDs in the Closed Candidate Corpus\n\n")
lines.append("### 2.1 Verified Fact\n")
lines.append("- Across the 500 evaluated claims in `data/fever_dev.jsonl`, there are exactly **540 annotated evidence references**, representing **534 distinct (unique) evidence passage IDs**.\n")
lines.append("- When partitioned by verifiable ground truth:\n")
lines.append("  - **SUPPORTS and REFUTES (334 claims):** Contain exactly **374 annotated gold evidence references**, which deduplicate to exactly **369 unique evidence passage IDs**.\n")
lines.append("  - **NOT ENOUGH INFO (166 claims):** Contain exactly **166 pseudo-evidence references**, which deduplicate to exactly **165 unique passage IDs**.\n")
lines.append("- The candidate retrieval corpus `data/evidence_corpus.json` contains exactly **808 passages**.\n")
lines.append("- This 808-passage pool was generated by deduplicating all evidence references across all 750 sampled claims in `data/fever_dev.jsonl` (534 unique passages from the 500 evaluated claims + 274 passages from the remaining 250 claims).\n")
lines.append("- Exactly **369 out of 369 (100.0%)** of the unique gold passage IDs for the evaluated verifiable claims exist within the 808-passage candidate pool.\n\n")

lines.append("### 2.2 Interpretation or Limitation\n")
lines.append("- The retrieval evaluation operates under a **closed, gold-conditioned candidate pool** where 100% of target evidence passages are guaranteed to be present.\n")
lines.append("- Reporting evidence counts requires distinguishing between **claim-passage references** (tokens of association; $N = 374$) and **unique text passages** (types; $N = 369$) to avoid apparent discrepancies between claim-level and instance-level audits.\n\n")

lines.append("### 2.3 Recommended Correction\n")
lines.append("- Whenever quoting dataset sizes, state both metrics explicitly:  \n")
lines.append("  *\"The 334 verifiable claims contain 374 gold evidence references representing 369 unique Wikipedia passages, all 369 of which are present within the closed 808-passage candidate corpus.\"*\n\n")

lines.append("----\n\n")

# Item 3: NEI Evidence Label Semantics
lines.append("## 3. Evidence-Label Semantics for NOT ENOUGH INFO (NEI) Claims\n\n")
lines.append("### 3.1 Verified Fact\n")
lines.append("- In `experiments/experiment2_formal.py` (lines 99–100 and 469–470), gold labels are derived using:  \n")
lines.append("  `is_gold = 1 if ev.evidence_id in {ep.evidence_id for ep in rec.evidence} else 0`.\n")
lines.append("- Because `copenlu/fever_gold_evidence` attached retrieved candidate passages (from Papelo / Stammbach et al.) to NEI claims, and `data/fever_loader.py` normalized them into `rec.evidence`, exactly **147 of the 443 positive labels (33.18%)** in the official frozen Experiment 2 dataset belong to NEI claims.\n")
lines.append("- Two rigorous sensitivity protocols were established to audit this defect without modifying frozen files:\n")
lines.append("  - **Sensitivity Protocol A (All 500 Claims):** Reclassifies all 147 NEI passages as non-gold ($y = 0$), yielding 296 gold instances (19.73%) and 1,204 distractors (80.27%).\n")
lines.append("  - **Sensitivity Protocol B (334 Verifiable Claims):** Excludes NEI claims entirely, evaluating only SUPPORTS and REFUTES claims, yielding 296 gold instances (29.54%) and 706 distractors (70.46%).\n\n")

lines.append("### 3.2 Interpretation or Limitation\n")
lines.append("- In FEVER, an NEI claim cannot be supported or refuted by Wikipedia. Reclassifying its retrieved passages as 'not annotated gold evidence' (**$y = 0$**) does **NOT** imply that the passage is 'factually false' or 'intrinsically irrelevant'. It simply denotes that the passage does not constitute annotator-verified evidence substantiating the claim.\n")
lines.append("- Evaluating gold discrimination on NEI pseudo-evidence measured retrieval alignment with a third-party retriever rather than human ground truth.\n\n")

lines.append("### 3.3 Recommended Correction\n")
lines.append("- Document that the official Experiment 2 results evaluated a mixed label proxy, and present Protocol A and Protocol B as the definitive sensitivity benchmarks for pure gold-evidence discrimination.\n")
lines.append("- Strictly define negative labels ($y = 0$) as *'not annotator-verified gold evidence'* rather than *'false information'*.\n\n")

lines.append("----\n\n")

# Item 4: Retrieval Terminology
lines.append("## 4. Retrieval Terminology and Evaluation Bounds\n\n")
lines.append("### 4.1 Verified Fact\n")
lines.append("- Across the 334 verifiable claims (SUPPORTS and REFUTES):\n")
lines.append("  - Exactly **283 out of 334 claims** retrieve at least one annotated gold passage in their Top-3 retrieved candidates. The **Claim-Level Top-3 Gold Hit Rate** is:  \n")
lines.append("    $$\\text{Claim-Level Hit Rate} = \\frac{283}{334} = 84.73\\%$$\n")
lines.append("  - Across all 374 gold claim-passage references, exactly **296 references** were retrieved in the Top-3 candidates. The **Gold Reference Recall** is:  \n")
lines.append("    $$\\text{Gold Reference Recall} = \\frac{296}{374} = 79.14\\%$$\n")
lines.append("- In the official frozen 500-claim pool (which counted 147 NEI passages), the claim-level hit rate was reported as $430 / 500 = 86.00\\%$.\n\n")

lines.append("### 4.2 Interpretation or Limitation\n")
lines.append("- These metrics evaluate dense bi-encoder retrieval (`all-MiniLM-L6-v2`) against a **closed candidate pool of 808 passages**, wherein 100% of gold passages were pre-seeded.\n")
lines.append("- Conflating this setup with open-domain Wikipedia retrieval is a severe threat to external validity. In true open-domain FEVER retrieval over 5.4 million Wikipedia articles (>30 million sentences), standard bi-encoders achieve Top-3 sentence recall of approximately 50%–65%.\n\n")

lines.append("### 4.3 Recommended Correction\n")
lines.append("- Never refer to 84.73% or 86.00% as 'retrieval recall' without stating the exact denominator and scope.\n")
lines.append("- Always label 84.73% as **'Claim-Level Top-3 Gold Hit Rate (283/334 verifiable claims)'** and 79.14% as **'Gold Reference Recall (296/374 references)'**, accompanied by the mandatory caveat:  \n")
lines.append("  *\"Conditioned on a closed candidate pool of 808 pre-indexed passages; not open-domain Wikipedia retrieval.\"*\n\n")

lines.append("----\n\n")

# Item 5: Sensitivity Metrics
lines.append("## 5. Sensitivity Analysis Metrics for Experiment 2 (Protocols A & B)\n\n")
lines.append("### 5.1 Verified Fact\n")
lines.append("The table below documents the official frozen results alongside the re-evaluated metrics under Sensitivity Protocols A and B (computed from the frozen model scores in `outputs/phase11_formal/results.csv` using $B = 1,000$ paired claim-cluster bootstrap replicates):\n\n")

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
lines.append(f"| **AUROC Holm-Adjusted $p$** | 0.0240 (sig at $\\alpha=0.05$) | **{p_a['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** (sig at $\\alpha=0.025$) | **{p_b['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** (sig at $\\alpha=0.025$) |\n")
lines.append("| **Dynamic AP: Fuzzy** | 0.4884 | 0.3537 | 0.5091 |\n")
lines.append("| **Dynamic AP: Baseline** | 0.6280 | 0.4217 | 0.6189 |\n")
lines.append(f"| **AP Difference ($\\hat{{\\Delta}}$)** | **-0.1396** | **{p_a['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** | **{p_b['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** |\n")
lines.append(f"| **AP 95% Marginal CI** | [-0.1777, -0.1025] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] |\n")
lines.append(f"| **AP 97.5% Bonferroni CI** | [-0.1856, -0.0988] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] |\n")
lines.append(f"| **AP Raw $p$-value** | 0.0010 | {p_a['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} | {p_b['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} |\n")
lines.append(f"| **AP Holm-Adjusted $p$** | 0.0020 (sig at $\\alpha=0.025$) | **{p_a['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** (sig at $\\alpha=0.05$) | **{p_b['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** (sig at $\\alpha=0.05$) |\n")
lines.append("| **Top-1 Gold Hit: All Claims (Metric A)** | Fuzzy 73.0% vs Base 69.0% | Fuzzy 49.6% vs Base 47.2% (+2.4%) | Fuzzy 74.3% vs Base 70.7% (+3.6%) |\n")
lines.append("| **Top-1 Gold Hit: Gold Present (Metric B)**| Fuzzy 84.9% vs Base 80.2% | Fuzzy 87.6% vs Base 83.4% (+4.2%) | Fuzzy 87.6% vs Base 83.4% (+4.2%) |\n")
lines.append("| **Top-1 Direct Agreement (Metric C)** | 87.4% (437/500) | 87.4% (437/500) | 89.2% (298/334) |\n")
lines.append("| **Static Ablation AUROC** | Fuzzy 0.7946 vs Base 0.8889 | Fuzzy 0.7888 vs Base 0.8661 | Fuzzy 0.8210 vs Base 0.9107 |\n")
lines.append("| **Static Ablation AP** | Fuzzy 0.5550 vs Base 0.7600 | Fuzzy 0.4757 vs Base 0.6360 | Fuzzy 0.6038 vs Base 0.8086 |\n\n")

lines.append("### 5.2 Interpretation or Limitation\n")
lines.append("- The fuzzy AUROC advantage observed in the official run is **not an artifact of NEI pseudo-evidence**. When labels are purged of NEI pseudo-evidence, the fuzzy AUROC advantage more than doubles (from $+0.0365$ to $+0.0898$ in Protocol A and $+0.0827$ in Protocol B), with both lower CI bounds strictly exceeding zero.\n")
lines.append("- Simultaneously, the baseline's Average Precision advantage narrows significantly under clean gold labels (from $-0.1396$ to $-0.0680$ in Protocol A and $-0.1098$ in Protocol B).\n")
lines.append("- However, the baseline retains superior Average Precision under dynamic conditions and superior AUROC under static conditions. The comparative finding remains a **nuanced trade-off**, not universal superiority.\n\n")

lines.append("### 5.3 Recommended Correction\n")
lines.append("- Present Protocol A and Protocol B alongside the official frozen results in all final tables, explicitly explaining how label semantics impact AUROC and AP.\n\n")

lines.append("----\n\n")

# Item 6: Statistical Code Note
lines.append("## 6. Statistical Inference Code Note: Holm-Adjusted P-Values vs. Sequential Alpha Thresholds\n\n")
lines.append("### 6.1 Verified Fact\n")
lines.append("- In `experiments/experiment2_formal.py` (lines 380–388):\n")
lines.append("  ```python\n")
lines.append("  step_alpha = alpha / (m - rank + 1)\n")
lines.append("  p_adj = min(1.0, p_raw * (m - rank + 1))\n")
lines.append("  p_adj = max(p_adj, prev_adj_p)  # enforce monotonicity\n")
lines.append("  rejected = (p_adj < alpha)\n")
lines.append("  ```\n")
lines.append("- In standard multiple hypothesis testing theory, there are two equivalent ways to formulate the Holm procedure:\n")
lines.append("  1. **Raw P-Value Formulation:** Compare sorted raw $p_{(k)}$ against the sequential threshold $\\alpha_{(k)} = \\alpha / (m - k + 1)$.\n")
lines.append("  2. **Adjusted P-Value Formulation:** Compare adjusted $p_{(k)}^{\\text{adj}} = \\max_{j \\le k} ((m - j + 1) \\cdot p_{(j)})$ against the fixed nominal $\\alpha = 0.05$.\n")
lines.append("- The code correctly implements Method 2 (`rejected = (p_adj < alpha)`).\n")
lines.append("- However, in `summary.json`, the field `\"step_alpha\"` ($0.025$ for the smallest p-value, $0.05$ for the second) is reported alongside `\"p_value_holm_adjusted\"`.\n\n")

lines.append("### 6.2 Interpretation or Limitation\n")
lines.append("- Reporting `step_alpha` next to `p_value_holm_adjusted` can mislead a reader into mistakenly testing whether $p^{\\text{adj}} < \\text{step\\_alpha}$, which would double-penalize the hypothesis.\n")
lines.append("- Fortunately, for all evaluated conditions (Official, Protocol A, and Protocol B), all raw and adjusted p-values ($0.0010$, $0.0020$, $0.0240$) are strictly below **both** $\\alpha / 2 = 0.025$ and $\\alpha = 0.05$. Therefore, **all statistical significance determinations remain 100% mathematically valid and unchanged**.\n\n")

lines.append("### 6.3 Recommended Correction\n")
lines.append("- In documentation tables, report either:  \n")
lines.append("  - Raw p-values tested against `step_alpha` ($\\alpha / 2 = 0.025, \\alpha = 0.050$), OR  \n")
lines.append("  - Holm-adjusted p-values tested against fixed nominal $\\alpha = 0.05$.\n")
lines.append("- Do not display `step_alpha` as the comparison threshold for `p_value_holm_adjusted`.\n\n")

lines.append("----\n\n")

# Item 7: Source-Level Limitations
lines.append("## 7. Remaining Source-Level Limitations: Stability Caching and Contradiction Saturation\n\n")
lines.append("### 7.1 Verified Fact\n")
lines.append("Two unresolved implementation characteristics exist in the frozen experiment pipeline:\n")
lines.append("1. **Binary Collapse of Dynamic Contradiction:**\n")
lines.append("   In `conflict/detector.py` (lines 60–68):\n")
lines.append("   ```python\n")
lines.append("   conflict_detected = (max_sup > 0.5 and max_ref > 0.5)\n")
lines.append("   conflict_score = min(1.0, 2.0 * min(max_sup, max_ref)) if conflict_detected else 0.0\n")
lines.append("   ```\n")
lines.append("   Because $\\min(max\\_sup, max\\_ref) > 0.5$ whenever conflict is detected, $2.0 \\times \\min(...) > 1.0$. Thus, `min(1.0, ...)` **strictly evaluates to 1.0 in 100% of conflict cases**.\n")
lines.append("   This binary step-function causes exactly $104 \\times 3 = 312$ baseline adjusted scores (20.8%) to saturate at 0.0.\n")
lines.append("2. **Dynamic Signals Caching:**\n")
lines.append("   In `experiments/experiment2_formal.py` (lines 158–178), dynamic contradiction and stability signals are loaded directly from a pre-computed cache file (`outputs/phase11_formal/.signals_cache.json`) if present, rather than evaluating live NLI models during execution.\n")
lines.append("   Furthermore, in `stability/analyzer.py`, stability evaluates only a single 3-item order reversal ($k=3$), constraining Spearman correlation to only 4 discrete values $\\{-1.0, -0.5, 0.5, 1.0\\}$, with 90.2% of claims evaluating to $s = 1.0$.\n\n")

lines.append("### 7.2 Interpretation or Limitation\n")
lines.append("- These characteristics represent **unresolved source-level implementation limitations** in the frozen codebase.\n")
lines.append("- In strict adherence to audit rules, the code must not be modified, and hypothetical 'corrected' scores must not be guessed or imputed without a formal rerun.\n\n")

lines.append("### 7.3 Recommended Correction\n")
lines.append("- Transparently document both limitations in the Phase 14 technical report and limitations section as verified properties of the Phase 11 implementation.\n")
lines.append("- Propose unclipped continuous contradiction scaling (e.g. $2.0 \\times (\\min(p_s, p_r) - 0.5)$) and multi-permutation stability testing as priorities for future post-Phase-14 work.\n\n")

lines.append("----\n\n")

# Item 8: Bounded Epistemic Language
lines.append("## 8. Epistemic Language and Bounded Claims Specification\n\n")
lines.append("### 8.1 Verified Fact\n")
lines.append("- The empirical results across Experiments 1 and 2 reveal specific trade-offs:\n")
lines.append("  - Experiment 1: LIME achieved higher mean and median rank correlation with intervention influence than SHAP (LIME $\\rho = 0.5877$ vs. SHAP $\\rho = 0.5369$), while both exhibited median $\\rho = 0.5000$.\n")
lines.append("  - Experiment 2: Fuzzy logic achieved significantly higher AUROC under dynamic conditions ($+0.0365$ official; $+0.0898$ Protocol A; $+0.0827$ Protocol B) and higher Top-1 gold selection (+4.2%), whereas the Weighted Baseline achieved significantly higher Average Precision under dynamic conditions ($-0.1396$ official; $-0.0680$ Protocol A; $-0.1098$ Protocol B) and higher AUROC under static conditions ($-0.0943$ official).\n\n")

lines.append("### 8.2 Interpretation or Limitation\n")
lines.append("- Absolute terms such as 'proves', 'optimizes', 'guarantees', or 'demonstrates universal superiority' are scientifically unfounded.\n")
lines.append("- Counterfactual leave-one-out erasure measures post-hoc model sensitivity to feature removal; it does not constitute Pearl structural causal DAG mediation analysis.\n")
lines.append("- Evaluated gold labels are FEVER annotator proxy labels, not infallible real-world factual ground truth.\n\n")

lines.append("### 8.3 Recommended Correction (Wording Prescriptions)\n\n")
lines.append("| Unbounded Phrasing (To Be Avoided) | Rigorous Bounded Phrasing (Mandatory in Phase 14) |\n")
lines.append("|:---|:---|\n")
lines.append("| *\"Fuzzy logic proves superior evidence reliability.\"* | *\"Fuzzy logic achieves statistically significant higher AUROC and Top-1 gold hit rates under dynamic contradiction, but lower Average Precision than the baseline.\"* |\n")
lines.append("| *\"Causal evidence attribution.\"* | *\"Counterfactual leave-one-out sensitivity analysis.\"* |\n")
lines.append("| *\"Retrieval achieves 86% recall across FEVER.\"* | *\"Retrieval achieves 84.7% claim-level Top-3 gold hit rate within a closed 808-passage candidate pool.\"* |\n")
lines.append("| *\"The system optimizes fact-checking accuracy.\"* | *\"The system refines evidence rank discrimination under simulated multi-source conflict.\"* |\n")
lines.append("| *\"The bug was repaired in the code.\"* | *\"The semantic discrepancy was audited via sensitivity reclassification on frozen scores.\"* |\n\n")

lines.append("----\n\n")

lines.append("## 9. Conclusion & Mandatory Phase Gate Freeze\n\n")
lines.append("This Errata Document formalizes the verified technical and methodological boundaries of the project across all 8 required audit domains.\n\n")
lines.append("- **No Existing Files Were Edited:** Earlier reports and code remain bit-for-bit unchanged.\n")
lines.append("- **No Experiments Were Rerun:** All evaluations use frozen scores.\n")
lines.append("- **Phase 14 Gate Respected:** Phase 14 has **NOT** been started.\n")

with open('PRE_PHASE_14_AUDIT_ERRATA.md', 'w', encoding='utf-8') as fp:
    fp.writelines(lines)

print('PRE_PHASE_14_AUDIT_ERRATA.md successfully created.')
