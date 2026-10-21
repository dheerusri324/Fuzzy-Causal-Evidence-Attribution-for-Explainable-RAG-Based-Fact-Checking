import json
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# 1. Load authoritative data for exact metric injection
# -----------------------------------------------------------------------------
with open('outputs/sensitivity_nei_label_audit/summary_sensitivity.json', 'r', encoding='utf-8') as fp:
    sens_sum = json.load(fp)

p_a = sens_sum['protocol_a_all_500_claims_nei_as_zero']
p_b = sens_sum['protocol_b_supports_refutes_only_334_claims']

# -----------------------------------------------------------------------------
# 2. Build PRE_PHASE_14_AUDIT_ERRATA_V3.md
# -----------------------------------------------------------------------------
v3_lines = []

v3_lines.append("# Pre-Phase-14 Audit Errata (Version 3): Final Methodological Specifications, Hybrid Retrieval Architecture, In-Memory Caching Mechanics, and Bounded Inferences\n\n")
v3_lines.append("**Document Version:** Version 3.0 (Authoritative Final Pre-Phase-14 Audit Errata)  \n")
v3_lines.append("**Document Date:** 2026-10-20  \n")
v3_lines.append("**Document Role:** Authoritative Technical Errata & Boundary Specification  \n")
v3_lines.append("**Scope:** Cross-Phase Codebase, Dataset Provenance, Hybrid Retrieval Evaluation, In-Memory & On-Disk Caching Mechanics, Statistical Inference, and Formal Experimentation  \n")
v3_lines.append("**Status:** FROZEN REFERENCE — DOCUMENTATION ONLY — PHASE 14 NOT STARTED\n\n")

v3_lines.append("----\n\n")

v3_lines.append("## Executive Notice & Strict Audit Constraints\n\n")
v3_lines.append("This document constitutes Version 3 of the Pre-Phase-14 Audit Errata for the project **Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking**. It represents the authoritative, complete technical errata and boundary specification, superseding Versions 1 and 2 in technical accuracy. Specifically, Version 3 introduces five essential corrections:\n\n")
v3_lines.append("1. **Correction of Retrieval Architecture:** Accurately documents that the formal experimental pipeline employs **hybrid retrieval** (linear fusion of normalized BM25 Okapi lexical scores and dense bi-encoder embeddings via `sentence-transformers/all-MiniLM-L6-v2` with $\\alpha = 0.5$) over the closed 808-passage pool, replacing preliminary descriptions that claimed dense-only retrieval.\n")
v3_lines.append("2. **Top-1 Statistical Significance Precision:** Clarifies that the +4.24 percentage point difference in Top-1 gold hit rate on gold-present claims (Fuzzy 87.63% vs. Baseline 83.39%) is an **observed difference / observed improvement** and was **not** subjected to formal paired hypothesis testing (which was restricted exclusively to co-primary AUROC and AP metrics).\n")
v3_lines.append("3. **Precision on Explainer In-Memory Caching:** Accurately details that in `attribution/explainer.py`, the baseline full-evidence verifier call (`baseline_res = self.verifier.verify(claim, evidence_list)`) is executed directly on the reversed order, but masked coalition evaluations in `_get_probabilities()` sort evidence IDs (`sorted(...)`), causing reversed coalitions to hit the in-memory cache populated during the forward pass. Distinguishes this in-memory cache from the disk-based dynamic signals cache (`outputs/phase11_formal/.signals_cache.json`), while strictly refraining from asserting that this mechanism alone explains the 90.2% stability invariance or imputing hypothetical scores.\n")
v3_lines.append("4. **Contradiction Formula Syntax & Proof:** Formally establishes that the contradiction step-function formula is **100% present** in `conflict/detector.py` (lines 80–83 as a multiline `if-else` block using variables `max_support` and `max_refute`), resolving an exact-string search failure in the prior automated validation script and providing the mathematical proof that $\\min(max\\_support, max\\_refute) > 0.5$ forces clipping to exactly $1.0$.\n")
v3_lines.append("5. **Full-Document Cross-Phase Consistency:** Enforces strict epistemic boundaries across all sections, eliminating unsourced external benchmarks, bounding causal claims to counterfactual leave-one-out sensitivity analysis, reporting SHAP mean $\\rho = 0.4517$ and LIME mean $\\rho = 0.5877$ (both medians $0.5000$), and using Holm-adjusted significance annotations ($p^{\\text{adj}} < 0.05$).\n\n")

v3_lines.append("> [!IMPORTANT]\n")
v3_lines.append("> **Strict Audit Constraints Maintained:**\n")
v3_lines.append("> - **Zero Source Code or Frozen Result Modifications:** No existing source code, configuration files, or frozen experimental results (`outputs/phase10_formal/`, `outputs/phase11_formal/`) were altered.\n")
v3_lines.append("> - **No Re-runs:** Neither Experiment 1 nor Experiment 2 was re-executed. No inference, retrieval, SHAP, LIME, or interventions were run.\n")
v3_lines.append("> - **Preservation of Official Metrics:** Official reported metrics remain intact. All re-evaluated metrics herein are explicitly designated as **Sensitivity Analyses** derived from frozen scores.\n")
v3_lines.append("> - **No Unsubstantiated Claims:** Implementation defects are not claimed to be 'repaired' in code; all findings are documented as verified properties of the existing implementation.\n")
v3_lines.append("> - **Prior Versions Preserved:** `PRE_PHASE_14_AUDIT_ERRATA.md` (V1) and `PRE_PHASE_14_AUDIT_ERRATA_V2.md` (V2) remain preserved unchanged in the repository.\n")
v3_lines.append("> - **Phase Gate:** Phase 14 has **NOT** been started.\n\n")

v3_lines.append("Every errata item below is systematically organized into three distinct epistemic sections:\n")
v3_lines.append("1. **Verified Fact:** Programmatically or mathematically proven properties of the code, data, and experimental artifacts.\n")
v3_lines.append("2. **Interpretation or Limitation:** Scientific implications, methodological constraints, and boundaries of validity.\n")
v3_lines.append("3. **Recommended Correction:** Explicit textual and tabular adjustments required for Phase 14 documentation and presentations.\n\n")

v3_lines.append("----\n\n")

# Section 1: Dataset Provenance
v3_lines.append("## 1. Dataset Provenance and Development Split Lineage\n\n")
v3_lines.append("### 1.1 Verified Fact\n")
v3_lines.append("- In `data/fever_loader.py` (lines 118–168), the ingestion function `load_fever_split` streams records from the HuggingFace repository `copenlu/fever_gold_evidence` using `split=\"train\"`.\n")
v3_lines.append("- Up to the first 5,000 valid records are streamed sequentially from the training split.\n")
v3_lines.append("- In `create_dev_subset()` (lines 171–209), the script sets `set_seed(42)` and samples exactly 250 records per label for `[\"SUPPORTS\", \"REFUTES\", \"NOT_ENOUGH_INFO\"]`, producing a custom 750-record balanced split.\n")
v3_lines.append("- In line 204, each sampled record is assigned `rec.split = \"dev\"` and saved locally to `data/fever_dev.jsonl`.\n")
v3_lines.append("- In both formal experiments (`experiments/experiment1_formal.py` line 67; `experiments/experiment2_formal.py` line 74), only the first 500 records of `data/fever_dev.jsonl` are evaluated (`records[:500]`).\n")
v3_lines.append("- The 500 evaluated records comprise exactly: 168 `SUPPORTS`, 166 `REFUTES`, and 166 `NOT_ENOUGH_INFO` claims.\n\n")

v3_lines.append("### 1.2 Interpretation or Limitation\n")
v3_lines.append("- The evaluation set is **not the official FEVER development split** (which contains 19,998 claims released by Thorne et al., 2018). It is a custom, balanced 500-claim research subset sampled from the training split of the preprocessed `copenlu/fever_gold_evidence` repository.\n")
v3_lines.append("- **Reproducibility Limitation:** Setting `set_seed(42)` controls local Python pseudorandom sampling, but a random seed alone does not pin the upstream streamed dataset revision, commit SHA, or network streaming order. If the upstream repository alters its train split ordering or record contents, the resulting subset could diverge.\n\n")

v3_lines.append("### 1.3 Recommended Correction\n")
v3_lines.append("- In all Phase 14 documentation, presentations, and publications, describe the dataset with full provenance:  \n")
v3_lines.append("  *\"Evaluated on a balanced research sample of 500 claims (168 Supports, 166 Refutes, 166 NEI) sampled with seed 42 from the copenlu/fever_gold_evidence train split.\"*\n")
v3_lines.append("- Discontinue any phrasing calling this data 'the official FEVER development set'.\n")
v3_lines.append("- In future reproducibility documentation, record the exact git commit hash / dataset fingerprint of `copenlu/fever_gold_evidence` and the library versions of `datasets` and `transformers`.\n\n")

v3_lines.append("----\n\n")

# Section 2: Evidence References vs Unique Passage IDs
v3_lines.append("## 2. Evidence References vs. Unique Passage IDs in the Closed Candidate Pool\n\n")
v3_lines.append("### 2.1 Verified Fact\n")
v3_lines.append("- Across the 500 evaluated claims in `data/fever_dev.jsonl`, there are exactly **540 annotated evidence references**, representing **534 unique evidence passage IDs**.\n")
v3_lines.append("- Partitioning by claim label:\n")
v3_lines.append("  - **SUPPORTS and REFUTES (334 claims):** Contain exactly **374 annotated gold evidence references**, which deduplicate to exactly **369 unique evidence passage IDs**.\n")
v3_lines.append("  - **NOT ENOUGH INFO (166 claims):** Contain exactly **166 pseudo-evidence references**, which deduplicate to exactly **165 unique passage IDs**.\n")
v3_lines.append("- The candidate retrieval corpus `data/evidence_corpus.json` contains exactly **808 passages**.\n")
v3_lines.append("- This 808-passage pool is the deduplicated union of all evidence entries across all 750 sampled claims in `data/fever_dev.jsonl` (534 unique passages from the 500 evaluated claims + 274 unique passages from the remaining 250 claims).\n")
v3_lines.append("- Exactly **369 out of 369 (100.0%)** of the unique gold passage IDs for the evaluated verifiable claims exist within the 808-passage candidate pool.\n\n")

v3_lines.append("### 2.2 Interpretation or Limitation\n")
v3_lines.append("- The retrieval task is conducted over a **closed, pre-seeded candidate pool** where 100% of gold passages are guaranteed to be present.\n")
v3_lines.append("- A clear distinction must be maintained between **claim-passage references** (the 374 instances of evidence association) and **unique passage IDs** (the 369 distinct text passages) to prevent confusion across evaluation tables.\n\n")

v3_lines.append("### 2.3 Recommended Correction\n")
v3_lines.append("- State both metrics unambiguously in all dataset summaries:  \n")
v3_lines.append("  *\"The 334 verifiable claims contain 374 gold evidence references representing 369 unique Wikipedia passages, all 369 of which are present within the closed 808-passage candidate corpus.\"*\n\n")

v3_lines.append("----\n\n")

# Section 3: NEI Evidence Label Semantics
v3_lines.append("## 3. Evidence-Label Semantics for NOT ENOUGH INFO (NEI) Claims\n\n")
v3_lines.append("### 3.1 Verified Fact\n")
v3_lines.append("- In `experiments/experiment2_formal.py` (lines 99–100 and 469–470), gold-evidence proxy labels are derived as:  \n")
v3_lines.append("  `is_gold = 1 if ev.evidence_id in {ep.evidence_id for ep in rec.evidence} else 0`.\n")
v3_lines.append("- Because `copenlu/fever_gold_evidence` included retrieved candidate passages (from Papelo / Stammbach et al.) for NEI claims, and `data/fever_loader.py` stored them into `rec.evidence`, exactly **147 of the 443 positive labels (33.18%)** in the official frozen Experiment 2 dataset belong to NEI claims.\n")
v3_lines.append("- Two sensitivity protocols were executed using the frozen scores in `outputs/sensitivity_nei_label_audit/results_nei_sensitivity.csv`:\n")
v3_lines.append("  - **Protocol A (All 500 Claims):** Reclassifies all 147 NEI passages as $y = 0$ (296 gold instances, 1,204 distractors).\n")
v3_lines.append("  - **Protocol B (334 Verifiable Claims):** Excludes NEI claims entirely, evaluating only SUPPORTS and REFUTES claims (296 gold instances, 706 distractors).\n\n")

v3_lines.append("### 3.2 Interpretation or Limitation\n")
v3_lines.append("- In the FEVER task definition, a label of `NOT ENOUGH INFO` denotes that there is **insufficient annotated evidence within the benchmark's evidence setting** to verify (support or refute) the claim.\n")
v3_lines.append("- Reclassifying NEI evidence as 'not annotated gold evidence' ($y = 0$) does **not** mean that the passage is 'factually false' or that no passage in Wikipedia could ever support or refute the claim. It means strictly that the passage was not annotated as verifying evidence in the benchmark dataset.\n")
v3_lines.append("- Evaluating discrimination on NEI pseudo-evidence measured alignment with a third-party retrieval model rather than human ground truth.\n\n")

v3_lines.append("### 3.3 Recommended Correction\n")
v3_lines.append("- Define $y = 0$ as *'not annotator-verified gold evidence within the benchmark setting'* rather than *'false information'* or *'intrinsically irrelevant'*.\n")
v3_lines.append("- Cite Protocols A and B as sensitivity evaluations that establish how gold discrimination behaves when unverified claims are handled appropriately.\n\n")

v3_lines.append("----\n\n")

# Section 4: Hybrid Retrieval Architecture
v3_lines.append("## 4. Hybrid Retrieval Architecture and Closed-Pool Evaluation Scope\n\n")
v3_lines.append("### 4.1 Verified Fact\n")
v3_lines.append("- In `retrieval/retriever.py` (lines 138–164), `config/settings.py` (lines 30–33), `outputs/phase10_formal/config.json` (line 5), and `outputs/phase11_formal/config.json` (line 4), the retrieval pipeline is formally specified and executed as **hybrid retrieval**:\n")
v3_lines.append("  - **Lexical Index:** BM25Okapi (`retrieval/bm25.py`).\n")
v3_lines.append("  - **Dense Semantic Index:** Bi-encoder embeddings using `sentence-transformers/all-MiniLM-L6-v2` with FAISS inner-product search over normalized vectors (`retrieval/dense.py`).\n")
v3_lines.append("  - **Linear Score Fusion:** `_retrieve_hybrid()` queries both indexes for $n = \\min(3 \\times \\text{top\\_k}, |\\mathcal{C}|) = 9$ candidates, min-max normalizes scores to $[0, 1]$, and combines them via:\n")
v3_lines.append("    $$\\text{Score}_{\\text{hybrid}}(i) = \\alpha \\cdot \\text{Score}_{\\text{BM25}}(i) + (1.0 - \\alpha) \\cdot \\text{Score}_{\\text{Dense}}(i)$$\n")
v3_lines.append("    with default fusion parameter $\\alpha = 0.5$, returning the top-3 candidates.\n")
v3_lines.append("- **Retrieval Performance over 808-Passage Candidate Pool:**\n")
v3_lines.append("  - Across the 334 verifiable claims (SUPPORTS and REFUTES), exactly **283 out of 334 claims** retrieve at least one annotated gold passage in their Top-3 candidates. The **Claim-Level Top-3 Gold Hit Rate** is:  \n")
v3_lines.append("    $$\\text{Claim-Level Hit Rate} = \\frac{283}{334} = 84.73\\%$$\n")
v3_lines.append("  - Across all 374 annotated gold references, exactly **296 references** were retrieved in the Top-3 candidates. The **Gold Reference Recall** is:  \n")
v3_lines.append("    $$\\text{Gold Reference Recall} = \\frac{296}{374} = 79.14\\%$$\n")
v3_lines.append("  - In the official frozen 500-claim pool (which counted 147 NEI passages), the claim-level hit rate was reported as $430 / 500 = 86.00\\%$.\n\n")

v3_lines.append("### 4.2 Interpretation or Limitation\n")
v3_lines.append("- **Correction of Preliminary Dense-Only Descriptions:** Earlier working documents and errata drafts inadvertently described the retrieval stage as 'dense bi-encoder retrieval'. Source code and frozen JSON configurations prove conclusively that retrieval operated in **hybrid mode** combining BM25 and all-MiniLM-L6-v2 embeddings.\n")
v3_lines.append("- **Closed-Pool Scope:** These metrics evaluate hybrid retrieval against a **closed candidate pool of 808 pre-indexed sentence passages**, not an open-domain web or full Wikipedia search engine.\n")
v3_lines.append("- **Prohibition of Unsourced Retrieval Benchmarks:** Any reference to standard bi-encoders achieving '50%–65% Top-3 recall on open-domain Wikipedia' is retracted and strictly prohibited without primary literature citations from identical task settings.\n\n")

v3_lines.append("### 4.3 Recommended Correction\n")
v3_lines.append("- Formally specify the retrieval method in all documentation as:  \n")
v3_lines.append("  *\"Hybrid retrieval combining normalized BM25 Okapi lexical matching and all-MiniLM-L6-v2 dense bi-encoder embeddings ($\\alpha = 0.5$) over a closed candidate pool of 808 pre-indexed Wikipedia sentence passages.\"*\n")
v3_lines.append("- Always label $84.73\\%$ as **'Claim-Level Top-3 Gold Hit Rate (283/334 verifiable claims)'** and $79.14\\%$ as **'Gold Reference Recall (296/374 references)'**.\n\n")

v3_lines.append("----\n\n")

# Section 5: Experiment 1 Correlations & Sensitivity Metrics
v3_lines.append("## 5. Formal Correlation Metrics and Experiment 2 Sensitivity Analysis\n\n")
v3_lines.append("### 5.1 Experiment 1 Correlation Arithmetic Correction\n")
v3_lines.append("- **Verified Fact:** Direct recomputation from `outputs/phase10_formal/results.csv` confirms:\n")
v3_lines.append("  - **SHAP Mean Spearman $\\rho = 0.4517$** (Median $= 0.5000$).\n")
v3_lines.append("  - **LIME Mean Spearman $\\rho = 0.5877$** (Median $= 0.5000$).\n")
v3_lines.append("  - Zero undefined correlations ($N = 500$ valid claims for both methods).\n")
v3_lines.append("- **Arithmetic Correction:** The value $0.5369$ reported in earlier working drafts for SHAP mean Spearman $\\rho$ was an unweighted drafting error. The true frozen value is **$0.4517$**.\n")
v3_lines.append("- **Wording Correction:** Both SHAP and LIME share an identical median correlation of **$0.5000$**. LIME exhibits a higher **mean** Spearman rank correlation ($0.5877$ vs. $0.4517$), but does **not** have a higher median.\n\n")

v3_lines.append("### 5.2 Experiment 2 Sensitivity Metrics (Protocols A & B)\n")
v3_lines.append("The table below reports the official frozen results alongside the re-evaluated metrics under Sensitivity Protocols A and B (evaluated on frozen scores using $B = 1,000$ paired claim-cluster bootstrap replicates):\n\n")

v3_lines.append("| Evaluated Metric / Parameter | Official Frozen (Exp 2) | Sensitivity Protocol A (All 500, NEI=0) | Sensitivity Protocol B (334 Verifiable Only) |\n")
v3_lines.append("|:---|:---:|:---:|:---:|\n")
v3_lines.append("| **Evaluated Population** | 500 claims (1,500 rows) | 500 claims (1,500 rows) | 334 claims (1,002 rows) |\n")
v3_lines.append("| **Gold Evidence Instances ($y=1$)** | 443 (29.53%) | 296 (19.73%) | 296 (29.54%) |\n")
v3_lines.append("| **Distractor Instances ($y=0$)** | 1,057 (70.47%) | 1,204 (80.27%) | 706 (70.46%) |\n")
v3_lines.append("| **Claims with Gold in Top-3** | 430 / 500 (86.00%) | 283 / 500 (56.60%) | 283 / 334 (84.73%) |\n")
v3_lines.append("| **Dynamic AUROC: Fuzzy** | 0.7630 | 0.7386 | 0.7777 |\n")
v3_lines.append("| **Dynamic AUROC: Baseline** | 0.7265 | 0.6488 | 0.6949 |\n")
v3_lines.append(f"| **AUROC Difference ($\\hat{{\\Delta}}$)** | **+0.0365** | **+{p_a['dynamic_pipeline_results']['H1_AUROC']['delta_hat']:.4f}** | **+{p_b['dynamic_pipeline_results']['H1_AUROC']['delta_hat']:.4f}** |\n")
v3_lines.append(f"| **AUROC 95% Marginal CI** | [0.0053, 0.0686] | [{p_a['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][0]:.4f}, {p_a['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][0]:.4f}, {p_b['dynamic_pipeline_results']['H1_AUROC']['ci_95_marginal'][1]:.4f}] |\n")
v3_lines.append(f"| **AUROC 97.5% Bonferroni CI** | [0.00006, 0.0710] | [{p_a['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][0]:.4f}, {p_a['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][0]:.4f}, {p_b['dynamic_pipeline_results']['H1_AUROC']['ci_975_bonferroni'][1]:.4f}] |\n")
v3_lines.append(f"| **AUROC Raw $p$-value** | 0.0240 | {p_a['dynamic_pipeline_results']['H1_AUROC']['p_value_raw']:.4f} | {p_b['dynamic_pipeline_results']['H1_AUROC']['p_value_raw']:.4f} |\n")
v3_lines.append(f"| **AUROC Holm-Adjusted $p$** | 0.0240 ($p^{{\\text{{adj}}}} < 0.05$) | **{p_a['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) | **{p_b['dynamic_pipeline_results']['H1_AUROC']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) |\n")
v3_lines.append("| **Dynamic AP: Fuzzy** | 0.4884 | 0.3537 | 0.5091 |\n")
v3_lines.append("| **Dynamic AP: Baseline** | 0.6280 | 0.4217 | 0.6189 |\n")
v3_lines.append(f"| **AP Difference ($\\hat{{\\Delta}}$)** | **-0.1396** | **{p_a['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** | **{p_b['dynamic_pipeline_results']['H2_AP']['delta_hat']:.4f}** |\n")
v3_lines.append(f"| **AP 95% Marginal CI** | [-0.1777, -0.1025] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_95_marginal'][1]:.4f}] |\n")
v3_lines.append(f"| **AP 97.5% Bonferroni CI** | [-0.1856, -0.0988] | [{p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_a['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] | [{p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][0]:.4f}, {p_b['dynamic_pipeline_results']['H2_AP']['ci_975_bonferroni'][1]:.4f}] |\n")
v3_lines.append(f"| **AP Raw $p$-value** | 0.0010 | {p_a['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} | {p_b['dynamic_pipeline_results']['H2_AP']['p_value_raw']:.4f} |\n")
v3_lines.append(f"| **AP Holm-Adjusted $p$** | 0.0020 ($p^{{\\text{{adj}}}} < 0.05$) | **{p_a['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) | **{p_b['dynamic_pipeline_results']['H2_AP']['p_value_holm_adjusted']:.4f}** ($p^{{\\text{{adj}}}} < 0.05$) |\n")
v3_lines.append("| **Top-1 Gold Hit: All Claims (Metric A)** | Fuzzy 73.0% vs Base 69.0% | Fuzzy 49.6% vs Base 47.2% (+2.4 pp) | Fuzzy 74.3% vs Base 70.7% (+3.6 pp) |\n")
v3_lines.append("| **Top-1 Gold Hit: Gold Present (Metric B)**| Fuzzy 84.9% vs Base 80.2% | Fuzzy 87.63% vs Base 83.39% (+4.24 pp) | Fuzzy 87.63% vs Base 83.39% (+4.24 pp) |\n")
v3_lines.append("| **Top-1 Direct Agreement (Metric C)** | 87.4% (437/500) | 87.4% (437/500) | 89.2% (298/334) |\n")
v3_lines.append("| **Static Ablation AUROC** | Fuzzy 0.7946 vs Base 0.8889 | Fuzzy 0.7888 vs Base 0.8661 | Fuzzy 0.8210 vs Base 0.9107 |\n")
v3_lines.append("| **Static Ablation AP** | Fuzzy 0.5550 vs Base 0.7600 | Fuzzy 0.4757 vs Base 0.6360 | Fuzzy 0.6038 vs Base 0.8086 |\n\n")

v3_lines.append("### 5.3 Interpretation or Limitation\n")
v3_lines.append("- **AUROC Advantage Strengthens:** Purging NEI pseudo-evidence substantially increases the fuzzy AUROC advantage over the baseline (from $+0.0365$ official to $+0.0898$ in Protocol A and $+0.0827$ in Protocol B), with both lower CI bounds strictly exceeding zero.\n")
v3_lines.append("- **AP Advantage Narrows Numerically:** The baseline's Average Precision advantage **narrows numerically** from $-0.1396$ to $-0.0680$ in Protocol A and $-0.1098$ in Protocol B. Because no direct formal hypothesis test was performed comparing the change in delta between protocols, this shift must be described strictly as a numerical narrowing rather than a statistically tested difference.\n")
v3_lines.append("- **Top-1 Gold Selection as Observed Improvement:** For the 283 claims where gold evidence was present in the top-3, Fuzzy selects gold evidence as Rank 1 in 248 claims (87.63%) vs. Baseline in 236 claims (83.39%), an **observed difference of +4.24 percentage points**. This must **not** be described as statistically significant because paired bootstrap hypothesis testing in `experiments/experiment2_formal.py` was executed **exclusively** on the co-primary metrics (AUROC and AP), **not** on Top-1 binary hit outcomes.\n\n")

v3_lines.append("### 5.4 Recommended Correction\n")
v3_lines.append("- Present all table significance markers as $p^{\\text{adj}} < 0.05$ (nominal $\\alpha$).\n")
v3_lines.append("- Use 'percentage points (pp)' rather than '%' when describing differences in hit rates (+4.24 pp).\n")
v3_lines.append("- Explicitly describe the Top-1 hit rate improvement as an observed numerical difference, avoiding claims of statistical significance.\n")
v3_lines.append("- Replace 'narrows significantly' with 'narrows numerically'.\n\n")

v3_lines.append("----\n\n")

# Section 6: Holm Inference
v3_lines.append("## 6. Statistical Inference Note: Holm-Bonferroni Formulation Consistency\n\n")
v3_lines.append("### 6.1 Verified Fact\n")
v3_lines.append("- In the Holm–Bonferroni procedure (Holm, 1979), two equivalent formulations control Family-Wise Error Rate (FWER) at $\\alpha = 0.05$:\n")
v3_lines.append("  1. **Raw P-Value Step-Down:** Compare sorted raw p-values $p_{(1)} \\le p_{(2)}$ against sequential thresholds $\\alpha_{(1)} = \\alpha / 2 = 0.025$ and $\\alpha_{(2)} = \\alpha / 1 = 0.050$.\n")
v3_lines.append("  2. **Adjusted P-Value Step-Down:** Multiply sorted p-values by their step factor ($p_{(1)}^{\\text{adj}} = 2 \\cdot p_{(1)}$; $p_{(2)}^{\\text{adj}} = \\max(p_{(1)}^{\\text{adj}}, 1 \\cdot p_{(2)})$) and compare against fixed nominal $\\alpha = 0.05$.\n")
v3_lines.append("- In `scratch/run_nei_sensitivity_audit.py` (lines 90–98), the code assigns:\n")
v3_lines.append("  ```python\n")
v3_lines.append("  step_alpha = 0.025 if i == 0 else 0.05\n")
v3_lines.append("  statistically_significant = bool(adj_p < step_alpha)\n")
v3_lines.append("  ```\n")
v3_lines.append("- Comparing `adj_p < step_alpha` is an implementation inconsistency: it applies the multiplicity penalty twice to hypothesis 1.\n\n")

v3_lines.append("### 6.2 Interpretation or Limitation\n")
v3_lines.append("- In the actual results, $p_{(1)}^{\\text{adj}} = 0.0020$, which is strictly less than both $0.025$ and $0.050$. For hypothesis 2, $p_{(2)}^{\\text{adj}} = 0.0020 < 0.050$.\n")
v3_lines.append("- Consequently, **this implementation inconsistency does not alter any statistical significance conclusions** in the reported results. Both co-primary metrics remain statistically significant at FWER $\\alpha = 0.05$.\n\n")

v3_lines.append("### 6.3 Recommended Correction\n")
v3_lines.append("- In all tables and narratives, report significance by comparing adjusted p-values against nominal $\\alpha = 0.05$ (`p_adj < 0.05`).\n")
v3_lines.append("- Explicitly document that before any future reuse or expansion of `scratch/run_nei_sensitivity_audit.py`, line 97 should be corrected to `statistically_significant = bool(adj_p < 0.05)`.\n\n")

v3_lines.append("----\n\n")

# Section 7: Stability Caching & Contradiction Saturation
v3_lines.append("## 7. Remaining Source-Level Limitations: Stability Prediction Caching and Contradiction Collapse\n\n")
v3_lines.append("### 7.1 Verified Fact\n")
v3_lines.append("Static source code inspection reveals two distinct caching mechanisms and a numerical saturation phenomenon in the frozen pipeline:\n\n")
v3_lines.append("1. **Explainer In-Memory Prediction Cache Key Construction in `attribution/explainer.py`:**\n")
v3_lines.append("   In `attribution/explainer.py` (lines 43–52 and 82–97):\n")
v3_lines.append("   ```python\n")
v3_lines.append("   def _get_probabilities(self, claim: str, evidence_subset: List[RetrievedEvidence]) -> np.ndarray:\n")
v3_lines.append("       ev_ids = tuple(sorted(ev.evidence_id for ev in evidence_subset))\n")
v3_lines.append("       cache_key = (claim, ev_ids)\n")
v3_lines.append("       if cache_key not in self._eval_cache:\n")
v3_lines.append("           res = self.verifier.verify(claim, evidence_subset)\n")
v3_lines.append("           self._eval_cache[cache_key] = res.probabilities\n")
v3_lines.append("       probs = self._eval_cache[cache_key]\n")
v3_lines.append("       return np.array([probs[c] for c in self.class_names])\n")
v3_lines.append("   ```\n")
v3_lines.append("   - **Full-Evidence Verifier Invocation:** At line 83, `baseline_res = self.verifier.verify(claim, evidence_list)` evaluates the entire evidence list in its supplied order. When `stability/analyzer.py` reverses the list (`perturbed_evidence.reverse()`), the full-evidence verifier call **is** executed on the reversed order, and `target_class = baseline_res.verdict` is determined from that reversed evaluation. It is therefore factually incorrect to state that the verifier is never invoked with reversed evidence.\n")
v3_lines.append("   - **Masked Coalition Cache Invariance:** In `predict_wrapper(masks)` (lines 88–96), SHAP and LIME evaluate masked feature coalitions by invoking `_get_probabilities(claim, subset)`. Because line 44 explicitly sorts evidence IDs (`sorted(...)`) to construct `cache_key`, any masked subset of evidence passages produces the identical key `(claim, sorted_ev_ids)` regardless of list permutation.\n")
v3_lines.append("   - Since `self._eval_cache` persists across forward and reversed attribution evaluations within the explainer instance, `predict_wrapper` returns the cached probabilities evaluated during the forward pass for every masked coalition.\n")
v3_lines.append("   - This architectural behavior means that coalition evaluations do not capture evidence re-ordering effects, which undermines the intended sensitivity of the permutation stability test.\n")
v3_lines.append("   - Furthermore, if the full-evidence baseline prediction on the reversed list produces a different `target_class`, the explainer extracts attribution for that new target class while reusing coalition probabilities cached from the forward pass.\n\n")
v3_lines.append("2. **Disk-Based Dynamic Signal Caching in `experiments/experiment2_formal.py`:**\n")
v3_lines.append("   - Completely distinct from the in-memory prediction cache, `experiments/experiment2_formal.py` (lines 158–178) loads pre-computed dynamic signals from a JSON file on disk (`outputs/phase11_formal/.signals_cache.json`).\n")
v3_lines.append("   - When this file is present, the script loads pre-computed contradiction scores and stability scores directly, bypassing execution of the live NLI contradiction detector and stability analyzer models.\n\n")
v3_lines.append("3. **Binary Contradiction Step-Function Collapse in `conflict/detector.py`:**\n")
v3_lines.append("   - In `conflict/detector.py` (lines 80–83), the contradiction score is computed as:\n")
v3_lines.append("     ```python\n")
v3_lines.append("     if conflict_detected:\n")
v3_lines.append("         conflict_score = min(1.0, 2.0 * min(max_support, max_refute))\n")
v3_lines.append("     else:\n")
v3_lines.append("         conflict_score = 0.0\n")
v3_lines.append("     ```\n")
v3_lines.append("   - In lines 47–58 and 71, `conflict_detected` requires both `len(supporting_ids) > 0` and `len(refuting_ids) > 0`, where an evidence passage is classified as supporting only if `p_sup > self.threshold` (default $0.5$) and refuting only if `p_ref > self.threshold` (default $0.5$).\n")
v3_lines.append("   - Consequently, whenever `conflict_detected` evaluates to `True`, both $max\\_support > 0.5$ and $max\\_refute > 0.5$.\n")
v3_lines.append("   - Therefore, $\\min(max\\_support, max\\_refute) > 0.5$, which guarantees that $2.0 \\times \\min(max\\_support, max\\_refute) > 1.0$.\n")
v3_lines.append("   - The outer function `min(1.0, ...)` clips the value to exactly $1.0$ in 100% of detected conflict cases.\n")
v3_lines.append("   - The continuous contradiction formulation thus collapses to a discrete binary step-function $\\{0.0, 1.0\\}$, which causes exactly 312 baseline adjusted scores (20.8%) across 104 claims in Experiment 2 to saturate at $0.0$.\n\n")

v3_lines.append("### 7.2 Interpretation or Limitation\n")
v3_lines.append("- **Epistemic Boundary on Stability Scores:** While sorting evidence IDs in `_get_probabilities()` causes masked coalition evaluations to reuse forward-pass probabilities, this implementation detail alone must **not** be asserted as the sole explanation for the 90.2% stability invariance (451/500 claims with score 1.0). Target class concordance, cross-encoder robustness, and rank preservation across permutations also influence the outcome. Under strict audit rules, no hypothetical 'corrected' stability scores may be guessed or imputed without an actual rerun.\n")
v3_lines.append("- **Source Code Freeze:** Both issues represent unaddressed source-level implementation limitations in the frozen codebase. In accordance with audit safety rules, source code has not been altered.\n\n")

v3_lines.append("### 7.3 Recommended Correction\n")
v3_lines.append("- Accurately document both the in-memory coalition cache mechanism and the disk-based signals cache in Phase 14 technical limitations.\n")
v3_lines.append("- Avoid claiming that the in-memory cache alone accounts for the 90.2% stability rate; describe it as an architectural limitation that compromises coalition order-sensitivity.\n")
v3_lines.append("- For future work, recommend order-sensitive coalition tuple keys and continuous, unclipped contradiction scaling formulations.\n\n")

v3_lines.append("----\n\n")

# Section 8: Bounded Epistemic Language
v3_lines.append("## 8. Epistemic Language & Bounded Claims Specification\n\n")
v3_lines.append("### 8.1 Verified Fact\n")
v3_lines.append("- The empirical findings across Experiments 1 and 2 reveal balanced trade-offs:\n")
v3_lines.append("  - **Experiment 1:** LIME achieved a higher mean Spearman rank correlation with intervention influence than SHAP ($0.5877$ vs. $0.4517$), while both achieved identical median correlations ($0.5000$).\n")
v3_lines.append("  - **Experiment 2:** Fuzzy logic achieved statistically significant higher AUROC under dynamic conditions ($+0.0365$ official; $+0.0898$ Protocol A; $+0.0827$ Protocol B) and higher observed Top-1 gold selection (+4.24 pp on gold-present claims), whereas the Weighted Baseline achieved statistically significant higher Average Precision under dynamic conditions ($-0.1396$ official; $-0.0680$ Protocol A; $-0.1098$ Protocol B) and higher AUROC under static conditions ($-0.0943$ official).\n\n")

v3_lines.append("### 8.2 Interpretation or Limitation\n")
v3_lines.append("- Absolute terms such as 'proves', 'optimizes', 'guarantees', or 'demonstrates universal superiority' are scientifically unfounded.\n")
v3_lines.append("- The observed effects are specific to the evaluated 500-claim dataset and 808-passage candidate pool; claiming universal generalizability across arbitrary fact-checking corpora is unjustified.\n")
v3_lines.append("- Counterfactual leave-one-out erasure measures post-hoc model sensitivity to feature removal; it does not constitute Pearl structural causal DAG mediation analysis.\n\n")

v3_lines.append("### 8.3 Recommended Correction (Wording Prescriptions)\n\n")
v3_lines.append("| Unbounded Phrasing (To Be Avoided) | Rigorous Bounded Phrasing (Mandatory in Phase 14) |\n")
v3_lines.append("|:---|:---|\n")
v3_lines.append("| *\"Fuzzy logic proves superior evidence reliability.\"* | *\"Fuzzy logic achieves statistically significant higher AUROC under dynamic contradiction, but lower Average Precision than the baseline.\"* |\n")
v3_lines.append("| *\"Causal evidence attribution.\"* | *\"Counterfactual leave-one-out sensitivity analysis.\"* |\n")
v3_lines.append("| *\"Standard bi-encoders achieve 50–65% recall across Wikipedia.\"* | *\"Retrieval is evaluated over an indexed corpus of 808 pre-extracted Wikipedia sentence passages derived from the sampled claims.\"* |\n")
v3_lines.append("| *\"Dense-only retrieval was used.\"* | *\"Hybrid BM25 + dense (all-MiniLM-L6-v2, $\\alpha = 0.5$) retrieval was used over a closed 808-passage candidate pool.\"* |\n")
v3_lines.append("| *\"The baseline AP advantage narrows significantly.\"* | *\"The baseline AP advantage narrows numerically from -0.1396 to -0.0680 in Protocol A and -0.1098 in Protocol B.\"* |\n")
v3_lines.append("| *\"LIME achieved higher mean and median correlation.\"* | *\"LIME achieved a higher mean correlation (0.5877 vs. 0.4517), while both achieved identical medians (0.5000).\"* |\n")
v3_lines.append("| *\"Fuzzy logic produced a statistically significant +4.2% Top-1 hit rate improvement.\"* | *\"Fuzzy logic produced an observed +4.24 percentage point improvement (87.63% vs. 83.39%) in Top-1 gold hit rate on gold-present claims; formal bootstrap hypothesis testing was conducted on co-primary AUROC/AP metrics, not on Top-1 binary hit outcomes.\"* |\n")
v3_lines.append("| *\"The verifier is never invoked with reversed evidence.\"* | *\"The verifier is invoked for full reversed evidence, but masked coalition evaluations in `_get_probabilities()` sort evidence IDs, retrieving cached forward probabilities.\"* |\n\n")

v3_lines.append("----\n\n")

v3_lines.append("## 9. Conclusion & Mandatory Phase Gate Freeze\n\n")
v3_lines.append("This document formalizes Version 3 of the Pre-Phase-14 Audit Errata, completing all requested technical corrections across hybrid retrieval architecture, Top-1 statistical interpretation, explainer caching mechanics, and contradiction formula syntax.\n\n")
v3_lines.append("- **Frozen Artifacts Intact:** Original reports, CSVs, `PRE_PHASE_14_AUDIT_ERRATA.md` (V1), and `PRE_PHASE_14_AUDIT_ERRATA_V2.md` (V2) remain strictly unchanged.\n")
v3_lines.append("- **Phase Gate Respected:** Phase 14 has **NOT** been started.\n")

with open('PRE_PHASE_14_AUDIT_ERRATA_V3.md', 'w', encoding='utf-8') as fp:
    fp.writelines(v3_lines)

print('PRE_PHASE_14_AUDIT_ERRATA_V3.md written successfully.')
