# Pre-Phase-14 Independent Technical Audit — Read-Only Evidence Collection & Technical Audit Report

**Audit Date:** 2026-10-20  
**Auditor:** Independent Technical Auditor (Automated Read-Only Inspection)  
**Audit Scope:** Full Project Lifecycle (Phase 0 through Phase 13)  
**Project Title:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Audit Status:** COMPLETE — EVIDENCE PRESERVED — READY FOR INDEPENDENT REVIEW — PHASE 14 NOT STARTED

----

## 1. Executive Summary & Audit Safety Constraints

This technical audit report and accompanying audit bundle have been compiled to support an independent evaluation of the **Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking** project before commencing Phase 14.

The objective is to gather an exhaustive, read-only evidence dossier across the entire codebase, experimental outputs, and analytical literature; to independently verify quantitative reproducibility; and to explicitly report all genuine methodological, mathematical, statistical, dataset, and documentation defects discovered.

### 1.1 Strict Safety & Constraint Compliance Verification

In strict adherence to the audit mandate, the following safety constraints were maintained throughout:
1. **Zero Modifications to Existing Code and Artifacts:** No existing project file was modified, reformatted, regenerated, or overwritten. Original files remain bit-for-bit identical to their pre-audit state.
2. **Zero Re-execution of Experiments:** Neither Experiment 1 (`experiments/experiment1_formal.py`) nor Experiment 2 (`experiments/experiment2_formal.py`) was re-run. No models were invoked, no random seeds were altered, and no scores were recomputed for output storage.
3. **Frozen Directory Integrity:** Files under `outputs/phase10_formal/` and `outputs/phase11_formal/` were inspected in place and copied into the bundle without modification.
4. **Zero Silent Repairs:** All identified defects—including mathematical formula collapsing in conflict detection, baseline zero-saturation, and missing presentation files—are fully exposed and documented rather than silently patched.
5. **No Raw Data Dumps:** Raw datasets (such as the 808-passage evidence corpus and the 3 MB raw FEVER jsonl split) are summarized via validated counts, schemas, and summary distributions rather than printed in full.
6. **Rigorous Epistemic Classification:** Every finding is explicitly classified into **Verified Fact**, **Suspected Problem**, or **Unresolved Question**.

## 2. Audit Bundle Directory Layout & Preservation

A dedicated read-only directory `audit_bundle/` has been created at the root of the workspace. Its internal layout mirrors the functional domains of the project:

```
audit_bundle/
├── source_files/               # 29 Authoritative source code files
│   ├── experiments/            # Formal experiment execution scripts (Exp 1 & Exp 2)
│   ├── fuzzy/                  # Mamdani engine, weighted baseline, normalization, schemas
│   ├── intervention/           # Counterfactual erasure and intervention evaluator
│   ├── attribution/            # Unified SHAP and LIME model explainers
│   ├── retrieval/              # Dense retriever and corpus loader
│   ├── data/                   # FEVER dataset loaders, schemas, and metadata
│   ├── conflict/               # NLI cross-encoder contradiction detector
│   ├── stability/              # Permutation-based evidence order stability analyzer
│   ├── app/                    # Web interface and end-to-end pipeline orchestrator
│   ├── config/                 # Settings dataclasses and YAML configurations
│   └── scratch/                # Analysis scripts and artifact generators
├── artifacts/                  # 34 Formal experimental results and reports
│   ├── outputs/
│   │   ├── phase10_formal/     # Frozen Exp 1 results.csv, config.json, and plots
│   │   ├── phase11_formal/     # Frozen Exp 2 results.csv, config.json, summary.json, plots
│   │   └── phase12_analysis/   # Phase 12 analysis_summary.csv, audit JSON, and figures
│   ├── docs/                   # Formal markdown documentation chapters
│   ├── *.md                    # Phase-level formal reports, audits, and specifications
│   └── *.pptx                  # Phase 0 proposal presentation
├── integrity_manifest.json     # Machine-readable SHA-256 cryptographic manifest
└── PRE_PHASE_14_INDEPENDENT_AUDIT_BUNDLE.md # Authoritative technical audit report
```

## 3. Task 1 — Authoritative Implementation Files Inventory

The table below details each requested implementation component, its actual relative path in the repository, byte size, and functional implementation status:

| Component Description | Requested Path | Actual Repository Path | Size (Bytes) | Architectural Status & Notes |
|:---|:---|:---|:---:|:---|
| Formal Experiment 1 Script | `experiments/experiment1_formal.py` | `experiments/experiment1_formal.py` | 12,182 | Present. Executes SHAP, LIME, and counterfactual intervention on 500 claims. |
| Formal Experiment 2 Script | `experiments/experiment2_formal.py` | `experiments/experiment2_formal.py` | 49,672 | Present. Executes Fuzzy vs. Baseline evaluation with claim-cluster bootstrap. |
| Fuzzy Logic Engine | `fuzzy/engine.py` | `fuzzy/engine.py` | 6,018 | Present. Mamdani fuzzy inference system with 9 linguistic rules and centroid defuzzification. |
| Weighted Baseline Engine | `fuzzy/baseline.py` | `fuzzy/baseline.py` | 1,384 | Present. Implements static weighted sum and dynamic multiplicative adjustment. |
| Fuzzy Normalization | `fuzzy/normalization.py` | `fuzzy/normalization.py` | 1,442 | Present. Normalizes attribution magnitudes and inputs to [0, 1]. |
| Fuzzy Schemas | `fuzzy/schema.py` | `fuzzy/schema.py` | 932 | Present. Defines `FuzzyInput` and `FuzzyOutput` dataclasses. |
| Counterfactual Intervener | `intervention/` | `intervention/intervener.py` | 5,583 | Present as `intervener.py` (not `evaluator.py`). Computes probability drops under leave-one-out erasure. |
| Attribution Explainer | `attribution/` | `attribution/explainer.py` | 10,398 | Present as `explainer.py` (not separate shap/lime files). Unified SHAP and LIME explainer. |
| Evidence Retriever | `retrieval/retriever.py` | `retrieval/retriever.py` | 7,266 | Present. Dense bi-encoder retrieval using `sentence-transformers/all-MiniLM-L6-v2`. |
| Dataset Loader | `data/fever_loader.py` | `data/fever_loader.py` | 11,163 | Present. Parses FEVER JSONL records and extracts gold evidence annotations. |
| Contradiction Detector | `conflict/detector.py` | `conflict/detector.py` | 3,809 | Present. Uses NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) for pairwise contradiction. |
| Stability Analyzer | `stability/analyzer.py` | `stability/analyzer.py` | 3,821 | Present. Computes permutation stability across reversed evidence orders. |
| Pipeline Orchestrator | `pipeline/pipeline.py` | `app/pipeline.py` | 5,868 | **Path Discrepancy.** Located at `app/pipeline.py`; directory `pipeline/` does not exist. |
| Config Definitions | `config/config.py` | `config/settings.py` & `config/default_config.yaml` | 3,500 / 555 | **Path Discrepancy.** Located at `config/settings.py` and `config/default_config.yaml`. |
| Bootstrap Inference Code | (inference engine) | `experiments/experiment2_formal.py` (lines 276–398) | — | Embedded directly within Experiment 2 formal execution script. |
| Figure & Table Scripts | (analysis scripts) | `scratch/generate_phase12_artifacts.py` & `scratch/run_phase12_analysis.py` | 17,676 / 10,343 | Present in `scratch/`. Generates Phase 12 analysis tables, summaries, and publication figures. |

## 4. Task 2 — Authoritative Experimental Artifacts Inventory

The table below details all preserved experimental data, summaries, reports, and presentation artifacts:

| Requested Artifact | Actual Repository Path | Size (Bytes) | Artifact Role & Audit Notes |
|:---|:---|:---:|:---|
| Exp 1 Frozen Results CSV | `outputs/phase10_formal/results.csv` | 640,584 | Primary data: 1,500 evidence rows across 500 claims; 17 evaluated metrics. |
| Exp 1 Frozen Config JSON | `outputs/phase10_formal/config.json` | 566 | Execution parameters: 500 claims, top-3 retrieval, random seed 42. |
| Exp 1 Publication Plots | `outputs/phase10_formal/*.png` | 101,580 (2 files) | Scatter and calibration plots: `shap_vs_influence.png`, `lime_vs_influence.png`. |
| Exp 2 Frozen Results CSV | `outputs/phase11_formal/results.csv` | 865,078 | Primary data: 1,500 evidence rows; 39 columns comparing Fuzzy vs. Baseline. |
| Exp 2 Frozen Config JSON | `outputs/phase11_formal/config.json` | 878 | Execution parameters: B=1,000 bootstrap replicates, seed 42, dual ablation. |
| Exp 2 Formal Summary JSON | `outputs/phase11_formal/summary.json` | 7,205 | Statistical outcomes: Point estimates, CIs, p-values, Holm corrections, saturation rates. |
| Exp 2 Publication Plots | `outputs/phase11_formal/plots/*.png` | 873,224 (4 files) | ROC curves, distributions, hit rates, and attribution gap behavior plots. |
| Phase 12 Analysis Summary CSV | `outputs/phase12_analysis/analysis_summary.csv` | 6,080 | Consolidated statistical table for Exp 1: 57 verified metrics and percentiles. |
| Phase 12 Dataset Overlap JSON | `outputs/phase12_analysis/dataset_overlap_audit.json` | 5,679 | Rigorous audit of corpus construction, retrieval coverage, and contingency tables. |
| Phase 12 Publication Plots | `outputs/phase12_analysis/*.png` | 716,546 (4 files) | Influence ranks, Spearman distributions, agreement breakdowns, flip dynamics. |
| Exp 1 Formal Report | `EXPERIMENT_1_FORMAL_REPORT.md` | 3,643 | Initial formal execution report for Experiment 1. |
| Exp 2 Formal Report | `EXPERIMENT_2_FORMAL_REPORT.md` | 19,217 | Authoritative formal execution report for Experiment 2. |
| Exp 2 Post-Run Validation | `EXPERIMENT_2_POSTRUN_VALIDATION.md` | 18,930 | Independent validation confirming reproduction and identifying baseline saturation. |
| Phase 12 Research Analysis | `PHASE_12_RESEARCH_ANALYSIS.md` | 27,397 | In-depth scientific analysis of Experiment 1 rank correlation and verdict flips. |
| Phase 13 Research Synthesis | `PHASE_13_FINAL_RESEARCH_SYNTHESIS.md` | 22,834 | Master research synthesis integrating both experiments and scientific trade-offs. |
| Phase 13 Technical Audit | `PHASE_13_FINAL_TECHNICAL_AUDIT.md` | 12,193 | Prior cross-phase technical consistency audit report. |
| Phase 13 Documentation Audit | `PHASE_13_DOCUMENTATION_AUDIT.md` | 10,875 | Prior cross-phase documentation synchronization audit report. |
| Dataset Overlap Audit Report | `DATASET_OVERLAP_AUDIT.md` | 12,765 | Comprehensive analysis of gold-evidence availability and closed-pool bounds. |
| Docs: Experiments | `docs/EXPERIMENTS.md` | 7,731 | Master specification of formal experimental protocols and baselines. |
| Docs: Datasets & Provenance | `docs/DATASETS_AND_PROVENANCE.md` | 4,892 | Documenting FEVER dev split streaming, corpus pooling, and licensing. |
| Docs: Limitations & Future Work | `docs/LIMITATIONS_AND_FUTURE_WORK.md` | 5,925 | Documenting scientific boundaries, proxy labels, and closed candidate pool. |
| Project Specification | `PROJECT_SPECIFICATION.md` | 13,706 | Architectural specifications, planned modules, and evaluation criteria. |
| Final Presentation PPTX | `final_ppt.pptx` | **MISSING** | **CRITICAL DEFECT.** File does not exist anywhere in the repository. |
| Proposal Presentation PPTX | `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx` | 58,844 | Phase 0 Proposal slide deck (6 slides; pre-experimental; dated August 2026). |

## 5. Task 3 — Read-Only Cryptographic Integrity Manifest

To ensure strict data integrity and cryptographic chain-of-custody, SHA-256 digests were computed for all 29 source files and 34 preserved artifacts.

> [!IMPORTANT]
> **Cryptographic Baseline Confirmation:** An exhaustive search of all commit histories, markdown reports, JSON summaries, and logs across Phases 0 through 13 verified that **no prior SHA-256 hash manifest or hash store existed**. Prior reports certified integrity using row counts and summary statistics rather than cryptographic hashes. Therefore, this manifest establishes the project's official, immutable cryptographic baseline.

### 5.1 Source Code Files Integrity Manifest

| Relative Path | Size (Bytes) | Last Modified (UTC) | SHA-256 Hash |
|:---|:---:|:---:|:---|
| `experiments/experiment1_formal.py` | 12,182 | 2026-09-25T07:41:15.677427Z | `fe086a026c12993e2dc42a9e4aaf905dc5dfd77ddbba10245dfcb1bf4b38639f` |
| `experiments/experiment2_formal.py` | 49,672 | 2026-10-19T14:08:14.094733Z | `8721c2b8eb5a4e053a01bb1fe22a9a58b6d64bcc1ea81bf67cc1c4ac59252a82` |
| `fuzzy/engine.py` | 6,018 | 2026-08-26T15:18:10.103759Z | `db4bd8023b3cebbb326cd3e5c5b05e181d714e48e9384145c40017834a53b2dd` |
| `fuzzy/baseline.py` | 1,384 | 2026-08-26T15:18:18.794640Z | `74c4f6820229f6f22e742a0670e90e737364e59518952ddbd1b21f9ab291604a` |
| `fuzzy/normalization.py` | 1,442 | 2026-08-26T15:17:52.907170Z | `343065827de1eade5a38f761dc7d4fcc96740debe214c55118cf4493c6400650` |
| `fuzzy/schema.py` | 932 | 2026-08-26T15:17:44.361326Z | `f2ba412fe48393a7dd4e42ee1219548bbe5a794773b9c42e778c069f80d18ce6` |
| `fuzzy/__init__.py` | 586 | 2026-08-26T15:18:32.627339Z | `b4146742391da51d4541248ba9a7ea2ab03da6582c40d997a88a78db0fe6d14d` |
| `intervention/intervener.py` | 5,583 | 2026-08-26T14:57:33.450362Z | `fa1e22d5a453faf1a831878509b09e514ff788df57fbd0948a0f44d11fbf4222` |
| `intervention/schema.py` | 1,218 | 2026-08-26T14:57:16.405530Z | `fadfabb1659320612337cef1281342bc51375c1fc2d347fb6e30324604f5e159` |
| `intervention/__init__.py` | 291 | 2026-08-26T14:57:39.729492Z | `0034f39186f5b178facb94287e5c82431b6657263e70bb096516651587435833` |
| `attribution/explainer.py` | 10,398 | 2026-08-26T14:51:11.230807Z | `147d300ebd50e043afb300dd15ac37b447cc107d1f7dfaf8b5575b7053bd0afe` |
| `attribution/schema.py` | 1,950 | 2026-08-26T14:43:25.742997Z | `a87f1ebf248166ba986ded61d20363d199bfa28a2921efa90d7aa2be89930a53` |
| `attribution/__init__.py` | 273 | 2026-08-26T14:44:12.169810Z | `e79000d95eb5519bce4ee2036518134fe6fbe0b18b619038c4b4c4fc86fb9118` |
| `retrieval/retriever.py` | 7,266 | 2026-08-21T14:13:16.426939Z | `836e2517e5e0a0fee5aa4b1ce031c3855c530b4000562cb619f4aeae06ee9551` |
| `data/fever_loader.py` | 11,163 | 2026-08-21T04:26:45.734313Z | `0904397eddce2c094c4c0764acdfe641da17a7e9b66760f1c34a94ac5a8d9c50` |
| `data/schema.py` | 2,969 | 2026-08-21T04:26:02.185053Z | `2c0c0d61240ea47b455090296b6a097c093945ac7a7f4d5f495f59c8e3459fa1` |
| `data/dataset_meta.json` | 290 | 2026-08-21T04:28:31.285898Z | `ec9f2e4a1cb9c992267c82048d1ef789c7a711686ef86deb2409dc99b30c7be5` |
| `conflict/detector.py` | 3,809 | 2026-08-26T15:31:44.174398Z | `07b21b4c1c1c9b0d01c53e873baf4fe4d0d1760e23dd2ec4caf52c13b2e19ed3` |
| `stability/analyzer.py` | 3,821 | 2026-08-26T15:32:16.301768Z | `156601abe26d882ebd84bd6d0ff59d6a07258db99c87bd31de0dd79ae3b29038` |
| `app/pipeline.py` | 5,868 | 2026-08-27T03:26:05.997048Z | `e1878f24158c79834933254361df28347c711c11ecbd8a3d14b39871253dc141` |
| `app/main.py` | 4,334 | 2026-08-27T03:24:14.405820Z | `7530e9fa4daea8381c2e94f8625755f877068617d2467ff6fe9a7c67bf4e737a` |
| `app/schema.py` | 1,359 | 2026-08-27T03:23:43.015987Z | `491f3f3c7aba70a3a5165134917d9cb6ec0dd14a2cfd8f820bb4512c50ed77eb` |
| `config/settings.py` | 3,500 | 2026-08-26T15:31:23.330827Z | `07ccd8690caa95a00b52ce5668f5a9ab392932c36128e5c7caa50561e357bbfd` |
| `config/default_config.yaml` | 555 | 2026-08-20T06:55:49.537373Z | `8a2fb8cbdfb4b91b636566d8c501d9be3144b8697817c574872246d27cb8abde` |
| `scratch/generate_phase12_artifacts.py` | 17,676 | 2026-10-18T16:06:35.504854Z | `1dc236cad24894bb7285b0b8f3ed8230d474eee1a99a8dbee408b83baec3ddc3` |
| `scratch/run_phase12_analysis.py` | 10,343 | 2026-10-18T16:05:06.211919Z | `fe22cf015717c05af07ad2470a16d318b218166e0807e762e803a869fbe984fa` |
| `scratch/update_analysis_summary.py` | 10,766 | 2026-10-18T16:27:59.705916Z | `0cb55d55f92dcddc8fd4218ee126dc23493282013ea93a8786339f5a52ff0c1d` |
| `scratch/update_audit_json.py` | 5,801 | 2026-10-18T16:27:21.862834Z | `1dbf31f57faa4eceb15597040ba6f6f1336f4977b7eef976b37ae9931a71f768` |
| `scratch/verify_exp2.py` | 16,761 | 2026-10-19T15:51:35.198043Z | `73a67fd51554471ff5e76b98d899bcec04cae6e05100a9e27177e68f377931fc` |

### 5.2 Experimental Artifacts Integrity Manifest

| Relative Path | Size (Bytes) | Last Modified (UTC) | SHA-256 Hash |
|:---|:---:|:---:|:---|
| `outputs/phase10_formal/results.csv` | 640,584 | 2026-09-25T07:56:13.885935Z | `75746ff340e8a5165547208bf4123ed8147790ab3a5bddba938b2d393cff1ea6` |
| `outputs/phase10_formal/config.json` | 566 | 2026-09-25T07:41:38.480286Z | `d94a316bb8f9b7f5823a8f930fe8afd24a10f49c44ad8b381f5a7e34adf2ada0` |
| `outputs/phase10_formal/lime_vs_influence.png` | 48,178 | 2026-09-25T07:56:14.393041Z | `0470c4856d4b2b7929ed76324c771c0141936a1d806b998351de855dfb384190` |
| `outputs/phase10_formal/shap_vs_influence.png` | 53,402 | 2026-09-25T07:56:14.199993Z | `7ee0f4ad7a78e5bea9a1525e5c8f8719a33c587be4d9a07369ebd3428262817d` |
| `outputs/phase11_formal/results.csv` | 865,078 | 2026-10-19T15:25:13.403883Z | `f3d6578c572db62cbdcb9afb299502add5148b02fb4d8b8190faca5b795208aa` |
| `outputs/phase11_formal/config.json` | 878 | 2026-10-19T15:38:49.739351Z | `7e67552e49d9a920367388c9774007962c636419a2cd5d1c4df3f774d6112fb5` |
| `outputs/phase11_formal/summary.json` | 7,205 | 2026-10-19T15:38:49.737976Z | `e03321d99a9c2580810baaaca567c1f326fee745b04a0012ecbe22b31337ddf6` |
| `outputs/phase11_formal/independent_validation.json` | 4,211 | 2026-10-19T15:52:49.535122Z | `40226a7085e1ea357a2d0336780583328cf94463efbc36e383840d9d80d546ad` |
| `outputs/phase11_formal/plots/attribution_gap_behavior.png` | 210,890 | 2026-10-19T15:38:49.720740Z | `dd6cdfbe81ae07689bdcd536db91f6bb5389cfa52d28a67406626cb4fa08e8c2` |
| `outputs/phase11_formal/plots/gold_discrimination_roc.png` | 379,981 | 2026-10-19T15:38:47.061324Z | `091cbac534310773ad8e61f514b6799f1d33f7fb1fac48b56a306ecfc5c22c61` |
| `outputs/phase11_formal/plots/score_distribution_comparison.png` | 167,847 | 2026-10-19T15:38:48.136170Z | `aa50ab062fb0bf3aef480efdcbea95ad8bfdd093e4911c171ffb0ce9fb700f05` |
| `outputs/phase11_formal/plots/top1_hit_rate_comparison.png` | 114,506 | 2026-10-19T15:38:49.010420Z | `382b022240d287e621b0d3278a30ebd379945d8907517a9d594f27c633f840e7` |
| `outputs/phase12_analysis/analysis_summary.csv` | 6,080 | 2026-10-18T16:28:28.579497Z | `71a09a310568bbd4bcd6179d1222e0c27175c005a197ef624b867092c3d17194` |
| `outputs/phase12_analysis/dataset_overlap_audit.json` | 5,679 | 2026-10-18T16:27:32.413923Z | `771c795d30c829a8f3b6ec95df6984ce4e863c378da3d1bef49f1ac3da300d06` |
| `outputs/phase12_analysis/influence_by_attribution_rank.png` | 211,449 | 2026-10-18T16:07:06.798135Z | `6a3a5a025fbee42de88d5e479a7acb0f55b46e673426515a4d153beea89f230e` |
| `outputs/phase12_analysis/spearman_rho_distribution.png` | 171,116 | 2026-10-18T16:07:05.603211Z | `c445e660263f513eeefe5050b1eb8caa190df1633795aeae447e0e972a876d0d` |
| `outputs/phase12_analysis/top1_agreement_breakdown.png` | 163,759 | 2026-10-18T16:07:05.988930Z | `3464f18463f7e72cec66d3cd66a422d84dd6390469a44a3872411490c6026431` |
| `outputs/phase12_analysis/verdict_flip_dynamics.png` | 170,222 | 2026-10-18T16:07:06.402618Z | `19b8968b24267b273548fc56ad3ebc5546429fd68515e1c6b5f7c647ae801a5c` |
| `EXPERIMENT_1_FORMAL_REPORT.md` | 3,643 | 2026-09-25T07:56:14.402073Z | `2c0d5ed513412702731fc2e886fcb509f6bf7728b7dc3cc1bbe0dcf959397ee2` |
| `EXPERIMENT_2_FORMAL_REPORT.md` | 19,217 | 2026-10-19T15:58:41.233165Z | `e6a9cc7952fbd316101e4e344e8c41aec61b49171e465aacef73899e514d7577` |
| `EXPERIMENT_2_POSTRUN_VALIDATION.md` | 18,930 | 2026-10-19T15:59:11.572814Z | `a892fc33b463ec3509a98d4e9bcf0e3e399b06811d6947ca9557aff764ea43d8` |
| `PHASE_12_RESEARCH_ANALYSIS.md` | 27,397 | 2026-10-18T16:30:25.560087Z | `0b2a76cb01dcf3ac09c331a34d7823151ae14aa124e3875c342b93f3d4c5ed50` |
| `PHASE_13_FINAL_RESEARCH_SYNTHESIS.md` | 22,834 | 2026-10-19T16:22:51.222663Z | `f0fe3cd2729d34581505d258f1eeb1499bcf9c18165beb2870342c9da0cdf6c6` |
| `PHASE_13_FINAL_TECHNICAL_AUDIT.md` | 12,193 | 2026-10-19T16:24:18.192688Z | `9e7a816da92c964616a4d9f4a2d181e41bc5d25d3fe6502558d9083e970e2671` |
| `PHASE_13_DOCUMENTATION_AUDIT.md` | 10,875 | 2026-10-19T16:23:34.998866Z | `d074f5481a0004dabc8918653e5ed88d3930fcec35aa73fee2d7f55ffdb21f88` |
| `DATASET_OVERLAP_AUDIT.md` | 12,765 | 2026-10-18T16:28:59.422458Z | `2f6b8825c8028bb3db4d53229fa4b8bb4157f9c7474368d80bee65ab2e1ad6aa` |
| `docs/EXPERIMENTS.md` | 7,731 | 2026-10-19T16:21:00.314076Z | `b342545bf4bcd3aec0e9abc45c341aac3f143779016a2ddd651b205dd258f67b` |
| `docs/DATASETS_AND_PROVENANCE.md` | 4,892 | 2026-10-19T16:01:42.674212Z | `13af450e21e778772df33a6eebe91b21def2681cba9ab464037a79c80f8aa36e` |
| `docs/LIMITATIONS_AND_FUTURE_WORK.md` | 5,925 | 2026-10-19T16:21:22.797465Z | `537f1e324fd9f4af9784eb22b2322448240781f8afd4d6edbe2f9dca0005aeb7` |
| `docs/ARCHITECTURE.md` | 15,732 | 2026-10-19T16:19:34.853972Z | `70776ca1b9946286cb569874f5fbc4ec5cae6593c03674b11734676b4e7ae9c1` |
| `docs/REPRODUCIBILITY.md` | 5,262 | 2026-10-19T16:02:31.434101Z | `d78e47729140cfcf77ecb5289135788f87c3ef7d8f317d453a45d6f7cb8b369d` |
| `PROJECT_SPECIFICATION.md` | 13,706 | 2026-08-20T06:50:38.522868Z | `2d498968c953e9029240f10bf1fa5eed1eaec2fa54fbb65ddea5207d36f1726c` |
| `README.md` | 9,696 | 2026-10-19T16:23:18.557153Z | `90d0bf67c3a84f25c2bd26fbd4034cbe4d9f23807c077689169ba9551040bb92` |
| `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx` | 58,844 | 2026-08-20T04:47:07.289499Z | `14d3228a51886fb7fd8e3d839d1e1cba3988d8452b01445359afbaaf0024f11e` |


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
