import os
import json
import datetime
import pandas as pd
import numpy as np

# Load manifest
with open('audit_bundle/integrity_manifest.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)

src_files = manifest['source_files']
art_files = manifest['artifacts']

doc = []

# Title & Metadata
doc.append("# Pre-Phase-14 Independent Technical Audit — Read-Only Evidence Collection & Technical Audit Report\n\n")
doc.append("**Audit Date:** 2026-10-20  \n")
doc.append("**Auditor:** Independent Technical Auditor (Automated Read-Only Inspection)  \n")
doc.append("**Audit Scope:** Full Project Lifecycle (Phase 0 through Phase 13)  \n")
doc.append("**Project Title:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  \n")
doc.append("**Audit Status:** COMPLETE — EVIDENCE PRESERVED — READY FOR INDEPENDENT REVIEW — PHASE 14 NOT STARTED\n\n")

doc.append("----\n\n")

# Section 1: Executive Summary & Safety Compliance
doc.append("## 1. Executive Summary & Audit Safety Constraints\n\n")
doc.append("This technical audit report and accompanying audit bundle have been compiled to support an independent evaluation of the **Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking** project before commencing Phase 14.\n\n")
doc.append("The objective is to gather an exhaustive, read-only evidence dossier across the entire codebase, experimental outputs, and analytical literature; to independently verify quantitative reproducibility; and to explicitly report all genuine methodological, mathematical, statistical, dataset, and documentation defects discovered.\n\n")

doc.append("### 1.1 Strict Safety & Constraint Compliance Verification\n\n")
doc.append("In strict adherence to the audit mandate, the following safety constraints were maintained throughout:\n")
doc.append("1. **Zero Modifications to Existing Code and Artifacts:** No existing project file was modified, reformatted, regenerated, or overwritten. Original files remain bit-for-bit identical to their pre-audit state.\n")
doc.append("2. **Zero Re-execution of Experiments:** Neither Experiment 1 (`experiments/experiment1_formal.py`) nor Experiment 2 (`experiments/experiment2_formal.py`) was re-run. No models were invoked, no random seeds were altered, and no scores were recomputed for output storage.\n")
doc.append("3. **Frozen Directory Integrity:** Files under `outputs/phase10_formal/` and `outputs/phase11_formal/` were inspected in place and copied into the bundle without modification.\n")
doc.append("4. **Zero Silent Repairs:** All identified defects—including mathematical formula collapsing in conflict detection, baseline zero-saturation, and missing presentation files—are fully exposed and documented rather than silently patched.\n")
doc.append("5. **No Raw Data Dumps:** Raw datasets (such as the 808-passage evidence corpus and the 3 MB raw FEVER jsonl split) are summarized via validated counts, schemas, and summary distributions rather than printed in full.\n")
doc.append("6. **Rigorous Epistemic Classification:** Every finding is explicitly classified into **Verified Fact**, **Suspected Problem**, or **Unresolved Question**.\n\n")

# Section 2: Bundle Structure
doc.append("## 2. Audit Bundle Directory Layout & Preservation\n\n")
doc.append("A dedicated read-only directory `audit_bundle/` has been created at the root of the workspace. Its internal layout mirrors the functional domains of the project:\n\n")
doc.append("```\naudit_bundle/\n├── source_files/               # 29 Authoritative source code files\n│   ├── experiments/            # Formal experiment execution scripts (Exp 1 & Exp 2)\n│   ├── fuzzy/                  # Mamdani engine, weighted baseline, normalization, schemas\n│   ├── intervention/           # Counterfactual erasure and intervention evaluator\n│   ├── attribution/            # Unified SHAP and LIME model explainers\n│   ├── retrieval/              # Dense retriever and corpus loader\n│   ├── data/                   # FEVER dataset loaders, schemas, and metadata\n│   ├── conflict/               # NLI cross-encoder contradiction detector\n│   ├── stability/              # Permutation-based evidence order stability analyzer\n│   ├── app/                    # Web interface and end-to-end pipeline orchestrator\n│   ├── config/                 # Settings dataclasses and YAML configurations\n│   └── scratch/                # Analysis scripts and artifact generators\n├── artifacts/                  # 34 Formal experimental results and reports\n│   ├── outputs/\n│   │   ├── phase10_formal/     # Frozen Exp 1 results.csv, config.json, and plots\n│   │   ├── phase11_formal/     # Frozen Exp 2 results.csv, config.json, summary.json, plots\n│   │   └── phase12_analysis/   # Phase 12 analysis_summary.csv, audit JSON, and figures\n│   ├── docs/                   # Formal markdown documentation chapters\n│   ├── *.md                    # Phase-level formal reports, audits, and specifications\n│   └── *.pptx                  # Phase 0 proposal presentation\n├── integrity_manifest.json     # Machine-readable SHA-256 cryptographic manifest\n└── PRE_PHASE_14_INDEPENDENT_AUDIT_BUNDLE.md # Authoritative technical audit report\n```\n\n")

# Section 3: Task 1 File Inventory
doc.append("## 3. Task 1 — Authoritative Implementation Files Inventory\n\n")
doc.append("The table below details each requested implementation component, its actual relative path in the repository, byte size, and functional implementation status:\n\n")
doc.append("| Component Description | Requested Path | Actual Repository Path | Size (Bytes) | Architectural Status & Notes |\n")
doc.append("|:---|:---|:---|:---:|:---|\n")
doc.append("| Formal Experiment 1 Script | `experiments/experiment1_formal.py` | `experiments/experiment1_formal.py` | 12,182 | Present. Executes SHAP, LIME, and counterfactual intervention on 500 claims. |\n")
doc.append("| Formal Experiment 2 Script | `experiments/experiment2_formal.py` | `experiments/experiment2_formal.py` | 49,672 | Present. Executes Fuzzy vs. Baseline evaluation with claim-cluster bootstrap. |\n")
doc.append("| Fuzzy Logic Engine | `fuzzy/engine.py` | `fuzzy/engine.py` | 6,018 | Present. Mamdani fuzzy inference system with 9 linguistic rules and centroid defuzzification. |\n")
doc.append("| Weighted Baseline Engine | `fuzzy/baseline.py` | `fuzzy/baseline.py` | 1,384 | Present. Implements static weighted sum and dynamic multiplicative adjustment. |\n")
doc.append("| Fuzzy Normalization | `fuzzy/normalization.py` | `fuzzy/normalization.py` | 1,442 | Present. Normalizes attribution magnitudes and inputs to [0, 1]. |\n")
doc.append("| Fuzzy Schemas | `fuzzy/schema.py` | `fuzzy/schema.py` | 932 | Present. Defines `FuzzyInput` and `FuzzyOutput` dataclasses. |\n")
doc.append("| Counterfactual Intervener | `intervention/` | `intervention/intervener.py` | 5,583 | Present as `intervener.py` (not `evaluator.py`). Computes probability drops under leave-one-out erasure. |\n")
doc.append("| Attribution Explainer | `attribution/` | `attribution/explainer.py` | 10,398 | Present as `explainer.py` (not separate shap/lime files). Unified SHAP and LIME explainer. |\n")
doc.append("| Evidence Retriever | `retrieval/retriever.py` | `retrieval/retriever.py` | 7,266 | Present. Dense bi-encoder retrieval using `sentence-transformers/all-MiniLM-L6-v2`. |\n")
doc.append("| Dataset Loader | `data/fever_loader.py` | `data/fever_loader.py` | 11,163 | Present. Parses FEVER JSONL records and extracts gold evidence annotations. |\n")
doc.append("| Contradiction Detector | `conflict/detector.py` | `conflict/detector.py` | 3,809 | Present. Uses NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) for pairwise contradiction. |\n")
doc.append("| Stability Analyzer | `stability/analyzer.py` | `stability/analyzer.py` | 3,821 | Present. Computes permutation stability across reversed evidence orders. |\n")
doc.append("| Pipeline Orchestrator | `pipeline/pipeline.py` | `app/pipeline.py` | 5,868 | **Path Discrepancy.** Located at `app/pipeline.py`; directory `pipeline/` does not exist. |\n")
doc.append("| Config Definitions | `config/config.py` | `config/settings.py` & `config/default_config.yaml` | 3,500 / 555 | **Path Discrepancy.** Located at `config/settings.py` and `config/default_config.yaml`. |\n")
doc.append("| Bootstrap Inference Code | (inference engine) | `experiments/experiment2_formal.py` (lines 276–398) | — | Embedded directly within Experiment 2 formal execution script. |\n")
doc.append("| Figure & Table Scripts | (analysis scripts) | `scratch/generate_phase12_artifacts.py` & `scratch/run_phase12_analysis.py` | 17,676 / 10,343 | Present in `scratch/`. Generates Phase 12 analysis tables, summaries, and publication figures. |\n")

# Section 4: Task 2 Artifacts Inventory
doc.append("\n## 4. Task 2 — Authoritative Experimental Artifacts Inventory\n\n")
doc.append("The table below details all preserved experimental data, summaries, reports, and presentation artifacts:\n\n")
doc.append("| Requested Artifact | Actual Repository Path | Size (Bytes) | Artifact Role & Audit Notes |\n")
doc.append("|:---|:---|:---:|:---|\n")
doc.append("| Exp 1 Frozen Results CSV | `outputs/phase10_formal/results.csv` | 640,584 | Primary data: 1,500 evidence rows across 500 claims; 17 evaluated metrics. |\n")
doc.append("| Exp 1 Frozen Config JSON | `outputs/phase10_formal/config.json` | 566 | Execution parameters: 500 claims, top-3 retrieval, random seed 42. |\n")
doc.append("| Exp 1 Publication Plots | `outputs/phase10_formal/*.png` | 101,580 (2 files) | Scatter and calibration plots: `shap_vs_influence.png`, `lime_vs_influence.png`. |\n")
doc.append("| Exp 2 Frozen Results CSV | `outputs/phase11_formal/results.csv` | 865,078 | Primary data: 1,500 evidence rows; 39 columns comparing Fuzzy vs. Baseline. |\n")
doc.append("| Exp 2 Frozen Config JSON | `outputs/phase11_formal/config.json` | 878 | Execution parameters: B=1,000 bootstrap replicates, seed 42, dual ablation. |\n")
doc.append("| Exp 2 Formal Summary JSON | `outputs/phase11_formal/summary.json` | 7,205 | Statistical outcomes: Point estimates, CIs, p-values, Holm corrections, saturation rates. |\n")
doc.append("| Exp 2 Publication Plots | `outputs/phase11_formal/plots/*.png` | 873,224 (4 files) | ROC curves, distributions, hit rates, and attribution gap behavior plots. |\n")
doc.append("| Phase 12 Analysis Summary CSV | `outputs/phase12_analysis/analysis_summary.csv` | 6,080 | Consolidated statistical table for Exp 1: 57 verified metrics and percentiles. |\n")
doc.append("| Phase 12 Dataset Overlap JSON | `outputs/phase12_analysis/dataset_overlap_audit.json` | 5,679 | Rigorous audit of corpus construction, retrieval coverage, and contingency tables. |\n")
doc.append("| Phase 12 Publication Plots | `outputs/phase12_analysis/*.png` | 716,546 (4 files) | Influence ranks, Spearman distributions, agreement breakdowns, flip dynamics. |\n")
doc.append("| Exp 1 Formal Report | `EXPERIMENT_1_FORMAL_REPORT.md` | 3,643 | Initial formal execution report for Experiment 1. |\n")
doc.append("| Exp 2 Formal Report | `EXPERIMENT_2_FORMAL_REPORT.md` | 19,217 | Authoritative formal execution report for Experiment 2. |\n")
doc.append("| Exp 2 Post-Run Validation | `EXPERIMENT_2_POSTRUN_VALIDATION.md` | 18,930 | Independent validation confirming reproduction and identifying baseline saturation. |\n")
doc.append("| Phase 12 Research Analysis | `PHASE_12_RESEARCH_ANALYSIS.md` | 27,397 | In-depth scientific analysis of Experiment 1 rank correlation and verdict flips. |\n")
doc.append("| Phase 13 Research Synthesis | `PHASE_13_FINAL_RESEARCH_SYNTHESIS.md` | 22,834 | Master research synthesis integrating both experiments and scientific trade-offs. |\n")
doc.append("| Phase 13 Technical Audit | `PHASE_13_FINAL_TECHNICAL_AUDIT.md` | 12,193 | Prior cross-phase technical consistency audit report. |\n")
doc.append("| Phase 13 Documentation Audit | `PHASE_13_DOCUMENTATION_AUDIT.md` | 10,875 | Prior cross-phase documentation synchronization audit report. |\n")
doc.append("| Dataset Overlap Audit Report | `DATASET_OVERLAP_AUDIT.md` | 12,765 | Comprehensive analysis of gold-evidence availability and closed-pool bounds. |\n")
doc.append("| Docs: Experiments | `docs/EXPERIMENTS.md` | 7,731 | Master specification of formal experimental protocols and baselines. |\n")
doc.append("| Docs: Datasets & Provenance | `docs/DATASETS_AND_PROVENANCE.md` | 4,892 | Documenting FEVER dev split streaming, corpus pooling, and licensing. |\n")
doc.append("| Docs: Limitations & Future Work | `docs/LIMITATIONS_AND_FUTURE_WORK.md` | 5,925 | Documenting scientific boundaries, proxy labels, and closed candidate pool. |\n")
doc.append("| Project Specification | `PROJECT_SPECIFICATION.md` | 13,706 | Architectural specifications, planned modules, and evaluation criteria. |\n")
doc.append("| Final Presentation PPTX | `final_ppt.pptx` | **MISSING** | **CRITICAL DEFECT.** File does not exist anywhere in the repository. |\n")
doc.append("| Proposal Presentation PPTX | `Fuzzy_Causal_Fact_Checking_Proposal_V2.pptx` | 58,844 | Phase 0 Proposal slide deck (6 slides; pre-experimental; dated August 2026). |\n")

# Section 5: Task 3 Cryptographic Integrity Manifest
doc.append("\n## 5. Task 3 — Read-Only Cryptographic Integrity Manifest\n\n")
doc.append("To ensure strict data integrity and cryptographic chain-of-custody, SHA-256 digests were computed for all 29 source files and 34 preserved artifacts.\n\n")
doc.append("> [!IMPORTANT]\n")
doc.append("> **Cryptographic Baseline Confirmation:** An exhaustive search of all commit histories, markdown reports, JSON summaries, and logs across Phases 0 through 13 verified that **no prior SHA-256 hash manifest or hash store existed**. Prior reports certified integrity using row counts and summary statistics rather than cryptographic hashes. Therefore, this manifest establishes the project's official, immutable cryptographic baseline.\n\n")

doc.append("### 5.1 Source Code Files Integrity Manifest\n\n")
doc.append("| Relative Path | Size (Bytes) | Last Modified (UTC) | SHA-256 Hash |\n")
doc.append("|:---|:---:|:---:|:---|\n")
for p, info in src_files.items():
    if 'error' in info:
        doc.append(f"| `{p}` | — | — | **MISSING** |\n")
    else:
        doc.append(f"| `{info['relative_path']}` | {info['size_bytes']:,} | {info['last_modified_utc']} | `{info['sha256']}` |\n")

doc.append("\n### 5.2 Experimental Artifacts Integrity Manifest\n\n")
doc.append("| Relative Path | Size (Bytes) | Last Modified (UTC) | SHA-256 Hash |\n")
doc.append("|:---|:---:|:---:|:---|\n")
for p, info in art_files.items():
    if 'error' in info:
        doc.append(f"| `{p}` | — | — | **MISSING** |\n")
    else:
        doc.append(f"| `{info['relative_path']}` | {info['size_bytes']:,} | {info['last_modified_utc']} | `{info['sha256']}` |\n")

with open('scratch/report_base.md', 'w', encoding='utf-8') as f:
    f.writelines(doc)

print('report_base.md successfully generated.')
