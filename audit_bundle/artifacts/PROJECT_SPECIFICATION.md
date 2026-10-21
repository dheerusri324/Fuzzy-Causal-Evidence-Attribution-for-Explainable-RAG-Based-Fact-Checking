# PROJECT SPECIFICATION

## Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking

**Status:** FROZEN — Phase 0 Complete  
**Date:** 2026-08-20  
**Team:** Bala Dheeraj Chennavaram (1602-23-733-006), A Saketh Vamshee (1602-23-733-042), V Rohit Reddy (1602-23-733-065)

---

## 1. Project Objective

Build an end-to-end explainable RAG-based fact-checking system that:
1. Retrieves evidence for a user claim via hybrid retrieval.
2. Verifies the claim against retrieved evidence.
3. Estimates evidence attribution using SHAP and LIME.
4. Tests whether attributed evidence actually influences the verifier's decision via intervention (evidence removal).
5. Combines imperfect signals through fuzzy reasoning to produce continuous reliability/confidence scores.
6. Detects conflicting evidence explicitly.
7. Measures explanation stability under perturbations.
8. Outputs a reliability-aware explanation with verdict, confidence, evidence influence, fuzzy scores, conflict status, and stability.

### Core Research Question

> Which evidence appears influential, does that evidence actually change the verifier's decision under intervention, and how reliable should the resulting explanation be?

### Central Contribution

A reliability-aware evidence attribution framework that combines post-hoc attribution (SHAP/LIME) with intervention-based evidence influence and fuzzy evidence reasoning for explainable RAG-based fact-checking.

SHAP, LIME, RAG, and fuzzy logic individually are **not** claimed as novel. The novelty lies in their combination around the evidence-grounding problem.

---

## 2. Base IEEE Paper

**Title:** "Explaining the Role of Evidence in Data-Driven Fact-Checking"

**What it does:** Evaluates whether SHAP and LIME can correctly identify which evidence pieces drove a fact-checking model's verdict. Tests on FEVEROUS, SciFact, FM2, AVeriTeC with RoBERTa, GFCE, LLaMA-3.1-70B, GPT-4.1 Mini, Claude 3.5 Sonnet.

**Key finding:** SHAP generally outperforms LIME at labeling evidence as Useful/Noise, but models often rely on internal priors rather than evidence.

---

## 3. Confirmed Research Gaps (from IEEE Paper)

| # | Gap | Operational Meaning |
|---|-----|---------------------|
| 1 | Attribution ≠ Causal Influence | SHAP/LIME may capture correlation, not whether evidence actually drives the decision |
| 2 | Binary Useful/Noise Classification | Rigid threshold-based binary labels; no continuous degrees of relevance/support |
| 3 | Conflicting Evidence Excluded | AVeriTeC "conflicting" label dropped; mixed-evidence scenarios underrepresented |
| 4 | Static Retrieval Pipelines | Fixed datasets only; no dynamic retrieval evaluation |
| 5 | Explanation Reliability Unquantified | No stability/uncertainty metrics for explanations |
| 6 | Baseline Prior Reliance Undetected | Models may ignore evidence and rely on internal priors without detection |

---

## 4. Implementation Modules (All 10 Required)

Every module below is part of the **full implementation scope**. None are removed because they lack a dedicated experiment.

### Module 1 — User Claim Input
- Natural language claim as system entry point.

### Module 2 — Hybrid RAG Retrieval
- **BM25** lexical retrieval.
- **Dense retrieval** via sentence-transformer embeddings + FAISS.
- **Fusion/reranking** to combine scores.
- Configurable retrieval mode and top-k.
- Cached embeddings.

### Module 3 — Top-k Evidence Selection
- Evidence passages with relevance scores.
- Source metadata (evidence ID, source type, retrieval score, rank).

### Module 4 — Claim-Evidence Verification Model
- Pretrained NLI model (DeBERTa or RoBERTa-based).
- Outputs: SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION + class probabilities.
- Clean interface allowing future verifier replacement.
- **Not** dependent on a 7B+ generative LLM.

### Module 5 — SHAP + LIME Evidence Attribution
- Evidence-level perturbation-based attribution.
- Each evidence piece treated as an interpretable unit.
- Configurable perturbation count (start low for dev, increase for experiments).
- Outputs: SHAP score, LIME score, ranking per evidence piece.

### Module 6 — Intervention-Based Evidence Influence
- For top-ranked evidence: remove evidence → re-run verifier → record ΔP and verdict change.
- Optionally test small evidence combinations.
- Does **not** enumerate all subsets.
- Does **not** claim mathematical proof of causality.
- Terminology: "intervention-based influence" / "counterfactual influence."

### Module 7 — Fuzzy Evidence Reasoning
- Inputs: semantic relevance, SHAP score, LIME score, intervention ΔP, contradiction signal, source reliability, stability.
- Outputs: evidence relevance degree, support degree, contradiction degree, evidence reliability, explanation confidence.
- Interpretable membership functions (triangular/trapezoidal) and IF-THEN rules.
- **Both** a fuzzy system AND a non-fuzzy baseline (simple hard/weighted aggregation) are implemented for Experiment 2.
- Example rules:
  - IF relevance HIGH AND intervention influence HIGH AND source reliability HIGH → evidence reliability VERY HIGH
  - IF relevance HIGH AND contradiction HIGH → evidence state CONFLICTING
  - IF SHAP influence HIGH BUT intervention influence LOW → attribution confidence LOW

### Module 8 — Conflict Detection
- Classifies evidence: supporting / refuting / neutral-contextual.
- Detects explicit disagreement between evidence pieces.
- Uses pretrained NLI for contradiction detection.
- Implemented feature, not a separate experiment.

### Module 9 — Explanation Stability / Reliability Analysis
- Controlled perturbations: evidence-order changes, irrelevant evidence insertion/removal, meaning-preserving variations.
- Measures consistency of evidence rankings/contributions (e.g., rank correlation).
- Produces stability score contributing to fuzzy reliability.
- Implemented feature, not a separate experiment.

### Module 10 — Final Reliability-Aware Explanation
- Aggregates all upstream signals.
- Output contains: verdict, confidence, retrieved evidence with rankings, SHAP/LIME influence, intervention influence, fuzzy scores, conflict status, stability/reliability, human-readable explanation.

---

## 5. Research Experiments

### 5A. Core Mandatory Experiments (2)

#### Experiment 1 — Attribution vs. Intervention

**Question:** Do evidence pieces identified as important by SHAP/LIME actually influence the verifier's decision when removed?

**Compares:** SHAP attribution, LIME attribution, intervention-based influence.

**Metrics:**
- Top-k evidence agreement (overlap between SHAP/LIME ranking and intervention ranking)
- Prediction probability change (ΔP) when top-attributed evidence is removed
- Verdict-flip rate (fraction of claims where removing top evidence changes verdict)
- Attribution–intervention correlation (Spearman/Kendall between SHAP/LIME scores and ΔP)

#### Experiment 2 — Fuzzy vs. Non-Fuzzy Reliability

**Question:** Does fuzzy evidence reasoning improve reliability/confidence assessment compared to a simpler non-fuzzy baseline?

**Compares:**
- Baseline: simple hard/weighted evidence scoring
- Proposed: fuzzy evidence reasoning

**Metrics:** To be finalized during Phase 7 implementation when the fuzzy system's outputs are concretely defined. Will include quantitative comparison of reliability/confidence scoring between the two approaches.

**Rule:** No improvement is claimed until experimental data demonstrates it.

### 5B. Optional Additional Experiments

Additional experiments may be added later with explicit user approval, provided they:
1. Are directly motivated by the implemented system or identified research gaps.
2. Reuse existing pipeline outputs (retrieval results, SHAP/LIME scores, intervention ΔP, fuzzy scores, etc.) rather than requiring major new architecture.
3. Do not substantially expand implementation scope.
4. Provide meaningful research value.
5. Fit within available development time and compute budget.
6. Do not interfere with the two core experiments.

The implementation architecture exposes reusable module outputs to support this extensibility.

---

## 6. Model Strategy

| Component | Model | Rationale |
|-----------|-------|-----------|
| Dense Retrieval Embeddings | `all-MiniLM-L6-v2` or `all-mpnet-base-v2` | Fast, well-tested, runs on CPU/consumer GPU |
| Claim-Evidence Verifier | `cross-encoder/nli-deberta-v3-base` or similar NLI model | Pretrained, no fine-tuning needed, practical compute |
| NLI for Conflict Detection | Same NLI model or `cross-encoder/nli-deberta-v3-large` | Reuse existing model where possible |
| SHAP/LIME | `shap`, `lime` libraries | Standard XAI libraries |
| Fuzzy Logic | `scikit-fuzzy` | Lightweight, well-documented |
| Optional LLM Verifier | User-approved only; not a hard dependency | — |

No model >1B parameters is a hard requirement.

---

## 7. Dataset Strategy

| Priority | Dataset | Purpose | Access |
|----------|---------|---------|--------|
| Primary | **FEVER** | Claims + Wikipedia evidence; large, well-studied | HuggingFace `datasets` |
| Secondary | **SciFact** | Scientific claims; smaller, domain-specific | HuggingFace `datasets` |
| Optional | AVeriTeC | Real-world claims with conflicting evidence | GitHub |

- Start with FEVER.
- Use a manageable subset (e.g., 500–1000 claims) for development; evaluation subset size decided during experiment phases.
- Cache all processed data.

---

## 8. Technology Assumptions

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| Retrieval - Lexical | `rank_bm25` |
| Retrieval - Dense | `sentence-transformers` + `faiss-cpu` |
| Verification | `transformers` (HuggingFace) |
| XAI | `shap`, `lime` |
| Fuzzy Logic | `scikit-fuzzy` |
| NLP Utilities | `nltk` or `spacy` (lightweight) |
| Perturbations | `nlpaug` or manual |
| Data | `datasets` (HuggingFace), `pandas` |
| Metrics/Stats | `scipy`, `scikit-learn` |
| Plotting | `matplotlib`, `seaborn` |
| UI | `streamlit` |
| Testing | `pytest` |
| Config | `pydantic` or YAML-based |

---

## 9. Evaluation Strategy

| What | How | When |
|------|-----|------|
| Module-level sanity | Unit tests + small examples after each phase | Phases 1–9 |
| End-to-end integration | Full pipeline test on sample claims | Phase 9 |
| Experiment 1 | Attribution vs. intervention on evaluation subset | Phase 10 |
| Experiment 2 | Fuzzy vs. non-fuzzy on evaluation subset | Phase 11 |
| Artifact export | CSV/JSON results, plots, tables, reproduction commands | Phase 12 |

---

## 10. Non-Goals

The following are explicitly **out of scope**:

- Training or fine-tuning foundation models
- Knowledge graphs / GraphRAG
- Multi-agent systems
- Computer vision / multimodal models
- Reinforcement learning
- Autonomous web agents
- Domain-shift evaluation (future work only)
- Additional research experiments beyond the 2 core without explicit approval
- Drafting the research paper (teammate's responsibility)
- Claims of mathematical causal proof
- Dependency on 7B+ LLMs

---

## 11. Detected Contradictions / Ambiguities

| # | Item | Source A | Source B | Resolution |
|---|------|---------|---------|------------|
| 1 | Verifier type | PPT Slide 5 says "VERIFICATION LLM" | Master prompt specifies NLI model | **Resolved:** Use pretrained NLI model (DeBERTa). LLM is optional extension. |
| 2 | Experiment count | Proposal PDF implies broad evaluation | Master prompt limits experiments | **Resolved:** 2 core mandatory experiments; additional experiments extensible with explicit approval. |
| 3 | "Causal" claim strength | Proposal PDF uses "causal" loosely | Master prompt restricts terminology | **Resolved:** Use "intervention-based influence" / "counterfactual influence." Title retains "Fuzzy-Causal." |

No blocking contradictions found.

---

## 12. Phased Roadmap

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| **0** | Audit + Specification | `PROJECT_SPECIFICATION.md` ← **current** |
| **1** | Project Foundation | Runnable skeleton + smoke test |
| **2** | Dataset / Evidence Corpus | Preprocessing pipeline + cached data |
| **3** | Hybrid RAG Retrieval | BM25 + dense + fusion → top-k evidence |
| **4** | Claim-Evidence Verification | Working NLI verifier |
| **5** | SHAP + LIME Attribution | Attribution module + example output |
| **6** | Intervention-Based Influence | Intervention module |
| **7** | Fuzzy Reasoning | Fuzzy system + non-fuzzy baseline |
| **8** | Conflict Detection + Stability | Conflict + stability modules |
| **9** | Full Integration + UI | End-to-end Streamlit app |
| **10** | Experiment 1 | Attribution vs. intervention results |
| **11** | Experiment 2 | Fuzzy vs. non-fuzzy results |
| **12** | Experiment Artifact Export | `EXPERIMENT_RESULTS.md` + data files |
| **13** | Final Technical Documentation | README + architecture docs |
| **14** | PPT Technical Review | PPT corrections (on request only) |

Each phase requires explicit user approval before the next begins.

---

## 13. Illustrative System Output (Pre-Implementation)

The following is **illustrative only** and must not be presented as actual results:

```
Claim:     "JWST was launched in 2021."
Verdict:   SUPPORTED
Confidence: 94%

Evidence:
  E1 — Supporting — high relevance — high intervention influence
  E2 — Contextual — low influence
  E3 — Noise — negligible influence

Fuzzy Evidence Reliability: High
Explanation Stability:      High
Conflict Level:             Low
Most Influential:           E1

Reason: Removing E1 changes the verifier's decision substantially.
```

---

*This specification is frozen. Changes require explicit user approval.*
