# System Architecture & Technical Specifications

**Project:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Phase:** 13 — Final Research Documentation  
**Status:** Verified & Complete  

---

## 1. High-Level Architectural Overview

The framework addresses a critical limitation in current Retrieval-Augmented Generation (RAG) and automated fact-checking pipelines: **feature attribution methods (e.g., SHAP, LIME) identify which input tokens or evidence sentences the verifier attended to, but do not directly measure whether removing that evidence changes the factual verdict.** Furthermore, raw attribution ignores essential epistemic dimensions such as **source reliability**, **mutual evidence contradiction**, and **explanation stability**.

To resolve this, the system implements an end-to-end modular pipeline consisting of eight primary components:

```
[User Claim] 
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. Hybrid RAG Retriever                 │  (BM25 Keyword + Dense Sentence-Transformers)
│    Candidate Pool: 808 Passages         │
└────────────────────┬────────────────────┘
                     │ Top-k Retrieved Passages (k=3)
                     ▼
┌─────────────────────────────────────────┐
│ 2. Claim-Evidence Cross-Encoder         │  (DeBERTa-v3-base NLI: SUPPORTS / REFUTES / NEI)
│    Joint Token Serialization            │
└────────────────────┬────────────────────┘
                     │ Predicted Verdict & Class Probabilities P(y|E)
                     ├─────────────────────────────────────────┐
                     ▼                                         ▼
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ 3. Post-Hoc Feature Attribution         │   │ 4. Counterfactual Intervention Engine   │
│    SHAP (Partition) & LIME (Surrogate)  │   │    Leave-One-Out (LOO) Evidence Deletion│
└────────────────────┬────────────────────┘   └────────────────────┬────────────────────┘
                     │ Feature Attribution Strengths               │ Influence ΔP & Verdict Flips
                     └────────────────────┬────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ 5. Mutual Conflict Detector             │   │ 6. Explanation Stability Analyzer       │
│    Pairwise Cross-Encoder Contradiction │   │    Evidence-Order Permutation (Spearman)│
└────────────────────┬────────────────────┘   └────────────────────┬────────────────────┘
                     │ Contradiction Degree (c_deg)            │ Stability Score (s_stab)
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 7. Fuzzy Evidence Reasoning Engine vs. Linear Baseline                        │
│    - Mamdani System (11 Rules across 5 Outputs, Centroid Defuzzification)     │
│    - Linear Weighted Baseline (Additive Stability, Multiplicative Penalty)    │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ Graded Reliability & Confidence
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 8. User Interface & Explainability Presentation Layer                         │
│    Interactive Streamlit UI (app/main.py) with Ranked Verifiable Attribution   │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Specifications & Data Flow

### 2.1 Component 1: Hybrid RAG Retriever ([`retrieval/retriever.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/retrieval/retriever.py))
- **Responsibilities**: Ingests claims and retrieves the top-$k$ ($k=3$) most relevant evidence passages from the local evidence corpus.
- **Sparse Inverted Index**: BM25 (`rank-bm25`) over tokenized document passages.
- **Dense Vector Embedding**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic representations.
- **Fusion Method**: Hybrid score combination using min-max score normalization and weighted interpolation:
  $$S_{\text{hybrid}}(c, e) = \alpha_{\text{ret}} \cdot \tilde{S}_{\text{dense}}(c, e) + (1 - \alpha_{\text{ret}}) \cdot \tilde{S}_{\text{bm25}}(c, e)$$
  where $\alpha_{\text{ret}} = 0.5$ by default.
- **Output**: Ranked list of `Evidence` objects containing `evidence_id`, `text`, `doc_title`, `retrieval_score`, and `retrieval_rank`.

### 2.2 Component 2: Claim-Evidence Cross-Encoder ([`verification/verifier.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/verification/verifier.py))
- **Responsibilities**: Performs joint natural language inference (NLI) between the claim and the concatenated Top-$k$ evidence set.
- **Base Model**: `cross-encoder/nli-deberta-v3-base` fine-tuned on MNLI/SNLI.
- **Classification Head**: 3-class softmax output over:
  1. `SUPPORTS` (entailment)
  2. `REFUTES` (contradiction)
  3. `NOT_ENOUGH_INFO` (neutral)
- **Serialization Format**: `[CLS] Claim: {claim} [SEP] Evidence: [E1] {text_1} [E2] {text_2} [E3] {text_3} [SEP]`.
- **Output**: `VerificationResult` containing the predicted discrete `verdict` and probability distribution $P(\hat{y} \mid E)$.

### 2.3 Component 3: Post-Hoc Feature Attribution ([`attribution/explainer.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/attribution/explainer.py))
- **Responsibilities**: Quantifies token- and passage-level feature attribution towards the predicted verifier verdict.
- **SHAP Engine** ([`attribution/shap_attribution.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/attribution/shap_attribution.py)): Uses partition-based Shapley value estimation targeting the predicted class logit.
- **LIME Engine** ([`attribution/lime_attribution.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/attribution/lime_attribution.py)): Uses localized perturbation with cosine-distance kernel weights and sparse linear surrogate models.
- **Passage Aggregation**: Sum of attribution values across tokens belonging to passage $e_i$, normalized to $[0, 1]$ via min-max scaling across the Top-$k$ set.
- **Output**: `EvidenceAttributionResult` with per-piece `shap_target` and `lime_target` scores.

### 2.4 Component 4: Counterfactual Intervention Engine ([`intervention/intervener.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/intervention/intervener.py))
- **Responsibilities**: Measures the direct empirical sensitivity of the verifier when evidence pieces are counterfactually perturbed.
- **Intervention Protocol**: Leave-One-Out (LOO) evidence ablation:
  $$E_{-i} = E \setminus \{e_i\}$$
- **Target Metrics**:
  1. **Probability Influence ($\Delta P$)**:
     $$\Delta P_i = |P(\hat{y} \mid E) - P(\hat{y} \mid E_{-i})|$$
  2. **Discrete Verdict Flip**:
     $$\mathbb{I}(\text{verdict}(E) \neq \text{verdict}(E_{-i}))$$
- **Output**: `InterventionResult` detailing post-intervention probability, influence magnitude $\Delta P_i$, and boolean verdict flip indicators.

### 2.5 Component 5: Mutual Conflict Detector ([`conflict/detector.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/conflict/detector.py))
- **Responsibilities**: Evaluates whether retrieved evidence passages contain internally contradictory assertions or directly oppose the claim.
- **Contradiction Scoring**: Computes pairwise NLI contradiction logits between retrieved evidence passages, as well as individual evidence-versus-claim contradiction probabilities.
- **Aggregation**: Derives a continuous contradiction degree $c_{\text{deg}} \in [0, 1]$ and assigns evidence roles (`SUPPORTING`, `REFUTING`, `NEUTRAL`).
- **Output**: `ConflictResult` containing `conflict_score` and evidence role mappings.

### 2.6 Component 6: Explanation Stability Analyzer ([`stability/analyzer.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/stability/analyzer.py))
- **Responsibilities**: Measures the robustness of post-hoc explanations against evidence-order permutation.
- **Perturbation Mechanism**: Evidence-order permutation: reverses the sequence of retrieved evidence passages presented to the explainer ($E = [e_1, e_2, e_3] \to [e_3, e_2, e_1]$).
- **Compared Attributions**: SHAP target class attribution scores (`shap_target`) aligned strictly by `evidence_id`.
- **Stability Metric**: Spearman rank correlation ($\rho$) between baseline and perturbed attribution vectors, mapped into $[0, 1]$ via linear min-max scaling:
  $$s_{\text{stab}} = \max\left(0.0, \min\left(1.0, \frac{\rho + 1.0}{2.0}\right)\right)$$
  (defaults to $1.0$ if standard deviation is zero).
- **Output**: `StabilityResult` containing continuous `stability_score` $s_{\text{stab}} \in [0, 1]$.

### 2.7 Component 7: Fuzzy Evidence Reasoning vs. Baseline ([`fuzzy/engine.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/fuzzy/engine.py), [`fuzzy/baseline.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/fuzzy/baseline.py))
- **Fuzzy Engine**:
  - Implements a Mamdani fuzzy inference system with 6 continuous input dimensions:
    1. Relevance ($r \in [0, 1]$)
    2. Attribution Strength ($a \in [0, 1]$, derived from $\frac{1}{2}(\text{SHAP} + \text{LIME})$)
    3. Intervention Influence ($i \in [0, 1]$)
    4. Source Reliability ($s \in [0, 1]$, fixed at 0.75 for Wikipedia)
    5. Contradiction Degree ($c_{\text{deg}} \in [0, 1]$)
    6. Stability Score ($s_{\text{stab}} \in [0, 1]$)
  - Membership functions: Triangular and trapezoidal sets (`LOW`, `MEDIUM`, `HIGH`).
  - Rule base: **11 fuzzy IF-THEN rules** across **5 output variables**:
    - **8 primary domain synergy rules** (`rule1` through `rule8`): capture non-linear interactions across inputs.
    - **3 baseline passthrough rules** (`rule9` through `rule11`): guarantee active inference coverage across all output variables.
  - Consequent Output Variables:
    1. `evidence_reliability` (Primary score, targeted by Rules 1, 3, 8)
    2. `explanation_confidence` (Diagnostic score, targeted by Rules 2, 6, 7)
    3. `evidence_relevance` (Targeted by Rules 9, 10)
    4. `support_degree` (Targeted by Rule 5)
    5. `contradiction_output` (Targeted by Rules 4, 11)
  - Defuzzification: Centroid (center-of-gravity) integration over $[0, 1]$.
- **Linear Weighted Baseline**:
  - Two-stage scoring implementation in [`fuzzy/baseline.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/fuzzy/baseline.py):
    1. **Weighted Positive Score ($S_{\text{raw}}$)**:
       $$S_{\text{raw}} = 0.30 \cdot r + 0.25 \cdot a + 0.25 \cdot i + 0.10 \cdot s + 0.10 \cdot s_{\text{stab}}$$
       bounded within $[0, 1]$. Stability ($s_{\text{stab}}$) is incorporated directly as an additive weighted component ($w = 0.10$).
    2. **Contradiction-Adjusted Score ($S_{\text{adj}}$)**:
       $$S_{\text{adj}} = S_{\text{raw}} \cdot (1 - c_{\text{deg}})$$
       Penalized multiplicatively by the contradiction degree $(1 - c_{\text{deg}})$. It is **not** multiplied by $s_{\text{stab}}$ a second time.

### 2.8 Component 8: End-to-End Orchestrator & UI ([`app/pipeline.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/app/pipeline.py), [`app/main.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/app/main.py))
- **Pipeline Orchestrator**: Executes the sequential flow, caches intermediate models, and formats results.
- **Streamlit Interface**: Interactive web dashboard allowing user claim entry, live retrieval visualization, interactive attribution breakdown, and reliability-weighted verdict presentation.

---

## 3. Configuration Management ([`config/settings.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/config/settings.py))

All pipeline parameters are configured via `Config` dataclass and YAML files:
- `top_k`: Number of retrieved evidence passages per claim (default: 3).
- `retrieval_mode`: Hybrid, Dense-only, or BM25-only (default: `hybrid`).
- `verifier_model`: HuggingFace model identifier (default: `cross-encoder/nli-deberta-v3-base`).
- `dense_model`: SentenceTransformers model identifier (default: `sentence-transformers/all-MiniLM-L6-v2`).
- `device`: Compute device (`cpu` or `cuda`).
- `random_seed`: Master seed for reproducibility (default: 42).
