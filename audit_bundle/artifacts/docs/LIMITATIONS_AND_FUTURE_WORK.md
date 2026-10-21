# Limitations, Threats to Validity & Future Research Directions

**Project:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Phase:** 13 — Final Research Documentation  
**Status:** Verified & Complete  

---

## 1. Threats to Validity & Methodological Boundaries

To ensure scientific integrity and prevent overclaiming, this section documents the critical boundaries and threats to validity identified across the research project.

### 1.1 Retrieval Candidate Pool Constraints
- **Gold-Seeded Closed Corpus**: The retrieval candidate pool consists of 808 Wikipedia passages extracted directly from the gold annotations of the 750 FEVER dev claims.
- **Threat**: The high claim-level retrieval coverage (86.00% of claims retrieved $\ge 1$ gold evidence in the Top-3) is an artifact of this closed pool. In an unconstrained open-domain retrieval setting with millions of Wikipedia articles, retrieval recall would be substantially lower. The closed corpus was deliberately chosen to isolate attribution and reasoning from catastrophic retrieval failure, but its findings cannot be generalized directly to open-web search.

### 1.2 Context Window Constraints ($k = 3$)
- **Context Size**: The pipeline evaluates exactly $k = 3$ retrieved passages per claim.
- **Threat**: In complex real-world fact-checking scenarios, verification frequently requires synthesizing 5 to 10 distinct passages across diverse sources. Restricting context to $k = 3$ limits the combinatorial space of evidence interactions and makes rank correlation metrics discrete.

### 1.3 Nature of FEVER Gold Annotations
- **Annotation Scope**: FEVER gold annotations represent human annotators' *minimal sufficient sets* of sentences required to justify a verdict.
- **Threat**: Gold annotations do not represent an objective, comprehensive ground truth of multi-attribute evidence reliability or source credibility. Distractor passages (unannotated retrieved sentences) are frequently factual, topically relevant Wikipedia statements that were simply non-minimal. Thus, evaluating "reliability" against gold annotation status treats human relevance selection as an extrinsic proxy for reliability.

### 1.4 Leave-One-Out (LOO) Intervention Limits
- **Intervention Protocol**: The causal intervention engine measures the effect of single-passage deletion ($E \setminus \{e_i\}$).
- **Threat**: Single-evidence removal cannot capture higher-order non-linear synergies or redundancies between multiple evidence pieces. For example, if two retrieved passages contain identical supporting facts, removing either one in isolation may produce $\Delta P \approx 0$, masking their collective necessity.

### 1.5 Model Sensitivity vs. Real-World Causality
- **Verifier Sensitivity**: The intervention influence metric $\Delta P_i = |P(\hat{y} \mid E) - P(\hat{y} \mid E_{-i})|$ measures the empirical sensitivity of the fine-tuned DeBERTa cross-encoder to input feature removal.
- **Threat**: This metric reflects **internal model decision dynamics**, not real-world factual causality. A high $\Delta P$ proves that the verifier's classification head relies on that passage; it does not prove that the real-world truth of the claim causally depends on that document.

### 1.6 Stability Metric Scope
- **Perturbation Model**: The stability analyzer specifically evaluates evidence-order permutation (reversing the presentation order of retrieved evidence passages $E = [e_1, e_2, e_3] \to [e_3, e_2, e_1]$) and measures attribution rank preservation via normalized Spearman correlation: $s_{\text{stab}} = (\rho + 1)/2$.
- **Threat**: While this directly quantifies cross-encoder positional invariance and sensitivity to document sequence, it does not assess lexical masking, syntactic paraphrasing, or token-level noise within individual passages, nor does it explore the full factorial permutation set ($k!$) beyond order reversal.

### 1.7 Exploratory vs. Confirmatory Subgroup Testing
- **Subgroup Findings**:
  - The pre-registered membership-based attribution-gap subgroups ($S_{\text{gap}}^{\text{std}}, S_{\text{gap}}^{\text{ext}}$) were completely empty ($N = 0$).
  - The rank-divergence subgroup ($S_{\text{gap}}^{\text{rank}}$, $N = 87$) was evaluated as an **exploratory descriptive diagnostic only**.
- **Boundary**: No confirmatory statistical claims regarding attribution-gap suppression can be made on this dataset.

---

## 2. Promising Directions for Future Work

### 2.1 Scaling to Unconstrained Open-Domain Corpora
Future research should evaluate the fuzzy-causal reasoning engine over large-scale open-domain corpora (e.g., KILT benchmark, full Wikipedia dump with 5.4M articles). This will test how the system behaves under extreme retrieval noise, low semantic similarity, and widespread distractor presence.

### 2.2 Multi-Hop and Multi-Document Fact-Checking
Extending the pipeline to multi-hop datasets (such as **HoVer** or **FEVEROUS**) where individual claims require chain-of-reasoning across multiple independent documents. In these settings, multi-evidence subset ablation ($k \ge 5$) will be critical to reveal combinatorial synergies.

### 2.3 Integration of Objective Source Credibility Metadata
The current framework held source reliability constant at $s = 0.75$ due to Wikipedia's uniform origin. Future implementations should incorporate real-world source metadata, such as:
- Publisher domain authority (e.g., NewsGuard, Media Bias/Fact Check).
- Cryptographic provenance (C2PA content credentials).
- Temporal freshness and publication lag relative to the claim timestamp.

### 2.4 Beyond Centroid Defuzzification
Investigate alternative defuzzification and fuzzy aggregation mechanisms (such as Sugeno fuzzy systems or calibrated non-linear squashing functions) to mitigate the Average Precision dampening observed in Experiment 2 while preserving the superior AUROC ranking discrimination.
