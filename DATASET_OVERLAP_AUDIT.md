# Dataset-Overlap & Retrieval Corpus Provenance Audit

## Executive Summary

This document presents a rigorous, read-only audit of the dataset relationships, evidence corpus construction, gold-evidence representation, and attribution agreement metrics for **Experiment 1 (Formal Evaluation)**.

The primary research question of Experiment 1 evaluates the within-context fidelity of post-hoc attribution (SHAP and LIME) compared to empirical leave-one-out intervention influence:
$$\Delta P = | P_{\text{baseline}} - P_{\text{intervened}} |$$

Because this evaluation sits downstream of retrieval, this audit establishes the exact relationship between the evaluated claims and the retrieval corpus, distinguishing between:
1. **Model-training data leakage** (none exists; models are frozen off-the-shelf checkpoints evaluated zero-shot),
2. **Gold-evidence availability in the candidate corpus** (100% of annotated gold passages exist in the closed pool),
3. **Top-3 retrieval coverage** (the proportion of annotated gold IDs successfully retrieved into the Top-3 context), and
4. **Within-context XAI fidelity** (the rank concordance and Top-1 agreement between attribution and intervention influence conditional on the retrieved evidence).

---

## 1. Dataset Relationships & Corpus Construction

### 1.1 Source Provenance and Sampling Pipeline

An inspection of `data/fever_loader.py` and `data/dataset_meta.json` reveals the exact construction pipeline:

1. **Upstream Source**: The project streamed records from the Hugging Face dataset [`copenlu/fever_gold_evidence`](https://huggingface.co/datasets/copenlu/fever_gold_evidence), using the split labeled `"train"`.
2. **Full Dataset Cache**: The first 5,000 streamed records were normalized to the internal `ClaimRecord` schema and saved locally as `data/fever_full.jsonl`.
3. **Stratified Dev Subset**: From these 5,000 records, a balanced development subset of 750 claims was sampled using a fixed random seed (`seed=42`), containing exactly 250 claims per label (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`). This was saved as `data/fever_dev.jsonl`.
4. **Corpus Extraction (`extract_evidence_corpus`)**: The retrieval corpus `data/evidence_corpus.json` was generated directly by extracting and pooling all unique evidence passages annotated within the 750 claims of `data/fever_dev.jsonl`.
5. **Formal Experiment Slice**: During the formal 500-claim evaluation (`experiments/experiment1_formal.py`), the pipeline loaded `data/fever_dev.jsonl` and evaluated the first 500 records (`formal_records = records[:cfg.max_claims]`, with `cfg.max_claims = 500`).

```
copenlu/fever_gold_evidence (Streamed train split)
           │
           ▼ (First 5,000 records)
   data/fever_full.jsonl (5,000 records)
           │
           ▼ (Stratified sample: 250 SUPPORTS, 250 REFUTES, 250 NEI; seed=42)
   data/fever_dev.jsonl (750 records)
      ├── Claims 1 to 500   ──────► Evaluated in Experiment 1 (500 claims)
      └── Claims 501 to 750 ──────► Retained in dev set (250 claims)
           │
           ▼ (extract_evidence_corpus over all 750 records)
   data/evidence_corpus.json (808 unique passages)
```

### 1.2 Evidence Deduplication Handling

Evidence sentences were deduplicated at two distinct stages:
1. **Intra-Claim Deduplication**: In `_normalize_record()`, evidence sentences within the same claim were tracked via a `seen_ids` set on `{doc_title}_{sentence_idx}` to prevent duplicate entries within a single claim record.
2. **Inter-Claim Deduplication**: In `extract_evidence_corpus()`, evidence passages across all 750 claims were stored in a dictionary keyed by `ev.evidence_id`. Multiple claims referencing the same Wikipedia sentence were merged into a single unique corpus passage.

### 1.3 Stable Identifier Overlap Check

Every claim record possesses a deterministic 32-character hexadecimal MD5 `claim_id`. Every evidence piece possesses a deterministic `evidence_id` formatted as:
$$\text{evidence\_id} = \{\text{doc\_title}\}\_\{\text{sentence\_idx}\}$$
Because these identifiers are stable and deterministic across all files, an exact 1:1 match was performed with zero reliance on heuristics or fuzzy text matching.

---

## 2. Gold-Evidence Representation in the Candidate Corpus

### 2.1 Candidate-Corpus Availability

An audit comparing the 500 evaluated claims against `data/evidence_corpus.json` establishes:

| Metric | Measured Value | Percentage | Context / Description |
| :--- | :---: | :---: | :--- |
| **Evaluated Claims ($N$)** | 500 | 100.0% | First 500 records of `data/fever_dev.jsonl` |
| **Claims with Non-Empty Gold Annotations** | **500** | **100.0%** | All 500 evaluated claims contain non-empty gold evidence |
| **Claims with Empty Gold Annotations** | **0** | **0.0%** | Zero evaluated claims possess empty evidence sets |
| **Distinct Gold Evidence IDs in Evaluated Claims** | **534** | — | Unique annotated evidence passages across the 500 claims |
| **Evaluated Gold Passages Present in Corpus** | **534** | **100.0%** | All 534 passages exist in `evidence_corpus.json` |
| **Corpus Passages from Remaining 250 Claims** | **274** | 33.91% | Distractor passages originating from claims 501–750 |
| **Total Evidence Corpus Size** | **808** | 100.0% | $534 \text{ (eval gold)} + 274 \text{ (distractors)} = 808$ |

All 534 annotated gold evidence passages for the evaluated claims exist inside the 808-passage candidate pool. This characterizes the experimental setup as a **gold-conditioned, closed candidate-pool evaluation**.

---

## 3. Top-3 Retrieval Coverage with Explicit Denominators

Although the candidate pool contains all 534 gold evidence passages, the hybrid retriever (BM25 + Dense) was constrained to select only the **Top-3** evidence pieces ($k=3$) from among the 808 candidates.

### 3.1 Mutually Exclusive Claim Partition

To prevent ambiguous subset interpretations (e.g., vacuum-truth errors where an empty set is treated as a subset), the 500 evaluated claims are partitioned into mutually exclusive, non-overlapping coverage categories:

| Coverage Category | Claim Count ($N$) | Percentage (%) | Denominator | Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Full Gold Coverage** | **409** | **81.80%** | 500 | All annotated gold-evidence IDs were retrieved in the Top-3 |
| **Partial Gold Coverage** | **21** | **4.20%** | 500 | At least 1, but not all, annotated gold-evidence IDs were retrieved in Top-3 |
| **Zero Gold Coverage** | **70** | **14.00%** | 500 | Annotated gold evidence exists, but zero gold pieces were retrieved in Top-3 |
| **Empty Gold Annotations** | **0** | **0.00%** | 500 | Claims lacking annotated gold evidence in the source dataset |
| **Sum (Mutually Exclusive Partition)** | **500** | **100.00%** | **500** | Exact reconciliation across all evaluated claims ($409 + 21 + 70 + 0 = 500$) |

### 3.2 Cumulative Retrieval Coverage

Combining the mutually exclusive categories yields the cumulative retrieval metric:
- **Claims with at least one annotated gold-evidence ID retrieved in Top-3**:
  $$\text{Full (409)} + \text{Partial (21)} = \mathbf{430 \text{ claims}} \quad (\mathbf{86.00\%} \text{ of 500 claims})$$

### 3.3 Evidence-Instance Retrieval Composition

At the individual evidence instance level across all 1,500 retrieved pieces ($500 \text{ claims} \times 3$):
- **Retrieved Pieces Matching Annotated Gold**: **443 pieces** (**29.53%**)
- **Retrieved Pieces as Distractors**: **1,057 pieces** (**70.47%**)

### 3.4 Downstream Impact: Gold vs. Distractor Evidence

The downstream claim verifier and attribution modules treated gold evidence versus distractor evidence with stark differences:

| Evidence Type | Instances ($N$) | Mean Influence ($\Delta P$) | Median Influence ($\Delta P$) | Mean $| \text{SHAP} |$ | Mean $| \text{LIME} |$ | Proportion Ranked #1 by Influence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annotated Gold Evidence** | 443 | **0.4114** | **0.0651** | **0.4825** | **0.1108** | **62.08%** (275 / 443) |
| **Distractor Evidence** | 1,057 | **0.1280** | **0.0012** | **0.2600** | **0.0435** | **21.38%** (226 / 1,057) |

When an annotated gold passage was retrieved, the verifier model relied on it heavily (mean $\Delta P = 0.4114$), and SHAP/LIME assigned it high magnitude ($0.4825$ vs. $0.2600$). Conversely, distractor passages exhibited negligible intervention influence (median $\Delta P = 0.0012$).

---

## 4. Verification of Direct SHAP–LIME Top-1 Disagreement

A dedicated pairwise verification was conducted comparing the Top-1 evidence ID selected by $| \text{SHAP} |$ against the Top-1 evidence ID selected by $| \text{LIME} |$ for each claim.

### 4.1 Direct Pairwise Agreement vs. Disagreement

- **Tie Status**: Exactly **0 claims** had ties for Rank 1 in SHAP, and exactly **0 claims** had ties for Rank 1 in LIME. Every claim produced a unique top-1 evidence piece for both methods.
- **Direct Top-1 Agreement**: **412 claims** (**82.40%** of 500 claims)
  Both $| \text{SHAP} |$ and $| \text{LIME} |$ assigned Rank 1 to the exact same evidence ID.
- **Direct Top-1 Disagreement**: **88 claims** (**17.60%** of 500 claims)
  $| \text{SHAP} |$ and $| \text{LIME} |$ assigned Rank 1 to different evidence IDs.

### 4.2 Reconciling Direct Disagreement (17.60%) with the 13.20% Figure

It is critical to distinguish **direct pairwise SHAP–LIME disagreement** from the **4-way contingency against empirical intervention influence**:

```
Deconstruction of the 500 Evaluated Claims:

1. Direct SHAP–LIME Agreement (412 claims, 82.40%):
   ├── Both Agree with Intervention Top-1:           354 claims (70.80%)
   └── Neither Agrees with Intervention Top-1:        58 claims (11.60%)  <- Shared blind spot

2. Direct SHAP–LIME Disagreement (88 claims, 17.60%):
   ├── LIME Matches Intervention, SHAP Fails:         66 claims (13.20%)  <- "LIME-Only Agreement"
   ├── SHAP Matches Intervention, LIME Fails:          7 claims  (1.40%)  <- "SHAP-Only Agreement"
   └── Neither Matches Intervention:                  15 claims  (3.00%)  <- 3-way divergent
```

> [!NOTE]
> **Resolution**: The **13.20%** figure ($66 / 500$ claims) represents the proportion of claims where **LIME alone agrees with the empirical intervention while SHAP does not**. 
> The **direct pairwise disagreement between SHAP and LIME** is **17.60%** ($88 / 500$ claims).

---

## 5. Scientific Framing & Evaluation Validity

### 5.1 No Model-Training Data Leakage

Conventional data leakage occurs when test data (or its ground truth) is inadvertently used to train or tune model parameters, producing artificially inflated benchmark accuracy.
In this project:
1. **Zero Fine-Tuning**: Both the verifier (`cross-encoder/nli-deberta-v3-base`) and the dense embedder (`all-MiniLM-L6-v2`) are frozen, pre-trained models downloaded directly from Hugging Face. Neither model was trained, fine-tuned, or adapted on this corpus or FEVER claims.
2. **Evaluation Objective**: Experiment 1 does not benchmark model accuracy or claim classification scores. It audits the **relative fidelity of post-hoc explanations (SHAP/LIME) against leave-one-out omissions**.

### 5.2 Tripartite Scientific Boundaries

To ensure complete scientific rigor in reporting, three operational stages must be explicitly separated:

1. **Candidate-Corpus Availability (Closed-Pool Pre-Seeding)**:
   The 808-passage corpus contains gold evidence for evaluated claims alongside 274 realistic distractors. This is a gold-conditioned candidate pool, not an open-domain web corpus.
2. **Top-3 Retrieval Coverage**:
   Within this closed candidate pool, the hybrid retriever successfully captured all annotated gold evidence in 81.80% of claims (409/500) and at least one gold piece in 86.00% of claims (430/500), while failing completely on 14.00% of claims (70/500).
3. **Within-Context XAI Fidelity**:
   Given whatever Top-3 evidence pieces are retrieved (whether gold or distractor), Experiment 1 evaluates whether post-hoc attribution reflects empirical model sensitivity under leave-one-out omission. Because explainer and intervener receive the **identical Top-3 evidence triplet**, this internal comparison is mathematically sound and uncompromised.

### 5.3 Unsupported Claims

- **Unsupported**: Claiming that this pipeline achieves an 86.0% recall on open-domain, full-Wikipedia FEVER retrieval. Searching through an 808-passage candidate pool is fundamentally different from searching 5.4 million Wikipedia articles.
- **Paper Framing**: The retrieval stage must be described as a **"dense candidate ranking setting over a curated 808-passage pool"** rather than open-domain document discovery.
