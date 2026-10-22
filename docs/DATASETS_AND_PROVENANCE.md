# Dataset Provenance, Construction & Evaluation Corpus

**Project:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Phase:** 13 — Final Research Documentation  
**Status:** Verified & Complete  

---

## 1. Overview of Data Sources

The empirical evaluation of both Experiment 1 (attribution-intervention alignment) and Experiment 2 (fuzzy vs. linear baseline scoring) relies on the **FEVER (Fact Extraction and VERification)** benchmark ([Thorne et al., 2018](https://aclanthology.org/N18-1074/)).

Two primary data files reside in `data/`:
1. [`data/fever_dev.jsonl`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/data/fever_dev.jsonl): Contains 750 structured FEVER claims with corresponding ground-truth verdicts (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) and human-annotated gold-evidence references.
2. [`data/evidence_corpus.json`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/data/evidence_corpus.json): Contains the 808-passage retrieval candidate pool used by the hybrid retriever.

---

## 2. Construction of the 808-Passage Retrieval Corpus

To establish complete experimental transparency, the provenance of `data/evidence_corpus.json` was audited during Phase 12 (documented in `DATASET_OVERLAP_AUDIT.md`):

### 2.1 Extraction and Deduplication
- **Source Split**: All 750 claims in `data/fever_dev.jsonl` were parsed for annotated gold-evidence tuples `(doc_title, sentence_id, text)`.
- **Deduplication**: Claims that share identical Wikipedia evidence sentences were unified by matching `(doc_title, sentence_id)` pairs.
- **Corpus Size**: This deduplication produced exactly **808 unique evidence passages**.
- **Storage Schema**: Each passage is stored as a JSON object with:
  - `evidence_id`: String formatted as `{doc_title}_{sentence_id}`.
  - `doc_title`: Wikipedia article title.
  - `sentence_id`: Integer sentence index within the article.
  - `text`: Sentence string content.

### 2.2 Seeded Nature of the Corpus
For the **500 claims** evaluated in Formal Experiments 1 and 2:
- Exactly **534 gold evidence passages** are associated with these 500 claims.
- **100.0% of these 534 gold passages (534 / 534) are contained in the 808-passage corpus.**
- The remaining 274 passages in the corpus originated from the gold evidence of the 250 non-evaluated claims (claims 501–750) or shared Wikipedia pages.

---

## 3. Retrieval Performance in the Closed Pool

During formal evaluation, the hybrid retriever (BM25 + `sentence-transformers/all-MiniLM-L6-v2`) selected $k=3$ passages per claim across all 500 claims ($1,500$ total evaluations):

| Metric | Measured Value | Percentage / Context |
| :--- | :---: | :---: |
| Total Evaluated Claims | 500 | 100.0% |
| Total Evaluated Evidence Instances | 1,500 | Exactly $3$ per claim |
| Claims with $\ge 1$ Positive Proxy Passage in Top-3 (Legacy) | 430 | **86.00%** (430 / 500) |
| Verifiable Claims with $\ge 1$ Genuine Gold Evidence in Top-3 | 283 | **84.73%** (283 / 334) |
| Total Retrieved Passages Matching Legacy Positive Proxies | 443 | **29.53%** ($443 / 1,500$; 296 genuine gold + 147 NEI candidate proxies) |
| Total Retrieved Distractor Passages (Legacy Non-proxy) | 1,057 | **70.47%** ($1,057 / 1,500$) |

### Critical Scientific Nuance:
> [!WARNING]
> **The 86.00% retrieval coverage is conditioned on a closed, gold-seeded pool of 808 passages.**  
> It must **never** be cited or reported as open-domain Wikipedia retrieval performance. In a full open-domain Wikipedia setting (containing over 5 million articles and tens of millions of sentences), Top-3 retrieval recall without pre-filtering would be substantially lower. The closed pool serves to isolate the attribution and reasoning mechanisms from catastrophic retrieval failure.

---

## 4. Epistemic Nature of FEVER Gold Annotations

When evaluating evidence reliability (as in Experiment 2), ground-truth labels are defined by whether an evidence piece is an annotated FEVER gold sentence ($y_i \in \{0, 1\}$).

The research framework explicitly acknowledges the limitations of this proxy:
1. **Minimal Sufficient Sets**: FEVER annotators were instructed to identify the minimal set of sentences sufficient to justify a verdict. Other relevant, factual, and supportive sentences in the article were often omitted.
2. **Relevance vs. Reliability**: An unannotated retrieved passage ($y_i = 0$, distractor) is **not necessarily false or irrelevant**. It may be a factually true Wikipedia statement that was simply redundant or non-minimal.
3. **Absence of Independent Credibility Labels**: FEVER does not contain real-world source reliability or credibility ratings (e.g., domain authority, journalistic rating, bias metrics). In the experiments, source reliability is held constant ($s = 0.75$) to reflect uniform Wikipedia provenance.
