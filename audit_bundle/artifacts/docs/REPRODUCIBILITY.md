# Reproducibility Guide & Verification Instructions

**Project:** Fuzzy-Causal Evidence Attribution for Explainable RAG-Based Fact-Checking  
**Phase:** 13 — Final Research Documentation  
**Status:** Verified & Complete  

---

## 1. Overview of Reproducibility Levels

To maximize transparency while respecting computational constraints, the research framework provides two distinct levels of reproducibility:

1. **Level 1: Independent Tabular Reproduction (Zero Model Execution, < 30 seconds)**:
   Verifies all statistical point estimates, bootstrap inference distributions, confidence intervals, Top-1 hit rates, ablations, and condition isolation directly from the saved frozen artifacts.
2. **Level 2: Full End-to-End Pipeline Execution (Full Neural Inference, Several Hours)**:
   Executes the raw neural models (DeBERTa cross-encoder, MiniLM dense embedder, BM25, SHAP, LIME, and Mamdani fuzzy engine) from scratch.

---

## 2. Level 1: Reproducing Results from Frozen Outputs

The project includes an automated, independent verification script: [`scratch/verify_exp2.py`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/scratch/verify_exp2.py).

### How to Run:
```bash
# Activate your python virtual environment
python scratch/verify_exp2.py
```

### What Level 1 Verifies:
1. **Primary Metrics**: Recomputes AUROC, Average Precision, and delta values for Fuzzy and Baseline against ground truth from [`data/fever_dev.jsonl`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/data/fever_dev.jsonl).
2. **Bootstrap Inference**: Re-runs the exact $B = 1,000$ cluster bootstrap with NumPy PCG64 seed 42 to verify empirical $p$-values and 95%/97.5% confidence intervals.
3. **Secondary Metrics**: Validates Top-1 hit rates (Metrics A, B, C), static ablation scores, and the $20.80\%$ ($312/1,500$) baseline zero-saturation count.
4. **Condition Isolation**: Checks that raw attribution and intervention columns are 100% numerically identical to Experiment 1, ensuring zero data leakage.
5. **Output**: Writes [`outputs/phase11_formal/independent_validation.json`](file:///c:/Users/dheer/OneDrive/Documents/Fuzzy-Causal%20Evidence%20Attribution%20for%20Explainable%20RAG-Based%20Fact-Checking/outputs/phase11_formal/independent_validation.json).

---

## 3. Level 2: End-to-End Execution from Source Code

### 3.1 Environment Setup
```bash
# 1. Create and activate a Python 3.10+ virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux / macOS

# 2. Install pinned dependencies
pip install -r requirements.txt
```

### 3.2 Key Dependencies & Pinned Versions
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `sentence-transformers >= 2.2.2`
- `shap >= 0.42.0`
- `lime >= 0.2.0`
- `scikit-fuzzy >= 0.4.2`
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `rank-bm25 >= 0.2.2`
- `streamlit >= 1.25.0`

### 3.3 Running Formal Experiment 1
```bash
python experiments/experiment1_formal.py
```
- **Inputs**: `data/fever_dev.jsonl` (first 500 claims), `data/evidence_corpus.json`.
- **Outputs**: `outputs/phase10_formal/results.csv`, `outputs/phase10_formal/config.json`.
- **Expected Runtime**: 2–5 hours on modern GPU (dominated by token-level SHAP partition evaluation and LIME local surrogates across 1,500 instances).

### 3.4 Running Formal Experiment 2
```bash
python experiments/experiment2_formal.py
```
- **Inputs**: Reuses frozen attribution and intervention inputs from `outputs/phase10_formal/results.csv`.
- **Dynamic Signals**: Uses cached conflict and stability scores from `outputs/phase11_formal/.signals_cache.json` if present (execution takes ~2 minutes). If the cache is deleted, it re-runs DeBERTa pairwise conflict detection and stability analysis (~30–60 minutes on GPU).
- **Outputs**: `outputs/phase11_formal/results.csv`, `summary.json`, `config.json`, and diagnostic plots.

### 3.5 Launching the Interactive Web Application
```bash
streamlit run app/main.py
```
- Opens the browser UI at `http://localhost:8501`.
- Allows real-time claim entry, retrieval, verification, and reliability-weighted explainability presentation.

---

## 4. Known Environmental Nuances & Reproducibility Limitations

1. **Floating-Point Precision Nuance**:
   In `experiment2_formal.py`, summary metrics were evaluated in memory as 64-bit numpy floats before disk serialization. When exported to CSV via pandas `to_csv()` and re-read, a minor truncation of $\approx 2.14 \times 10^{-6}$ occurs in the fuzzy metrics. This is well below the statistical tolerance of $10^{-4}$ and does not affect rounded four-decimal-place values ($0.7630$ and $0.4884$).
2. **GPU Architecture Differences**:
   Cross-encoder logits from `cross-encoder/nli-deberta-v3-base` may exhibit minor floating-point differences ($\sim 10^{-5}$) across different CUDA architectures (e.g., Ampere vs. Ada Lovelace vs. Hopper) or between CPU and CUDA execution due to non-associative matrix multiplication kernels.
3. **Random Seeds**:
   - Master Python / PyTorch / NumPy seed: `42`.
   - Bootstrap generator: `numpy.random.Generator(PCG64(42))`.
