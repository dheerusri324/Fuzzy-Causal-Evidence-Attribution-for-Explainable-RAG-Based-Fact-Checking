# Experiment 1 — Sanity Evaluation

## Research Question
When SHAP/LIME identifies an evidence piece as influential, does removing that same evidence piece produce a measurable change in the verifier’s decision?

## Frozen Configuration
```json
{
    "dataset": "FEVER",
    "dataset_split": "dev",
    "number_of_claims_evaluated": 10,
    "retrieval_method": "hybrid",
    "top_k": 3,
    "embedding_model": "all-MiniLM-L6-v2",
    "verifier_model": "cross-encoder/nli-deberta-v3-base",
    "shap_sample_count": 25,
    "lime_sample_count": 25,
    "maximum_intervention_evidence": 5,
    "intervention_method": "Leave-One-Out (LOO)",
    "run_timestamp": "2026-09-25T13:02:36.980408",
    "python_version": "3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]"
}
```

## Dataset
- dataset: FEVER
- split: dev
- claim count: 10 (Sanity)
- retrieval settings: hybrid, Top-3

## Models
- embedding model: all-MiniLM-L6-v2
- verifier model: cross-encoder/nli-deberta-v3-base
- SHAP settings: 25 samples
- LIME settings: 25 samples

## Methodological Definitions
### Intervention Definition
```python
intervention_influence = abs(baseline_target_probability - intervened_target_probability)
```
### Attribution Comparison
Signed attribution is retained for raw analysis, while **absolute attribution magnitude** (`abs(SHAP)`) is used for evidence-importance ranking and correlation against the non-negative intervention influence.
### Correlation Method
Primary correlation is computed **per claim** using Spearman rank correlation, then aggregated (mean/median) across valid claims.
### Tie Handling
Ranking ties are assigned the minimum rank of the group. Undefined correlations caused by constant/tied vectors (where variance is 0) are explicitly excluded from the correlation aggregate and reported separately.
### Top-K Agreement
Top-1 Agreement: Calculates the percentage of claims where the top-ranked evidence by `abs(SHAP)` (or `abs(LIME)`) matches the top-ranked evidence by `intervention_influence` (resolving ties via set intersection).

## Data Integrity Checks
- **evidence-set consistency**: Verified. The exact same `RetrievedEvidence` instances are passed to Explainer and Intervener.
- **evidence-ID consistency**: Verified. All metrics are mapped back via strict `evidence_id` matching.
- **ranking alignment**: Verified. Ranks are derived via `scipy.stats.rankdata` (`method='min'`) per claim.
- **probability validity**: Verified. All probabilities are [0, 1].
- **intervention calculation validity**: Verified. ΔP strictly follows absolute difference.

## Aggregate Sanity Statistics
> **SANITY-EVALUATION STATISTICS — NOT FINAL RESEARCH RESULTS**

- Number of claims evaluated: 10
- Number of evidence pieces (interventions): 30
- Verdict-flip rate: 6 / 30 (20.0%)

### Top-1 Agreement (Evidence Importance)
- abs(SHAP) Top-1 Agreement: 70.0%
- abs(LIME) Top-1 Agreement: 90.0%

### Per-Claim Spearman Correlation (abs(Attribution) vs Influence)
- Valid SHAP correlations: 10 (Undefined due to ties/zeros: 0)
- Mean SHAP rho: 0.2000  |  Median SHAP rho: 0.5000
- Valid LIME correlations: 10 (Undefined due to ties/zeros: 0)
- Mean LIME rho: 0.5500  |  Median LIME rho: 0.5000

## Sample Claim Results
### Claim: Lincoln Motor Company was named by Henry M. Leland.
| Evidence ID | SHAP Rank | LIME Rank | Infl Rank | SHAP (raw) | abs(SHAP) | ΔP | Flipped? |
|---|---|---|---|---|---|---|---|
| Lincoln_Motor_Company_1 | 1 | 1 | 1 | 0.9893 | 0.9893 | 0.9966 | True |
| Henry_M._Leland_0 | 3 | 2 | 3 | 0.0009 | 0.0009 | 0.0010 | False |
| Ford_Motor_Company_0 | 2 | 3 | 2 | 0.0066 | 0.0066 | 0.0127 | False |

### Claim: Game of Thrones (season 3) is a movie.
| Evidence ID | SHAP Rank | LIME Rank | Infl Rank | SHAP (raw) | abs(SHAP) | ΔP | Flipped? |
|---|---|---|---|---|---|---|---|
| A_Game_of_Thrones_0 | 2 | 2 | 2 | 0.3333 | 0.3333 | 0.0006 | False |
| Game_of_Thrones_-LRB-season_1-RRB-_0 | 1 | 1 | 3 | 0.3333 | 0.3333 | 0.0004 | False |
| Stannis_Baratheon_0 | 3 | 3 | 1 | 0.3326 | 0.3326 | 0.0006 | False |

## Next Step
Awaiting explicit approval to run the formal Experiment 1 evaluation.