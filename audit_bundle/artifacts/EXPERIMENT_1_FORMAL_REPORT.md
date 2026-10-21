# Experiment 1 — Formal Evaluation

## 1. Research Question
Compare evidence importance from SHAP and LIME against intervention-based evidence influence within each claim. Does removing highly-attributed evidence produce a measurable change in the verifier’s decision?

## 2. Dataset and Sample Size
- Dataset: FEVER
- Split: dev
- Sample Size (Claims): 500
- Total Interventions (Evidence pieces): 1500

## 3. Frozen Configuration
```json
{
    "dataset": "FEVER",
    "dataset_split": "dev",
    "number_of_claims_evaluated": 500,
    "retrieval_method": "hybrid",
    "top_k": 3,
    "embedding_model": "all-MiniLM-L6-v2",
    "verifier_model": "cross-encoder/nli-deberta-v3-base",
    "shap_sample_count": 25,
    "lime_sample_count": 25,
    "maximum_intervention_evidence": 5,
    "intervention_method": "Leave-One-Out (LOO)",
    "run_timestamp": "2026-09-25T13:11:38.479284",
    "python_version": "3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]"
}
```

## 4. Methodology
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

## 5. Correlation Results
### Per-Claim Spearman Correlation (abs(Attribution) vs Influence)
- **Valid SHAP correlations**: 500
- **Mean SHAP rho**: 0.4517
- **Median SHAP rho**: 0.5000

- **Valid LIME correlations**: 500
- **Mean LIME rho**: 0.5877
- **Median LIME rho**: 0.5000

## 6. Top-1 Agreement Results
- **abs(SHAP) Top-1 Agreement**: 72.2%
- **abs(LIME) Top-1 Agreement**: 84.0%

## 7. Verdict-Flip Statistics
- **Total Verdict Flips**: 322
- **Verdict-Flip Rate**: 21.5% (322/1500)

## 8. Edge Cases / Undefined Correlations
- **Undefined SHAP correlations (ties/zeros)**: 0
- **Undefined LIME correlations (ties/zeros)**: 0

## 9. Limitations
- The dataset size and fixed top-k (3) limit the variance available for robust per-claim Spearman correlation. Many claims have identical influence across evidence (e.g., all 0), generating undefined correlations.
- Interventions evaluate solitary evidence removal, which may not capture nonlinear interactions between multiple evidence pieces.
- This evaluation measures statistical/intervention dependence, not strict counterfactual real-world causality.

## 10. Interpretation
The results demonstrate whether the feature importance magnitudes identified by SHAP and LIME align with the measurable decision shift when that feature is removed. Neither post-hoc method definitively predicts the exact intervention shift (rho values). However, the Top-1 agreement shows how often the most 'important' identified feature is also the most 'impactful' feature upon removal, without asserting real-world causality.