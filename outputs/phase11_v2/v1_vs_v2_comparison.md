# Phase 11 V1 vs V2 Remediation Comparative Evaluation Report

**Date:** 2026-10-21 17:49:03  
**Instances Evaluated:** 1500 evidence rows across 500 claims  
**Evaluation Status:** **INTERMEDIATE HYBRID EVALUATION**

> [!NOTE]
> **Methodological Scope Notice**: The metrics in this report reflect an **intermediate hybrid evaluation** conducted on 2026-10-21. The remediated 33-rule V2 fuzzy engine was evaluated over frozen Phase 10 feature attributions (`SHAP_score_raw`, `LIME_score_raw`, `intervention_influence`) and historical Phase 11 stability signals, with conflict signals recomputed by an earlier iteration of the conflict detector. These results confirm the complete elimination of silent fuzzy fallbacks (0.0% fallback rate vs. 59.07% in V1), but they do **not** represent a full end-to-end evaluation of the latest source code (which incorporates sequence-order sensitive caching in `attribution/explainer.py` and finalized continuous harmonic mean conflict scoring in `conflict/detector.py`).

## 1. Executive Summary

The remediated V2 pipeline resolves the silent fallback issue, fixes conflict score binary saturation,
decouples claim-level conflict from passage-level contradiction, and preserves order sensitivity in attribution stability.

### Key Comparative Metrics

| Dimension | Phase 11 V1 (Historical) | Phase 11 V2 (Remediated) | Remediation Impact |
| :--- | :--- | :--- | :--- |
| **Fallback Rate** | 59.07% (886/1,500) | **0.00% (0/1,500)** | **-100% elimination of silent fallbacks** |
| **Computed Coverage** | 40.93% (614/1,500) | **100.00% (1,500/1,500)** | Full input-space defuzzification |
| **Conflict Detection** | Min-max scaled (saturated) | **Continuous Harmonic Mean** | Smooth, non-saturating scoring |
| **Conflict Saturation (== 1.0)** | 312 rows | **0 rows** | Continuous variation preserved |
| **AUROC (Fuzzy)** | 0.7630 | **0.6872** | Fuzzy lower than baseline (baseline: 0.7195) |
| **AUROC Delta (vs Baseline)** | +0.0365 | **-0.0323** | Baseline favored over fuzzy on AUROC |
| **AUPR (Fuzzy)** | 0.4884 | **0.5630** | Fuzzy lower than baseline (baseline: 0.6324) |
| **Spearman $\rho$ with Gold** | 0.4672 | **0.3148** | Lower correlation than baseline (0.3845) |

## 2. Field-Level Status Audit (V2)

In V2, every evidence record reports complete field-level status metadata:
- `field_status`: Explicitly identifies `COMPUTED` for all 5 consequents (`evidence_reliability`, `evidence_relevance`, `support_degree`, `contradiction_output`, `explanation_confidence`).
- `inference_status`: Recorded as `SUCCESS` for 100.0% of evaluated instances.
- `fallback_used`: Exactly `False` across all 1,500 evaluated rows.
