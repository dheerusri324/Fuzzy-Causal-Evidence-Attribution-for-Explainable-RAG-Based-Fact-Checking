# Phase 8: Conflict Detection + Explanation Stability

## Implementation

### Conflict Detection
- **Approach**: Analyzes each retrieved evidence piece individually against the claim to determine its independent stance (`SUPPORTING`, `REFUTING`, or `NEUTRAL`).
- **NLI Model Reused**: Utilized the cached Phase 4 `cross-encoder/nli-deberta-v3-base` model. No additional models were downloaded.
- **Evidence-Role Classification**: An evidence piece is assigned `SUPPORTING` or `REFUTING` if its respective NLI probability is greater than both other classes AND exceeds `config.conflict_role_threshold`. Otherwise, it is assigned `NEUTRAL`.
- **Threshold**: Configurable via `conflict_role_threshold` (default `0.5`).
- **Conflict-Score Formulation**: Computed as `2.0 * min(max_support, max_refute)`, bounded to `[0.0, 1.0]`. This ensures the conflict score is only high when both meaningful supporting and meaningful refuting evidence coexist.

### Stability Analysis
- **Approach**: Assesses the robustness of evidence-level attribution.
- **Perturbations Implemented**: Evidence-order permutation. The evidence list is reversed and attributions are recomputed.
- **Attribution Comparison Metric**: Spearman rank correlation ($\rho$) is computed on the SHAP target-class attribution scores. Alignment is strictly mapped by `evidence_id`.
- **Normalization to [0,1]**: The correlation is normalized via `(rho + 1.0) / 2.0`, bounding the final stability score in `[0.0, 1.0]`.

### Fuzzy Integration
- **Contradiction Signal**: The `ConflictResult.conflict_score` can now directly populate the `contradiction_degree` field of `FuzzyInput`.
- **Stability Signal**: The `StabilityResult.stability_score` can now directly populate the `stability_score` field of `FuzzyInput`.
- **Placeholder Behavior Replacement**: When real Phase 8 results are computed, they perfectly replace the placeholders in the fuzzy engine without modifying the fuzzy logic itself.
- **Remaining Fallbacks**: `source_reliability` remains a configurable fallback prior (default `0.75` for Wikipedia). If a Phase 8 calculation is skipped for compute reasons, the defaults (`contradiction_degree=0.0`, `stability_score=1.0`) are safely maintained.

## Files Modified/Created
- `config/settings.py`: Added `conflict_role_threshold` and `stability_perturbation`.
- `conflict/schema.py`: Added `EvidenceRole` and `ConflictResult` structures.
- `conflict/detector.py`: Implemented `ConflictDetector`.
- `conflict/__init__.py`: Package initialization.
- `stability/schema.py`: Added `StabilityResult` structure.
- `stability/analyzer.py`: Implemented `StabilityAnalyzer`.
- `stability/__init__.py`: Package initialization.
- `tests/test_conflict.py`: 8 isolated tests.
- `tests/test_stability.py`: 4 isolated tests.

## Tests
- **Command**: `pytest tests/test_conflict.py tests/test_stability.py -v --tb=short`
- **Result**: 12 passed. All assertions covering logic, bounds, identity alignment, and thresholds are robust.

## Sanity Checks

### Conflict Sanity Check
*Claim: Lincoln Motor Company was named by Henry M. Leland.*
```text
[CONFLICT DETECTION]
Lincoln_Motor_Company_1 -> SUPPORTING (Sup: 0.9816, Ref: 0.0002, Neu: 0.0182)
Henry_M._Leland_0 -> NEUTRAL (Sup: 0.0003, Ref: 0.0057, Neu: 0.9940)
Ford_Motor_Company_0 -> REFUTING (Sup: 0.0000, Ref: 0.9998, Neu: 0.0002)
Conflict detected = YES
Conflict score = 1.0000
```
*Note: A conflict was successfully detected between the supporting and refuting evidence.*

### Stability Sanity Check
*Claim: Lincoln Motor Company was named by Henry M. Leland.*
```text
[STABILITY ANALYSIS]
Baseline Ranking:
  Lincoln_Motor_Company_1: 0.9893
  Henry_M._Leland_0: 0.0009
  Ford_Motor_Company_0: 0.0066

Perturbed Ranking (Evidence list reversed):
  Ford_Motor_Company_0: 0.0066
  Henry_M._Leland_0: 0.0009
  Lincoln_Motor_Company_1: 0.9893

Correlation = 1.0000
Stability Score = 1.0000
```
*Note: The explanation remained completely stable under the tested order perturbation.*

## Limitations
- The stability calculation is currently restricted to order-permutation to minimize token/compute overhead and prevent semantic drift from synthetic paraphrasing. 
- Stability measures the robustness of the explanation method under perturbation; it does not guarantee that the explanation itself is factually correct.
- `source_reliability` remains a naive configurable prior until properly integrated with a domain authority database.

---
**STOPPED**. Awaiting explicit confirmation before proceeding to Phase 9.
