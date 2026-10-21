# Phase 5: SHAP + LIME Evidence Attribution

## Implementation
I implemented evidence-level attribution using a wrapper over the Phase 4 verifier. The explainer constructs binary perturbation masks representing the presence or absence of individual evidence pieces.
- **SHAP:** Used `shap.KernelExplainer` applied directly over the binary evidence masks.
- **LIME:** Built a lightweight binary tabular LIME implementation directly inside the explainer. It randomly generates subset masks, weights them using a standard LIME exponential kernel based on normalized Hamming distance to the full evidence set, and fits a Ridge regression.

## Feature Unit
**Explicit confirmation: Evidence pieces are the interpretable features.**
The system evaluates the verifier's confidence exclusively by omitting or including intact evidence blocks. Token-level occlusion is not used.

## Files
- `attribution/schema.py`: Structured output for attribution containing `EvidenceAttribution` and `AttributionResult`.
- `attribution/explainer.py`: `EvidenceExplainer` class encapsulating SHAP and custom LIME. Implements an in-memory `_eval_cache` so that repeatedly sampled subsets (common in kernel attribution) don't trigger redundant model inferences.
- `tests/test_attribution.py`: Contains test suites to ensure attribution shapes and output requirements remain stable.

## Configuration
Added to `config/settings.py`:
- `shap_num_samples: 100`
- `lime_num_samples: 100`
- `max_attribution_evidence: 5` (truncates trailing evidence inputs safely before attribution to avoid exponential runtime explosions)

## Tests
- **Command:** `pytest tests/test_attribution.py -v --tb=short`
- **Result:** 8 tests passed.

## Sanity Check Example
```text
Claim: Lincoln Motor Company was named by Henry M. Leland.
Verifier verdict: SUPPORTS

Evidence:
Lincoln_Motor_Company_1: Founded in 1917 by Henry M. Leland , Lincoln has been a subsidiary of Ford since 1922 .
Henry_M._Leland_0: Henry Martyn Leland ( February 16 , 1843 -- March 26 , 1932 ) was an American machinist ...
Ford_Motor_Company_0: The Ford Motor Company ( commonly referred to simply as `` Ford '' ) is an American multinational automaker ...

SHAP (Target Class: SUPPORTS):
Lincoln_Motor_Company_1 = 0.9893
Ford_Motor_Company_0 = 0.0066
Henry_M._Leland_0 = 0.0009

LIME (Target Class: SUPPORTS):
Lincoln_Motor_Company_1 = 0.2670
Ford_Motor_Company_0 = -0.0335
Henry_M._Leland_0 = -0.0368
```
*Note: Both SHAP and LIME assign the highest attribution to E1 in this sanity example.*

## Known Limitations
- The underlying NLI model evaluates multiple independent facts somewhat clumsily when they are stacked linearly as plain text. However, this perfectly serves the purpose of testing XAI techniques in evaluating the importance of such evidence.
- While exact SHAP calculation is practically possible for tiny feature dimensions ($N \le 10$), standard KernelSHAP samples effectively approach the exact calculation bounds.

---
**STOPPED.** Awaiting explicit confirmation before proceeding to Phase 6.
