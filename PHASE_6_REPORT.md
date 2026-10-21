# Phase 6: Intervention-Based Evidence Influence

## Implementation
I implemented an `EvidenceIntervener` that systematically removes individual evidence pieces (Leave-One-Out) and measures the verifier's response. This isolates the actual influence of each piece from any mathematical attribution heuristics. To ensure efficiency, it integrates an internal caching mechanism to avoid redundant predictions on identical evidence subsets.

## Influence Definition
Intervention-based influence is calculated as the absolute difference ($\Delta P$) in the probability of the *baseline predicted class* after the evidence piece is removed.
```python
delta_probability = intervened_target_probability - baseline_target_probability
intervention_influence = abs(delta_probability)
```
The module also records `verdict_changed` (Boolean flip) if the removal alters the verifier's hard classification, which directly supports the project's verdict-flip rate metric.

## Files Created/Modified
- `PHASE_5_REPORT.md`: Updated to use non-causal language.
- `config/settings.py`: Added `max_intervention_evidence`.
- `intervention/schema.py`: Structured data classes `EvidenceIntervention` and `InterventionResult`.
- `intervention/intervener.py`: Implemented `EvidenceIntervener` logic.
- `intervention/__init__.py`: Package init.
- `tests/test_intervention.py`: Suite containing 9 tests.

## Tests
- **Command:** `pytest tests/test_intervention.py -v --tb=short`
- **Result:** 9 tests passed.

## Sanity Check
```text
--- Claim: Lincoln Motor Company was named by Henry M. Leland. ---
Baseline Verdict = SUPPORTS
Baseline P(SUPPORTS) = 0.9967

Lincoln_Motor_Company_1 removed:
  P(SUPPORTS) = 0.0002
  ΔP = 0.9966
  Verdict changed = YES

Henry_M._Leland_0 removed:
  P(SUPPORTS) = 0.9957
  ΔP = 0.0010
  Verdict changed = NO

Ford_Motor_Company_0 removed:
  P(SUPPORTS) = 0.9840
  ΔP = 0.0127
  Verdict changed = NO
```
*Scientific Note: Removing `Lincoln_Motor_Company_1` substantially changed the verifier's predicted probability and caused a verdict flip. This piece of evidence exhibited the largest intervention-based influence for this claim.*

## Scientific Notes
- **Limitation:** Leave-One-Out (LOO) interventions only capture first-order marginal influence. If two evidence pieces are redundant (e.g. both perfectly support the claim on their own), removing either *one* might result in $\Delta P \approx 0$. 
- **Causality:** This module measures intervention-based influence on the *model's* predictions. It does not prove true real-world causality, but rather the model's conditional dependency on the omitted information. 

---
**STOPPED.** Awaiting explicit confirmation before proceeding to Phase 7.
