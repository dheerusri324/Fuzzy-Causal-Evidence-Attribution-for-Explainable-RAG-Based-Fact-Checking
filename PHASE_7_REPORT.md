# Phase 7: Fuzzy Evidence Reasoning

## Implementation
I implemented `FuzzyEvidenceEngine` using `scikit-fuzzy`. The engine uses a configurable linguistic variable system (Antecedents and Consequents) bound to overlapping triangular and trapezoidal membership functions (`LOW`, `MEDIUM`, `HIGH`). It combines evidence signals smoothly using a human-interpretable rule base.

## Inputs
All inputs are normalized to `[0,1]`.
1. **relevance**: Extracted from Phase 3's hybrid retrieval system.
2. **attribution_strength**: Mean combination of normalized SHAP and LIME attribution vectors from Phase 5.
3. **intervention_influence**: Absolute $\Delta P$ from the Phase 6 intervention engine.
4. **contradiction_degree**: (Placeholder) Will come from Phase 8. Defaults to 0.0.
5. **source_reliability**: (Placeholder) Engineering prior default for Wikipedia. Defaults to 0.75.
6. **stability_score**: (Placeholder) Will come from Phase 8. Defaults to 1.0.

## Membership Functions
Each universe is bounded `[0.0, 1.0]`:
- `LOW`: `trimf(0.0, 0.0, 0.4)` — Peaks at 0, ramps down to 0 at 0.4.
- `MEDIUM`: `trimf(0.2, 0.5, 0.8)` — Centered exactly at 0.5.
- `HIGH`: `trapmf(0.6, 0.8, 1.0, 1.0)` — Begins at 0.6, plateaus fully at 0.8 to 1.0.

## Rules
The system enforces intuitive hierarchies without needing hundreds of rules:
1. `HIGH` (Relevance, Attribution, Intervention, Source) & `LOW` (Contradiction) $\rightarrow$ `HIGH` Evidence Reliability.
2. `HIGH` (Relevance, Attribution) but `LOW` (Intervention) $\rightarrow$ `LOW` Explanation Confidence.
3. `LOW` (Relevance, Attribution, Intervention) $\rightarrow$ `LOW` Evidence Reliability.
4. `HIGH` (Relevance, Contradiction) $\rightarrow$ `HIGH` Contradiction Degree.
5. `HIGH` (Relevance, Intervention, Stability) & `LOW` (Contradiction) $\rightarrow$ `HIGH` Support Degree.
6. `HIGH` (Attribution, Intervention, Stability) $\rightarrow$ `HIGH` Explanation Confidence.
7. `HIGH` (Attribution) & `LOW` (Stability) $\rightarrow$ `LOW` Explanation Confidence.
8. `MEDIUM` (Relevance, Attribution, Intervention) $\rightarrow$ `MEDIUM` Evidence Reliability.

## Outputs
1. **evidence_relevance**: Validates evidence relevance presence.
2. **evidence_reliability**: Combined trust metric.
3. **support_degree**: Likelihood that the evidence genuinely pushes the verifier output.
4. **contradiction_degree**: Represents detected internal conflict (driven heavily by future Phase 8).
5. **explanation_confidence**: Confidence in the explanation mechanisms (penalized heavily if attribution does not translate into actual intervention).

## Non-Fuzzy Baseline
To facilitate future validation (Experiment 2), I implemented a `compute_baseline_score` function which executes a simple weighted sum combination:
```python
score = (0.30 * relevance + 0.25 * attribution + 0.25 * intervention + 0.10 * source + 0.10 * stability)
adjusted_score = score * (1.0 - contradiction_degree)
```

## Files
- `requirements.txt`: Added `scikit-fuzzy`.
- `config/settings.py`: Added configuration logic for fuzzy weights.
- `fuzzy/schema.py`: Structured data classes for Inputs/Outputs.
- `fuzzy/normalization.py`: Bounds functions mapping `[-1, 1]` or unbounded spaces into `[0, 1]`.
- `fuzzy/engine.py`: Defines the `FuzzyEvidenceEngine`.
- `fuzzy/baseline.py`: Non-fuzzy evaluation formulas.
- `tests/test_fuzzy.py`: Contains 8 isolated test cases.

## Tests
- **Command:** `pytest tests/test_fuzzy.py -v --tb=short`
- **Result:** 8 tests passed covering determinism, mathematical bounds, Rule 2 & 4 execution logic, and baselines.

## Sanity Checks

### CASE A — Strong evidence
*Inputs: Rel=0.95, Attr=0.90, Int=0.88, Contradict=0.05, SrcRel=0.80, Stab=0.90*
```text
  Evidence Reliability:    0.8397
  Support Degree:          0.8397
  Explanation Confidence:  0.8444
```

### CASE B — Attribution without intervention
*Inputs: Rel=0.90, Attr=0.90, Int=0.10, Contradict=0.05, SrcRel=0.80, Stab=0.90*
```text
  Evidence Reliability:    0.5000
  Support Degree:          0.0000
  Explanation Confidence:  0.1400  <-- Successfully punished by Rule 2
```

### CASE C — Conflicting evidence
*Inputs: Rel=0.90, Attr=0.70, Int=0.80, Contradict=0.90, SrcRel=0.80, Stab=0.80*
```text
  Evidence Reliability:    0.5000
  Support Degree:          0.0000
  Contradiction Degree:    0.8444  <-- Successfully triggered by Rule 4
  Baseline Adjusted:       0.0805  <-- Heavily penalized by baseline contradiction formulation
```

## Limitations
- **Placeholders**: `contradiction_degree` and `stability_score` are currently fully synthetically driven placeholders. They carry no real meaning until Phase 8 is implemented. 
- **Validation**: This engine provides a strong, mathematically sound structure for reasoning under uncertainty. **It has NOT yet been proven to improve reliability**. That hypothesis will be formally tested in Experiment 2.

---
**STOPPED.** Awaiting explicit confirmation before proceeding to Phase 8.
