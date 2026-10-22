"""
Explanation Stability Analyzer.

Measures robustness of evidence attribution under controlled perturbations
(e.g., Evidence-order permutation). Computes Spearman rank correlation between
baseline and perturbed attributions to output a stability score in [0,1].
"""

import copy
import random
from typing import List, Dict, Optional
import scipy.stats as stats

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from attribution.explainer import EvidenceExplainer
from stability.schema import StabilityResult

class StabilityAnalyzer:
    def __init__(self, explainer: EvidenceExplainer, config: Optional[Config] = None):
        self.explainer = explainer
        self.config = config or Config()
        self.perturbation_type = getattr(self.config, "stability_perturbation", "permutation")
        
    def _get_attribution_scores(self, claim_id: str, claim: str, evidence: List[RetrievedEvidence]) -> Dict[str, float]:
        """Runs the explainer and returns SHAP target class scores mapped by evidence_id."""
        attr_result = self.explainer.explain(claim_id, claim, evidence)
        
        scores = {}
        for ev_attr in attr_result.evidence_attributions:
            # We use shap_target (absolute value is typically best, but since it's ranking, signed is fine.
            # Using shap_target directly to match how it influences the prediction).
            scores[ev_attr.evidence_id] = ev_attr.shap_target
        return scores
        
    def analyze(self, claim_id: str, claim: str, evidence_list: List[RetrievedEvidence]) -> StabilityResult:
        if len(evidence_list) <= 1:
            # Cannot truly measure perturbation stability on 0 or 1 item
            return StabilityResult(
                claim_id=claim_id,
                claim=claim,
                baseline_ranking={},
                perturbed_ranking={},
                spearman_correlation=1.0,
                stability_score=1.0
            )
            
        # 1. Compute baseline attribution
        baseline_scores = self._get_attribution_scores(f"{claim_id}_base", claim, evidence_list)
        
        # 2. Apply controlled perturbation: Evidence-order permutation
        # Limitation Note:
        # The default "permutation" (or "reverse") strategy performs a single deterministic list reversal
        # (pi = [k, ..., 1]). For k=3, this evaluates a single permutation rather than all 3! = 6 permutations.
        # This deterministic behavior is preserved for backwards compatibility with the Phase 8/11 protocols.
        # Alternatively, "random_shuffle" shuffles evidence order using a configured seed.
        perturbed_evidence = list(evidence_list)
        if self.perturbation_type in ("permutation", "reverse"):
            # Deterministic list reversal
            perturbed_evidence.reverse()
        elif self.perturbation_type == "random_shuffle":
            seed = getattr(self.config, "seed", 42)
            rng = random.Random(seed)
            rng.shuffle(perturbed_evidence)
        else:
            # Default fallback to reversal
            perturbed_evidence.reverse()
        
        # 3. Compute perturbed attribution
        perturbed_scores = self._get_attribution_scores(f"{claim_id}_pert", claim, perturbed_evidence)
        
        # 4. Compare rankings using Spearman rank correlation
        # We must align the IDs
        common_ids = sorted(list(baseline_scores.keys()))
        
        base_vals = [baseline_scores[eid] for eid in common_ids]
        pert_vals = [perturbed_scores[eid] for eid in common_ids]
        
        # If all values are identical (e.g., all 0), correlation is undefined/nan.
        # Handle nan gracefully.
        rho, p_value = stats.spearmanr(base_vals, pert_vals)
        if math.isnan(rho):
            # If standard deviation is 0, they are perfectly identical flats
            rho = 1.0
            
        # 5. Normalize to [0,1]
        # Spearman rho is in [-1, 1].
        # formula: (rho + 1) / 2
        stability_score = max(0.0, min(1.0, (rho + 1.0) / 2.0))
        
        return StabilityResult(
            claim_id=claim_id,
            claim=claim,
            baseline_ranking=baseline_scores,
            perturbed_ranking=perturbed_scores,
            spearman_correlation=rho,
            stability_score=stability_score
        )

import math
