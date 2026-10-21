"""
Evidence Intervener.

Evaluates intervention-based influence by explicitly removing individual 
evidence pieces and measuring the delta in the target class probability.
This isolates the model's actual reliance on the evidence piece from attribution 
heuristics.
"""

from typing import List, Dict, Tuple, Optional
import copy

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from intervention.schema import EvidenceIntervention, InterventionResult
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class EvidenceIntervener:
    """Computes intervention-based influence for evidence pieces."""

    def __init__(self, verifier: ClaimEvidenceVerifier, config: Optional[Config] = None):
        self.verifier = verifier
        self.config = config or Config()
        self.max_evidence = getattr(self.config, "max_intervention_evidence", 5)
        
        # Simple memory cache to avoid re-running the verifier on identical subsets
        self._eval_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}
        # Also cache the verdicts
        self._verdict_cache: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        
    def _get_prediction(self, claim: str, evidence_subset: List[RetrievedEvidence]) -> Tuple[str, Dict[str, float]]:
        """Call the verifier and return (verdict, probabilities) utilizing cache."""
        ev_ids = tuple(sorted(ev.evidence_id for ev in evidence_subset))
        cache_key = (claim, ev_ids)
        
        if cache_key not in self._eval_cache:
            res = self.verifier.verify(claim, evidence_subset)
            self._eval_cache[cache_key] = res.probabilities
            self._verdict_cache[cache_key] = res.verdict
            
        return self._verdict_cache[cache_key], self._eval_cache[cache_key]

    def intervene(
        self, 
        claim_id: str, 
        claim: str, 
        evidence_list: List[RetrievedEvidence],
        top_k_evidence: Optional[List[RetrievedEvidence]] = None
    ) -> InterventionResult:
        """Measure influence by performing leave-one-out interventions.
        
        Args:
            claim_id: Unique claim identifier.
            claim: The claim text.
            evidence_list: The full context of retrieved evidence.
            top_k_evidence: Optional subset of evidence to intervene on. If provided, 
                            we only intervene (remove) items in this list from the full list.
        """
        
        # Enforce maximum evidence limit for the full set
        if len(evidence_list) > self.max_evidence:
            logger.warning(
                f"Truncating baseline evidence for intervention from {len(evidence_list)} to {self.max_evidence}."
            )
            evidence_list = evidence_list[:self.max_evidence]

        # 1. Baseline prediction (all evidence)
        baseline_verdict, baseline_probs = self._get_prediction(claim, evidence_list)
        
        # The influence target is the probability of the *baseline predicted class*
        target_class = baseline_verdict
        baseline_target_prob = baseline_probs[target_class]
        
        # Edge case: No evidence available to intervene on
        if not evidence_list:
            return InterventionResult(
                claim_id=claim_id,
                claim=claim,
                baseline_verdict=baseline_verdict,
                baseline_probabilities=baseline_probs,
                interventions=[]
            )

        # 2. Determine which evidence pieces to intervene on
        # By default, intervene on all pieces in the baseline set.
        intervention_targets = evidence_list
        if top_k_evidence is not None:
            # Only intervene on the requested top-k (e.g. from SHAP/LIME)
            # Ensure they are actually in the baseline set to make removal valid
            baseline_ids = {e.evidence_id for e in evidence_list}
            intervention_targets = [e for e in top_k_evidence if e.evidence_id in baseline_ids]

        # 3. Perform leave-one-out interventions
        interventions = []
        for target_ev in intervention_targets:
            # Create subset excluding the target evidence piece
            intervened_subset = [ev for ev in evidence_list if ev.evidence_id != target_ev.evidence_id]
            
            # Predict on the intervened subset
            intervened_verdict, intervened_probs = self._get_prediction(claim, intervened_subset)
            intervened_target_prob = intervened_probs[target_class]
            
            # Calculate delta and influence
            delta = intervened_target_prob - baseline_target_prob
            influence = abs(delta)
            
            intervention = EvidenceIntervention(
                evidence_id=target_ev.evidence_id,
                baseline_target_probability=baseline_target_prob,
                intervened_target_probability=intervened_target_prob,
                delta_probability=delta,
                intervention_influence=influence,
                verdict_before=baseline_verdict,
                verdict_after=intervened_verdict,
                verdict_changed=(baseline_verdict != intervened_verdict)
            )
            interventions.append(intervention)

        return InterventionResult(
            claim_id=claim_id,
            claim=claim,
            baseline_verdict=baseline_verdict,
            baseline_probabilities=baseline_probs,
            interventions=interventions
        )
