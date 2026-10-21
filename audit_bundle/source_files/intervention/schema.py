"""
Intervention output schema.

Defines the structure for intervention-based evidence influence results,
recording probabilities before and after removing specific evidence pieces.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class EvidenceIntervention:
    """Intervention result for the removal of a single piece of evidence."""
    evidence_id: str
    
    # Target class probability before removal (baseline)
    baseline_target_probability: float
    
    # Target class probability after removal (intervened)
    intervened_target_probability: float
    
    # Raw delta (intervened - baseline). Can be negative or positive.
    delta_probability: float
    
    # Absolute difference used as the influence score
    intervention_influence: float
    
    verdict_before: str
    verdict_after: str
    verdict_changed: bool


@dataclass
class InterventionResult:
    """Complete intervention results for a claim across multiple evidence removals."""
    claim_id: str
    claim: str
    baseline_verdict: str
    baseline_probabilities: Dict[str, float]
    interventions: List[EvidenceIntervention]
    
    def to_dict(self) -> dict:
        return asdict(self)
