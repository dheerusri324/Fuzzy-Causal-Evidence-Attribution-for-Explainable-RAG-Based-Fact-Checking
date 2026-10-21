"""
Attribution output schema.

Defines the structure for evidence-level attribution results from SHAP and LIME.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class EvidenceAttribution:
    """Attribution scores and ranks for a single piece of evidence."""
    evidence_id: str
    text: str
    
    # SHAP scores for each class
    shap_support: float
    shap_refute: float
    shap_nei: float
    shap_target: float  # SHAP score for the predicted target class
    shap_rank: int      # Rank based on shap_target (1 = most important)
    
    # LIME scores for each class
    lime_support: float
    lime_refute: float
    lime_nei: float
    lime_target: float  # LIME score for the predicted target class
    lime_rank: int      # Rank based on lime_target (1 = most important)


@dataclass
class AttributionResult:
    """Complete attribution result for a claim."""
    claim_id: str
    claim: str
    verdict: str  # The verifier's predicted class
    baseline_probabilities: Dict[str, float]  # Probabilities with ALL evidence
    evidence_attributions: List[EvidenceAttribution]
    
    def get_ranked_evidence(self, method: str = "shap", top_k: int = None) -> List[EvidenceAttribution]:
        """Get evidence pieces ranked by the specified attribution method.
        
        Args:
            method: "shap" or "lime"
            top_k: Number of top evidence pieces to return
        """
        if method.lower() == "shap":
            ranked = sorted(self.evidence_attributions, key=lambda x: x.shap_rank)
        elif method.lower() == "lime":
            ranked = sorted(self.evidence_attributions, key=lambda x: x.lime_rank)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
            
        if top_k is not None:
            ranked = ranked[:top_k]
            
        return ranked

    def to_dict(self) -> dict:
        return asdict(self)
