"""
Verification output schema.

Defines the structured output of the claim-evidence verifier.
This object is consumed by the downstream attribution (SHAP/LIME) and intervention modules.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class VerificationResult:
    """A structured verification result."""

    claim: str
    verdict: str
    support_probability: float
    refute_probability: float
    nei_probability: float
    confidence: float
    evidence_ids: List[str]

    @property
    def probabilities(self) -> Dict[str, float]:
        """Convenience property to access full distribution."""
        return {
            "SUPPORTS": self.support_probability,
            "REFUTES": self.refute_probability,
            "NOT_ENOUGH_INFO": self.nei_probability,
        }

    def to_dict(self) -> dict:
        return asdict(self)
