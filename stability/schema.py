"""
Schemas for Explanation Stability Module.
"""
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class StabilityResult:
    claim_id: str
    claim: str
    baseline_ranking: Dict[str, float]
    perturbed_ranking: Dict[str, float]
    spearman_correlation: float
    stability_score: float
