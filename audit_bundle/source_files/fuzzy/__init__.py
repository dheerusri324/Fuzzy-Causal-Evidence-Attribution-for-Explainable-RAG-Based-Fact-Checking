"""Fuzzy Evidence Reasoning package."""

from fuzzy.schema import FuzzyInput, FuzzyOutput, BaselineOutput
from fuzzy.engine import FuzzyEvidenceEngine
from fuzzy.baseline import compute_baseline_score
from fuzzy.normalization import normalize_relevance, normalize_attribution, normalize_intervention, compute_attribution_strength

__all__ = [
    "FuzzyInput", 
    "FuzzyOutput", 
    "BaselineOutput", 
    "FuzzyEvidenceEngine", 
    "compute_baseline_score",
    "normalize_relevance",
    "normalize_attribution",
    "normalize_intervention",
    "compute_attribution_strength"
]
