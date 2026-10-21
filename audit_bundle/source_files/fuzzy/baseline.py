"""
Non-fuzzy baseline for evidence scoring.
Provides a simple weighted aggregation to compare against the fuzzy logic engine.
"""

from config.settings import Config
from fuzzy.schema import FuzzyInput, BaselineOutput
from fuzzy.normalization import compute_attribution_strength

def compute_baseline_score(inputs: FuzzyInput, config: Config = None) -> BaselineOutput:
    """
    Computes a non-fuzzy weighted aggregation of evidence signals.
    """
    config = config or Config()
    
    attr_strength = compute_attribution_strength(
        inputs.attribution_shap, 
        inputs.attribution_lime,
        config.fuzzy_shap_weight,
        config.fuzzy_lime_weight
    )
    
    w = config.fuzzy_baseline_weights
    
    # Weighted positive score
    score = (
        w["relevance"] * inputs.relevance +
        w["attribution_strength"] * attr_strength +
        w["intervention_influence"] * inputs.intervention_influence +
        w["source_reliability"] * inputs.source_reliability +
        w["stability"] * inputs.stability_score
    )
    
    # Ensure it's bounded [0, 1] (assuming weights sum to 1.0)
    score = max(0.0, min(1.0, score))
    
    # Adjust for contradiction penalty
    adjusted_score = score * (1.0 - inputs.contradiction_degree)
    
    return BaselineOutput(
        baseline_score=score,
        adjusted_baseline_score=adjusted_score
    )
