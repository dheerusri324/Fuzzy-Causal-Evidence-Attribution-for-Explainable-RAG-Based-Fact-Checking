"""
Normalization utilities for the fuzzy reasoning engine.
Ensures all signals fall within [0, 1].
"""

def normalize_relevance(score: float, max_score: float = 1.0) -> float:
    """
    Normalize retrieval score to [0,1]. 
    If dense retrieval cosine similarity is used, it should already be [0,1].
    If BM25, a max_score is needed.
    """
    return max(0.0, min(1.0, score / max_score if max_score > 0 else 0.0))

def normalize_attribution(value: float, max_expected: float = 1.0) -> float:
    """
    Normalize absolute attribution value to [0,1].
    Usually SHAP/LIME probabilities are bounded by [-1, 1],
    so absolute values are roughly in [0,1].
    """
    return max(0.0, min(1.0, abs(value) / max_expected))

def compute_attribution_strength(shap_val: float, lime_val: float, shap_weight: float = 0.5, lime_weight: float = 0.5) -> float:
    """
    Compute aggregate attribution strength.
    """
    shap_norm = normalize_attribution(shap_val)
    lime_norm = normalize_attribution(lime_val)
    # Re-weight in case they don't sum to 1
    total_w = shap_weight + lime_weight
    if total_w == 0:
        return 0.0
    return (shap_weight * shap_norm + lime_weight * lime_norm) / total_w

def normalize_intervention(delta_prob: float) -> float:
    """
    Normalize delta probability to [0,1].
    Delta prob is already an absolute change in probability [0,1].
    """
    return max(0.0, min(1.0, abs(delta_prob)))
