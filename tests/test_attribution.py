"""
Tests for Phase 5 — Evidence-Level Attribution (SHAP + LIME).
"""

import pytest

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from attribution.explainer import EvidenceExplainer
from attribution.schema import AttributionResult

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def verifier():
    """Module-scoped verifier to avoid reloading."""
    cfg = Config(device="cpu")
    v = ClaimEvidenceVerifier(config=cfg)
    v._load_model()
    return v

@pytest.fixture(scope="module")
def explainer(verifier):
    # Use small sample counts for fast testing
    cfg = Config(
        device="cpu", 
        shap_num_samples=10, 
        lime_num_samples=10,
        max_attribution_evidence=3
    )
    return EvidenceExplainer(verifier=verifier, config=cfg)

@pytest.fixture
def sample_evidence():
    return [
        RetrievedEvidence(
            evidence_id="ev_1",
            text="The capital of France is Paris.",
            doc_title="France",
            sentence_idx=0,
            source="test",
            source_type="test",
            retrieval_score=0.9,
            retrieval_rank=1,
            retrieval_method="dense"
        ),
        RetrievedEvidence(
            evidence_id="ev_2",
            text="The Eiffel Tower is located in Paris.",
            doc_title="Paris",
            sentence_idx=0,
            source="test",
            source_type="test",
            retrieval_score=0.8,
            retrieval_rank=2,
            retrieval_method="dense"
        )
    ]

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_evidence_id_preservation(explainer, sample_evidence):
    claim = "Paris is the capital of France."
    res = explainer.explain("claim_1", claim, sample_evidence)
    
    # Check that both evidence IDs are present in the attributions
    attr_ids = [attr.evidence_id for attr in res.evidence_attributions]
    assert "ev_1" in attr_ids
    assert "ev_2" in attr_ids
    assert len(attr_ids) == 2

def test_single_evidence_attribution(explainer, sample_evidence):
    claim = "Paris is the capital of France."
    res = explainer.explain("claim_2", claim, [sample_evidence[0]])
    
    assert len(res.evidence_attributions) == 1
    attr = res.evidence_attributions[0]
    
    assert attr.evidence_id == "ev_1"
    assert attr.shap_rank == 1
    assert attr.lime_rank == 1

def test_multi_evidence_ranking(explainer, sample_evidence):
    # "ev_1" directly proves the claim, "ev_2" is related but less direct
    claim = "The capital of France is Paris."
    res = explainer.explain("claim_3", claim, sample_evidence)
    
    # "ev_1" should likely have higher attribution for the SUPPORTS class
    # but we just test ranking mechanics here
    ranked_shap = res.get_ranked_evidence(method="shap")
    assert len(ranked_shap) == 2
    assert ranked_shap[0].shap_rank == 1
    assert ranked_shap[1].shap_rank == 2
    assert ranked_shap[0].shap_target >= ranked_shap[1].shap_target
    
def test_target_class_selection(explainer, sample_evidence):
    claim = "Paris is the capital of France."
    res = explainer.explain("claim_4", claim, sample_evidence)
    
    # Target scores should map to the verifier's predicted class
    target = res.verdict
    
    attr = res.evidence_attributions[0]
    if target == "SUPPORTS":
        assert attr.shap_target == attr.shap_support
        assert attr.lime_target == attr.lime_support
    elif target == "REFUTES":
        assert attr.shap_target == attr.shap_refute
        assert attr.lime_target == attr.lime_refute
    else:
        assert attr.shap_target == attr.shap_nei
        assert attr.lime_target == attr.lime_nei

def test_output_schema(explainer, sample_evidence):
    claim = "Testing schema."
    res = explainer.explain("claim_5", claim, sample_evidence)
    
    d = res.to_dict()
    assert "claim_id" in d
    assert "claim" in d
    assert "verdict" in d
    assert "baseline_probabilities" in d
    assert "evidence_attributions" in d
    
    attr = d["evidence_attributions"][0]
    required_keys = [
        "evidence_id", "text", 
        "shap_support", "shap_refute", "shap_nei", "shap_target", "shap_rank",
        "lime_support", "lime_refute", "lime_nei", "lime_target", "lime_rank"
    ]
    for key in required_keys:
        assert key in attr

def test_empty_evidence_handling(explainer):
    claim = "The sky is blue."
    res = explainer.explain("claim_6", claim, [])
    
    assert res.verdict == "NOT_ENOUGH_INFO"
    assert len(res.evidence_attributions) == 0

def test_cache_behavior(explainer, sample_evidence):
    claim = "Caching test."
    
    # Clear cache for isolated test
    explainer._eval_cache.clear()
    
    explainer.explain("claim_7", claim, sample_evidence)
    cache_size_after_first = len(explainer._eval_cache)
    assert cache_size_after_first > 0
    
    # Second explain should not increase cache size for identical subsets
    explainer.explain("claim_8", claim, sample_evidence)
    cache_size_after_second = len(explainer._eval_cache)
    
    assert cache_size_after_first == cache_size_after_second

def test_max_evidence_truncation(explainer, sample_evidence):
    claim = "Truncation test."
    # Explainer fixture limits to 3
    many_evs = sample_evidence * 2  # 4 pieces
    
    res = explainer.explain("claim_9", claim, many_evs)
    
    assert len(res.evidence_attributions) == 3
