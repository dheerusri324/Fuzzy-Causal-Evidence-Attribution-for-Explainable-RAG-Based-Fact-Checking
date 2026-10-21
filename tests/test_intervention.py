"""
Tests for Phase 6 — Intervention-Based Evidence Influence.
"""

import pytest
import math

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from intervention.intervener import EvidenceIntervener
from intervention.schema import InterventionResult

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
def intervener(verifier):
    cfg = Config(device="cpu", max_intervention_evidence=3)
    return EvidenceIntervener(verifier=verifier, config=cfg)

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

def test_baseline_prediction_generation(intervener, sample_evidence):
    claim = "Paris is the capital of France."
    res = intervener.intervene("claim_1", claim, sample_evidence)
    
    # Check baseline properties
    assert res.claim_id == "claim_1"
    assert res.claim == claim
    assert res.baseline_verdict in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    assert len(res.baseline_probabilities) == 3
    assert "SUPPORTS" in res.baseline_probabilities

def test_single_evidence_removal(intervener, sample_evidence):
    claim = "Paris is the capital of France."
    res = intervener.intervene("claim_2", claim, sample_evidence)
    
    # We should have an intervention for each piece of evidence
    assert len(res.interventions) == 2
    
    ev_ids = [inv.evidence_id for inv in res.interventions]
    assert "ev_1" in ev_ids
    assert "ev_2" in ev_ids

def test_delta_and_influence_calculation(intervener, sample_evidence):
    claim = "Paris is the capital of France."
    res = intervener.intervene("claim_3", claim, sample_evidence)
    
    inv = res.interventions[0]
    expected_delta = inv.intervened_target_probability - inv.baseline_target_probability
    
    # Check delta
    assert math.isclose(inv.delta_probability, expected_delta, rel_tol=1e-5)
    
    # Check absolute influence
    assert math.isclose(inv.intervention_influence, abs(expected_delta), rel_tol=1e-5)

def test_verdict_change_detection(intervener, sample_evidence):
    claim = "Paris is the capital of France."
    res = intervener.intervene("claim_4", claim, sample_evidence)
    
    for inv in res.interventions:
        if inv.verdict_before != inv.verdict_after:
            assert inv.verdict_changed is True
        else:
            assert inv.verdict_changed is False

def test_top_k_intervention_selection(intervener, sample_evidence):
    claim = "Paris is the capital of France."
    # Only intervene on ev_1
    top_k = [sample_evidence[0]]
    
    res = intervener.intervene("claim_5", claim, sample_evidence, top_k_evidence=top_k)
    
    # Should only have one intervention
    assert len(res.interventions) == 1
    assert res.interventions[0].evidence_id == "ev_1"

def test_empty_evidence_handling(intervener):
    claim = "The sky is blue."
    res = intervener.intervene("claim_6", claim, [])
    
    assert res.baseline_verdict == "NOT_ENOUGH_INFO"
    assert len(res.interventions) == 0

def test_cache_behavior(intervener, sample_evidence):
    claim = "Cache test."
    intervener._eval_cache.clear()
    intervener._verdict_cache.clear()
    
    # First call
    intervener.intervene("claim_7", claim, sample_evidence)
    cache_size = len(intervener._eval_cache)
    assert cache_size > 0
    
    # Second call identical
    intervener.intervene("claim_8", claim, sample_evidence)
    assert len(intervener._eval_cache) == cache_size

def test_max_evidence_truncation(intervener, sample_evidence):
    claim = "Truncation test."
    # Config is set to 3. Create 4 pieces.
    many_evs = sample_evidence * 2
    
    res = intervener.intervene("claim_9", claim, many_evs)
    
    # Baseline truncated to 3, so interventions = 3
    assert len(res.interventions) == 3

def test_schema_compliance(intervener, sample_evidence):
    claim = "Schema compliance test."
    res = intervener.intervene("claim_10", claim, sample_evidence)
    
    d = res.to_dict()
    assert "baseline_verdict" in d
    assert "interventions" in d
    assert len(d["interventions"]) > 0
    
    inv_d = d["interventions"][0]
    required = [
        "evidence_id", "baseline_target_probability", "intervened_target_probability",
        "delta_probability", "intervention_influence", "verdict_before", 
        "verdict_after", "verdict_changed"
    ]
    for key in required:
        assert key in inv_d
