"""
Tests for Phase 4 — Claim-Evidence Verification.

Uses a real but small test against the loaded NLI model.
"""

import pytest
import math

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from verification.schema import VerificationResult

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def verifier():
    """Module-scoped verifier to avoid reloading the model per test."""
    cfg = Config(device="cpu", max_evidence=3)
    v = ClaimEvidenceVerifier(config=cfg)
    # Pre-load the model so tests run faster after the first one
    v._load_model()
    return v

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
            text="Paris is known for the Eiffel Tower.",
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

def test_model_initialization(verifier):
    assert verifier.model is not None
    assert verifier.tokenizer is not None
    assert verifier.model_name == "cross-encoder/nli-deberta-v3-base"

def test_label_mapping(verifier):
    assert verifier.id_to_project_label[0] == "REFUTES"
    assert verifier.id_to_project_label[1] == "SUPPORTS"
    assert verifier.id_to_project_label[2] == "NOT_ENOUGH_INFO"
    
    # Check underlying model's mapping
    actual_labels = verifier.model.config.id2label
    assert actual_labels[0].lower() == "contradiction"
    assert actual_labels[1].lower() == "entailment"
    assert actual_labels[2].lower() == "neutral"

def test_single_evidence_verification(verifier, sample_evidence):
    # Entailment case
    claim = "Paris is the capital of France."
    res = verifier.verify(claim, [sample_evidence[0]])
    
    assert isinstance(res, VerificationResult)
    assert res.claim == claim
    assert res.verdict == "SUPPORTS"
    assert "ev_1" in res.evidence_ids

def test_multi_evidence_verification(verifier, sample_evidence):
    # Should use both evidence pieces
    claim = "The Eiffel Tower is in the capital of France."
    res = verifier.verify(claim, sample_evidence)
    
    assert len(res.evidence_ids) == 2
    assert "ev_1" in res.evidence_ids
    assert "ev_2" in res.evidence_ids
    assert res.verdict == "SUPPORTS"

def test_probability_distribution_validity(verifier, sample_evidence):
    claim = "London is the capital of France." # Should refute
    res = verifier.verify(claim, [sample_evidence[0]])
    
    assert res.verdict == "REFUTES"
    assert res.refute_probability > 0.5
    assert res.refute_probability > res.support_probability
    assert res.refute_probability > res.nei_probability

def test_probability_sum(verifier, sample_evidence):
    claim = "Water is wet."
    res = verifier.verify(claim, sample_evidence)
    
    total = res.support_probability + res.refute_probability + res.nei_probability
    assert math.isclose(total, 1.0, rel_tol=1e-5)

def test_confidence_equals_predicted_class(verifier, sample_evidence):
    claim = "Paris is the capital of France."
    res = verifier.verify(claim, sample_evidence)
    
    probs = res.probabilities
    expected_conf = probs[res.verdict]
    
    assert math.isclose(res.confidence, expected_conf, rel_tol=1e-5)

def test_output_schema(verifier, sample_evidence):
    claim = "Test claim."
    res = verifier.verify(claim, sample_evidence)
    
    d = res.to_dict()
    assert "claim" in d
    assert "verdict" in d
    assert "support_probability" in d
    assert "refute_probability" in d
    assert "nei_probability" in d
    assert "confidence" in d
    assert "evidence_ids" in d

def test_empty_claim_raises(verifier, sample_evidence):
    with pytest.raises(ValueError, match="Claim cannot be empty"):
        verifier.verify("", sample_evidence)

def test_empty_evidence_handling(verifier):
    claim = "The sky is blue."
    res = verifier.verify(claim, [])
    
    assert res.verdict == "NOT_ENOUGH_INFO"
    assert res.confidence == 1.0
    assert res.nei_probability == 1.0
    assert len(res.evidence_ids) == 0

def test_repeated_calls_reuse_model(verifier, sample_evidence):
    claim = "Paris is a city."
    
    # First call
    res1 = verifier.verify(claim, sample_evidence)
    model_id1 = id(verifier.model)
    
    # Second call
    res2 = verifier.verify(claim, sample_evidence)
    model_id2 = id(verifier.model)
    
    assert model_id1 == model_id2  # Same object in memory
    assert math.isclose(res1.confidence, res2.confidence, rel_tol=1e-5)

def test_max_evidence_limit(verifier, sample_evidence):
    claim = "Paris is a city."
    
    # Create 5 evidence pieces
    many_evs = sample_evidence * 3  # 6 pieces
    
    res = verifier.verify(claim, many_evs)
    # The verifier max_evidence is set to 3 in the fixture
    assert len(res.evidence_ids) == 3
