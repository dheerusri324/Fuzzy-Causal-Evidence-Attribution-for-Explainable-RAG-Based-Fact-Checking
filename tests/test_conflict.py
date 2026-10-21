"""
Tests for Phase 8 Conflict Detection.
"""

import pytest

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from conflict.detector import ConflictDetector

@pytest.fixture(scope="module")
def verifier():
    cfg = Config(device="cpu")
    v = ClaimEvidenceVerifier(config=cfg)
    v._load_model()
    return v

@pytest.fixture(scope="module")
def detector(verifier):
    cfg = Config(conflict_role_threshold=0.5)
    return ConflictDetector(verifier=verifier, config=cfg)

def test_clearly_supporting_evidence(detector):
    claim = "The sky is blue."
    ev = RetrievedEvidence(
        evidence_id="e1", text="The sky is blue during the day.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""
    )
    res = detector.detect("c1", claim, [ev])
    
    assert not res.conflict_detected
    assert res.conflict_score == 0.0
    assert len(res.supporting_evidence_ids) == 1
    assert res.evidence_roles[0].assigned_role == "SUPPORTING"

def test_clearly_refuting_evidence(detector):
    claim = "The sky is green."
    ev = RetrievedEvidence(
        evidence_id="e2", text="The sky is completely blue and not green.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""
    )
    res = detector.detect("c2", claim, [ev])
    
    assert not res.conflict_detected
    assert res.conflict_score == 0.0
    assert len(res.refuting_evidence_ids) == 1
    assert res.evidence_roles[0].assigned_role == "REFUTING"

def test_mixed_evidence_conflict(detector):
    claim = "Apples are toxic."
    ev1 = RetrievedEvidence(
        evidence_id="e1", text="Apples are highly toxic and dangerous.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""
    )
    ev2 = RetrievedEvidence(
        evidence_id="e2", text="Apples are healthy and completely non-toxic.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=2, retrieval_method=""
    )
    res = detector.detect("c3", claim, [ev1, ev2])
    
    assert res.conflict_detected is True
    assert res.conflict_score > 0.0
    assert "e1" in res.supporting_evidence_ids
    assert "e2" in res.refuting_evidence_ids

def test_neutral_evidence(verifier):
    claim = "Apples are toxic."
    ev = RetrievedEvidence(
        evidence_id="e3", text="Apples grow on trees.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""
    )
    # Use a strict detector to ensure it falls to NEUTRAL if the model assigns ~0.6 support
    cfg = Config(conflict_role_threshold=0.99)
    strict_detector = ConflictDetector(verifier=verifier, config=cfg)
    res = strict_detector.detect("c4", claim, [ev])
    
    assert not res.conflict_detected
    assert res.conflict_score == 0.0
    assert len(res.neutral_evidence_ids) == 1
    assert res.evidence_roles[0].assigned_role == "NEUTRAL"

def test_conflict_score_range(detector):
    claim = "Mixed."
    ev1 = RetrievedEvidence(evidence_id="e1", text="Mixed is completely true.", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method="")
    ev2 = RetrievedEvidence(evidence_id="e2", text="Mixed is completely false.", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=1.0, retrieval_rank=2, retrieval_method="")
    res = detector.detect("c5", claim, [ev1, ev2])
    
    assert 0.0 <= res.conflict_score <= 1.0

def test_evidence_id_preservation(detector):
    claim = "Test"
    ev = RetrievedEvidence(evidence_id="magic_id_123", text="Test text.", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method="")
    res = detector.detect("c6", claim, [ev])
    
    assert res.evidence_roles[0].evidence_id == "magic_id_123"

def test_configurable_threshold_behavior(verifier):
    # A threshold so high that everything becomes neutral
    cfg = Config(conflict_role_threshold=0.9999)
    strict_detector = ConflictDetector(verifier=verifier, config=cfg)
    
    claim = "The sky is blue."
    ev = RetrievedEvidence(
        evidence_id="e1", text="The sky is blue.", doc_title="", sentence_idx=0,
        source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""
    )
    res = strict_detector.detect("c7", claim, [ev])
    # With typical probabilities (e.g. 0.95), 0.9999 threshold forces NEUTRAL
    assert res.evidence_roles[0].assigned_role == "NEUTRAL"

def test_empty_evidence(detector):
    res = detector.detect("c8", "claim", [])
    assert not res.conflict_detected
    assert res.conflict_score == 0.0
    assert len(res.evidence_roles) == 0
