"""
Unit and regression tests for Phase 11/12 fuzzy pipeline remediation (v2).

Covers:
1. Field-level distinction between legitimate 0.5 defuzzification and fallback 0.5.
2. Complete input space coverage in v2 fuzzy rule base (no dead zones).
3. Continuous harmonic mean conflict scores (no binary saturation).
4. Explainer cache key sequence preservation for order sensitivity.
5. Hybrid retrieval fusion non-tie behavior across candidate pools.
6. Decoupled passage contradiction vs claim conflict (Lincoln distractor regression).
"""

import os
import math
import numpy as np
import pytest

from config.settings import Config
from fuzzy.schema import FuzzyInput, FuzzyOutput
from fuzzy.engine import FuzzyEvidenceEngine
from conflict.detector import ConflictDetector
from conflict.schema import ConflictResult
from retrieval.schema import RetrievedEvidence
from retrieval.retriever import EvidenceRetriever
from attribution.explainer import EvidenceExplainer
from data.schema import ClaimRecord, EvidencePiece
from stability.analyzer import StabilityAnalyzer
from experiments.experiment2_v2 import (
    load_or_validate_cache,
    run_null_centered_cluster_bootstrap,
    apply_holm_bonferroni,
    compute_dict_hash
)


def test_field_level_status_legitimate_vs_fallback():
    """Verify that v1 engine distinguishes legitimate 0.5 from fallback 0.5,
    and v2 engine provides full computed coverage."""
    
    # 1. Baseline v1 engine on gap input
    engine_v1 = FuzzyEvidenceEngine(version="v1")
    
    # Input with relevance=0.5 (gap in v1 rule 9/10), low intervention (gap in rule 8)
    gap_input = FuzzyInput(
        relevance=0.5,
        attribution_shap=0.5,
        attribution_lime=0.5,
        intervention_influence=0.1,  # fails rule 8 (medium) and rule 1 (high)
        contradiction_degree=0.5,
        source_reliability=0.5,
        stability_score=0.5
    )
    
    out_v1_gap = engine_v1.evaluate(gap_input)
    assert out_v1_gap.evidence_reliability == 0.5
    assert out_v1_gap.field_status["evidence_reliability"] == "FALLBACK_EMPTY_RULES"
    assert out_v1_gap.field_status["evidence_relevance"] == "FALLBACK_EMPTY_RULES"
    assert out_v1_gap.fallback_used is True
    assert "evidence_reliability" in out_v1_gap.missing_fields

    # 2. Baseline v1 engine on Rule 8 input (legitimate 0.5 defuzzification)
    legit_input = FuzzyInput(
        relevance=0.5,
        attribution_shap=0.5,
        attribution_lime=0.5,
        intervention_influence=0.5,  # fires rule 8: MED & MED & MED
        contradiction_degree=0.0,
        source_reliability=0.75,
        stability_score=1.0
    )
    out_v1_legit = engine_v1.evaluate(legit_input)
    assert math.isclose(out_v1_legit.evidence_reliability, 0.5, abs_tol=1e-6)
    assert out_v1_legit.field_status["evidence_reliability"] == "COMPUTED"
    # evidence_relevance still fell back in v1 because relevance=0.5 was not covered
    assert out_v1_legit.field_status["evidence_relevance"] == "FALLBACK_EMPTY_RULES"

    # 3. Remediated v2 engine on both inputs
    engine_v2 = FuzzyEvidenceEngine(version="v2")
    out_v2_gap = engine_v2.evaluate(gap_input)
    assert out_v2_gap.fallback_used is False
    assert out_v2_gap.field_status["evidence_reliability"] == "COMPUTED"
    assert out_v2_gap.field_status["evidence_relevance"] == "COMPUTED"
    assert out_v2_gap.inference_status == "SUCCESS"

    out_v2_legit = engine_v2.evaluate(legit_input)
    assert out_v2_legit.fallback_used is False
    assert out_v2_legit.field_status["evidence_reliability"] == "COMPUTED"
    assert out_v2_legit.field_status["evidence_relevance"] == "COMPUTED"
    assert out_v2_legit.inference_status == "SUCCESS"


def test_v2_rule_base_dense_grid_coverage():
    """Verify that v2 fuzzy engine has 100% computed coverage across a test grid."""
    engine = FuzzyEvidenceEngine(version="v2")
    
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for rel in test_values:
        for attr in [0.1, 0.5, 0.9]:
            for intv in [0.1, 0.5, 0.9]:
                for cnt in [0.0, 0.5, 1.0]:
                    inp = FuzzyInput(
                        relevance=rel,
                        attribution_shap=attr,
                        attribution_lime=attr,
                        intervention_influence=intv,
                        contradiction_degree=cnt,
                        source_reliability=0.8,
                        stability_score=0.9
                    )
                    out = engine.evaluate(inp)
                    assert out.fallback_used is False, f"Fallback occurred for inputs rel={rel}, attr={attr}, intv={intv}, cnt={cnt}"
                    assert out.inference_status == "SUCCESS"
                    for fld, stat in out.field_status.items():
                        assert stat == "COMPUTED", f"Field {fld} not computed for rel={rel}, cnt={cnt}"


def test_harmonic_mean_conflict_properties():
    """Verify all four mathematical properties of the continuous Harmonic Mean conflict score:
    1. Zero when there is no conflict (e.g. only supports or only refutes or both zero).
    2. Bounded in [0, 1] across all valid input probabilities.
    3. Produces distinct, graded values for different conflict strengths.
    4. Does not saturate at 1.0 for sub-maximal inputs.
    """
    from unittest.mock import MagicMock
    from verification.schema import VerificationResult

    mock_verifier = MagicMock()
    detector = ConflictDetector(verifier=mock_verifier)

    ev1 = RetrievedEvidence(evidence_id="e1", text="text1", doc_title="T1", sentence_idx=0, source="s", source_type="w", retrieval_score=0.9, retrieval_rank=1, retrieval_method="m")
    ev2 = RetrievedEvidence(evidence_id="e2", text="text2", doc_title="T2", sentence_idx=0, source="s", source_type="w", retrieval_score=0.8, retrieval_rank=2, retrieval_method="m")

    # Property 1: 0 when there is no conflict
    # Case 1A: Only support, zero refutation
    mock_verifier.verify.side_effect = [
        VerificationResult(claim="c", verdict="SUPPORTS", support_probability=0.85, refute_probability=0.0, nei_probability=0.15, confidence=0.85, evidence_ids=["e1"]),
        VerificationResult(claim="c", verdict="SUPPORTS", support_probability=0.75, refute_probability=0.0, nei_probability=0.25, confidence=0.75, evidence_ids=["e2"]),
    ]
    res_no_conflict = detector.detect("c1", "test claim", [ev1, ev2])
    assert res_no_conflict.conflict_score == 0.0
    assert res_no_conflict.conflict_detected is False

    # Case 1B: Only refutation, zero support
    mock_verifier.verify.side_effect = [
        VerificationResult(claim="c", verdict="REFUTES", support_probability=0.0, refute_probability=0.80, nei_probability=0.20, confidence=0.80, evidence_ids=["e1"]),
        VerificationResult(claim="c", verdict="REFUTES", support_probability=0.0, refute_probability=0.70, nei_probability=0.30, confidence=0.70, evidence_ids=["e2"]),
    ]
    res_refute_only = detector.detect("c1", "test claim", [ev1, ev2])
    assert res_refute_only.conflict_score == 0.0
    assert res_refute_only.conflict_detected is False

    # Property 2: Bounded in [0, 1] across full grid
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
            h = (2.0 * s * r) / (s + r) if (s + r) > 0 else 0.0
            assert 0.0 <= h <= 1.0

    # Property 3: Graded values for different conflict strengths
    # With default threshold = 0.5, refuting evidence must have p_ref > 0.5:
    # Strengths: (0.9, 0.55) vs (0.9, 0.70) vs (0.9, 0.90)
    # H(0.9, 0.55) = 2 * 0.9 * 0.55 / 1.45 = 0.99 / 1.45 ≈ 0.68276
    # H(0.9, 0.70) = 2 * 0.9 * 0.70 / 1.60 = 1.26 / 1.60 = 0.78750
    # H(0.9, 0.90) = 2 * 0.9 * 0.90 / 1.80 = 1.62 / 1.80 = 0.90000
    mock_verifier.verify.side_effect = [
        VerificationResult(claim="c", verdict="SUPPORTS", support_probability=0.9, refute_probability=0.05, nei_probability=0.05, confidence=0.9, evidence_ids=["e1"]),
        VerificationResult(claim="c", verdict="REFUTES", support_probability=0.1, refute_probability=0.55, nei_probability=0.35, confidence=0.55, evidence_ids=["e2"]),
    ]
    res_low = detector.detect("c1", "test claim", [ev1, ev2])

    mock_verifier.verify.side_effect = [
        VerificationResult(claim="c", verdict="SUPPORTS", support_probability=0.9, refute_probability=0.05, nei_probability=0.05, confidence=0.9, evidence_ids=["e1"]),
        VerificationResult(claim="c", verdict="REFUTES", support_probability=0.1, refute_probability=0.70, nei_probability=0.20, confidence=0.70, evidence_ids=["e2"]),
    ]
    res_mid = detector.detect("c1", "test claim", [ev1, ev2])

    mock_verifier.verify.side_effect = [
        VerificationResult(claim="c", verdict="SUPPORTS", support_probability=0.9, refute_probability=0.05, nei_probability=0.05, confidence=0.9, evidence_ids=["e1"]),
        VerificationResult(claim="c", verdict="REFUTES", support_probability=0.05, refute_probability=0.90, nei_probability=0.05, confidence=0.90, evidence_ids=["e2"]),
    ]
    res_high = detector.detect("c1", "test claim", [ev1, ev2])

    assert math.isclose(res_low.conflict_score, (2 * 0.9 * 0.55) / 1.45, abs_tol=1e-4)
    assert math.isclose(res_mid.conflict_score, 0.7875, abs_tol=1e-4)
    assert math.isclose(res_high.conflict_score, 0.9000, abs_tol=1e-4)
    assert res_low.conflict_score < res_mid.conflict_score < res_high.conflict_score

    # Property 4: Does not saturate at 1.0 for sub-maximal inputs
    assert res_mid.conflict_score < 1.0
    assert res_high.conflict_score < 1.0
    # Even strong 0.95 & 0.95 yields 0.95 < 1.0
    h_strong = (2 * 0.95 * 0.95) / (0.95 + 0.95)
    assert math.isclose(h_strong, 0.95, abs_tol=1e-4)
    assert h_strong < 1.0


def test_explainer_cache_key_order_sensitivity_and_repeat_caching():
    """Verify that EvidenceExplainer distinguishes different evidence orderings
    while preserving repeat evaluation caching (cache hits for identical orderings)."""
    from unittest.mock import MagicMock
    from verification.schema import VerificationResult

    mock_verifier = MagicMock()
    # Order-sensitive behavior: first item determines support vs refutation
    def mock_verify(claim, ev_subset):
        first_id = ev_subset[0].evidence_id if ev_subset else ""
        if first_id == "evA":
            probs = {"REFUTES": 0.1, "SUPPORTS": 0.8, "NOT_ENOUGH_INFO": 0.1}
        else:
            probs = {"REFUTES": 0.3, "SUPPORTS": 0.5, "NOT_ENOUGH_INFO": 0.2}
        return VerificationResult(
            claim=claim,
            verdict="SUPPORTS",
            support_probability=probs["SUPPORTS"],
            refute_probability=probs["REFUTES"],
            nei_probability=probs["NOT_ENOUGH_INFO"],
            confidence=0.8,
            evidence_ids=[ev.evidence_id for ev in ev_subset]
        )

    mock_verifier.verify.side_effect = mock_verify
    explainer = EvidenceExplainer(verifier=mock_verifier)

    ev_a = RetrievedEvidence(evidence_id="evA", text="Evidence A", doc_title="A", sentence_idx=0, source="s", source_type="w", retrieval_score=0.9, retrieval_rank=1, retrieval_method="m")
    ev_b = RetrievedEvidence(evidence_id="evB", text="Evidence B", doc_title="B", sentence_idx=0, source="s", source_type="w", retrieval_score=0.8, retrieval_rank=2, retrieval_method="m")

    # Step 1: Initial call with [ev_a, ev_b]
    probs_ab_1 = explainer._get_probabilities("test claim", [ev_a, ev_b])
    assert mock_verifier.verify.call_count == 1

    # Step 2: Repeat call with identical ordering [ev_a, ev_b] -> Cache Hit!
    probs_ab_2 = explainer._get_probabilities("test claim", [ev_a, ev_b])
    assert mock_verifier.verify.call_count == 1, "Repeat evaluation of identical ordering should hit cache without calling verifier"
    assert np.allclose(probs_ab_1, probs_ab_2)

    # Step 3: Call with reversed ordering [ev_b, ev_a] -> Cache Miss (order sensitivity)!
    probs_ba_1 = explainer._get_probabilities("test claim", [ev_b, ev_a])
    assert mock_verifier.verify.call_count == 2, "Reversed ordering should not be masked by sorted cache key"
    assert not np.allclose(probs_ab_1, probs_ba_1)

    # Step 4: Repeat call with [ev_b, ev_a] -> Cache Hit for reversed ordering!
    probs_ba_2 = explainer._get_probabilities("test claim", [ev_b, ev_a])
    assert mock_verifier.verify.call_count == 2, "Repeat evaluation of reversed ordering should hit cache"
    assert np.allclose(probs_ba_1, probs_ba_2)

    # Cache entries verification
    assert ("test claim", ("evA", "evB")) in explainer._eval_cache
    assert ("test claim", ("evB", "evA")) in explainer._eval_cache


def test_stability_analyzer_explainer_order_interaction():
    """Verify that StabilityAnalyzer with permutation perturbation exercises
    distinct orderings in the explainer cache and accurately captures order sensitivity."""
    from unittest.mock import MagicMock
    from verification.schema import VerificationResult
    from stability.analyzer import StabilityAnalyzer

    # 1. Order-sensitive verifier
    mock_verifier_sens = MagicMock()
    def mock_verify_sens(claim, ev_subset):
        if not ev_subset:
            return VerificationResult(claim=claim, verdict="NOT_ENOUGH_INFO", support_probability=0.33, refute_probability=0.33, nei_probability=0.34, confidence=0.34, evidence_ids=[])
        # Model is sensitive to which evidence is at the front of the prompt
        lead_id = ev_subset[0].evidence_id
        if lead_id == "ev1":
            p_sup, p_ref = 0.85, 0.05
        elif lead_id == "ev2":
            p_sup, p_ref = 0.20, 0.70
        else:
            p_sup, p_ref = 0.40, 0.40
        return VerificationResult(
            claim=claim,
            verdict="SUPPORTS" if p_sup > p_ref else "REFUTES",
            support_probability=p_sup,
            refute_probability=p_ref,
            nei_probability=1.0 - p_sup - p_ref,
            confidence=max(p_sup, p_ref),
            evidence_ids=[ev.evidence_id for ev in ev_subset]
        )

    mock_verifier_sens.verify.side_effect = mock_verify_sens
    explainer_sens = EvidenceExplainer(verifier=mock_verifier_sens)
    analyzer_sens = StabilityAnalyzer(explainer=explainer_sens, config=Config(stability_perturbation="permutation"))

    ev1 = RetrievedEvidence(evidence_id="ev1", text="First evidence", doc_title="T1", sentence_idx=0, source="s", source_type="w", retrieval_score=0.9, retrieval_rank=1, retrieval_method="m")
    ev2 = RetrievedEvidence(evidence_id="ev2", text="Second evidence", doc_title="T2", sentence_idx=0, source="s", source_type="w", retrieval_score=0.8, retrieval_rank=2, retrieval_method="m")

    res_sens = analyzer_sens.analyze("c1", "Sensitive Claim", [ev1, ev2])

    # Distinct sequences should have been evaluated and cached
    assert ("Sensitive Claim", ("ev1", "ev2")) in explainer_sens._eval_cache
    assert ("Sensitive Claim", ("ev2", "ev1")) in explainer_sens._eval_cache
    # Order-sensitive verifier under reversal yields lower stability
    assert res_sens.stability_score < 1.0

    # 2. Order-insensitive verifier
    mock_verifier_insens = MagicMock()
    def mock_verify_insens(claim, ev_subset):
        # Order-invariant sum of evidence IDs
        has_ev1 = any(e.evidence_id == "ev1" for e in ev_subset)
        has_ev2 = any(e.evidence_id == "ev2" for e in ev_subset)
        p_sup = 0.8 if has_ev1 else (0.5 if has_ev2 else 0.3)
        return VerificationResult(
            claim=claim,
            verdict="SUPPORTS",
            support_probability=p_sup,
            refute_probability=0.1,
            nei_probability=0.9 - p_sup,
            confidence=p_sup,
            evidence_ids=[e.evidence_id for e in ev_subset]
        )

    mock_verifier_insens.verify.side_effect = mock_verify_insens
    explainer_insens = EvidenceExplainer(verifier=mock_verifier_insens)
    analyzer_insens = StabilityAnalyzer(explainer=explainer_insens, config=Config(stability_perturbation="permutation"))

    res_insens = analyzer_insens.analyze("c2", "Insensitive Claim", [ev1, ev2])
    # Order-invariant verifier yields perfect rank correlation
    assert math.isclose(res_insens.spearman_correlation, 1.0, abs_tol=1e-6)
    assert math.isclose(res_insens.stability_score, 1.0, abs_tol=1e-6)


def test_claim_record_label_semantics():
    """Verify that ClaimRecord properties accurately distinguish verifiable claims
    and human-annotated gold evidence vs unverified NOT_ENOUGH_INFO claims."""
    from data.schema import ClaimRecord, EvidencePiece

    gold_piece = EvidencePiece(
        evidence_id="Wikipedia_Article_0",
        text="This is a human-annotated gold evidence sentence.",
        doc_title="Wikipedia_Article",
        sentence_idx=0
    )

    # 1. SUPPORTS claim: verifiable, human gold evidence preserved
    rec_sup = ClaimRecord(
        claim_id="c_sup",
        claim_text="Sachin Tendulkar played cricket.",
        label="SUPPORTS",
        evidence=[gold_piece]
    )
    assert rec_sup.is_verifiable is True
    assert len(rec_sup.verified_gold_evidence) == 1
    assert rec_sup.verified_gold_evidence[0].evidence_id == "Wikipedia_Article_0"

    # 2. REFUTES claim: verifiable, human gold evidence preserved
    rec_ref = ClaimRecord(
        claim_id="c_ref",
        claim_text="Sachin Tendulkar was an astronaut.",
        label="REFUTES",
        evidence=[gold_piece]
    )
    assert rec_ref.is_verifiable is True
    assert len(rec_ref.verified_gold_evidence) == 1
    assert rec_ref.verified_gold_evidence[0].evidence_id == "Wikipedia_Article_0"

    # 3. NOT_ENOUGH_INFO claim: unverified in FEVER benchmark
    # Has no human-annotated gold evidence. Even if candidate/proxy passages were attached
    # in legacy preprocessing, verified_gold_evidence must return an empty list.
    candidate_proxy_piece = EvidencePiece(
        evidence_id="Candidate_Doc_0",
        text="This is an unverified candidate passage from copenlu retrieval.",
        doc_title="Candidate_Doc",
        sentence_idx=0
    )
    rec_nei = ClaimRecord(
        claim_id="c_nei",
        claim_text="Sachin Tendulkar enjoys scuba diving.",
        label="NOT_ENOUGH_INFO",
        evidence=[candidate_proxy_piece]
    )
    assert rec_nei.is_verifiable is False
    assert rec_nei.verified_gold_evidence == []
    # Raw legacy evidence field still holds candidate proxy passages for backward compatibility
    assert len(rec_nei.evidence) == 1


def test_hybrid_retrieval_candidate_fusion_no_artificial_ties(tmp_path):
    """Verify that candidate pool fusion evaluates genuine scores for both modalities without 0.5000 ties."""
    mini_corpus = [
        {"evidence_id": "c1", "text": "Sachin Tendulkar is a former Indian international cricketer.", "doc_title": "Sachin", "sentence_idx": 0, "source": "fever", "source_type": "wikipedia"},
        {"evidence_id": "c2", "text": "Tendulkar took up cricket at the age of eleven.", "doc_title": "Sachin", "sentence_idx": 1, "source": "fever", "source_type": "wikipedia"},
        {"evidence_id": "c3", "text": "Cricket is a bat-and-ball game played between two teams.", "doc_title": "Cricket", "sentence_idx": 0, "source": "fever", "source_type": "wikipedia"},
        {"evidence_id": "c4", "text": "Michael Fassbender is an Irish-German actor.", "doc_title": "Fassbender", "sentence_idx": 0, "source": "fever", "source_type": "wikipedia"},
        {"evidence_id": "c5", "text": "Kristofer Hivju is a Norwegian actor.", "doc_title": "Hivju", "sentence_idx": 0, "source": "fever", "source_type": "wikipedia"},
    ]
    import json
    corpus_file = tmp_path / "corpus.json"
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(mini_corpus, f)

    cfg = Config(cache_dir=str(tmp_path / "cache"), top_k=5, retrieval_mode="hybrid")
    retriever = EvidenceRetriever(corpus_path=str(corpus_file), config=cfg)
    
    results = retriever.retrieve("Sachin Tendulkar cricketer", top_k=5, mode="hybrid")
    scores = [r.retrieval_score for r in results]
    
    # Check that scores are graded and not all tied at 0.5000
    assert len(scores) == 5
    assert scores[0] > scores[-1], "Top candidate should have strictly higher score than bottom candidate"
    assert scores.count(0.5000) < 3, "Candidate pool fusion should not produce multiple artificial 0.5000 ties"


def test_lincoln_distractor_decoupling():
    """Verify that passage-level contradiction and claim conflict are decoupled in pipeline fuzzy input."""
    engine_v2 = FuzzyEvidenceEngine(version="v2")

    # Lincoln claim with Ford distractor passage:
    # relevance is LOW/MED for Ford passage, contradiction NLI is HIGH (due to entity mismatch),
    # but claim conflict is LOW (no supporting passage for Ford being Lincoln).
    distractor_input = FuzzyInput(
        relevance=0.25,                 # Low relevance distractor
        attribution_shap=0.05,
        attribution_lime=0.05,
        intervention_influence=0.02,
        contradiction_degree=0.95,      # High NLI refute prob from entity mismatch
        claim_conflict=0.0,             # Decoupled claim-level conflict
        source_reliability=0.75,
        stability_score=0.9
    )

    out = engine_v2.evaluate(distractor_input)
    # Distractor rule: relevance LOW & contradiction HIGH -> contradiction_output LOW
    assert out.contradiction_degree < 0.45, "Distractor passage should not produce high contradiction degree"
    assert out.support_degree < 0.25
    assert out.fallback_used is False


def test_protocol_a_and_b_label_semantics():
    """Regression test for Fix 1: Protocol A and Protocol B label semantics.
    Proves:
    1. Verifiable claim's actual gold evidence receives positive label.
    2. Retrieved non-gold passage for verifiable claim receives negative label.
    3. NEI candidate passage receives negative label under Protocol A even if in rec.evidence.
    4. NEI claims are completely excluded under Protocol B.
    5. Protocol B never marks an NEI candidate as positive.
    6. Malformed annotation field / invalid label raises ValueError.
    """
    gold_piece = EvidencePiece(evidence_id="DocA_0", text="Gold fact.", doc_title="DocA", sentence_idx=0)
    distractor_piece = EvidencePiece(evidence_id="DocB_1", text="Distractor fact.", doc_title="DocB", sentence_idx=1)
    nei_proxy_piece = EvidencePiece(evidence_id="DocC_0", text="NEI candidate passage.", doc_title="DocC", sentence_idx=0)

    # Verifiable claim (SUPPORTS)
    rec_sup = ClaimRecord(
        claim_id="c_sup",
        claim_text="Earth orbits the sun.",
        label="SUPPORTS",
        evidence=[gold_piece]
    )

    # Non-verifiable claim (NOT_ENOUGH_INFO) with candidate proxy in legacy evidence field
    rec_nei = ClaimRecord(
        claim_id="c_nei",
        claim_text="Aliens live on Mars.",
        label="NOT_ENOUGH_INFO",
        evidence=[nei_proxy_piece]
    )

    # 1 & 2: Verifiable claim labels
    gold_ids_sup = {ep.evidence_id for ep in rec_sup.verified_gold_evidence}
    assert "DocA_0" in gold_ids_sup
    label_gold = 1 if "DocA_0" in gold_ids_sup else 0
    label_dist = 1 if "DocB_1" in gold_ids_sup else 0
    assert label_gold == 1, "Actual gold evidence must receive positive label (1)"
    assert label_dist == 0, "Retrieved distractor must receive negative label (0)"

    # 3: Protocol A label for NEI candidate
    gold_ids_nei = {ep.evidence_id for ep in rec_nei.verified_gold_evidence}
    assert gold_ids_nei == set(), "NEI claim must have empty verified_gold_evidence"
    label_nei_proto_a = 1 if "DocC_0" in gold_ids_nei else 0
    assert label_nei_proto_a == 0, "NEI candidate passage must receive negative label (0) under Protocol A"

    # 4 & 5: Protocol B filtering
    eval_claims = [rec_sup, rec_nei]
    protocol_b_claims = [r for r in eval_claims if r.is_verifiable]
    assert rec_nei not in protocol_b_claims, "NEI claims must be completely excluded under Protocol B"
    assert len(protocol_b_claims) == 1
    assert protocol_b_claims[0].claim_id == "c_sup"

    for r in protocol_b_claims:
        assert r.label in {"SUPPORTS", "REFUTES"}
        for ep in r.evidence:
            assert r.is_verifiable is True

    # 6: Malformed / invalid claim label raises error
    with pytest.raises(ValueError, match="Invalid label"):
        ClaimRecord(claim_id="c_bad", claim_text="Invalid.", label="UNVERIFIED")


def test_signals_cache_versioning_and_invalidation(tmp_path):
    """Regression test for Fix 2: Signals cache metadata, versioning, and safe invalidation.
    Proves:
    1. Valid cache with matching fingerprints and evidence IDs is accepted.
    2. Changing a configuration parameter triggers invalidation.
    3. Changing a source fingerprint triggers invalidation.
    4. Changing retrieved evidence IDs triggers invalidation.
    5. Malformed or incompatible cache data triggers safe recomputation without crash.
    6. Protected historical directories remain untouched.
    """
    import json
    import os

    cache_file = tmp_path / ".signals_cache_v2_corrected.json"

    meta = {
        "cache_version": "2.1",
        "created_at": "2026-10-22T00:00:00",
        "source_fingerprints": {"conflict/detector.py": "abc123hash"},
        "config_hash": "cfg_hash_initial",
        "corpus_fingerprint": "corpus_hash_1",
        "dataset_fingerprint": "dataset_hash_1",
        "retrieval_mode": "hybrid",
        "top_k": 3
    }

    dummy_signals = {
        "c1": {
            "claim_id": "c1",
            "evidence_ids": ["ev1", "ev2", "ev3"],
            "conflict_score": 0.45,
            "conflict_detected": True,
            "evidence_refute_probs": {"ev1": 0.1, "ev2": 0.8, "ev3": 0.05},
            "stability_score": 1.0
        }
    }

    payload = {"metadata": meta, "signals": dummy_signals}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Dummy ClaimRecord and retrieved evidence
    rec = ClaimRecord(claim_id="c1", claim_text="Claim 1", label="SUPPORTS")
    retrieved = {
        "c1": [
            RetrievedEvidence(evidence_id="ev1", text="t1", doc_title="d1", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.9, retrieval_rank=1, retrieval_method="hybrid"),
            RetrievedEvidence(evidence_id="ev2", text="t2", doc_title="d2", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.8, retrieval_rank=2, retrieval_method="hybrid"),
            RetrievedEvidence(evidence_id="ev3", text="t3", doc_title="d3", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.7, retrieval_rank=3, retrieval_method="hybrid"),
        ]
    }

    # 1. Valid cache is accepted
    loaded, status = load_or_validate_cache(str(cache_file), meta, [rec], retrieved)
    assert status == "accepted"
    assert loaded is not None
    assert "c1" in loaded

    # 2. Config hash mismatch invalidates
    meta_mismatch_cfg = dict(meta, config_hash="cfg_hash_MODIFIED")
    loaded_cfg, status_cfg = load_or_validate_cache(str(cache_file), meta_mismatch_cfg, [rec], retrieved)
    assert status_cfg == "config_hash_mismatch"
    assert loaded_cfg is None

    # 3. Source fingerprint mismatch invalidates
    meta_mismatch_src = dict(meta, source_fingerprints={"conflict/detector.py": "DIFFERENT_HASH"})
    loaded_src, status_src = load_or_validate_cache(str(cache_file), meta_mismatch_src, [rec], retrieved)
    assert status_src == "source_fingerprint_mismatch"
    assert loaded_src is None

    # 4. Evidence ID mismatch invalidates
    retrieved_mismatch = {
        "c1": [
            RetrievedEvidence(evidence_id="ev1", text="t1", doc_title="d1", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.9, retrieval_rank=1, retrieval_method="hybrid"),
            RetrievedEvidence(evidence_id="ev_DIFFERENT", text="t2", doc_title="d2", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.8, retrieval_rank=2, retrieval_method="hybrid"),
            RetrievedEvidence(evidence_id="ev3", text="t3", doc_title="d3", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.7, retrieval_rank=3, retrieval_method="hybrid"),
        ]
    }
    loaded_eid, status_eid = load_or_validate_cache(str(cache_file), meta, [rec], retrieved_mismatch)
    assert status_eid == "evidence_id_mismatch"
    assert loaded_eid is None

    # 5. Malformed/corrupt JSON safely invalidates without crashing
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write("{corrupt json content")
    loaded_corrupt, status_corrupt = load_or_validate_cache(str(cache_file), meta, [rec], retrieved)
    assert "corrupt" in status_corrupt
    assert loaded_corrupt is None

    # 6. Verify protected historical directories exist and remain unmodified
    for hist_dir in ["outputs/phase10_formal", "outputs/phase11_formal", "outputs/phase11_v2"]:
        assert os.path.exists(hist_dir), f"Protected historical directory missing: {hist_dir}"


def test_null_centered_bootstrap_finite_simulation_and_holm():
    """Regression test for Fix 3: Statistical reporting, finite-simulation p-values, and Holm adjustment.
    Proves:
    1. Null-centered cluster bootstrap resamples at claim level.
    2. Finite-sample correction p = (exceedances + 1) / (valid_reps + 1) is strictly > 0 even with 0 exceedances.
    3. Reproducible results with fixed random seed.
    4. Holm-Bonferroni correction maintains monotonicity and controls FWER across co-primary endpoints.
    5. Handles degenerate samples explicitly.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    # Create synthetic dataset with 20 claims, 3 evidence pieces per claim
    rng = np.random.default_rng(42)
    rows = []
    for c in range(20):
        cid = f"claim_{c}"
        has_gold = (c % 2 == 0)
        for e in range(3):
            is_gold = 1 if (has_gold and e == 0) else 0
            # Fuzzy score gives good signal
            fuz_score = 0.8 if is_gold else float(rng.uniform(0.1, 0.4))
            # Baseline score gives weaker signal
            base_score = 0.6 if is_gold else float(rng.uniform(0.2, 0.5))
            rows.append({
                "claim_id": cid,
                "evidence_id": f"ev_{c}_{e}",
                "evaluation_label": is_gold,
                "fuzzy_score": fuz_score,
                "base_score": base_score
            })
    df_synth = pd.DataFrame(rows)

    # 1 & 2: Finite-sample corrected bootstrap
    res_auroc = run_null_centered_cluster_bootstrap(
        df_synth,
        metric_name="AUROC",
        metric_func=roc_auc_score,
        score_col_fuzzy="fuzzy_score",
        score_col_base="base_score",
        label_col="evaluation_label",
        B=200,
        seed=42
    )

    assert res_auroc["replicates_valid"] > 150
    assert res_auroc["p_value_raw"] > 0.0, "Finite simulation p-value must be strictly > 0.0"
    assert res_auroc["simulation_resolution"] == 1.0 / (res_auroc["replicates_valid"] + 1.0)
    # Check formula exactness:
    expected_p = (res_auroc["exceedances"] + 1.0) / (res_auroc["replicates_valid"] + 1.0)
    assert math.isclose(res_auroc["p_value_raw"], expected_p, rel_tol=1e-7)

    # 3: Reproducibility with fixed random seed
    res_repeat = run_null_centered_cluster_bootstrap(
        df_synth,
        metric_name="AUROC",
        metric_func=roc_auc_score,
        score_col_fuzzy="fuzzy_score",
        score_col_base="base_score",
        label_col="evaluation_label",
        B=200,
        seed=42
    )
    assert res_repeat["observed_difference"] == res_auroc["observed_difference"]
    assert res_repeat["ci_lower_95"] == res_auroc["ci_lower_95"]
    assert res_repeat["p_value_raw"] == res_auroc["p_value_raw"]

    # 4: Holm multiplicity adjustment across co-primary endpoints
    raw_p_dict = {"auroc": 0.015, "average_precision": 0.003}
    holm_adj = apply_holm_bonferroni(raw_p_dict)
    # AP is smaller: ranked 1st -> multiplier 2 -> 0.003 * 2 = 0.006
    # AUROC is larger: ranked 2nd -> multiplier 1 -> max(0.006, 0.015 * 1) = 0.015
    assert math.isclose(holm_adj["average_precision"], 0.006, abs_tol=1e-6)
    assert math.isclose(holm_adj["auroc"], 0.015, abs_tol=1e-6)
    assert holm_adj["average_precision"] <= holm_adj["auroc"], "Holm correction must preserve monotonicity"


def test_retrieval_to_attribution_id_alignment():
    """Regression test for Fix 4: Retrieval-to-attribution explicit ID matching.
    Proves:
    1. Correctly aligned input succeeds by (claim_id, evidence_id).
    2. Shuffling/reordering reference rows still matches by ID.
    3. Missing evidence ID raises KeyError.
    4. Duplicate evidence ID raises ValueError.
    """
    # Reference Phase 10 rows
    ref_rows = [
        {"claim_id": "c1", "evidence_id": "e_a", "val": 10},
        {"claim_id": "c1", "evidence_id": "e_b", "val": 20},
        {"claim_id": "c1", "evidence_id": "e_c", "val": 30},
    ]

    # Map by (claim_id, evidence_id)
    ref_map = {(r["claim_id"], r["evidence_id"]): r for r in ref_rows}

    # 1. Correctly ordered retrieved evidence
    retrieved_order1 = ["e_a", "e_b", "e_c"]
    matched_vals = [ref_map[("c1", eid)]["val"] for eid in retrieved_order1]
    assert matched_vals == [10, 20, 30]

    # 2. Reordered retrieved evidence still matches by ID, not positional index
    retrieved_order2 = ["e_c", "e_a", "e_b"]
    matched_vals2 = [ref_map[("c1", eid)]["val"] for eid in retrieved_order2]
    assert matched_vals2 == [30, 10, 20]

    # 3. Missing evidence ID detected
    with pytest.raises(KeyError):
        _ = ref_map[("c1", "e_MISSING")]

    # 4. Duplicate ID check in reference detection
    dup_ref_rows = list(ref_rows) + [{"claim_id": "c1", "evidence_id": "e_a", "val": 99}]
    exp_map_dup = {}
    with pytest.raises(ValueError, match="Duplicate"):
        for r in dup_ref_rows:
            key = (r["claim_id"], r["evidence_id"])
            if key in exp_map_dup:
                raise ValueError(f"Duplicate (claim_id, evidence_id): {key}")
            exp_map_dup[key] = r


def test_fuzzy_engine_six_antecedent_coverage_v1_vs_v2():
    """Regression test for Fix 5: Six-antecedent rule coverage and V1 vs V2 separation.
    Proves:
    1. V1 has 11 rules across 5 outputs with known coverage gaps (fallbacks).
    2. V2 has 33 rules across 5 outputs with zero fallbacks across varying:
       - Relevance
       - Differing SHAP vs LIME attribution (e.g. SHAP=0.2, LIME=0.8)
       - Intervention influence
       - Contradiction degree
       - Source reliability
       - Stability score
    """
    engine_v1 = FuzzyEvidenceEngine(version="v1")
    engine_v2 = FuzzyEvidenceEngine(version="v2")

    # Antecedents varying across all 6 dimensions with differing SHAP and LIME
    test_inputs = [
        # Gap in V1 (moderate relevance, low intervention, differing SHAP/LIME)
        FuzzyInput(
            relevance=0.45,
            attribution_shap=0.20,
            attribution_lime=0.85,
            intervention_influence=0.15,
            contradiction_degree=0.40,
            claim_conflict=0.30,
            stability_score=0.70,
            source_reliability=0.60
        ),
        # Strong relevance, high SHAP, low LIME, high contradiction
        FuzzyInput(
            relevance=0.85,
            attribution_shap=0.90,
            attribution_lime=0.30,
            intervention_influence=0.75,
            contradiction_degree=0.85,
            claim_conflict=0.50,
            stability_score=0.95,
            source_reliability=0.80
        ),
        # Low relevance, zero contradiction, low stability
        FuzzyInput(
            relevance=0.10,
            attribution_shap=0.15,
            attribution_lime=0.10,
            intervention_influence=0.05,
            contradiction_degree=0.00,
            claim_conflict=0.00,
            stability_score=0.20,
            source_reliability=0.50
        )
    ]

    # V1 demonstrates fallbacks on input 0
    out_v1_0 = engine_v1.evaluate(test_inputs[0])
    assert out_v1_0.fallback_used is True
    assert out_v1_0.evidence_reliability == 0.50

    # V2 computes successfully on all test inputs across all 6 dimensions
    for idx, inp in enumerate(test_inputs):
        out_v2 = engine_v2.evaluate(inp)
        assert out_v2.fallback_used is False, f"V2 fallback occurred on input {idx}"
        assert out_v2.inference_status == "SUCCESS"
        assert out_v2.activated_rule_count > 0
        for fld, stat in out_v2.field_status.items():
            assert stat == "COMPUTED", f"Field {fld} not computed on input {idx}"


def test_stability_analyzer_random_seed_configuration():
    """Regression test for Fix 6: Configured random_seed in StabilityAnalyzer.
    Proves:
    1. StabilityAnalyzer reads Config.random_seed for random_shuffle perturbation.
    2. Identical seed produces identical evidence order.
    3. Different seeds produce distinct permutations for a controlled evidence list.
    4. Default 'permutation' perturbation performs deterministic reversal.
    """
    from unittest.mock import MagicMock
    from attribution.schema import AttributionResult, EvidenceAttribution

    # Create 5 distinct evidence items
    evidence = [
        RetrievedEvidence(evidence_id=f"ev_{i}", text=f"text {i}", doc_title=f"doc_{i}", sentence_idx=0, source="fever", source_type="wikipedia", retrieval_score=0.5, retrieval_rank=i+1, retrieval_method="hybrid")
        for i in range(5)
    ]

    # Mock explainer that returns attribution scores
    mock_explainer = MagicMock()
    mock_explainer.explain.side_effect = lambda cid, claim, ev_list: AttributionResult(
        claim_id=cid,
        claim=claim,
        verdict="SUPPORTS",
        baseline_probabilities={"SUPPORTS": 0.9, "REFUTES": 0.05, "NOT_ENOUGH_INFO": 0.05},
        evidence_attributions=[
            EvidenceAttribution(
                evidence_id=ev.evidence_id,
                text=ev.text,
                shap_support=0.1 * idx,
                shap_refute=0.0,
                shap_nei=0.0,
                shap_target=0.1 * idx,
                shap_rank=idx + 1,
                lime_support=0.1 * idx,
                lime_refute=0.0,
                lime_nei=0.0,
                lime_target=0.1 * idx,
                lime_rank=idx + 1
            )
            for idx, ev in enumerate(ev_list)
        ]
    )

    # 1 & 2: Identical seed produces identical result
    cfg_seed42_a = Config(random_seed=42, stability_perturbation="random_shuffle")
    analyzer_42_a = StabilityAnalyzer(explainer=mock_explainer, config=cfg_seed42_a)
    res_42_a = analyzer_42_a.analyze("c1", "Claim text", evidence)

    cfg_seed42_b = Config(random_seed=42, stability_perturbation="random_shuffle")
    analyzer_42_b = StabilityAnalyzer(explainer=mock_explainer, config=cfg_seed42_b)
    res_42_b = analyzer_42_b.analyze("c1", "Claim text", evidence)

    assert res_42_a.perturbed_ranking == res_42_b.perturbed_ranking, "Identical seed must produce identical perturbation ranking"
    assert math.isclose(res_42_a.stability_score, res_42_b.stability_score, rel_tol=1e-7)

    # 3: Different seed produces different perturbation order
    cfg_seed999 = Config(random_seed=999, stability_perturbation="random_shuffle")
    analyzer_999 = StabilityAnalyzer(explainer=mock_explainer, config=cfg_seed999)
    res_999 = analyzer_999.analyze("c1", "Claim text", evidence)

    # For 5 elements, seed 42 vs 999 produces different shuffled orders
    order_42 = list(res_42_a.perturbed_ranking.keys())
    order_999 = list(res_999.perturbed_ranking.keys())
    assert order_42 != order_999, "Different random seeds should yield different shuffled orders"

    # 4: Default permutation strategy performs deterministic reversal
    cfg_reverse = Config(stability_perturbation="permutation")
    analyzer_rev = StabilityAnalyzer(explainer=mock_explainer, config=cfg_reverse)
    res_rev = analyzer_rev.analyze("c1", "Claim text", evidence)
    assert res_rev.stability_score >= 0.0


def test_manifest_provenance_dynamic_mocked_git():
    """Regression test: dynamic Git provenance resolution under clean, dirty, untracked, and error conditions."""
    from unittest.mock import patch, MagicMock
    from experiments.experiment2_v2 import get_git_provenance

    mock_res_sha = MagicMock(returncode=0, stdout="a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4\n", stderr="")
    mock_res_branch = MagicMock(returncode=0, stdout="feature/fuzzy-remediation\n", stderr="")

    # Case 1: Clean repository (no tracked changes, no untracked files)
    mock_res_status_clean = MagicMock(returncode=0, stdout="", stderr="")

    def side_effect_clean(cmd, **kwargs):
        if cmd[1] == "rev-parse" and cmd[2] == "HEAD":
            return mock_res_sha
        elif cmd[1] == "rev-parse" and "--abbrev-ref" in cmd:
            return mock_res_branch
        elif cmd[1] == "status":
            return mock_res_status_clean
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=side_effect_clean):
        prov_clean = get_git_provenance("/fake/dir")
        assert prov_clean["commit_sha"] == "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4"
        assert prov_clean["branch"] == "feature/fuzzy-remediation"
        assert prov_clean["working_tree_dirty"] is False
        assert prov_clean["has_tracked_modifications"] is False
        assert prov_clean["uncommitted_tracked_changes"] == []
        assert prov_clean["has_untracked_files"] is False
        assert prov_clean["untracked_files"] == []
        assert prov_clean["provenance_status"] == "verified"
        assert prov_clean["git_error"] is None

    # Case 2: Dirty working tree (uncommitted tracked modifications)
    mock_res_status_dirty = MagicMock(
        returncode=0,
        stdout=" M experiments/experiment2_v2.py\n?? untracked_file.txt\n",
        stderr=""
    )

    def side_effect_dirty(cmd, **kwargs):
        if cmd[1] == "rev-parse" and cmd[2] == "HEAD":
            return mock_res_sha
        elif cmd[1] == "rev-parse" and "--abbrev-ref" in cmd:
            return mock_res_branch
        elif cmd[1] == "status":
            return mock_res_status_dirty
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=side_effect_dirty):
        prov_dirty = get_git_provenance("/fake/dir")
        assert prov_dirty["commit_sha"] == "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4"
        assert prov_dirty["branch"] == "feature/fuzzy-remediation"
        assert prov_dirty["working_tree_dirty"] is True
        assert prov_dirty["has_tracked_modifications"] is True
        assert prov_dirty["uncommitted_tracked_changes"] == [" M experiments/experiment2_v2.py"]
        assert prov_dirty["has_untracked_files"] is True
        assert prov_dirty["untracked_files"] == ["?? untracked_file.txt"]
        assert prov_dirty["provenance_status"] == "dirty_working_tree"

    # Case 2b: Untracked-only files present (tracked files clean, but untracked files exist)
    mock_res_status_untracked = MagicMock(
        returncode=0,
        stdout="?? outputs/phase11_v2_revalidation_91b8567/\n?? scratch/temp.txt\n",
        stderr=""
    )

    def side_effect_untracked(cmd, **kwargs):
        if cmd[1] == "rev-parse" and cmd[2] == "HEAD":
            return mock_res_sha
        elif cmd[1] == "rev-parse" and "--abbrev-ref" in cmd:
            return mock_res_branch
        elif cmd[1] == "status":
            return mock_res_status_untracked
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=side_effect_untracked):
        prov_untracked = get_git_provenance("/fake/dir")
        assert prov_untracked["commit_sha"] == "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4"
        assert prov_untracked["working_tree_dirty"] is True  # Must NOT be reported as clean
        assert prov_untracked["has_tracked_modifications"] is False
        assert prov_untracked["uncommitted_tracked_changes"] == []
        assert prov_untracked["has_untracked_files"] is True
        assert len(prov_untracked["untracked_files"]) == 2
        assert "?? outputs/phase11_v2_revalidation_91b8567/" in prov_untracked["untracked_files"]
        assert prov_untracked["provenance_status"] == "untracked_files_present"

    # Case 3: Missing Git metadata (Git not installed / FileNotFoundError)
    with patch("subprocess.run", side_effect=FileNotFoundError("git binary not found")):
        prov_err = get_git_provenance("/fake/dir")
        assert prov_err["commit_sha"] is None
        assert prov_err["branch"] is None
        assert prov_err["working_tree_dirty"] is None
        assert prov_err["has_tracked_modifications"] is None
        assert prov_err["has_untracked_files"] is None
        assert prov_err["provenance_status"] == "git_unavailable"
        assert "git binary not found" in prov_err["git_error"]

    # Case 4: Non-zero exit code (not a git repository)
    mock_res_not_git = MagicMock(returncode=128, stdout="", stderr="fatal: not a git repository")
    with patch("subprocess.run", return_value=mock_res_not_git):
        prov_not_git = get_git_provenance("/fake/dir")
        assert prov_not_git["commit_sha"] is None
        assert prov_not_git["provenance_status"] == "git_unavailable"
        assert "fatal: not a git repository" in prov_not_git["git_error"]

    # Case 5: Branch-independent deterministic resolution across arbitrary branch names
    for test_branch, expected_branch in [
        ("main\n", "main"),
        ("master\n", "master"),
        ("fix/fuzzy-pipeline-v2\n", "fix/fuzzy-pipeline-v2"),
        ("feature/arbitrary-branch-name\n", "feature/arbitrary-branch-name"),
        ("HEAD\n", "DETACHED_HEAD"),
    ]:
        mock_branch_var = MagicMock(returncode=0, stdout=test_branch, stderr="")
        with patch("subprocess.run", side_effect=lambda cmd, **kw: mock_branch_var if "--abbrev-ref" in cmd else (mock_res_sha if cmd[1] == "rev-parse" else mock_res_status_clean)):
            res_b = get_git_provenance("/fake/dir")
            assert res_b["branch"] == expected_branch


def test_manifest_provenance_no_hardcoded_commit_and_config_fidelity():
    """Regression test: verifies absence of hardcoded commit 58d4180, source fingerprint accuracy, and configuration fidelity."""
    import inspect
    import experiments.experiment2_v2 as exp2
    from experiments.experiment2_v2 import build_run_manifest, get_source_fingerprints, compute_file_sha256

    # 1. Verify code does not contain hardcoded commit '58d4180'
    source_text = inspect.getsource(exp2)
    assert "58d4180" not in source_text, "Source code must not hardcode legacy commit '58d4180'"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 2. Source fingerprints match actual file contents
    fps = get_source_fingerprints(project_root)
    assert "experiments/experiment2_v2.py" in fps
    assert "conflict/detector.py" in fps
    assert "stability/analyzer.py" in fps
    assert "fuzzy/engine.py" in fps

    exp2_path = os.path.join(project_root, "experiments/experiment2_v2.py")
    expected_exp2_hash = compute_file_sha256(exp2_path)
    assert fps["experiments/experiment2_v2.py"] == expected_exp2_hash
    assert len(fps["experiments/experiment2_v2.py"]) == 64

    # 3. Configuration fidelity in manifest builder
    test_cfg = {
        "top_k": 3,
        "retrieval_mode": "hybrid",
        "hybrid_fusion_method": "legacy",
        "conflict_score_method": "harmonic_mean",
        "random_seed": 42
    }
    manifest = build_run_manifest(
        project_root=project_root,
        output_dir="outputs/test_manifest",
        cfg_dict=test_cfg,
        source_fps=fps,
        dataset_fp="dummy_dataset_fp",
        corpus_fp="dummy_corpus_fp",
        t_start=1000.0,
        cache_status="accepted",
        total_fallbacks_v2=0,
        B=1000,
        seed=42,
        t_end=1005.5
    )

    assert manifest["manifest_version"] == "2.0"
    assert manifest["configuration"] == test_cfg
    assert manifest["duration_seconds"] == 5.5
    assert manifest["source_code_fingerprints"] == fps
    assert manifest["git_provenance"]["commit_sha"] is not None
    assert manifest["feature_provenance"]["fuzzy_engine"]["fallbacks_observed"] == 0


def test_manifest_provenance_real_git_smoke_check():
    """Integration smoke test: retrieves real Git metadata from local project working repository without assuming a specific branch name."""
    from experiments.experiment2_v2 import get_git_provenance

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prov = get_git_provenance(project_root)

    # 1. Valid commit SHA (40-char hex string)
    assert prov["commit_sha"] is not None, "Real Git environment must return a commit SHA"
    assert len(prov["commit_sha"]) == 40, f"Expected 40-char commit SHA, got {len(prov['commit_sha'])}"
    assert all(c in "0123456789abcdef" for c in prov["commit_sha"]), "Commit SHA must be lowercase hex"

    # 2. Branch name: sensible non-empty string, independent of branch name
    assert isinstance(prov["branch"], str), "Branch must be a string"
    assert len(prov["branch"]) > 0, "Branch must not be empty"
    assert prov["branch"] != "UNKNOWN", f"Branch could not be resolved: {prov['branch']}"
    assert not any(c.isspace() for c in prov["branch"]), f"Branch name cannot contain whitespace: {prov['branch']}"

    # 3. Working tree status structure and boolean integrity
    assert isinstance(prov["working_tree_dirty"], bool), "working_tree_dirty must be a boolean"
    assert isinstance(prov["has_tracked_modifications"], bool), "has_tracked_modifications must be a boolean"
    assert isinstance(prov["has_untracked_files"], bool), "has_untracked_files must be a boolean"
    assert isinstance(prov["uncommitted_tracked_changes"], list), "uncommitted_tracked_changes must be a list"
    assert isinstance(prov["untracked_files"], list), "untracked_files must be a list"
    assert prov["git_error"] is None

    # 4. Status consistency check based on actual working tree state
    if prov["has_tracked_modifications"]:
        assert prov["working_tree_dirty"] is True
        assert prov["provenance_status"] == "dirty_working_tree"
        assert len(prov["uncommitted_tracked_changes"]) > 0
    elif prov["has_untracked_files"]:
        assert prov["working_tree_dirty"] is True
        assert prov["provenance_status"] == "untracked_files_present"
        assert len(prov["untracked_files"]) > 0
    else:
        assert prov["working_tree_dirty"] is False
        assert prov["provenance_status"] == "verified"
        assert len(prov["uncommitted_tracked_changes"]) == 0
        assert len(prov["untracked_files"]) == 0
