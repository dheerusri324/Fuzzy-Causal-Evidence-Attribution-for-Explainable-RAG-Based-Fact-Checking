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
