"""
Tests for Phase 7 — Fuzzy Evidence Reasoning.
"""

import pytest
import math

from config.settings import Config
from fuzzy.schema import FuzzyInput
from fuzzy.engine import FuzzyEvidenceEngine
from fuzzy.baseline import compute_baseline_score
from fuzzy.normalization import normalize_relevance, normalize_attribution, normalize_intervention, compute_attribution_strength


@pytest.fixture(scope="module")
def config():
    return Config()

@pytest.fixture(scope="module")
def engine(config):
    return FuzzyEvidenceEngine(config=config)


def test_normalization():
    assert normalize_relevance(1.5, max_score=2.0) == 0.75
    assert normalize_relevance(-0.5) == 0.0
    assert normalize_relevance(1.5) == 1.0  # default max_score=1.0

    assert normalize_attribution(0.8) == 0.8
    assert normalize_attribution(-0.4) == 0.4
    
    # 0.5 * 0.4 + 0.5 * 0.8 = 0.6
    assert math.isclose(compute_attribution_strength(0.4, -0.8), 0.6, rel_tol=1e-5)

def test_missing_input_defaults(engine):
    # Test that not providing contradiction/stability uses defaults
    inp = FuzzyInput(
        relevance=0.9,
        attribution_shap=0.8,
        attribution_lime=0.8,
        intervention_influence=0.8
    )
    # Default is 0.0 for contradiction, 0.75 for source, 1.0 for stability
    assert inp.contradiction_degree == 0.0
    assert inp.source_reliability == 0.75
    assert inp.stability_score == 1.0

def test_fuzzy_engine_determinism(engine):
    inp = FuzzyInput(
        relevance=0.8,
        attribution_shap=0.8,
        attribution_lime=0.8,
        intervention_influence=0.8,
    )
    out1 = engine.evaluate(inp)
    out2 = engine.evaluate(inp)
    
    assert out1.evidence_reliability == out2.evidence_reliability

def test_output_range_validity(engine):
    inp = FuzzyInput(relevance=0.5, attribution_shap=0.5, attribution_lime=0.5, intervention_influence=0.5)
    out = engine.evaluate(inp)
    
    assert 0.0 <= out.evidence_relevance <= 1.0
    assert 0.0 <= out.evidence_reliability <= 1.0
    assert 0.0 <= out.support_degree <= 1.0
    assert 0.0 <= out.contradiction_degree <= 1.0
    assert 0.0 <= out.explanation_confidence <= 1.0

def test_weighted_baseline(config):
    inp = FuzzyInput(
        relevance=1.0, 
        attribution_shap=1.0, attribution_lime=1.0, 
        intervention_influence=1.0,
        source_reliability=1.0,
        stability_score=1.0,
        contradiction_degree=0.0
    )
    # Weights sum to 1.0, so output should be 1.0
    base_out = compute_baseline_score(inp, config)
    assert math.isclose(base_out.baseline_score, 1.0, rel_tol=1e-5)
    assert math.isclose(base_out.adjusted_baseline_score, 1.0, rel_tol=1e-5)
    
    # Test contradiction penalty
    inp.contradiction_degree = 0.5
    base_out_adj = compute_baseline_score(inp, config)
    assert math.isclose(base_out_adj.adjusted_baseline_score, 0.5, rel_tol=1e-5)

def test_rule_execution_strong_evidence(engine):
    # CASE A: Strong evidence
    inp = FuzzyInput(
        relevance=0.95,
        attribution_shap=0.90,
        attribution_lime=0.90,
        intervention_influence=0.88,
        contradiction_degree=0.05,
        source_reliability=0.80,
        stability_score=0.90
    )
    out = engine.evaluate(inp)
    
    # Should result in HIGH reliability, support, and confidence
    assert out.evidence_reliability > 0.6
    assert out.support_degree > 0.6
    assert out.explanation_confidence > 0.6
    assert out.evidence_relevance > 0.6

def test_attribution_vs_intervention_confidence_behavior(engine):
    # CASE B: Attribution without intervention
    inp = FuzzyInput(
        relevance=0.90,
        attribution_shap=0.90,
        attribution_lime=0.90,
        intervention_influence=0.10,  # LOW intervention
        contradiction_degree=0.05,
        source_reliability=0.80,
        stability_score=0.90
    )
    out = engine.evaluate(inp)
    
    # Confidence should be low due to Rule 2
    assert out.explanation_confidence < 0.5

def test_contradiction_behavior(engine):
    # CASE C: Conflicting evidence
    inp = FuzzyInput(
        relevance=0.90,
        attribution_shap=0.70,
        attribution_lime=0.70,
        intervention_influence=0.80,
        contradiction_degree=0.90,  # HIGH contradiction
        source_reliability=0.80,
        stability_score=0.80
    )
    out = engine.evaluate(inp)
    
    # Rule 4 should fire giving HIGH contradiction output
    assert out.contradiction_degree > 0.6
