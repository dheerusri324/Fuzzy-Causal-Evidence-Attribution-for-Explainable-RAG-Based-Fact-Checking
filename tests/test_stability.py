"""
Tests for Phase 8 Explanation Stability.
"""

import pytest
import math
from unittest.mock import MagicMock

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from attribution.explainer import EvidenceExplainer
from attribution.schema import AttributionResult, EvidenceAttribution
from stability.analyzer import StabilityAnalyzer

# Mock Explainer to control the SHAP outputs for deterministic stability tests
class MockExplainer(EvidenceExplainer):
    def __init__(self, fixed_attributions=None):
        self.fixed_attributions = fixed_attributions or {}
        
    def explain(self, claim_id, claim, evidence):
        # Generate mock attribution result based on the evidence passed in
        attrs = []
        for ev in evidence:
            val = self.fixed_attributions.get(ev.evidence_id, 0.5)
            attrs.append(
                EvidenceAttribution(
                    evidence_id=ev.evidence_id,
                    text=ev.text,
                    shap_target=val,
                    lime_target=val,
                    shap_support=val, shap_refute=0, shap_nei=0,
                    lime_support=val, lime_refute=0, lime_nei=0,
                    shap_rank=1, lime_rank=1
                )
            )
        return AttributionResult(
            claim_id=claim_id, claim=claim,
            verdict="SUPPORTS", baseline_probabilities={},
            evidence_attributions=attrs
        )

@pytest.fixture
def sample_evidence():
    return [
        RetrievedEvidence(evidence_id="e1", text="text1", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=1.0, retrieval_rank=1, retrieval_method=""),
        RetrievedEvidence(evidence_id="e2", text="text2", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=0.9, retrieval_rank=2, retrieval_method=""),
        RetrievedEvidence(evidence_id="e3", text="text3", doc_title="", sentence_idx=0, source="", source_type="", retrieval_score=0.8, retrieval_rank=3, retrieval_method="")
    ]

def test_identical_attribution_vectors_stability(sample_evidence):
    # Fixed attributions mean the explanation will output the same values regardless of perturbation
    explainer = MockExplainer(fixed_attributions={"e1": 0.9, "e2": 0.5, "e3": 0.1})
    analyzer = StabilityAnalyzer(explainer=explainer, config=Config(stability_perturbation="permutation"))
    
    res = analyzer.analyze("c1", "Claim", sample_evidence)
    
    # Perfect stability
    assert math.isclose(res.spearman_correlation, 1.0, rel_tol=1e-5)
    assert math.isclose(res.stability_score, 1.0, rel_tol=1e-5)

def test_output_range_validation(sample_evidence):
    explainer = MockExplainer(fixed_attributions={"e1": 0.9, "e2": 0.5, "e3": 0.1})
    analyzer = StabilityAnalyzer(explainer=explainer)
    res = analyzer.analyze("c2", "Claim", sample_evidence)
    
    assert 0.0 <= res.stability_score <= 1.0
    assert -1.0 <= res.spearman_correlation <= 1.0

def test_evidence_id_alignment(sample_evidence):
    explainer = MockExplainer(fixed_attributions={"e1": 0.9, "e2": 0.5, "e3": 0.1})
    analyzer = StabilityAnalyzer(explainer=explainer)
    res = analyzer.analyze("c3", "Claim", sample_evidence)
    
    assert "e1" in res.baseline_ranking
    assert "e1" in res.perturbed_ranking

def test_empty_or_single_evidence(sample_evidence):
    explainer = MockExplainer()
    analyzer = StabilityAnalyzer(explainer=explainer)
    
    # Single evidence
    res1 = analyzer.analyze("c4", "Claim", sample_evidence[:1])
    assert res1.stability_score == 1.0
    
    # Empty evidence
    res2 = analyzer.analyze("c5", "Claim", [])
    assert res2.stability_score == 1.0
