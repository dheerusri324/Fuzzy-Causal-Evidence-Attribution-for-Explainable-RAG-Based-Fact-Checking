"""
End-to-End Tests for the Phase 9 Full Application Pipeline.
"""

import pytest
from app.pipeline import FactCheckingPipeline
from config.settings import Config

@pytest.fixture(scope="module")
def pipeline():
    # Use a small configuration for fast testing
    cfg = Config(top_k=2, shap_num_samples=10, lime_num_samples=10)
    return FactCheckingPipeline(config=cfg)

def test_pipeline_end_to_end(pipeline):
    claim = "The sky is blue."
    
    # Run process
    res = pipeline.process(claim)
    
    # 1. Verification presence
    assert res.verification is not None
    assert res.verification.verdict in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    
    # 2. Evidence presence
    assert len(res.evidence_results) > 0
    for ev_res in res.evidence_results:
        assert ev_res.evidence is not None
        assert ev_res.role in ["SUPPORTING", "REFUTING", "NEUTRAL"]
        # SHAP, LIME, Intervention, Fuzzy should all be populated
        assert isinstance(ev_res.shap_target, float)
        assert isinstance(ev_res.delta_p, float)
        assert ev_res.fuzzy_output is not None
        
    # 3. Global modules
    assert res.conflict is not None
    assert isinstance(res.conflict.conflict_score, float)
    
    assert res.stability is not None
    assert isinstance(res.stability.stability_score, float)
    
    # 4. Final summary
    assert isinstance(res.final_explanation, str)
    assert len(res.final_explanation) > 0
    
def test_pipeline_empty_evidence_handling():
    # If a claim retrieves no evidence, it should gracefully raise an error
    cfg = Config()
    pipe = FactCheckingPipeline(config=cfg)
    
    # Mock retriever to return empty
    pipe.retriever.retrieve = lambda claim, top_k, mode: []
    
    with pytest.raises(ValueError, match="No evidence retrieved"):
        pipe.process("Bogus claim that gets nothing.")
