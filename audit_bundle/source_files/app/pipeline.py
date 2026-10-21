"""
Full System Integration Pipeline.
Orchestrates Phase 3-8 into a single unified pass.
"""

from typing import List, Optional
import uuid

from config.settings import Config
from retrieval.retriever import EvidenceRetriever
from verification.verifier import ClaimEvidenceVerifier
from attribution.explainer import EvidenceExplainer
from intervention.intervener import EvidenceIntervener
from conflict.detector import ConflictDetector
from stability.analyzer import StabilityAnalyzer
from fuzzy.engine import FuzzyEvidenceEngine
from fuzzy.schema import FuzzyInput

from app.schema import EvidencePipelineResult, PipelineResult

class FactCheckingPipeline:
    """Orchestrates the entire fact-checking process."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize all modules once to leverage caching and single model loads
        self.retriever = EvidenceRetriever(config=self.config)
        self.verifier = ClaimEvidenceVerifier(config=self.config)
        self.explainer = EvidenceExplainer(verifier=self.verifier, config=self.config)
        self.intervener = EvidenceIntervener(verifier=self.verifier, config=self.config)
        self.conflict_detector = ConflictDetector(verifier=self.verifier, config=self.config)
        self.stability_analyzer = StabilityAnalyzer(explainer=self.explainer, config=self.config)
        self.fuzzy_engine = FuzzyEvidenceEngine(config=self.config)

    def process(self, claim: str, claim_id: Optional[str] = None) -> PipelineResult:
        if not claim_id:
            claim_id = str(uuid.uuid4())[:8]

        # 1. Retrieval
        top_k = self.config.top_k
        mode = self.config.retrieval_mode
        evs = self.retriever.retrieve(claim, top_k=top_k, mode=mode)
        
        # Guard for empty evidence
        if not evs:
            raise ValueError("No evidence retrieved. Cannot proceed.")

        # 2. Verification
        ver_res = self.verifier.verify(claim, evs)
        
        # 3. Attribution (SHAP/LIME)
        attr_res = self.explainer.explain(claim_id, claim, evs)
        attr_map = {ea.evidence_id: ea for ea in attr_res.evidence_attributions}
        
        # 4. Intervention (Leave-One-Out)
        int_res = self.intervener.intervene(claim_id, claim, evs)
        int_map = {ei.evidence_id: ei for ei in int_res.interventions}
        
        # 5. Conflict Detection
        conf_res = self.conflict_detector.detect(claim_id, claim, evs)
        role_map = {er.evidence_id: er.assigned_role for er in conf_res.evidence_roles}
        
        # 6. Stability Analysis
        stab_res = self.stability_analyzer.analyze(claim_id, claim, evs)
        
        # 7. Fuzzy Reasoning & Assembly
        evidence_results = []
        for ev in evs:
            eid = ev.evidence_id
            
            # Extract piece-level data
            ea = attr_map.get(eid)
            ei = int_map.get(eid)
            role = role_map.get(eid, "NEUTRAL")
            
            shap_val = ea.shap_target if ea else 0.0
            lime_val = ea.lime_target if ea else 0.0
            prob_after = ei.intervened_target_probability if ei else 0.0
            delta_p = ei.intervention_influence if ei else 0.0
            verdict_changed = ei.verdict_changed if ei else False
            
            # Construct Fuzzy Inputs
            fuz_input = FuzzyInput(
                relevance=ev.retrieval_score,
                attribution_shap=shap_val,
                attribution_lime=lime_val,
                intervention_influence=delta_p,
                contradiction_degree=conf_res.conflict_score,
                stability_score=stab_res.stability_score
            )
            
            # Run Fuzzy Engine
            fuz_output = self.fuzzy_engine.evaluate(fuz_input)
            
            evidence_results.append(
                EvidencePipelineResult(
                    evidence=ev,
                    shap_target=shap_val,
                    lime_target=lime_val,
                    probability_after_removal=prob_after,
                    delta_p=delta_p,
                    verdict_changed=verdict_changed,
                    role=role,
                    fuzzy_output=fuz_output
                )
            )

        # 8. Final Natural Language Summary
        final_explanation = self._generate_summary(ver_res, evidence_results, conf_res, stab_res)

        return PipelineResult(
            claim_id=claim_id,
            claim=claim,
            verification=ver_res,
            evidence_results=evidence_results,
            conflict=conf_res,
            stability=stab_res,
            final_explanation=final_explanation
        )

    def _generate_summary(
        self, 
        ver_res, 
        evidence_results: List[EvidencePipelineResult], 
        conf_res, 
        stab_res
    ) -> str:
        verdict = ver_res.verdict
        prob = ver_res.probabilities.get(verdict, 0.0) * 100
        
        # Sort evidence by delta_p to find the most influential
        sorted_ev = sorted(evidence_results, key=lambda x: x.delta_p, reverse=True)
        most_influential = sorted_ev[0] if sorted_ev else None
        
        summary = f"The verifier predicted the claim is {verdict} with {prob:.1f}% confidence. "
        
        if most_influential:
            inf_id = most_influential.evidence.evidence_id
            inf_role = most_influential.role.lower()
            summary += f"The most influential evidence ({inf_id}) played a {inf_role} role. "
            
        if conf_res.conflict_detected:
            summary += f"However, conflict detection found mixed signals (Score: {conf_res.conflict_score:.2f}). "
            
        summary += f"The explanation robustness (stability) was {stab_res.stability_score:.2f}."
        
        return summary
