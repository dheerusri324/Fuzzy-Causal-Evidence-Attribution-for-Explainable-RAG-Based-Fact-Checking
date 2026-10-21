"""
Conflict Detection Module.

Classifies individual evidence pieces as SUPPORTING, REFUTING, or NEUTRAL
against a claim. Computes a conflict score based on whether meaningful 
supporting and refuting evidence coexist.
"""

from typing import List, Optional

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from conflict.schema import EvidenceRole, ConflictResult


class ConflictDetector:
    """Detects contradiction/conflict among a set of evidence for a claim."""

    def __init__(self, verifier: ClaimEvidenceVerifier, config: Optional[Config] = None):
        self.verifier = verifier
        self.config = config or Config()
        self.threshold = getattr(self.config, "conflict_role_threshold", 0.5)

    def detect(self, claim_id: str, claim: str, evidence_list: List[RetrievedEvidence]) -> ConflictResult:
        """Analyze evidence list for conflicts."""
        
        evidence_roles = []
        supporting_ids = []
        refuting_ids = []
        neutral_ids = []
        
        max_support = 0.0
        max_refute = 0.0
        
        for ev in evidence_list:
            # We verify the claim against this SINGLE piece of evidence
            # to determine its independent role.
            res = self.verifier.verify(claim, [ev])
            
            p_sup = res.probabilities.get("SUPPORTS", 0.0)
            p_ref = res.probabilities.get("REFUTES", 0.0)
            p_neu = res.probabilities.get("NOT_ENOUGH_INFO", 0.0)
            
            # Assignment logic using configured threshold
            # If the max prob is < threshold, default to NEUTRAL
            if p_sup > self.threshold and p_sup > p_ref and p_sup > p_neu:
                role = "SUPPORTING"
                supporting_ids.append(ev.evidence_id)
                max_support = max(max_support, p_sup)
            elif p_ref > self.threshold and p_ref > p_sup and p_ref > p_neu:
                role = "REFUTING"
                refuting_ids.append(ev.evidence_id)
                max_refute = max(max_refute, p_ref)
            else:
                role = "NEUTRAL"
                neutral_ids.append(ev.evidence_id)
                
            evidence_roles.append(
                EvidenceRole(
                    evidence_id=ev.evidence_id,
                    text=ev.text,
                    support_probability=p_sup,
                    refute_probability=p_ref,
                    neutral_probability=p_neu,
                    assigned_role=role
                )
            )
            
        # Conflict exists if there is at least one meaningful piece of evidence on both sides
        conflict_detected = len(supporting_ids) > 0 and len(refuting_ids) > 0
        
        # Conflict Score Formulation:
        # 0 = no conflict, 1 = strong conflict.
        # We use a simple harmonic mean-style bounding or min-max intersection.
        # The simplest deterministic formulation is to take the minimum of the highest 
        # support and highest refute, then scale it up (multiply by 2) bounded at 1.0.
        # If max_sup=0.9 and max_ref=0.8, min is 0.8, score = min(1.0, 1.6) = 1.0.
        # If max_sup=0.9 and max_ref=0.0, score = 0.0.
        if conflict_detected:
            conflict_score = min(1.0, 2.0 * min(max_support, max_refute))
        else:
            conflict_score = 0.0
            
        return ConflictResult(
            claim_id=claim_id,
            claim=claim,
            evidence_roles=evidence_roles,
            conflict_detected=conflict_detected,
            conflict_score=conflict_score,
            supporting_evidence_ids=supporting_ids,
            refuting_evidence_ids=refuting_ids,
            neutral_evidence_ids=neutral_ids
        )
