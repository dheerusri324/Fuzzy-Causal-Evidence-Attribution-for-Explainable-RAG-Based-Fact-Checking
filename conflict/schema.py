"""
Conflict detection schemas.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class EvidenceRole:
    evidence_id: str
    text: str
    support_probability: float
    refute_probability: float
    neutral_probability: float
    assigned_role: str  # "SUPPORTING", "REFUTING", "NEUTRAL"

@dataclass
class ConflictResult:
    claim_id: str
    claim: str
    evidence_roles: List[EvidenceRole]
    conflict_detected: bool
    conflict_score: float
    supporting_evidence_ids: List[str]
    refuting_evidence_ids: List[str]
    neutral_evidence_ids: List[str]
