"""
Schemas for the Phase 9 Full Application Pipeline.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from retrieval.schema import RetrievedEvidence
from verification.schema import VerificationResult
from attribution.schema import AttributionResult
from intervention.schema import InterventionResult
from conflict.schema import ConflictResult
from stability.schema import StabilityResult
from fuzzy.schema import FuzzyOutput


@dataclass
class EvidencePipelineResult:
    """Combines all piece-level information for a single piece of evidence."""
    evidence: RetrievedEvidence
    # Explanations
    shap_target: float
    lime_target: float
    # Intervention
    probability_after_removal: float
    delta_p: float
    verdict_changed: bool
    # Conflict role
    role: str  # SUPPORTING, REFUTING, NEUTRAL
    # Fuzzy metrics
    fuzzy_output: FuzzyOutput


@dataclass
class PipelineResult:
    """The complete result of the end-to-end fact-checking pipeline."""
    claim_id: str
    claim: str
    
    # 1. Verification
    verification: VerificationResult
    
    # 2. Evidence-level results
    evidence_results: List[EvidencePipelineResult]
    
    # 3. Global Conflict/Stability
    conflict: ConflictResult
    stability: StabilityResult
    
    # 4. Final Natural Language Summary
    final_explanation: str
