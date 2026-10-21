"""
Claim-Evidence Verifier.

Uses a pretrained NLI cross-encoder model to classify a claim against
retrieved evidence. Exposes a reusable interface for downstream attribution.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.schema import VerificationResult
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class ClaimEvidenceVerifier:
    """NLI-based Verifier for Fact-Checking.
    
    Consumes a claim and a list of RetrievedEvidence objects, formats them,
    and returns a structured VerificationResult with a probability distribution.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Configuration for verification
        self.model_name = getattr(self.config, "verifier_model", "cross-encoder/nli-deberta-v3-base")
        self.device = self.config.device
        self.max_length = getattr(self.config, "max_seq_length", 512)
        self.max_evidence = getattr(self.config, "max_evidence", 5)
        
        self.tokenizer = None
        self.model = None
        
        # 0: contradiction, 1: entailment, 2: neutral (typical for NLI)
        self.id_to_project_label = {
            0: "REFUTES",
            1: "SUPPORTS",
            2: "NOT_ENOUGH_INFO"
        }

    def _load_model(self):
        """Lazy load the model and tokenizer once."""
        if self.model is None:
            logger.info(f"Loading NLI verifier model: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Verify label mapping from model config to catch unexpected schemas
            actual_labels = self.model.config.id2label
            expected = {0: "contradiction", 1: "entailment", 2: "neutral"}
            
            for idx, label in expected.items():
                actual = actual_labels.get(idx, "").lower()
                if actual != label:
                    logger.warning(
                        f"NLI label mismatch at idx {idx}! Expected '{label}', got '{actual}'"
                    )

    def _format_evidence(self, evidence_list: List[RetrievedEvidence]) -> str:
        """Format the evidence list into a single context string."""
        if not evidence_list:
            return ""
            
        formatted_parts = []
        for ev in evidence_list:
            formatted_parts.append(ev.text.strip())
            
        return " ".join(formatted_parts)

    def verify(
        self, 
        claim: str, 
        evidence_list: List[RetrievedEvidence]
    ) -> VerificationResult:
        """Verify the claim against the provided evidence list.
        
        Args:
            claim: The claim text.
            evidence_list: List of RetrievedEvidence objects (from Phase 3).
            
        Returns:
            VerificationResult containing verdict, probabilities, and evidence IDs used.
        """
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")
            
        self._load_model()
        
        # Respect max evidence limit
        evidence_subset = evidence_list[:self.max_evidence]
        evidence_ids = [ev.evidence_id for ev in evidence_subset]
        evidence_text = self._format_evidence(evidence_subset)
        
        # Edge case: No evidence provided
        if not evidence_text:
            return VerificationResult(
                claim=claim,
                verdict="NOT_ENOUGH_INFO",
                support_probability=0.0,
                refute_probability=0.0,
                nei_probability=1.0,
                confidence=1.0,
                evidence_ids=[]
            )

        # Tokenize cross-encoder style (evidence as premise, claim as hypothesis)
        inputs = self.tokenizer(
            evidence_text,
            claim,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
        # Map probabilities
        # idx 0 -> REFUTES
        # idx 1 -> SUPPORTS
        # idx 2 -> NOT_ENOUGH_INFO
        refute_prob = float(probs[0])
        support_prob = float(probs[1])
        nei_prob = float(probs[2])
        
        pred_idx = int(probs.argmax())
        verdict = self.id_to_project_label[pred_idx]
        confidence = float(probs[pred_idx])
        
        return VerificationResult(
            claim=claim,
            verdict=verdict,
            support_probability=support_prob,
            refute_probability=refute_prob,
            nei_probability=nei_prob,
            confidence=confidence,
            evidence_ids=evidence_ids
        )
