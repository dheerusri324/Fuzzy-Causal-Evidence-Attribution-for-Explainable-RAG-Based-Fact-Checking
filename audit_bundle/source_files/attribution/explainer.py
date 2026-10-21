"""
Evidence Explainer.

Implements evidence-level attribution using SHAP and LIME.
Features are defined strictly as the presence (1) or absence (0) of
individual evidence pieces, rather than individual tokens.
"""

import numpy as np
import shap
from sklearn.linear_model import Ridge
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from config.settings import Config
from retrieval.schema import RetrievedEvidence
from verification.verifier import ClaimEvidenceVerifier
from attribution.schema import EvidenceAttribution, AttributionResult
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class EvidenceExplainer:
    """Computes SHAP and LIME attribution scores for evidence pieces."""

    def __init__(self, verifier: ClaimEvidenceVerifier, config: Optional[Config] = None):
        self.verifier = verifier
        self.config = config or Config()
        
        self.shap_samples = getattr(self.config, "shap_num_samples", 100)
        self.lime_samples = getattr(self.config, "lime_num_samples", 100)
        self.max_evidence = getattr(self.config, "max_attribution_evidence", 5)
        
        # Internal cache for model predictions: (claim, tuple(evidence_ids)) -> prob_dict
        self._eval_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}
        
        # Map class names to indices to ensure consistent ordering
        self.class_names = ["REFUTES", "SUPPORTS", "NOT_ENOUGH_INFO"]
        
    def _get_probabilities(self, claim: str, evidence_subset: List[RetrievedEvidence]) -> np.ndarray:
        """Call the verifier and return probabilities, utilizing an in-memory cache."""
        # Use evidence IDs as cache key
        ev_ids = tuple(sorted(ev.evidence_id for ev in evidence_subset))
        cache_key = (claim, ev_ids)
        
        if cache_key not in self._eval_cache:
            res = self.verifier.verify(claim, evidence_subset)
            self._eval_cache[cache_key] = res.probabilities
            
        probs = self._eval_cache[cache_key]
        return np.array([probs[c] for c in self.class_names])

    def explain(
        self, 
        claim_id: str, 
        claim: str, 
        evidence_list: List[RetrievedEvidence]
    ) -> AttributionResult:
        """Explain the verifier's prediction by attributing importance to evidence pieces."""
        
        # Truncate evidence if it exceeds max_attribution_evidence to avoid exponential explosion
        if len(evidence_list) > self.max_evidence:
            logger.warning(
                f"Truncating evidence for attribution from {len(evidence_list)} to {self.max_evidence}."
            )
            evidence_list = evidence_list[:self.max_evidence]
            
        n_features = len(evidence_list)
        
        # Edge case: No evidence
        if n_features == 0:
            res = self.verifier.verify(claim, [])
            return AttributionResult(
                claim_id=claim_id,
                claim=claim,
                verdict=res.verdict,
                baseline_probabilities=res.probabilities,
                evidence_attributions=[]
            )

        # 1. Baseline prediction (all evidence)
        baseline_res = self.verifier.verify(claim, evidence_list)
        target_class = baseline_res.verdict
        target_idx = self.class_names.index(target_class)

        # Define the wrapper function for the explainers (takes binary masks -> probabilities)
        def predict_wrapper(masks: np.ndarray) -> np.ndarray:
            """Map binary masks (M, N) -> probabilities (M, C)."""
            out_probs = []
            for mask in masks:
                # Include evidence if mask value > 0.5
                subset = [ev for i, ev in enumerate(evidence_list) if mask[i] > 0.5]
                p = self._get_probabilities(claim, subset)
                out_probs.append(p)
            return np.array(out_probs)

        # ---------------------------------------------------------
        # SHAP ATTRIBUTION (using KernelSHAP on binary features)
        # ---------------------------------------------------------
        # Background is "no evidence" (all zeros)
        background = np.zeros((1, n_features))
        explainer_shap = shap.KernelExplainer(predict_wrapper, background)
        
        # Explain the "all evidence" instance (all ones)
        instance = np.ones((1, n_features))
        
        # Suppress the tqdm bar if n_features is small or it's unhelpful
        # Calculate shapley values for all classes
        shap_values = explainer_shap.shap_values(
            instance, 
            nsamples=self.shap_samples, 
            silent=True
        )
        
        # Extract the vector of scores for each class robustly
        shap_scores = {}
        if isinstance(shap_values, list):
            # Often shap returns a list of length num_classes
            # where each element has shape (num_instances, num_features)
            for i, class_name in enumerate(self.class_names):
                # Ensure we get the 1D array of features for the first instance
                val = shap_values[i]
                if len(val.shape) == 2:
                    shap_scores[class_name] = val[0]
                elif len(val.shape) == 1:
                    shap_scores[class_name] = val
                else:
                    shap_scores[class_name] = val.flatten()
        else:
            # Sometimes shap returns an array of shape (num_instances, num_features, num_classes)
            for i, class_name in enumerate(self.class_names):
                if len(shap_values.shape) == 3:
                    shap_scores[class_name] = shap_values[0, :, i]
                elif len(shap_values.shape) == 2:
                    # If it collapsed for some reason
                    shap_scores[class_name] = shap_values[0]
                else:
                    shap_scores[class_name] = shap_values.flatten()

        # ---------------------------------------------------------
        # LIME ATTRIBUTION (Custom Tabular LIME on binary features)
        # ---------------------------------------------------------
        lime_scores = self._compute_binary_lime(
            predict_wrapper, 
            n_features, 
            self.class_names
        )

        # ---------------------------------------------------------
        # Compile Results and Rankings
        # ---------------------------------------------------------
        attributions = []
        for i, ev in enumerate(evidence_list):
            attr = EvidenceAttribution(
                evidence_id=ev.evidence_id,
                text=ev.text,
                shap_support=float(shap_scores["SUPPORTS"][i]),
                shap_refute=float(shap_scores["REFUTES"][i]),
                shap_nei=float(shap_scores["NOT_ENOUGH_INFO"][i]),
                shap_target=float(shap_scores[target_class][i]),
                shap_rank=0,  # Computed below
                lime_support=float(lime_scores["SUPPORTS"][i]),
                lime_refute=float(lime_scores["REFUTES"][i]),
                lime_nei=float(lime_scores["NOT_ENOUGH_INFO"][i]),
                lime_target=float(lime_scores[target_class][i]),
                lime_rank=0   # Computed below
            )
            attributions.append(attr)
            
        # Compute ranks (1 = highest score for the target class)
        # Sort descending by score, then assign rank
        shap_sorted = sorted(attributions, key=lambda x: x.shap_target, reverse=True)
        for rank, attr in enumerate(shap_sorted, start=1):
            attr.shap_rank = rank
            
        lime_sorted = sorted(attributions, key=lambda x: x.lime_target, reverse=True)
        for rank, attr in enumerate(lime_sorted, start=1):
            attr.lime_rank = rank

        return AttributionResult(
            claim_id=claim_id,
            claim=claim,
            verdict=target_class,
            baseline_probabilities=baseline_res.probabilities,
            evidence_attributions=attributions
        )

    def _compute_binary_lime(
        self, 
        predict_fn, 
        n_features: int, 
        class_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Custom Tabular LIME implementation for binary feature spaces.
        
        Generates binary perturbations, scores them with the model, calculates
        cosine-based locality weights, and fits a Ridge regression surrogate.
        This perfectly mirrors LIME's semantics for binary presence/absence.
        """
        # 1. Generate perturbations (random binary masks)
        n_samples = self.lime_samples
        
        # If the feature space is very small (e.g. 2^5 = 32), generate exact space
        if 2**n_features <= n_samples:
            import itertools
            masks = np.array(list(itertools.product([0, 1], repeat=n_features)))
        else:
            # Generate random masks, ensure the "all 1s" and "all 0s" are included
            masks = np.random.randint(0, 2, size=(n_samples - 2, n_features))
            masks = np.vstack([np.ones((1, n_features)), np.zeros((1, n_features)), masks])
            # Deduplicate just in case
            masks = np.unique(masks, axis=0)

        # 2. Get predictions for all perturbations
        probs = predict_fn(masks)  # shape: (num_masks, num_classes)

        # 3. Calculate distance and weights (locality)
        # Distance to the original instance (all 1s). 
        # Using normalized hamming distance (fraction of mismatched features)
        original = np.ones(n_features)
        distances = np.sum(np.abs(masks - original), axis=1) / n_features
        
        # LIME's standard exponential kernel weight
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))

        # 4. Fit local surrogate models (Ridge Regression) for each class
        lime_scores = {}
        for c_idx, c_name in enumerate(class_names):
            y = probs[:, c_idx]
            
            # Fit weighted ridge regression
            # alpha=1.0 is default regularization
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(masks, y, sample_weight=weights)
            
            # The coefficients are the feature importances (LIME scores)
            lime_scores[c_name] = model.coef_

        return lime_scores
