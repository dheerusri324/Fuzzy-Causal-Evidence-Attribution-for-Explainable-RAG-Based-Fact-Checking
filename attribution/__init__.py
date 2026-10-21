"""Attribution package — Evidence-level SHAP and LIME Explanations."""

from attribution.schema import AttributionResult, EvidenceAttribution
from attribution.explainer import EvidenceExplainer

__all__ = ["AttributionResult", "EvidenceAttribution", "EvidenceExplainer"]
