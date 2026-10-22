"""
Project configuration system.

Provides a dataclass-based configuration with YAML file support.
All expensive parameters (model names, top-k, perturbation counts, etc.)
are configurable rather than hard-coded.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    """Central project configuration."""

    # --- Project paths ---
    project_root: str = ""
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "outputs"

    # --- Dataset ---
    dataset_name: str = "fever"
    dataset_split: str = "train"
    max_claims: int = 500  # dev subset size; increase for experiments

    # --- Retrieval ---
    retrieval_mode: str = "hybrid"  # bm25 | dense | hybrid
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    hybrid_fusion_method: str = "pool_union"  # pool_union | legacy

    # --- Verification ---
    verifier_model: str = "cross-encoder/nli-deberta-v3-base"
    max_evidence: int = 5
    max_seq_length: int = 512

    # --- Attribution & Intervention ---
    shap_num_samples: int = 100
    lime_num_samples: int = 100
    max_attribution_evidence: int = 5
    max_intervention_evidence: int = 5

    # --- Fuzzy Reasoning ---
    fuzzy_engine_version: str = "v2"  # v1 | v2
    fuzzy_shap_weight: float = 0.5
    fuzzy_lime_weight: float = 0.5
    fuzzy_baseline_weights: dict = None  # defined post_init

    # --- Conflict & Stability (Phase 8) ---
    conflict_role_threshold: float = 0.5
    conflict_score_method: str = "harmonic_mean"  # harmonic_mean | min_max
    stability_perturbation: str = "permutation"

    # --- Device / Compute ---
    device: str = "cpu"  # cpu | cuda | mps

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Development ---
    debug: bool = False

    def __post_init__(self):
        if not self.project_root:
            self.project_root = str(Path(__file__).resolve().parent.parent)
        # Resolve relative paths against project root
        root = Path(self.project_root)
        for attr in ("data_dir", "cache_dir", "output_dir"):
            val = getattr(self, attr)
            if not Path(val).is_absolute():
                setattr(self, attr, str(root / val))
                
        if self.fuzzy_baseline_weights is None:
            self.fuzzy_baseline_weights = {
                "relevance": 0.30,
                "attribution_strength": 0.25,
                "intervention_influence": 0.25,
                "source_reliability": 0.10,
                "stability": 0.10
            }

    # --- I/O ---

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.
        Only keys matching dataclass fields are used; unknown keys are ignored.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)

    def to_yaml(self, path: str) -> None:
        """Save current configuration to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        """Create data, cache, and output directories if they don't exist."""
        for attr in ("data_dir", "cache_dir", "output_dir"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


# Singleton default config (importable as `from config.settings import cfg`)
cfg = Config()
