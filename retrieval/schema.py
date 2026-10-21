"""
Retrieved evidence schema.

Defines the output format for all retrieval methods (BM25, dense, hybrid).
This is the contract between the retrieval module and all downstream consumers
(verification, attribution, intervention, etc.).
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class RetrievedEvidence:
    """A single retrieved evidence item with full metadata."""

    evidence_id: str          # Unique evidence identifier
    text: str                 # Evidence text
    doc_title: str            # Source document title
    sentence_idx: int         # Sentence index within document
    source: str               # Dataset source (e.g. "fever")
    source_type: str          # Source type (e.g. "wikipedia")
    retrieval_score: float    # Score from the retrieval method
    retrieval_rank: int       # Rank (1 = best)
    retrieval_method: str     # "bm25" | "dense" | "hybrid"

    def to_dict(self) -> dict:
        return asdict(self)
