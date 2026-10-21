"""
Data schema for the normalized internal representation.

Defines the canonical record format used across all modules:
retrieval, verification, attribution, intervention, etc.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# Valid labels for the claim-evidence verification task
VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


@dataclass
class EvidencePiece:
    """A single piece of evidence associated with a claim."""

    evidence_id: str          # Unique ID: "{doc_title}_{sentence_idx}"
    text: str                 # The evidence sentence text
    doc_title: str            # Source document title
    sentence_idx: int         # Sentence index within the document
    source: str = "fever"     # Dataset source
    source_type: str = "wikipedia"  # Source type (wikipedia for FEVER)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidencePiece":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClaimRecord:
    """A normalized claim record with associated evidence."""

    claim_id: str             # Unique claim identifier
    claim_text: str           # The claim string
    label: str                # SUPPORTS / REFUTES / NOT_ENOUGH_INFO
    evidence: List[EvidencePiece] = field(default_factory=list)
    original_id: Optional[int] = None     # Original dataset ID
    verifiable: Optional[str] = None      # VERIFIABLE / NOT VERIFIABLE
    dataset: str = "fever"                # Source dataset name
    split: str = "train"                  # Dataset split

    def __post_init__(self):
        # Normalize label
        self.label = self.label.upper().replace(" ", "_")
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{self.label}'. Must be one of {VALID_LABELS}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimRecord":
        evidence_dicts = d.pop("evidence", [])
        evidence = [EvidencePiece.from_dict(e) for e in evidence_dicts]
        valid_keys = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(evidence=evidence, **valid_keys)


def save_records(records: List[ClaimRecord], path: str) -> None:
    """Save a list of ClaimRecords to a JSON Lines file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def load_records(path: str) -> List[ClaimRecord]:
    """Load a list of ClaimRecords from a JSON Lines file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ClaimRecord.from_dict(json.loads(line)))
    return records
