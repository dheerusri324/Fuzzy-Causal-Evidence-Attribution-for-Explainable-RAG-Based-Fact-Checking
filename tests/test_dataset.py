"""
Tests for Phase 2 — Dataset / Evidence Corpus.

Validates:
1. Schema dataclass creation and serialization.
2. FEVER normalization logic.
3. Label validation.
4. Evidence deduplication.
5. Dev subset balance.
6. JSONL round-trip (save → load).
"""

import json
import pytest
from data.schema import ClaimRecord, EvidencePiece, save_records, load_records, VALID_LABELS
from data.fever_loader import (
    _normalize_record,
    _clean_text,
    create_dev_subset,
    extract_evidence_corpus,
)


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------

class TestSchema:
    def test_evidence_piece_creation(self):
        ep = EvidencePiece(
            evidence_id="Test_Doc_3",
            text="Some evidence text.",
            doc_title="Test_Doc",
            sentence_idx=3,
        )
        assert ep.evidence_id == "Test_Doc_3"
        assert ep.source == "fever"
        assert ep.source_type == "wikipedia"

    def test_claim_record_creation(self):
        rec = ClaimRecord(
            claim_id="abc123",
            claim_text="The sky is blue.",
            label="SUPPORTS",
            evidence=[],
        )
        assert rec.label == "SUPPORTS"
        assert rec.dataset == "fever"

    def test_label_normalization(self):
        rec = ClaimRecord(
            claim_id="x", claim_text="test", label="not enough info"
        )
        assert rec.label == "NOT_ENOUGH_INFO"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid label"):
            ClaimRecord(claim_id="x", claim_text="test", label="MAYBE")

    def test_round_trip_dict(self):
        ep = EvidencePiece(
            evidence_id="D_1", text="Evidence.", doc_title="D", sentence_idx=1
        )
        rec = ClaimRecord(
            claim_id="c1", claim_text="Claim.", label="REFUTES",
            evidence=[ep], original_id=42,
        )
        d = rec.to_dict()
        rec2 = ClaimRecord.from_dict(d)
        assert rec2.claim_id == "c1"
        assert rec2.label == "REFUTES"
        assert len(rec2.evidence) == 1
        assert rec2.evidence[0].text == "Evidence."
        assert rec2.original_id == 42

    def test_jsonl_round_trip(self, tmp_path):
        records = [
            ClaimRecord(
                claim_id=f"c{i}", claim_text=f"Claim {i}",
                label=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"][i % 3],
                evidence=[EvidencePiece(
                    evidence_id=f"e{i}", text=f"Ev {i}",
                    doc_title=f"doc{i}", sentence_idx=0,
                )],
            )
            for i in range(10)
        ]
        path = str(tmp_path / "test.jsonl")
        save_records(records, path)
        loaded = load_records(path)
        assert len(loaded) == 10
        assert loaded[0].claim_id == "c0"
        assert loaded[0].evidence[0].text == "Ev 0"


# ------------------------------------------------------------------
# FEVER normalization tests
# ------------------------------------------------------------------

class TestFeverNormalization:
    def test_clean_text(self):
        raw = "founded in -LRB- 2010 -RRB- by  someone"
        assert _clean_text(raw) == "founded in ( 2010 ) by someone"

    def test_normalize_valid_record(self):
        raw = {
            "claim": "Test claim.",
            "label": "SUPPORTS",
            "evidence": [["Doc_Title", "5", "Evidence text."]],
            "id": "abc",
            "original_id": 100,
            "verifiable": "VERIFIABLE",
        }
        rec = _normalize_record(raw, split="train")
        assert rec is not None
        assert rec.claim_text == "Test claim."
        assert rec.label == "SUPPORTS"
        assert len(rec.evidence) == 1
        assert rec.evidence[0].evidence_id == "Doc_Title_5"
        assert rec.evidence[0].doc_title == "Doc_Title"
        assert rec.evidence[0].sentence_idx == 5

    def test_normalize_nei_label(self):
        raw = {
            "claim": "Test.", "label": "NOT ENOUGH INFO",
            "evidence": [["D", "0", "Ev."]], "id": "x",
        }
        rec = _normalize_record(raw)
        assert rec.label == "NOT_ENOUGH_INFO"

    def test_normalize_empty_claim_returns_none(self):
        raw = {"claim": "", "label": "SUPPORTS", "evidence": [], "id": "x"}
        assert _normalize_record(raw) is None

    def test_normalize_invalid_label_returns_none(self):
        raw = {"claim": "Test.", "label": "UNKNOWN", "evidence": [], "id": "x"}
        assert _normalize_record(raw) is None

    def test_normalize_deduplicates_evidence(self):
        raw = {
            "claim": "Test.", "label": "SUPPORTS", "id": "x",
            "evidence": [
                ["Doc", "1", "Same evidence."],
                ["Doc", "1", "Same evidence."],  # duplicate
                ["Doc", "2", "Different evidence."],
            ],
        }
        rec = _normalize_record(raw)
        assert len(rec.evidence) == 2  # deduped


# ------------------------------------------------------------------
# Dev subset tests
# ------------------------------------------------------------------

class TestDevSubset:
    def _make_records(self, n_per_label=100):
        records = []
        for i in range(n_per_label * 3):
            label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"][i % 3]
            records.append(ClaimRecord(
                claim_id=f"c{i}", claim_text=f"Claim {i}", label=label,
                evidence=[EvidencePiece(
                    evidence_id=f"e{i}", text=f"Ev {i}",
                    doc_title=f"d{i}", sentence_idx=0,
                )],
            ))
        return records

    def test_balanced_subset(self):
        records = self._make_records(100)
        subset = create_dev_subset(records, per_label=30, seed=42)
        from collections import Counter
        dist = Counter(r.label for r in subset)
        assert dist["SUPPORTS"] == 30
        assert dist["REFUTES"] == 30
        assert dist["NOT_ENOUGH_INFO"] == 30
        assert len(subset) == 90

    def test_deterministic(self):
        records = self._make_records(100)
        s1 = create_dev_subset(records, per_label=20, seed=42)
        s2 = create_dev_subset(records, per_label=20, seed=42)
        ids1 = [r.claim_id for r in s1]
        ids2 = [r.claim_id for r in s2]
        assert ids1 == ids2

    def test_handles_insufficient_data(self):
        records = self._make_records(10)
        subset = create_dev_subset(records, per_label=50, seed=42)
        # Should take all available (10 per label, not 50)
        assert len(subset) == 30


# ------------------------------------------------------------------
# Evidence corpus tests
# ------------------------------------------------------------------

class TestEvidenceCorpus:
    def test_deduplication(self):
        ep = EvidencePiece(
            evidence_id="d_1", text="Shared evidence.",
            doc_title="d", sentence_idx=1,
        )
        records = [
            ClaimRecord(claim_id="c1", claim_text="C1", label="SUPPORTS",
                        evidence=[ep]),
            ClaimRecord(claim_id="c2", claim_text="C2", label="REFUTES",
                        evidence=[ep]),
        ]
        corpus = extract_evidence_corpus(records)
        assert len(corpus) == 1  # same evidence_id → deduplicated

    def test_corpus_fields(self):
        ep = EvidencePiece(
            evidence_id="d_0", text="Text.", doc_title="d", sentence_idx=0,
        )
        records = [
            ClaimRecord(claim_id="c1", claim_text="C1", label="SUPPORTS",
                        evidence=[ep]),
        ]
        corpus = extract_evidence_corpus(records)
        assert corpus[0]["evidence_id"] == "d_0"
        assert corpus[0]["text"] == "Text."
        assert corpus[0]["source_type"] == "wikipedia"
