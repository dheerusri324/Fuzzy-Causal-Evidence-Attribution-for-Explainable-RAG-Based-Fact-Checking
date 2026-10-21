"""
Tests for Phase 3 — Hybrid RAG Retrieval.

Tests are split into two groups:
1. BM25-only tests (no model download, fast).
2. Integration tests (require sentence-transformers model, slower).

All tests use a small synthetic corpus fixture.
"""

import json
import pytest
from pathlib import Path

from retrieval.schema import RetrievedEvidence
from retrieval.bm25 import BM25Index, _tokenize


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

MINI_CORPUS = [
    {
        "evidence_id": "JWST_0",
        "text": "The James Webb Space Telescope was launched on December 25, 2021.",
        "doc_title": "JWST",
        "sentence_idx": 0,
        "source": "fever",
        "source_type": "wikipedia",
    },
    {
        "evidence_id": "JWST_1",
        "text": "JWST is the largest optical telescope in space.",
        "doc_title": "JWST",
        "sentence_idx": 1,
        "source": "fever",
        "source_type": "wikipedia",
    },
    {
        "evidence_id": "Moon_0",
        "text": "The Moon is Earth's only natural satellite.",
        "doc_title": "Moon",
        "sentence_idx": 0,
        "source": "fever",
        "source_type": "wikipedia",
    },
    {
        "evidence_id": "Python_0",
        "text": "Python is a high-level programming language.",
        "doc_title": "Python_(programming_language)",
        "sentence_idx": 0,
        "source": "fever",
        "source_type": "wikipedia",
    },
    {
        "evidence_id": "Einstein_0",
        "text": "Albert Einstein developed the theory of relativity.",
        "doc_title": "Albert_Einstein",
        "sentence_idx": 0,
        "source": "fever",
        "source_type": "wikipedia",
    },
]


@pytest.fixture
def mini_corpus_path(tmp_path):
    """Write mini corpus to a temp JSON file."""
    path = tmp_path / "evidence_corpus.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(MINI_CORPUS, f)
    return str(path)


@pytest.fixture
def mini_texts_and_ids():
    texts = [item["text"] for item in MINI_CORPUS]
    ids = [item["evidence_id"] for item in MINI_CORPUS]
    return texts, ids


# ------------------------------------------------------------------
# BM25 unit tests (no model download)
# ------------------------------------------------------------------

class TestBM25:
    def test_tokenize_deterministic(self):
        t1 = _tokenize("Hello, World! Test.")
        t2 = _tokenize("Hello, World! Test.")
        assert t1 == t2
        assert t1 == ["hello", "world", "test"]

    def test_index_creation(self, mini_texts_and_ids):
        texts, ids = mini_texts_and_ids
        idx = BM25Index(texts, ids)
        assert len(idx.corpus_ids) == 5

    def test_search_returns_results(self, mini_texts_and_ids):
        texts, ids = mini_texts_and_ids
        idx = BM25Index(texts, ids)
        results = idx.search("James Webb Space Telescope launched", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_top_k_respected(self, mini_texts_and_ids):
        texts, ids = mini_texts_and_ids
        idx = BM25Index(texts, ids)
        results = idx.search("telescope", top_k=2)
        assert len(results) == 2

    def test_search_deterministic(self, mini_texts_and_ids):
        texts, ids = mini_texts_and_ids
        idx = BM25Index(texts, ids)
        r1 = idx.search("Albert Einstein relativity")
        r2 = idx.search("Albert Einstein relativity")
        assert r1 == r2

    def test_jwst_ranked_first(self, mini_texts_and_ids):
        texts, ids = mini_texts_and_ids
        idx = BM25Index(texts, ids)
        results = idx.search("JWST launched 2021", top_k=5)
        top_idx = results[0][0]
        assert ids[top_idx] == "JWST_0"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            BM25Index(["a", "b"], ["id1"])


# ------------------------------------------------------------------
# Integration tests (require sentence-transformers + FAISS)
# ------------------------------------------------------------------

class TestDenseRetrieval:
    def test_dense_search_returns_results(self, mini_corpus_path, tmp_path):
        from retrieval.dense import DenseIndex

        idx = DenseIndex(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            cache_dir=str(tmp_path / "cache"),
        )
        texts = [item["text"] for item in MINI_CORPUS]
        ids = [item["evidence_id"] for item in MINI_CORPUS]
        idx.build(texts, ids)

        results = idx.search("James Webb Space Telescope", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)

    def test_dense_cache_reuse(self, mini_corpus_path, tmp_path):
        from retrieval.dense import DenseIndex

        cache_dir = str(tmp_path / "cache")
        texts = [item["text"] for item in MINI_CORPUS]
        ids = [item["evidence_id"] for item in MINI_CORPUS]

        # First build
        idx1 = DenseIndex(model_name="all-MiniLM-L6-v2", device="cpu", cache_dir=cache_dir)
        idx1.build(texts, ids)
        r1 = idx1.search("telescope", top_k=2)

        # Second build should load from cache
        idx2 = DenseIndex(model_name="all-MiniLM-L6-v2", device="cpu", cache_dir=cache_dir)
        idx2.build(texts, ids)
        r2 = idx2.search("telescope", top_k=2)

        assert [r[0] for r in r1] == [r[0] for r in r2]

    def test_dense_top_k_respected(self, tmp_path):
        from retrieval.dense import DenseIndex

        idx = DenseIndex(model_name="all-MiniLM-L6-v2", device="cpu", cache_dir=str(tmp_path))
        texts = [item["text"] for item in MINI_CORPUS]
        ids = [item["evidence_id"] for item in MINI_CORPUS]
        idx.build(texts, ids)
        results = idx.search("moon satellite", top_k=2)
        assert len(results) == 2


class TestEvidenceRetriever:
    def test_bm25_mode(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(
            cache_dir=str(tmp_path / "cache"),
            retrieval_mode="bm25",
            top_k=3,
        )
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        results = retriever.retrieve("JWST launched in 2021")

        assert len(results) == 3
        assert all(isinstance(r, RetrievedEvidence) for r in results)
        assert results[0].retrieval_rank == 1
        assert results[0].retrieval_method == "bm25"

    def test_dense_mode(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(
            cache_dir=str(tmp_path / "cache"),
            retrieval_mode="dense",
            top_k=3,
        )
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        results = retriever.retrieve("telescope in space")

        assert len(results) == 3
        assert results[0].retrieval_method == "dense"

    def test_hybrid_mode(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(
            cache_dir=str(tmp_path / "cache"),
            retrieval_mode="hybrid",
            top_k=3,
        )
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        results = retriever.retrieve("James Webb Space Telescope")

        assert len(results) == 3
        assert results[0].retrieval_method == "hybrid"

    def test_output_schema_complete(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(cache_dir=str(tmp_path / "cache"), top_k=2)
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        results = retriever.retrieve("Einstein relativity", mode="bm25")

        ev = results[0]
        required_fields = [
            "evidence_id", "text", "doc_title", "sentence_idx",
            "source", "source_type", "retrieval_score",
            "retrieval_rank", "retrieval_method",
        ]
        d = ev.to_dict()
        for field in required_fields:
            assert field in d, f"Missing field: {field}"

    def test_invalid_mode_raises(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(cache_dir=str(tmp_path / "cache"))
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        with pytest.raises(ValueError, match="Invalid mode"):
            retriever.retrieve("test", mode="invalid")

    def test_deterministic_retrieval(self, mini_corpus_path, tmp_path):
        from retrieval.retriever import EvidenceRetriever
        from config.settings import Config

        cfg = Config(cache_dir=str(tmp_path / "cache"), top_k=3)
        retriever = EvidenceRetriever(corpus_path=mini_corpus_path, config=cfg)
        r1 = retriever.retrieve("moon satellite earth", mode="hybrid")
        r2 = retriever.retrieve("moon satellite earth", mode="hybrid")
        ids1 = [r.evidence_id for r in r1]
        ids2 = [r.evidence_id for r in r2]
        assert ids1 == ids2
