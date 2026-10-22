"""
Main retrieval interface — EvidenceRetriever.

Provides a single clean API for all retrieval modes (bm25, dense, hybrid).
Downstream modules (verification, attribution, intervention) interact only
with this interface and do not need to know internal retrieval details.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from retrieval.schema import RetrievedEvidence
from retrieval.bm25 import BM25Index, _tokenize
from retrieval.dense import DenseIndex
from config.settings import Config
from utils.logging_setup import get_logger

logger = get_logger(__name__)

VALID_MODES = {"bm25", "dense", "hybrid"}


class EvidenceRetriever:
    """Unified retrieval interface over the evidence corpus.

    Supports three modes:
        bm25   — lexical retrieval via BM25Okapi
        dense  — semantic retrieval via sentence-transformers + FAISS
        hybrid — weighted fusion of normalized BM25 + dense scores

    Usage:
        retriever = EvidenceRetriever(corpus_path="data/evidence_corpus.json")
        results = retriever.retrieve("JWST was launched in 2021", top_k=5, mode="hybrid")
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or Config()

        if corpus_path is None:
            corpus_path = str(Path(self.config.data_dir) / "evidence_corpus.json")

        # Load corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus: List[dict] = json.load(f)

        self.corpus_texts = [item["text"] for item in self.corpus]
        self.corpus_ids = [item["evidence_id"] for item in self.corpus]
        self._corpus_lookup = {item["evidence_id"]: item for item in self.corpus}

        # Lazy-initialized indexes
        self._bm25: Optional[BM25Index] = None
        self._dense: Optional[DenseIndex] = None

        logger.info(
            f"EvidenceRetriever initialized: {len(self.corpus)} evidence pieces."
        )

    # ------------------------------------------------------------------
    # Lazy index accessors
    # ------------------------------------------------------------------

    @property
    def bm25(self) -> BM25Index:
        """BM25 index, built on first access."""
        if self._bm25 is None:
            self._bm25 = BM25Index(self.corpus_texts, self.corpus_ids)
        return self._bm25

    @property
    def dense(self) -> DenseIndex:
        """Dense index, built/loaded-from-cache on first access."""
        if self._dense is None:
            self._dense = DenseIndex(
                model_name=self.config.embedding_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
            )
            self._dense.build(self.corpus_texts, self.corpus_ids)
        return self._dense

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        claim: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        alpha: float = 0.5,
    ) -> List[RetrievedEvidence]:
        """Retrieve top-k evidence for a claim.

        Args:
            claim: The claim text to find evidence for.
            top_k: Number of results (default: config.top_k).
            mode:  "bm25", "dense", or "hybrid" (default: config.retrieval_mode).
            alpha: Hybrid fusion weight. 0 = pure dense, 1 = pure BM25.
                   Only used when mode="hybrid". Default: 0.5.

        Returns:
            List of RetrievedEvidence sorted by retrieval_score descending.
        """
        top_k = top_k or self.config.top_k
        mode = mode or self.config.retrieval_mode

        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Use one of {VALID_MODES}")

        if mode == "bm25":
            return self._retrieve_bm25(claim, top_k)
        elif mode == "dense":
            return self._retrieve_dense(claim, top_k)
        else:
            return self._retrieve_hybrid(claim, top_k, alpha)

    # ------------------------------------------------------------------
    # Internal retrieval methods
    # ------------------------------------------------------------------

    def _retrieve_bm25(
        self, claim: str, top_k: int
    ) -> List[RetrievedEvidence]:
        results = self.bm25.search(claim, top_k)
        return self._format_results(results, "bm25")

    def _retrieve_dense(
        self, claim: str, top_k: int
    ) -> List[RetrievedEvidence]:
        results = self.dense.search(claim, top_k)
        return self._format_results(results, "dense")

    def _retrieve_hybrid(
        self, claim: str, top_k: int, alpha: float
    ) -> List[RetrievedEvidence]:
        fusion_method = getattr(self.config, "hybrid_fusion_method", "pool_union")

        if fusion_method == "legacy":
            # Exact historical Phase 10 baseline fusion for backwards compatibility
            n_candidates = min(top_k * 3, len(self.corpus))
            bm25_raw = self.bm25.search(claim, n_candidates)
            dense_raw = self.dense.search(claim, n_candidates)
            bm25_norm = self._normalize_scores(bm25_raw)
            dense_norm = self._normalize_scores(dense_raw)
            combined = {}
            for idx, score in bm25_norm:
                combined[idx] = alpha * score
            for idx, score in dense_norm:
                combined[idx] = combined.get(idx, 0.0) + (1.0 - alpha) * score
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return self._format_results(sorted_results, "hybrid")

        # Fetch candidate pool from both methods (v2 remediated fusion)
        n_candidates = min(top_k * 3, len(self.corpus))

        bm25_raw = self.bm25.search(claim, n_candidates)
        dense_raw = self.dense.search(claim, n_candidates)

        # Build union of unique candidate indices
        candidate_indices = []
        seen = set()
        for idx, _ in bm25_raw + dense_raw:
            if idx not in seen:
                seen.add(idx)
                candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # Get genuine BM25 scores for all candidates in the candidate pool
        tokenized_query = _tokenize(claim)
        all_bm25_scores = self.bm25._index.get_scores(tokenized_query)
        bm25_cand_raw = [(idx, float(all_bm25_scores[idx])) for idx in candidate_indices]

        # Get genuine dense cosine similarities for all candidates in the candidate pool
        dense_dict = dict(dense_raw)
        missing_dense = [idx for idx in candidate_indices if idx not in dense_dict]
        if missing_dense and getattr(self.dense, "embeddings", None) is not None:
            self.dense._load_model()
            q_emb = self.dense._model.encode([claim], normalize_embeddings=True).astype(np.float32)
            sub_emb = self.dense.embeddings[missing_dense]
            missing_scores = np.dot(sub_emb, q_emb[0])
            for idx, sc in zip(missing_dense, missing_scores):
                dense_dict[idx] = float(sc)

        dense_cand_raw = [(idx, dense_dict.get(idx, 0.0)) for idx in candidate_indices]

        # Normalize each set of scores across the candidate pool to [0, 1]
        bm25_norm_dict = dict(self._normalize_scores(bm25_cand_raw))
        dense_norm_dict = dict(self._normalize_scores(dense_cand_raw))

        # Weighted fusion
        combined: dict[int, float] = {}
        for idx in candidate_indices:
            b_sc = bm25_norm_dict.get(idx, 0.0)
            d_sc = dense_norm_dict.get(idx, 0.0)
            combined[idx] = alpha * b_sc + (1.0 - alpha) * d_sc

        # Sort descending, take top-k
        sorted_results = sorted(
            combined.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        return self._format_results(sorted_results, "hybrid")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        results: List[tuple],
    ) -> List[tuple]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return results
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s
        if rng < 1e-9:
            return [(idx, 1.0) for idx, _ in results]
        return [(idx, (s - min_s) / rng) for idx, s in results]

    def _format_results(
        self, results: List[tuple], method: str
    ) -> List[RetrievedEvidence]:
        """Convert (corpus_index, score) pairs to RetrievedEvidence objects."""
        formatted = []
        for rank, (idx, score) in enumerate(results, start=1):
            ev_id = self.corpus_ids[idx]
            meta = self._corpus_lookup[ev_id]
            formatted.append(
                RetrievedEvidence(
                    evidence_id=ev_id,
                    text=meta["text"],
                    doc_title=meta["doc_title"],
                    sentence_idx=meta["sentence_idx"],
                    source=meta.get("source", "fever"),
                    source_type=meta.get("source_type", "wikipedia"),
                    retrieval_score=round(score, 6),
                    retrieval_rank=rank,
                    retrieval_method=method,
                )
            )
        return formatted
