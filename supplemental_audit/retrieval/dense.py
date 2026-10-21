"""
Dense semantic retrieval using sentence-transformers + FAISS.

Embeds the evidence corpus, builds a FAISS inner-product index,
and caches everything to disk so subsequent runs skip recomputation.

Cache is versioned by model name + corpus hash to prevent stale indexes.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from utils.logging_setup import get_logger

logger = get_logger(__name__)


class DenseIndex:
    """Dense retrieval index backed by sentence-transformers + FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: str = "cache",
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self._model = None
        self.index = None
        self.corpus_ids: Optional[List[str]] = None
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _safe_model_name(self) -> str:
        return self.model_name.replace("/", "__").replace("\\", "__")

    def _corpus_hash(self, corpus_ids: List[str]) -> str:
        """Short hash of sorted corpus IDs for cache versioning."""
        content = "|".join(sorted(corpus_ids))
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, corpus_hash: str) -> Path:
        return self.cache_dir / f"dense_{self._safe_model_name()}_{corpus_hash}"

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    # ------------------------------------------------------------------
    # Build / load index
    # ------------------------------------------------------------------

    def build(self, corpus_texts: List[str], corpus_ids: List[str]) -> None:
        """Build (or load from cache) the FAISS index for the given corpus.

        If a compatible cache exists (same model + corpus hash), loads from
        disk without recomputing embeddings.
        """
        import faiss

        self.corpus_ids = list(corpus_ids)
        corpus_hash = self._corpus_hash(corpus_ids)
        cache_path = self._cache_path(corpus_hash)

        emb_file = cache_path / "embeddings.npy"
        idx_file = cache_path / "index.faiss"
        ids_file = cache_path / "corpus_ids.json"

        # Try loading from cache
        if emb_file.exists() and idx_file.exists() and ids_file.exists():
            logger.info(f"Loading dense index from cache: {cache_path}")
            self.embeddings = np.load(str(emb_file))
            self.index = faiss.read_index(str(idx_file))
            with open(ids_file, "r", encoding="utf-8") as f:
                self.corpus_ids = json.load(f)
            logger.info(
                f"Dense index loaded: {self.embeddings.shape[0]} vectors, "
                f"dim={self.embeddings.shape[1]}"
            )
            return

        # Compute embeddings
        self._load_model()
        logger.info(f"Encoding {len(corpus_texts)} documents...")
        self.embeddings = self._model.encode(
            corpus_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        # Build FAISS index (inner product on L2-normalized vectors = cosine sim)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        # Save cache
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_file), self.embeddings)
        faiss.write_index(self.index, str(idx_file))
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(self.corpus_ids, f)

        logger.info(
            f"Dense index built and cached: {self.embeddings.shape[0]} vectors, "
            f"dim={dim} → {cache_path}"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search the FAISS index for the nearest evidence to the query.

        Returns list of (corpus_index, cosine_similarity) sorted descending.
        """
        if self.index is None:
            raise RuntimeError("Dense index not built. Call build() first.")

        self._load_model()
        q_emb = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx >= 0:  # FAISS uses -1 for padding
                results.append((idx, float(scores[0][i])))
        return results
