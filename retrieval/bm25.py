"""
BM25 lexical retrieval index.

Uses rank_bm25 (BM25Okapi) with simple reproducible tokenization.
Deterministic for a given corpus and query.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple, reproducible lowercase tokenization.

    Strips punctuation, lowercases, splits on whitespace.
    Deterministic for the same input.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


class BM25Index:
    """BM25Okapi index over a list of evidence texts."""

    def __init__(self, corpus_texts: List[str], corpus_ids: List[str]):
        if len(corpus_texts) != len(corpus_ids):
            raise ValueError("corpus_texts and corpus_ids must have equal length")

        self.corpus_ids = list(corpus_ids)
        self._tokenized = [_tokenize(t) for t in corpus_texts]
        self._index = BM25Okapi(self._tokenized)
        logger.info(f"BM25 index built over {len(corpus_texts)} documents.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search the index and return (corpus_index, score) pairs.

        Results are sorted by score descending, limited to top_k.
        """
        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Argsort descending, take top-k
        top_k = min(top_k, len(self.corpus_ids))
        top_indices = scores.argsort()[-top_k:][::-1]

        return [(int(idx), float(scores[idx])) for idx in top_indices]
