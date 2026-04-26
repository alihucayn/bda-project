"""
TF-IDF + cosine similarity baseline (exact retrieval).
"""

from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from base import BaseIndex


class TFIDFIndex(BaseIndex):
    """
    Exact retrieval baseline: TF-IDF vectorization + cosine similarity.

    O(n) per query — brute-force scan, no approximation.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )
        self._matrix = None

    # ---------------- public BaseIndex API ----------------

    def index(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        texts = [c["text"] for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)
        self._is_built = True

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        q_vec = self._vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            c = self._chunks[idx].copy()
            c["score"] = round(float(sims[idx]), 4)
            c["method"] = self.name
            results.append(c)
        return results

    def info(self) -> Dict:
        return {
            "name": self.name,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "n_chunks_indexed": self.n_chunks,
            "vocabulary_size": len(self._vectorizer.vocabulary_) if self._is_built else 0,
        }
