"""
Unified Retriever (OOP).

Holds a registry of BaseIndex implementations and exposes a single API
for indexing, querying, comparing methods, and profiling.
"""

import time
import tracemalloc
from typing import List, Dict, Tuple

from base import BaseIndex
from minhash_lsh import MinHashIndex
from simhash import SimHashIndex
from tfidf_baseline import TFIDFIndex


class Retriever:
    """
    Facade over multiple BaseIndex implementations.

    The registry maps a method name (e.g. 'tfidf') to a BaseIndex instance.
    Add or replace methods at construction time, or call `register()`.
    """

    def __init__(
        self,
        n_hashes: int = 128,
        n_bands: int = 16,
        shingle_k: int = 5,
        hamming_threshold: int = 10,
        indices: Dict[str, BaseIndex] = None,
    ):
        # Default registry: TF-IDF (exact) + MinHash+LSH + SimHash
        self._indices: Dict[str, BaseIndex] = indices or {
            "tfidf": TFIDFIndex(),
            "minhash": MinHashIndex(
                n_hashes=n_hashes, n_bands=n_bands, shingle_k=shingle_k
            ),
            "simhash": SimHashIndex(hamming_threshold=hamming_threshold),
        }
        self.chunks: List[Dict] = []
        self._build_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, index: BaseIndex) -> None:
        """Add or replace a method in the registry."""
        self._indices[name] = index

    @property
    def methods(self) -> List[str]:
        return list(self._indices.keys())

    def get_index(self, method: str) -> BaseIndex:
        if method not in self._indices:
            raise KeyError(
                f"Unknown method '{method}'. Available: {self.methods}"
            )
        return self._indices[method]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict]) -> Dict:
        """Index all registered methods. Returns build timing stats."""
        self.chunks = chunks
        stats = {}
        for name, idx in self._indices.items():
            t0 = time.perf_counter()
            idx.index(chunks)
            stats[f"{name}_build_s"] = round(time.perf_counter() - t0, 4)
        stats["n_chunks"] = len(chunks)
        self._build_stats = stats
        return stats

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, query_text: str, method: str = "tfidf", top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """Run a single query. Returns (results, latency_ms)."""
        idx = self.get_index(method)
        t0 = time.perf_counter()
        results = idx.query(query_text, top_k)
        return results, (time.perf_counter() - t0) * 1000

    def query_all(
        self, query_text: str, top_k: int = 5
    ) -> Dict[str, Tuple[List[Dict], float]]:
        """Query every registered method. Returns {method: (results, ms)}."""
        return {name: self.query(query_text, name, top_k) for name in self.methods}

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile_memory(self, query_text: str, method: str, top_k: int = 5) -> Dict:
        tracemalloc.start()
        results, latency = self.query(query_text, method, top_k)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "method": method,
            "latency_ms": round(latency, 3),
            "peak_memory_kb": round(peak / 1024, 2),
            "n_results": len(results),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        return {
            "n_chunks": len(self.chunks),
            "methods": {name: idx.info() for name, idx in self._indices.items()},
            "build_stats": self._build_stats,
        }
