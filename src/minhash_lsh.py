"""
MinHash + LSH retrieval index (OOP).

Composition:
  Shingler          → text → shingle set
  MinHashSignature  → shingle set → signature vector
  LSHBands          → signature → bucket keys per band
  MinHashIndex      → orchestrates all three for indexing & query
"""

import re
import random
import hashlib
import zlib
from typing import List, Dict, Set

import numpy as np

from base import BaseIndex


_LARGE_PRIME = (1 << 31) - 1  # Mersenne prime 2^31 - 1


# ----------------------------------------------------------------------
# Helper components
# ----------------------------------------------------------------------

class Shingler:
    """Convert raw text into a set of hashed k-character shingles."""

    def __init__(self, k: int = 5):
        self.k = k

    def shingles(self, text: str) -> Set[int]:
        text = re.sub(r'\s+', ' ', text.lower()).strip()
        out = set()
        k = self.k
        for i in range(max(1, len(text) - k + 1)):
            # zlib.crc32 is deterministic across processes (unlike built-in
            # hash(), which is PYTHONHASHSEED-randomized) — required so the
            # SON-style parallel builder produces the same signatures as
            # the serial path.
            out.add(zlib.crc32(text[i: i + k].encode("utf-8")))
        return out


class MinHashSignature:
    """
    Compute MinHash signatures using n random universal hash functions
    of the form h(x) = (a*x + b) mod prime.
    """

    def __init__(self, n_hashes: int = 128, seed: int = 42):
        self.n_hashes = n_hashes
        rng = random.Random(seed)
        self._a = np.array(
            [rng.randint(1, _LARGE_PRIME) for _ in range(n_hashes)], dtype=np.int64
        )
        self._b = np.array(
            [rng.randint(0, _LARGE_PRIME) for _ in range(n_hashes)], dtype=np.int64
        )

    def signature(self, shingle_set: Set[int]) -> np.ndarray:
        sig = np.full(self.n_hashes, _LARGE_PRIME, dtype=np.int64)
        for s in shingle_set:
            hashes = (self._a * int(s) + self._b) % _LARGE_PRIME
            sig = np.minimum(sig, hashes)
        return sig

    @staticmethod
    def jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity between two signatures."""
        return float(np.mean(sig1 == sig2))


class LSHBands:
    """Hash MinHash signatures into bucket keys, one per band."""

    def __init__(self, n_hashes: int, n_bands: int):
        assert n_hashes % n_bands == 0, "n_hashes must be divisible by n_bands"
        self.n_bands = n_bands
        self.n_rows = n_hashes // n_bands
        self.threshold = (1.0 / n_bands) ** (1.0 / self.n_rows)

    def band_keys(self, signature: np.ndarray) -> List[int]:
        keys = []
        for b in range(self.n_bands):
            start = b * self.n_rows
            band_bytes = signature[start: start + self.n_rows].tobytes()
            keys.append(int(hashlib.md5(band_bytes).hexdigest(), 16) & 0xFFFFFFFF)
        return keys


# ----------------------------------------------------------------------
# Concrete index
# ----------------------------------------------------------------------

class MinHashIndex(BaseIndex):
    """MinHash + LSH approximate retrieval."""

    name = "minhash_lsh"

    def __init__(
        self,
        n_hashes: int = 128,
        n_bands: int = 16,
        shingle_k: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.shingle_k = shingle_k

        # Composition
        self._shingler = Shingler(k=shingle_k)
        self._hasher = MinHashSignature(n_hashes=n_hashes, seed=seed)
        self._bander = LSHBands(n_hashes=n_hashes, n_bands=n_bands)

        # State
        self._signatures: Dict[int, np.ndarray] = {}
        self._buckets: List[Dict[int, List[int]]] = [
            {} for _ in range(n_bands)
        ]

    # ---------------- public BaseIndex API ----------------

    def index(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        for chunk in chunks:
            cid = chunk["chunk_id"]
            shingles = self._shingler.shingles(chunk["text"])
            sig = self._hasher.signature(shingles)
            self._signatures[cid] = sig
            for b, bkey in enumerate(self._bander.band_keys(sig)):
                self._buckets[b].setdefault(bkey, []).append(cid)
        self._is_built = True

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        shingles = self._shingler.shingles(query_text)
        q_sig = self._hasher.signature(shingles)
        q_keys = self._bander.band_keys(q_sig)

        # Candidate generation: any chunk colliding in ≥1 band
        candidates: Dict[int, int] = {}
        for b, bkey in enumerate(q_keys):
            for cid in self._buckets[b].get(bkey, []):
                candidates[cid] = candidates.get(cid, 0) + 1

        if not candidates:
            candidates = {cid: 0 for cid in self._signatures}

        scored = [
            (cid, self._hasher.jaccard(q_sig, self._signatures[cid]))
            for cid in candidates
        ]
        scored.sort(key=lambda x: -x[1])
        top = scored[:top_k]

        chunk_map = {c["chunk_id"]: c for c in self._chunks}
        results = []
        for cid, sim in top:
            c = chunk_map[cid].copy()
            c["score"] = round(sim, 4)
            c["method"] = self.name
            results.append(c)
        return results

    def info(self) -> Dict:
        return {
            "name": self.name,
            "n_hashes": self.n_hashes,
            "n_bands": self.n_bands,
            "n_rows": self._bander.n_rows,
            "shingle_k": self.shingle_k,
            "approx_threshold": round(self._bander.threshold, 4),
            "n_chunks_indexed": len(self._signatures),
        }
