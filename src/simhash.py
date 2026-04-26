"""
SimHash retrieval index (OOP).

Composition:
  Tokenizer       → text → weighted token map
  Fingerprinter   → token map → 64-bit fingerprint
  SimHashIndex    → orchestrates them for indexing & query
"""

import re
import hashlib
from collections import Counter
from typing import List, Dict

import numpy as np

from base import BaseIndex


FINGERPRINT_BITS = 64


# ----------------------------------------------------------------------
# Helper components
# ----------------------------------------------------------------------

class Tokenizer:
    """Tokenize text and return TF-weighted token weights."""

    def __init__(self, min_len: int = 2):
        self.min_len = min_len
        self._pattern = re.compile(rf'[a-z]{{{min_len},}}')

    def weights(self, text: str) -> Dict[str, float]:
        tokens = self._pattern.findall(text.lower())
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()}


class Fingerprinter:
    """Compute weighted SimHash fingerprints over a fixed-size bit vector."""

    def __init__(self, bits: int = FINGERPRINT_BITS):
        self.bits = bits

    def _token_bits(self, token: str) -> np.ndarray:
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        out = np.empty(self.bits, dtype=np.float64)
        for i in range(self.bits):
            out[i] = 1.0 if (digest >> i) & 1 else -1.0
        return out

    def fingerprint(self, weighted_tokens: Dict[str, float]) -> int:
        if not weighted_tokens:
            return 0
        vec = np.zeros(self.bits, dtype=np.float64)
        for tok, w in weighted_tokens.items():
            vec += w * self._token_bits(tok)
        fp = 0
        for i in range(self.bits):
            if vec[i] > 0:
                fp |= (1 << i)
        return fp

    @staticmethod
    def hamming_distance(fp1: int, fp2: int) -> int:
        x = fp1 ^ fp2
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    def hamming_similarity(self, fp1: int, fp2: int) -> float:
        return 1.0 - self.hamming_distance(fp1, fp2) / self.bits


# ----------------------------------------------------------------------
# Concrete index
# ----------------------------------------------------------------------

class SimHashIndex(BaseIndex):
    """SimHash approximate retrieval based on Hamming distance."""

    name = "simhash"

    def __init__(self, hamming_threshold: int = 10, bits: int = FINGERPRINT_BITS):
        super().__init__()
        self.hamming_threshold = hamming_threshold
        self.bits = bits

        self._tokenizer = Tokenizer()
        self._fingerprinter = Fingerprinter(bits=bits)

        self._fingerprints: Dict[int, int] = {}

    # ---------------- public BaseIndex API ----------------

    def index(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        for chunk in chunks:
            weights = self._tokenizer.weights(chunk["text"])
            self._fingerprints[chunk["chunk_id"]] = self._fingerprinter.fingerprint(weights)
        self._is_built = True

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        weights = self._tokenizer.weights(query_text)
        q_fp = self._fingerprinter.fingerprint(weights)

        scored = []
        for cid, fp in self._fingerprints.items():
            dist = Fingerprinter.hamming_distance(q_fp, fp)
            sim = self._fingerprinter.hamming_similarity(q_fp, fp)
            scored.append((cid, dist, sim))
        scored.sort(key=lambda x: x[1])
        top = scored[:top_k]

        chunk_map = {c["chunk_id"]: c for c in self._chunks}
        results = []
        for cid, dist, sim in top:
            c = chunk_map[cid].copy()
            c["score"] = round(sim, 4)
            c["hamming_distance"] = dist
            c["method"] = self.name
            results.append(c)
        return results

    def info(self) -> Dict:
        return {
            "name": self.name,
            "fingerprint_bits": self.bits,
            "hamming_threshold": self.hamming_threshold,
            "n_chunks_indexed": len(self._fingerprints),
        }

    # ---------------- bonus: near-duplicate detection ----------------

    def find_near_duplicates(self) -> List[tuple]:
        """Return all (id_a, id_b, hamming_distance) below the threshold."""
        ids = list(self._fingerprints.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = Fingerprinter.hamming_distance(
                    self._fingerprints[ids[i]], self._fingerprints[ids[j]]
                )
                if d <= self.hamming_threshold:
                    pairs.append((ids[i], ids[j], d))
        return pairs
