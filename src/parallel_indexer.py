"""
SON-style parallel index builder (MapReduce simulation).

Two-phase distributed build:
  Map phase:    Workers compute MinHash signatures / SimHash fingerprints
                on their assigned chunk partition, in parallel processes.
  Reduce phase: Partial results are merged into a global index — LSH band
                buckets for MinHash, flat fingerprint dict for SimHash.

This mirrors SON (Savasere–Omiecinski–Navathe): independent local passes
followed by a global aggregation step. Used to satisfy the
"MapReduce / SON" extension from the project rubric.

Both indices are deterministic from a fixed seed (MinHash) or MD5 hashing
(SimHash), so workers reconstruct identical hash families without any
shared state.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple

import numpy as np

from minhash_lsh import (
    MinHashIndex, MinHashSignature, Shingler, LSHBands,
)
from simhash import SimHashIndex, Fingerprinter, Tokenizer


# ----------------------------------------------------------------------
# Top-level worker functions (must be picklable for multiprocessing)
# ----------------------------------------------------------------------

def _minhash_map(args: Tuple) -> List[Tuple[int, np.ndarray]]:
    chunks, n_hashes, shingle_k, seed = args
    shingler = Shingler(k=shingle_k)
    signer = MinHashSignature(n_hashes=n_hashes, seed=seed)
    return [
        (c["chunk_id"], signer.signature(shingler.shingles(c["text"])))
        for c in chunks
    ]


def _simhash_map(args: Tuple) -> List[Tuple[int, int]]:
    chunks, bits = args
    tok = Tokenizer()
    fp = Fingerprinter(bits=bits)
    return [
        (c["chunk_id"], fp.fingerprint(tok.weights(c["text"])))
        for c in chunks
    ]


def _partition(items: List, n: int) -> List[List]:
    if n <= 1:
        return [items]
    size, rem = divmod(len(items), n)
    parts, i = [], 0
    for w in range(n):
        s = size + (1 if w < rem else 0)
        parts.append(items[i:i + s])
        i += s
    return [p for p in parts if p]


# ----------------------------------------------------------------------
# Parallel builders
# ----------------------------------------------------------------------

class ParallelMinHashBuilder:
    """SON-style parallel build for MinHash + LSH."""

    def __init__(
        self,
        n_workers: int = 4,
        n_hashes: int = 128,
        n_bands: int = 16,
        shingle_k: int = 5,
        seed: int = 42,
    ):
        self.n_workers = max(1, n_workers)
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.shingle_k = shingle_k
        self.seed = seed

    def build(self, chunks: List[Dict]) -> Tuple[MinHashIndex, Dict]:
        partitions = _partition(chunks, self.n_workers)
        worker_args = [
            (part, self.n_hashes, self.shingle_k, self.seed)
            for part in partitions
        ]

        # ----- MAP -----
        t = time.perf_counter()
        if self.n_workers == 1:
            partials = [_minhash_map(worker_args[0])]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                partials = list(ex.map(_minhash_map, worker_args))
        map_time = time.perf_counter() - t

        # ----- REDUCE -----
        t = time.perf_counter()
        idx = MinHashIndex(
            n_hashes=self.n_hashes,
            n_bands=self.n_bands,
            shingle_k=self.shingle_k,
            seed=self.seed,
        )
        idx._chunks = chunks
        for part in partials:
            for cid, sig in part:
                idx._signatures[cid] = sig
                for b, bkey in enumerate(idx._bander.band_keys(sig)):
                    idx._buckets[b].setdefault(bkey, []).append(cid)
        idx._is_built = True
        reduce_time = time.perf_counter() - t

        stats = {
            "n_workers": self.n_workers,
            "n_partitions": len(partitions),
            "n_chunks": len(chunks),
            "map_time_s": round(map_time, 4),
            "reduce_time_s": round(reduce_time, 4),
            "total_time_s": round(map_time + reduce_time, 4),
        }
        return idx, stats


class ParallelSimHashBuilder:
    """SON-style parallel build for SimHash."""

    def __init__(self, n_workers: int = 4, hamming_threshold: int = 10, bits: int = 64):
        self.n_workers = max(1, n_workers)
        self.hamming_threshold = hamming_threshold
        self.bits = bits

    def build(self, chunks: List[Dict]) -> Tuple[SimHashIndex, Dict]:
        partitions = _partition(chunks, self.n_workers)
        worker_args = [(part, self.bits) for part in partitions]

        t = time.perf_counter()
        if self.n_workers == 1:
            partials = [_simhash_map(worker_args[0])]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                partials = list(ex.map(_simhash_map, worker_args))
        map_time = time.perf_counter() - t

        t = time.perf_counter()
        idx = SimHashIndex(hamming_threshold=self.hamming_threshold, bits=self.bits)
        idx._chunks = chunks
        for part in partials:
            for cid, fp in part:
                idx._fingerprints[cid] = fp
        idx._is_built = True
        reduce_time = time.perf_counter() - t

        return idx, {
            "n_workers": self.n_workers,
            "n_partitions": len(partitions),
            "n_chunks": len(chunks),
            "map_time_s": round(map_time, 4),
            "reduce_time_s": round(reduce_time, 4),
            "total_time_s": round(map_time + reduce_time, 4),
        }


# ----------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import DocumentIngestor

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"
    chunks = DocumentIngestor().ingest(pdf)
    print(f"Loaded {len(chunks)} chunks")

    for nw in (1, 2, 4, 8):
        _, stats = ParallelMinHashBuilder(n_workers=nw).build(chunks)
        print(f"  workers={nw}: map={stats['map_time_s']}s "
              f"reduce={stats['reduce_time_s']}s "
              f"total={stats['total_time_s']}s")
