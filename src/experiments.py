"""
Experiment harness (OOP).

Class hierarchy:
  BaseExperiment            - abstract: defines run() + plot()
    ExactVsApproxExperiment - latency / memory / Precision@k
    ParameterSensitivity    - n_hashes, n_bands, hamming_threshold sweeps
    ScalabilityExperiment   - duplicate corpus, measure scaling

ExperimentRunner orchestrates them and persists results.
"""

import sys
import json
import time
import tracemalloc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from ingest import DocumentIngestor
from minhash_lsh import MinHashIndex
from simhash import SimHashIndex
from tfidf_baseline import TFIDFIndex
from base import BaseIndex


# ----------------------------------------------------------------------
# Default test queries
# ----------------------------------------------------------------------

TEST_QUERIES = [
    "What is the minimum GPA requirement to graduate?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is probation and when is it given?",
    "What are the conditions for withdrawal from the university?",
    "How is CGPA calculated?",
    "What is the grading scheme for engineering programmes?",
    "Can a student defer a semester?",
    "What is the summer semester policy?",
    "How can I improve my CGPA by repeating a course?",
    "What is the XF grade?",
    "What medals are awarded at convocation?",
    "What are the conditions for suspension?",
    "What is the difference between warning and probation?",
]


# ----------------------------------------------------------------------
# Profiling helper
# ----------------------------------------------------------------------

class QueryProfiler:
    """Encapsulates timing + memory profiling of a single query."""

    @staticmethod
    def run(index: BaseIndex, query: str, top_k: int = 5):
        tracemalloc.start()
        t0 = time.perf_counter()
        results = index.query(query, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return results, latency_ms, peak / 1024  # KB


def precision_at_k(approx_results: List[Dict], exact_results: List[Dict], k: int) -> float:
    """Use exact (TF-IDF) results as ground truth."""
    exact_ids = {r["chunk_id"] for r in exact_results[:k]}
    approx_ids = {r["chunk_id"] for r in approx_results[:k]}
    return len(exact_ids & approx_ids) / k


# ----------------------------------------------------------------------
# Abstract experiment
# ----------------------------------------------------------------------

class BaseExperiment(ABC):
    """Abstract experiment: run() returns results dict; plot() draws figures."""

    name: str = "base"

    def __init__(self, chunks: List[Dict], plot_dir: Path, queries: List[str] = None):
        self.chunks = chunks
        self.plot_dir = plot_dir
        self.queries = queries or TEST_QUERIES
        self.results: Dict = {}

    @abstractmethod
    def run(self) -> Dict:
        ...

    def plot(self) -> None:
        """Optional override for visualization."""
        pass

    def execute(self) -> Dict:
        print(f"\n=== Experiment: {self.name} ===")
        self.results = self.run()
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot()
        return self.results


# ----------------------------------------------------------------------
# Experiment 1: Exact vs Approximate
# ----------------------------------------------------------------------

class ExactVsApproxExperiment(BaseExperiment):
    name = "Exact vs Approximate Retrieval"

    def __init__(self, chunks, plot_dir, queries=None, k: int = 5):
        super().__init__(chunks, plot_dir, queries)
        self.k = k

    def run(self) -> Dict:
        tf = TFIDFIndex(); tf.index(self.chunks)
        mh = MinHashIndex(); mh.index(self.chunks)
        sh = SimHashIndex(); sh.index(self.chunks)

        latencies = {"tfidf": [], "minhash": [], "simhash": []}
        memories = {"tfidf": [], "minhash": [], "simhash": []}
        precisions = {"minhash": [], "simhash": []}

        for q in self.queries:
            tf_res, tf_lat, tf_mem = QueryProfiler.run(tf, q, self.k)
            mh_res, mh_lat, mh_mem = QueryProfiler.run(mh, q, self.k)
            sh_res, sh_lat, sh_mem = QueryProfiler.run(sh, q, self.k)

            for name, lat, mem in [
                ("tfidf", tf_lat, tf_mem),
                ("minhash", mh_lat, mh_mem),
                ("simhash", sh_lat, sh_mem),
            ]:
                latencies[name].append(lat)
                memories[name].append(mem)

            precisions["minhash"].append(precision_at_k(mh_res, tf_res, self.k))
            precisions["simhash"].append(precision_at_k(sh_res, tf_res, self.k))

        result = {
            "k": self.k,
            "avg_latency_ms": {m: round(float(np.mean(v)), 3) for m, v in latencies.items()},
            "avg_memory_kb": {m: round(float(np.mean(v)), 2) for m, v in memories.items()},
            "avg_precision_at_k": {m: round(float(np.mean(v)), 4) for m, v in precisions.items()},
        }
        print(json.dumps(result, indent=2))
        return result

    def plot(self) -> None:
        r = self.results
        # Latency
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = ["TF-IDF\n(Exact)", "MinHash\n+LSH", "SimHash"]
        lats = [r["avg_latency_ms"][m] for m in ("tfidf", "minhash", "simhash")]
        bars = ax.bar(methods, lats, color=["steelblue", "coral", "mediumseagreen"])
        ax.bar_label(bars, fmt="%.2f ms", padding=3)
        ax.set_ylabel("Avg Query Latency (ms)")
        ax.set_title("Exact vs Approximate: Query Latency")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "exp1_latency.png", dpi=150)
        plt.close()

        # Precision
        fig, ax = plt.subplots(figsize=(5, 4))
        precs = [r["avg_precision_at_k"]["minhash"], r["avg_precision_at_k"]["simhash"]]
        bars = ax.bar(["MinHash+LSH", "SimHash"], precs, color=["coral", "mediumseagreen"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"Precision@{self.k} vs TF-IDF")
        ax.set_title("Approximate Retrieval Accuracy")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "exp1_precision.png", dpi=150)
        plt.close()


# ----------------------------------------------------------------------
# Experiment 2: Parameter sensitivity
# ----------------------------------------------------------------------

class ParameterSensitivityExperiment(BaseExperiment):
    name = "Parameter Sensitivity"

    def __init__(self, chunks, plot_dir, queries=None, k: int = 5):
        super().__init__(chunks, plot_dir, queries)
        self.k = k

    def _eval_minhash(self, n_hashes: int, n_bands: int, tf_index: TFIDFIndex):
        mh = MinHashIndex(n_hashes=n_hashes, n_bands=n_bands)
        mh.index(self.chunks)
        ps, ls = [], []
        for q in self.queries:
            tf_res, _, _ = QueryProfiler.run(tf_index, q, self.k)
            mh_res, lat, _ = QueryProfiler.run(mh, q, self.k)
            ps.append(precision_at_k(mh_res, tf_res, self.k))
            ls.append(lat)
        return float(np.mean(ps)), float(np.mean(ls)), mh

    def _eval_simhash(self, hamming: int, tf_index: TFIDFIndex):
        sh = SimHashIndex(hamming_threshold=hamming)
        sh.index(self.chunks)
        ps = []
        for q in self.queries:
            tf_res, _, _ = QueryProfiler.run(tf_index, q, self.k)
            sh_res, _, _ = QueryProfiler.run(sh, q, self.k)
            ps.append(precision_at_k(sh_res, tf_res, self.k))
        return float(np.mean(ps))

    def run(self) -> Dict:
        tf = TFIDFIndex(); tf.index(self.chunks)

        # Sweep n_hashes
        n_hash_vals = [32, 64, 128, 256]
        prec_h, lat_h = [], []
        for nh in n_hash_vals:
            p, l, _ = self._eval_minhash(nh, min(16, nh), tf)
            prec_h.append(p); lat_h.append(l)
            print(f"  n_hashes={nh}: precision={p:.3f}, latency={l:.2f}ms")

        # Sweep n_bands
        n_band_vals = [4, 8, 16, 32]
        prec_b, thr_b, vals_b = [], [], []
        for nb in n_band_vals:
            if 128 % nb != 0:
                continue
            p, _, mh = self._eval_minhash(128, nb, tf)
            prec_b.append(p); thr_b.append(round(mh._bander.threshold, 3)); vals_b.append(nb)
            print(f"  n_bands={nb}: threshold≈{mh._bander.threshold:.3f}, precision={p:.3f}")

        # Sweep hamming threshold
        ham_vals = [4, 8, 12, 16, 20]
        prec_ham = []
        for ht in ham_vals:
            p = self._eval_simhash(ht, tf)
            prec_ham.append(p)
            print(f"  hamming_threshold={ht}: precision={p:.3f}")

        return {
            "nhashes": {"values": n_hash_vals, "precision": prec_h, "latency_ms": lat_h},
            "nbands":  {"values": vals_b,      "precision": prec_b, "threshold": thr_b},
            "hamming": {"values": ham_vals,    "precision": prec_ham},
        }

    def plot(self) -> None:
        r = self.results
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(r["nhashes"]["values"], r["nhashes"]["precision"], "o-", color="coral")
        axes[0].set_xlabel("Number of Hash Functions")
        axes[0].set_ylabel("Avg Precision@5")
        axes[0].set_title("MinHash: n_hashes")
        axes[0].set_ylim(0, 1.05)

        axes[1].plot(r["nbands"]["values"], r["nbands"]["precision"], "s-", color="steelblue")
        axes[1].set_xlabel("Number of Bands")
        axes[1].set_ylabel("Avg Precision@5")
        axes[1].set_title("LSH: n_bands")
        axes[1].set_ylim(0, 1.05)

        axes[2].plot(r["hamming"]["values"], r["hamming"]["precision"], "^-", color="mediumseagreen")
        axes[2].set_xlabel("Hamming Threshold")
        axes[2].set_ylabel("Avg Precision@5")
        axes[2].set_title("SimHash: Hamming threshold")
        axes[2].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(self.plot_dir / "exp2_sensitivity.png", dpi=150)
        plt.close()


# ----------------------------------------------------------------------
# Experiment 3: Scalability
# ----------------------------------------------------------------------

class ScalabilityExperiment(BaseExperiment):
    name = "Scalability"

    def __init__(self, chunks, plot_dir, queries=None, multipliers=None):
        super().__init__(chunks, plot_dir, queries)
        self.multipliers = multipliers or [1, 2, 4, 8, 16]

    def _duplicated(self, mult: int) -> List[Dict]:
        n = len(self.chunks)
        out = []
        for i in range(mult):
            for c in self.chunks:
                nc = c.copy()
                nc["chunk_id"] = c["chunk_id"] + i * n
                out.append(nc)
        return out

    def run(self) -> Dict:
        build_times = {"minhash": [], "simhash": [], "tfidf": []}
        query_times = {"minhash": [], "simhash": [], "tfidf": []}
        sizes = []

        for mult in self.multipliers:
            big = self._duplicated(mult)
            sizes.append(len(big))

            indices = {"minhash": MinHashIndex(), "simhash": SimHashIndex(), "tfidf": TFIDFIndex()}
            for name, idx in indices.items():
                t = time.perf_counter()
                idx.index(big)
                build_times[name].append(time.perf_counter() - t)

            q = self.queries[0]
            for name, idx in indices.items():
                _, lat, _ = QueryProfiler.run(idx, q)
                query_times[name].append(lat)

            print(
                f"  size={len(big)}: "
                f"TF-IDF={query_times['tfidf'][-1]:.2f}ms  "
                f"MinHash={query_times['minhash'][-1]:.2f}ms  "
                f"SimHash={query_times['simhash'][-1]:.2f}ms"
            )

        return {
            "corpus_sizes": sizes,
            "build_times_s": {k: [round(v, 4) for v in vals] for k, vals in build_times.items()},
            "query_latency_ms": {k: [round(v, 3) for v in vals] for k, vals in query_times.items()},
        }

    def plot(self) -> None:
        r = self.results
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r["corpus_sizes"], r["query_latency_ms"]["tfidf"], "o-", label="TF-IDF (exact)", color="steelblue")
        ax.plot(r["corpus_sizes"], r["query_latency_ms"]["minhash"], "s-", label="MinHash+LSH", color="coral")
        ax.plot(r["corpus_sizes"], r["query_latency_ms"]["simhash"], "^-", label="SimHash", color="mediumseagreen")
        ax.set_xlabel("Corpus Size (# chunks)")
        ax.set_ylabel("Query Latency (ms)")
        ax.set_title("Scalability: Query Latency vs Corpus Size")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "exp3_scalability.png", dpi=150)
        plt.close()


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class ExperimentRunner:
    """Orchestrate all experiments and persist results to JSON + plots."""

    def __init__(self, pdf_path: str, project_root: Path = None):
        self.pdf_path = pdf_path
        self.project_root = project_root or Path(__file__).parent.parent
        self.plot_dir = self.project_root / "data" / "plots"
        self.results_path = self.project_root / "data" / "experiment_results.json"
        self.chunks_path = self.project_root / "data" / "chunks.json"

    def load_chunks(self) -> List[Dict]:
        ingestor = DocumentIngestor()
        return ingestor.ingest(self.pdf_path, str(self.chunks_path))

    def run_all(self) -> Dict:
        print(f"Loading and chunking: {self.pdf_path}")
        chunks = self.load_chunks()
        print(f"Chunks: {len(chunks)}")

        experiments = [
            ExactVsApproxExperiment(chunks, self.plot_dir),
            ParameterSensitivityExperiment(chunks, self.plot_dir),
            ScalabilityExperiment(chunks, self.plot_dir),
        ]

        all_results = {}
        for i, exp in enumerate(experiments, 1):
            all_results[f"experiment_{i}"] = exp.execute()

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {self.results_path}")
        print(f"Plots saved to {self.plot_dir}/")

        return all_results


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"
    ExperimentRunner(pdf).run_all()
