"""
Abstract base classes defining the contracts for the system's components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseIndex(ABC):
    """
    Abstract retrieval index.

    Every concrete index (MinHash+LSH, SimHash, TF-IDF, ...) implements:
      - index(chunks): build the in-memory data structures
      - query(query_text, top_k): return ranked top-k chunks
      - info(): return a dict describing index parameters / state
    """

    name: str = "base"

    def __init__(self):
        self._chunks: List[Dict] = []
        self._is_built: bool = False

    @abstractmethod
    def index(self, chunks: List[Dict]) -> None:
        """Build the index from a list of chunk dicts."""

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return top-k chunks ranked by similarity to the query."""

    @abstractmethod
    def info(self) -> Dict:
        """Return descriptive metadata about the index."""

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} chunks={self.n_chunks} built={self._is_built}>"


class BaseAnswerer(ABC):
    """
    Abstract answer generator (Strategy pattern).

    Concrete strategies: ExtractiveAnswerer, LLMAnswerer.
    """

    name: str = "base"

    @abstractmethod
    def answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate an answer string given the query and retrieved chunks."""

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Produce a full answer payload: answer text + supporting evidence.
        Concrete subclasses normally implement only `answer()`.
        """
        text = self.answer(query, chunks)
        evidence = []
        for c in chunks[:3]:
            snippet = c["text"][:400] + ("..." if len(c["text"]) > 400 else "")
            evidence.append({
                "text": snippet,
                "pages": f"{c['start_page']}–{c['end_page']}",
                "section": c.get("section", ""),
                "score": c.get("score", 0),
                "method": c.get("method", ""),
                "source": c.get("source", ""),
            })
        return {
            "answer": text,
            "mode": self.name,
            "supporting_chunks": evidence,
        }
