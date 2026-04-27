"""
Answer generation (Strategy pattern).

Concrete strategies:
  ExtractiveAnswerer  - pick the most query-relevant sentences from top chunks
  GroqAnswerer        - call Groq API (Llama) with retrieved chunks as context

A factory `make_answerer(mode)` builds the right strategy by name.
"""

import os
import re
from typing import List, Dict

from base import BaseAnswerer


# ----------------------------------------------------------------------
# Extractive strategy
# ----------------------------------------------------------------------

class ExtractiveAnswerer(BaseAnswerer):
    """Return the n_sentences most query-overlapping sentences from the top chunk."""

    name = "extractive"

    def __init__(self, n_sentences: int = 3):
        self.n_sentences = n_sentences
        self._word_re = re.compile(r'[a-z]{3,}')
        self._sentence_re = re.compile(r'(?<=[.!?])\s+')

    def answer(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant information found in the handbook."

        query_words = set(self._word_re.findall(query.lower()))
        best = chunks[0]
        sentences = self._sentence_re.split(best["text"])

        scored = []
        for sent in sentences:
            words = set(self._word_re.findall(sent.lower()))
            scored.append((len(query_words & words), sent))

        scored.sort(key=lambda x: -x[0])
        top = [s for _, s in scored[: self.n_sentences] if s.strip()]
        return " ".join(top)


# ----------------------------------------------------------------------
# Groq + Llama strategy
# ----------------------------------------------------------------------

class GroqAnswerer(BaseAnswerer):
    """Call the Groq API with a Llama model to generate a grounded answer."""

    name = "groq"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 512,
        api_key: str = None,
        fallback: BaseAnswerer = None,
        max_context_chunks: int = 5,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.fallback = fallback or ExtractiveAnswerer()
        self.max_context_chunks = max_context_chunks

    def _build_context(self, chunks: List[Dict]) -> str:
        n = min(self.max_context_chunks, len(chunks))
        parts = []
        for i, chunk in enumerate(chunks[:n], 1):
            ref = f"[Pages {chunk['start_page']}–{chunk['end_page']}]"
            src = f" ({chunk['source']})" if chunk.get("source") else ""
            parts.append(f"--- Chunk {i} {ref}{src} ---\n{chunk['text']}")
        return "\n\n".join(parts)

    def answer(self, query: str, chunks: List[Dict]) -> str:
        if not self.api_key:
            return self.fallback.answer(query, chunks)

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)

            system_prompt = (
                "You are a helpful assistant for NUST students. "
                "Answer the student's question STRICTLY based on the provided handbook excerpts. "
                "If the answer is not in the excerpts, say so. "
                "Be concise and cite the page numbers."
            )
            user_message = (
                f"HANDBOOK EXCERPTS:\n{self._build_context(chunks)}\n\n"
                f"STUDENT QUESTION: {query}\n\n"
                "Answer based only on the excerpts above:"
            )

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

        except Exception as e:
            return (
                f"[Groq error: {e}] Falling back to extractive answer.\n\n"
                + self.fallback.answer(query, chunks)
            )


# ----------------------------------------------------------------------
# Factory + backward-compatible wrappers
# ----------------------------------------------------------------------

_ANSWERERS = {
    "extractive": ExtractiveAnswerer,
    "groq": GroqAnswerer,
}


def make_answerer(mode: str = "extractive", **kwargs) -> BaseAnswerer:
    cls = _ANSWERERS.get(mode, ExtractiveAnswerer)
    return cls(**kwargs)


def generate_answer(query: str, chunks: List[Dict], mode: str = "extractive") -> Dict:
    """Backward-compatible procedural entry point."""
    answerer = make_answerer(mode)
    return answerer.generate(query, chunks)
