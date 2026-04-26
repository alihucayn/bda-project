"""
Document ingestion pipeline (OOP).

Classes:
  TextCleaner       - whitespace/header normalization
  ChapterDetector   - heuristic section/chapter labelling
  DocumentChunker   - sliding-window chunker with page tracking
  DocumentIngestor  - facade orchestrating PDF -> cleaned chunks
"""

import re
import json
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


# ----------------------------------------------------------------------
# Helper components
# ----------------------------------------------------------------------

class TextCleaner:
    """Normalize raw PDF text: strip footers, collapse whitespace."""

    _FOOTER = re.compile(r'NUST Undergraduate Student Handbook\s*\d*')
    _MULTI_NL = re.compile(r'\n{3,}')
    _MULTI_WS = re.compile(r'[ \t]{2,}')

    def clean(self, text: str) -> str:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = self._FOOTER.sub('', text)
        text = self._MULTI_NL.sub('\n\n', text)
        text = self._MULTI_WS.sub(' ', text)
        return text.strip()


class ChapterDetector:
    """Heuristic detector for chapter / numbered-section headings."""

    _PATTERNS = [
        re.compile(r'Chapter\s+\d+[:\s]+([^\n]{3,60})'),
        re.compile(r'^(\d+\.\s{1,4}[A-Z][^\n]{3,60})', re.MULTILINE),
    ]

    def detect(self, text: str) -> str:
        for pat in self._PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(0).strip()[:80]
        return ""


class DocumentChunker:
    """
    Sliding-window chunker over a token stream that preserves page numbers.

    Parameters
    ----------
    chunk_size : int
        Target number of words per chunk.
    overlap : int
        Word overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 350, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chapter_detector = ChapterDetector()

    def chunk(self, pages: List[Dict]) -> List[Dict]:
        # Flatten to (word, page) tuples
        word_page_pairs = []
        for p in pages:
            for w in p["text"].split():
                word_page_pairs.append((w, p["page"]))

        chunks = []
        chunk_id = 0
        i = 0
        n = len(word_page_pairs)
        current_chapter = ""

        while i < n:
            end = min(i + self.chunk_size, n)
            words = [wp[0] for wp in word_page_pairs[i:end]]
            page_nums = [wp[1] for wp in word_page_pairs[i:end]]

            text = " ".join(words)
            detected = self._chapter_detector.detect(text)
            if detected:
                current_chapter = detected

            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "start_page": page_nums[0],
                "end_page": page_nums[-1],
                "section": current_chapter,
                "word_count": len(words),
            })
            chunk_id += 1
            i += self.chunk_size - self.overlap

        return chunks


# ----------------------------------------------------------------------
# Facade
# ----------------------------------------------------------------------

class DocumentIngestor:
    """
    Top-level ingestion facade.

    Usage:
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest("handbook.pdf", "data/chunks.json")
    """

    def __init__(
        self,
        chunk_size: int = 350,
        overlap: int = 50,
        cleaner: TextCleaner = None,
        chunker: DocumentChunker = None,
    ):
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or DocumentChunker(chunk_size=chunk_size, overlap=overlap)

    # ---------------- core ----------------

    def extract_pages(self, pdf_path: str) -> List[Dict]:
        """Extract per-page text from a PDF, applying cleaning."""
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            cleaned = self.cleaner.clean(page.get_text("text"))
            if cleaned.strip():
                pages.append({"page": page_num, "text": cleaned})
        doc.close()
        return pages

    def ingest(self, pdf_path: str, output_path: str = None) -> List[Dict]:
        """Full pipeline: PDF -> chunks. Optionally persists to JSON."""
        pages = self.extract_pages(pdf_path)
        chunks = self.chunker.chunk(pages)
        if output_path:
            self.save_chunks(chunks, output_path)
        return chunks

    # ---------------- IO helpers ----------------

    @staticmethod
    def save_chunks(chunks: List[Dict], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {path}")

    @staticmethod
    def load_chunks(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ----------------------------------------------------------------------
# Backward-compatible module-level functions (used by older imports)
# ----------------------------------------------------------------------

def ingest(pdf_path: str, output_path: str = None) -> List[Dict]:
    return DocumentIngestor().ingest(pdf_path, output_path)


def load_chunks(path: str) -> List[Dict]:
    return DocumentIngestor.load_chunks(path)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.json"
    chunks = DocumentIngestor().ingest(pdf, out)
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg words/chunk: {sum(c['word_count'] for c in chunks) / len(chunks):.1f}")
    print("Sample chunk 0:")
    print(chunks[0]["text"][:200])
