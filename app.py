"""
Streamlit UI for the NUST Academic Policy QA System (OOP).

Run:
    streamlit run app.py
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

# Add src to path BEFORE importing project modules
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ingest import DocumentIngestor                      # noqa: E402
from retriever import Retriever                          # noqa: E402
from answer_gen import make_answerer                     # noqa: E402


# ----------------------------------------------------------------------
# Cached system loader (must be top-level for st.cache_resource)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Building indices... (first run only)")
def _build_retriever(pdf_path: str, cache_key: str, data_dir: str) -> Tuple[Retriever, List[Dict]]:
    """Cache-friendly factory: chunks the PDF and builds the indices."""
    chunks_path = Path(data_dir) / f"chunks_{cache_key}.json"
    if chunks_path.exists():
        chunks = DocumentIngestor.load_chunks(str(chunks_path))
    else:
        chunks = DocumentIngestor().ingest(pdf_path, str(chunks_path))

    retriever = Retriever(n_hashes=128, n_bands=16, shingle_k=5, hamming_threshold=10)
    retriever.build(chunks)
    return retriever, chunks


# ----------------------------------------------------------------------
# Upload manager
# ----------------------------------------------------------------------

class UploadManager:
    """Persist uploaded PDFs to disk and produce stable cache keys."""

    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir

    def save(self, uploaded_file) -> Tuple[str, str]:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        raw = uploaded_file.getvalue()
        digest = hashlib.md5(raw).hexdigest()[:12]
        target = self.upload_dir / f"{digest}_{uploaded_file.name}"
        if not target.exists():
            target.write_bytes(raw)
        return str(target), digest


# ----------------------------------------------------------------------
# UI components
# ----------------------------------------------------------------------

class UIConfig:
    """Static UI constants."""

    METHOD_LABELS = {
        "tfidf":   "TF-IDF (Exact Baseline)",
        "minhash": "MinHash + LSH (Approximate)",
        "simhash": "SimHash (Approximate)",
    }
    ANSWER_MODE_LABELS = {
        "extractive": "Extractive",
        "llm": "LLM (Claude)",
        "groq": "Llama via Groq",
    }
    METHOD_COLORS = {
        "tfidf":   "#2196F3",
        "minhash": "#FF5722",
        "simhash": "#4CAF50",
    }
    SAMPLE_QUERIES = [
        "What is the minimum GPA requirement to graduate?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the conditions for student withdrawal?",
        "What is the difference between probation and suspension?",
        "What is the XF grade and how is it cleared?",
        "Can a student defer a semester?",
        "What medals are awarded at convocation?",
        "How does the summer semester work?",
    ]
    DEFAULT_TOP_K = 5


class Sidebar:
    """Renders the sidebar and returns user-selected settings."""

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg

    def render(self) -> Dict:
        with st.sidebar:
            st.header("⚙️ Settings")

            uploaded_pdf = st.file_uploader(
                "Upload Handbook PDF",
                type=["pdf"],
                help="Upload the NUST UG/PG handbook (or any PDF) to index.",
            )
            top_k = st.slider("Top-k results", 1, 10, self.cfg.DEFAULT_TOP_K)
            method = st.selectbox(
                "Retrieval method",
                list(self.cfg.METHOD_LABELS.keys()),
                format_func=lambda m: self.cfg.METHOD_LABELS[m],
            )
            answer_mode = st.radio(
                "Answer mode",
                list(self.cfg.ANSWER_MODE_LABELS.keys()),
                format_func=lambda x: self.cfg.ANSWER_MODE_LABELS[x],
            )

            groq_api_key = ""
            if answer_mode == "groq":
                groq_api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    help="Get your free key at console.groq.com",
                )

            show_comparison = st.checkbox("Show all-method comparison", value=True)

            st.divider()
            st.header("Sample queries")
            sample_clicked = None
            for q in self.cfg.SAMPLE_QUERIES:
                if st.button(q[:55] + ("…" if len(q) > 55 else ""), use_container_width=True):
                    sample_clicked = q

        return {
            "uploaded_pdf": uploaded_pdf,
            "top_k": top_k,
            "method": method,
            "answer_mode": answer_mode,
            "groq_api_key": groq_api_key,
            "show_comparison": show_comparison,
            "sample_clicked": sample_clicked,
        }

    def show_index_metrics(self, n_chunks: int, avg_words: int) -> None:
        with st.sidebar:
            st.divider()
            st.metric("Chunks indexed", n_chunks)
            st.metric("Avg words/chunk", avg_words)


class ResultsView:
    """Renders retrieval results, evidence, and method comparison."""

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg

    def chunk_card(self, chunk: Dict, rank: int) -> None:
        method = chunk.get("method", "")
        score = chunk.get("score", 0)
        pages = f"Pages {chunk['start_page']}–{chunk['end_page']}"
        with st.expander(f"#{rank}  |  Score: {score:.4f}  |  {pages}  ({method.upper()})"):
            if chunk.get("section"):
                st.caption(f"Section: {chunk['section']}")
            text = chunk["text"]
            st.markdown(text[:700] + ("…" if len(text) > 700 else ""))

    def comparison_table(self, results_all: Dict[str, Tuple[List[Dict], float]]) -> None:
        cols = st.columns(len(results_all))
        for col, (method, (results, latency)) in zip(cols, results_all.items()):
            color = self.cfg.METHOD_COLORS.get(method, "#999")
            with col:
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding-left:8px'>"
                    f"<b>{self.cfg.METHOD_LABELS[method]}</b><br>"
                    f"<small>Latency: {latency:.2f} ms</small></div>",
                    unsafe_allow_html=True,
                )
                if results:
                    top = results[0]
                    st.markdown(f"**Score:** `{top['score']:.4f}`")
                    st.markdown(f"**Pages:** {top['start_page']}–{top['end_page']}")
                    st.markdown(top["text"][:300] + "…")
                else:
                    st.warning("No results")

    def evidence(self, supporting_chunks: List[Dict]) -> None:
        for i, ev in enumerate(supporting_chunks, 1):
            with st.expander(
                f"Evidence #{i} — Pages {ev['pages']} | Score: {ev['score']:.4f}"
            ):
                if ev.get("section"):
                    st.caption(f"Section: {ev['section']}")
                st.markdown(ev["text"])


# ----------------------------------------------------------------------
# Main controller
# ----------------------------------------------------------------------

class QAApp:
    """Top-level Streamlit controller."""

    def __init__(self):
        self.cfg = UIConfig()
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.uploads = UploadManager(self.data_dir / "uploaded")
        self.sidebar = Sidebar(self.cfg)
        self.results_view = ResultsView(self.cfg)

    # ---------- helpers ----------

    def _configure_page(self) -> None:
        st.set_page_config(
            page_title="NUST Academic Policy QA",
            page_icon="📚",
            layout="wide",
        )
        st.title("📚 NUST Academic Policy QA System")
        st.caption(
            "Scalable retrieval over the NUST Undergraduate Handbook using "
            "MinHash+LSH, SimHash, and TF-IDF (Big Data CS-404 Project)"
        )

    def _load_system(self, uploaded_pdf) -> Tuple[Retriever, List[Dict]]:
        pdf_path, cache_key = self.uploads.save(uploaded_pdf)
        return _build_retriever(pdf_path, cache_key, str(self.data_dir))

    # ---------- main flow ----------

    def run(self) -> None:
        self._configure_page()
        settings = self.sidebar.render()

        if settings["uploaded_pdf"] is None:
            st.info("👈 Upload a handbook PDF in the sidebar to begin.")
            st.stop()

        try:
            retriever, chunks = self._load_system(settings["uploaded_pdf"])
        except Exception as e:
            st.error(f"Failed to load handbook: {e}")
            st.stop()

        avg_words = sum(c["word_count"] for c in chunks) // len(chunks)
        self.sidebar.show_index_metrics(len(chunks), avg_words)

        # ---------- query input ----------
        default_query = settings["sample_clicked"] or st.session_state.get("last_query", "")
        query = st.text_input(
            "Ask a question about NUST academic policies:",
            value=default_query,
            placeholder="e.g. What is the minimum CGPA to graduate?",
        )
        col1, _ = st.columns([1, 5])
        search_clicked = col1.button("🔍 Search", type="primary")

        if not query:
            st.info("Enter a question above or click a sample query in the sidebar.")
            return

        if not (search_clicked or settings["sample_clicked"]):
            return

        st.session_state["last_query"] = query

        # ---------- retrieval ----------
        with st.spinner("Retrieving…"):
            results, latency = retriever.query(
                query, method=settings["method"], top_k=settings["top_k"]
            )
            results_all = (
                retriever.query_all(query, top_k=settings["top_k"])
                if settings["show_comparison"] else None
            )

        # ---------- answer generation ----------
        with st.spinner("Generating answer…"):
            mode = settings["answer_mode"]
            kwargs = {}
            if mode == "groq" and settings.get("groq_api_key"):
                kwargs["api_key"] = settings["groq_api_key"]
            answerer = make_answerer(mode, **kwargs)
            answer_data = answerer.generate(query, results)

        # ---------- display ----------
        st.divider()
        st.subheader("💡 Answer")
        st.success(answer_data["answer"])
        st.caption(
            f"Method: **{self.cfg.METHOD_LABELS[settings['method']]}** | "
            f"Latency: **{latency:.2f} ms** | "
            f"Mode: **{settings['answer_mode']}**"
        )

        st.subheader("📄 Supporting Evidence")
        self.results_view.evidence(answer_data["supporting_chunks"])

        st.subheader(f"🔎 Top-{settings['top_k']} Retrieved Chunks "
                     f"({self.cfg.METHOD_LABELS[settings['method']]})")
        for rank, chunk in enumerate(results, 1):
            self.results_view.chunk_card(chunk, rank)

        if results_all is not None:
            st.divider()
            st.subheader("📊 All-Method Comparison (Top-1)")
            self.results_view.comparison_table(results_all)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    QAApp().run()
