"""
Streamlit UI for the NUST Academic Policy QA System (OOP).

Run:
    streamlit run app.py
"""

import sys
import time
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
from parallel_indexer import (                           # noqa: E402
    ParallelMinHashBuilder, ParallelSimHashBuilder,
)
from tfidf_baseline import TFIDFIndex                    # noqa: E402


# ----------------------------------------------------------------------
# Cached system loader (must be top-level for st.cache_resource)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexing handbooks via SON / MapReduce…")
def _build_retriever(file_specs: Tuple[Tuple[str, str, str], ...],
                     data_dir: str,
                     n_workers: int = 4
                     ) -> Tuple[Retriever, List[Dict], Dict]:
    """
    Build a unified retriever over one or more uploaded PDFs.

    Each PDF is chunked into its own JSON (cached by content hash). Chunks
    from all files are then merged — with chunk_ids re-issued so they're
    unique across the corpus — and indexed in a single SON / MapReduce
    parallel pass that returns timing statistics.

    Parameters
    ----------
    file_specs : tuple of (pdf_path, digest, original_name)
        Made hashable so st.cache_resource can key on it.
    """
    per_file_meta: List[Dict] = []
    all_chunks: List[Dict] = []
    cid_offset = 0

    for pdf_path, digest, name in file_specs:
        chunks_path = Path(data_dir) / f"chunks_{digest}.json"
        if chunks_path.exists():
            chunks = DocumentIngestor.load_chunks(str(chunks_path))
        else:
            chunks = DocumentIngestor().ingest(pdf_path, str(chunks_path))

        # Tag and re-id for the merged corpus (per-file JSONs keep original IDs)
        merged = []
        for c in chunks:
            c2 = dict(c)
            c2["chunk_id"] = c2["chunk_id"] + cid_offset
            c2["source"] = name
            merged.append(c2)

        avg = sum(c["word_count"] for c in chunks) // max(1, len(chunks))
        per_file_meta.append({
            "name": name,
            "digest": digest,
            "n_chunks": len(chunks),
            "avg_words": avg,
            "max_page": max((c["end_page"] for c in chunks), default=0),
        })

        all_chunks.extend(merged)
        cid_offset += len(chunks)

    # ---- SON / MapReduce parallel build (MinHash + SimHash) ----
    mh_idx, mh_stats = ParallelMinHashBuilder(n_workers=n_workers).build(all_chunks)
    sh_idx, sh_stats = ParallelSimHashBuilder(n_workers=n_workers).build(all_chunks)

    # ---- TF-IDF (serial baseline) ----
    t = time.perf_counter()
    tf_idx = TFIDFIndex()
    tf_idx.index(all_chunks)
    tf_time = round(time.perf_counter() - t, 4)

    # Compose a Retriever from the pre-built indices
    retriever = Retriever(indices={
        "tfidf":   tf_idx,
        "minhash": mh_idx,
        "simhash": sh_idx,
    })
    retriever.chunks = all_chunks

    build_stats = {
        "n_files": len(file_specs),
        "n_chunks_total": len(all_chunks),
        "n_workers": n_workers,
        "per_file": per_file_meta,
        "minhash": mh_stats,
        "simhash": sh_stats,
        "tfidf_time_s": tf_time,
    }
    return retriever, all_chunks, build_stats


# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------

class Theme:
    """NUST-inspired palette + global CSS."""

    PRIMARY      = "#0E5C2F"   # deep NUST green
    PRIMARY_DARK = "#073B1D"
    ACCENT       = "#D4A017"   # academic gold
    BG_SOFT      = "#F7F5EE"
    INK          = "#1B1B1B"
    MUTED        = "#6B6B6B"
    BORDER       = "#E4E0D6"

    @classmethod
    def inject(cls) -> None:
        st.markdown(f"""
        <style>
            /* ---------- App background & typography ---------- */
            .stApp {{
                background:
                    radial-gradient(1100px 600px at 0% -10%, rgba(14,92,47,0.06) 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(212,160,23,0.05) 0%, transparent 55%),
                    {cls.BG_SOFT};
            }}
            html, body, [class*="css"] {{
                font-family: -apple-system, "SF Pro Text", "Helvetica Neue",
                             "Segoe UI", Inter, system-ui, sans-serif;
                color: {cls.INK};
            }}

            /* Hide default Streamlit chrome */
            #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
            .block-container {{ padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1180px; }}

            /* ---------- Hero ---------- */
            .nust-hero {{
                background: linear-gradient(120deg, {cls.PRIMARY} 0%, {cls.PRIMARY_DARK} 100%);
                color: white;
                padding: 28px 32px;
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(7,59,29,0.18);
                margin-bottom: 22px;
                position: relative;
                overflow: hidden;
            }}
            .nust-hero::after {{
                content: "";
                position: absolute; right: -40px; top: -40px;
                width: 220px; height: 220px;
                background: radial-gradient(circle, {cls.ACCENT}33 0%, transparent 70%);
                border-radius: 50%;
            }}
            .nust-hero h1 {{ font-size: 28px; margin: 0; font-weight: 700; letter-spacing: -0.4px; }}
            .nust-hero .tag {{ opacity: 0.85; margin-top: 6px; font-size: 14px; }}
            .nust-hero .badge-row {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
            .nust-badge {{
                display: inline-block;
                padding: 4px 10px; border-radius: 999px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.25);
                font-size: 12px; font-weight: 500; letter-spacing: 0.2px;
            }}
            .nust-badge.gold {{ background: {cls.ACCENT}; color: {cls.PRIMARY_DARK}; border: none; }}

            /* ---------- Cards ---------- */
            .nust-card {{
                background: white;
                border: 1px solid {cls.BORDER};
                border-radius: 12px;
                padding: 18px 22px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                margin-bottom: 14px;
            }}
            .nust-card.answer {{
                border-left: 4px solid {cls.PRIMARY};
                background: linear-gradient(180deg, #ffffff 0%, #fbfbf6 100%);
            }}
            .nust-card.answer .answer-label {{
                font-size: 12px; font-weight: 600; letter-spacing: 1.2px;
                color: {cls.PRIMARY}; text-transform: uppercase; margin-bottom: 8px;
            }}
            .nust-card.answer .answer-body {{
                font-size: 16px; line-height: 1.65; color: {cls.INK};
            }}
            .nust-card.meta {{
                display: flex; gap: 18px; flex-wrap: wrap;
                font-size: 13px; color: {cls.MUTED};
                background: transparent; border: none; box-shadow: none;
                padding: 6px 2px 0; margin-bottom: 22px;
            }}
            .nust-card.meta b {{ color: {cls.INK}; }}

            /* ---------- Section headings ---------- */
            .nust-section {{
                display: flex; align-items: baseline; gap: 10px;
                margin: 26px 0 12px;
            }}
            .nust-section h3 {{
                font-size: 18px; font-weight: 700; margin: 0; color: {cls.INK};
            }}
            .nust-section .rule {{
                flex: 1; height: 1px;
                background: linear-gradient(90deg, {cls.BORDER} 0%, transparent 100%);
            }}

            /* ---------- Method chips ---------- */
            .chip {{
                display: inline-block; padding: 3px 10px;
                border-radius: 999px; font-size: 11px; font-weight: 600;
                letter-spacing: 0.5px; text-transform: uppercase;
            }}

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #ffffff 0%, #faf8f1 100%);
                border-right: 1px solid {cls.BORDER};
            }}
            section[data-testid="stSidebar"] .stButton>button {{
                background: white;
                border: 1px solid {cls.BORDER};
                color: {cls.INK};
                text-align: left;
                font-size: 13px;
                padding: 8px 12px;
                border-radius: 8px;
                transition: all 0.15s ease;
            }}
            section[data-testid="stSidebar"] .stButton>button:hover {{
                border-color: {cls.PRIMARY};
                background: {cls.PRIMARY}0F;
                color: {cls.PRIMARY_DARK};
                transform: translateX(2px);
            }}

            /* Primary button (main panel) */
            .stButton>button[kind="primary"] {{
                background: {cls.PRIMARY};
                border: none;
                box-shadow: 0 2px 6px rgba(14,92,47,0.25);
            }}
            .stButton>button[kind="primary"]:hover {{
                background: {cls.PRIMARY_DARK};
            }}

            /* Expander styling */
            .streamlit-expanderHeader {{
                background: white !important;
                border: 1px solid {cls.BORDER} !important;
                border-radius: 10px !important;
                font-weight: 500 !important;
            }}

            /* Metric cards in sidebar */
            div[data-testid="stMetric"] {{
                background: white;
                border: 1px solid {cls.BORDER};
                border-radius: 10px;
                padding: 10px 14px;
            }}
            div[data-testid="stMetricValue"] {{
                font-size: 22px !important;
                color: {cls.PRIMARY_DARK} !important;
            }}

            /* Footer */
            .nust-footer {{
                text-align: center;
                color: {cls.MUTED};
                font-size: 12px;
                margin-top: 48px;
                padding-top: 18px;
                border-top: 1px solid {cls.BORDER};
            }}

            /* ---------- Widget readability (force light-theme contrast) ----------
               These rules guarantee dark text on light backgrounds for every
               native Streamlit widget, regardless of the user's theme setting.
               Scoped to AVOID touching .nust-hero (which is white-on-green). */

            /* Markdown body text */
            .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{
                color: {cls.INK};
            }}

            /* Inputs */
            .stTextInput input, .stTextArea textarea, .stNumberInput input {{
                color: {cls.INK} !important;
                background: white !important;
                border: 1px solid {cls.BORDER} !important;
            }}
            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder {{
                color: {cls.MUTED} !important;
                opacity: 1;
            }}

            /* Selectbox + multiselect */
            div[data-baseweb="select"] > div {{
                background: white !important;
                color: {cls.INK} !important;
                border-color: {cls.BORDER} !important;
            }}
            div[data-baseweb="select"] * {{ color: {cls.INK} !important; }}
            div[data-baseweb="popover"] li {{
                background: white !important;
                color: {cls.INK} !important;
            }}
            div[data-baseweb="popover"] li:hover {{
                background: {cls.PRIMARY}10 !important;
            }}

            /* Radio + checkbox labels */
            .stRadio label, .stCheckbox label,
            .stRadio label p, .stCheckbox label p {{
                color: {cls.INK} !important;
            }}

            /* Slider track + values */
            .stSlider label, .stSlider [data-testid="stTickBarMin"],
            .stSlider [data-testid="stTickBarMax"],
            .stSlider [role="slider"] {{
                color: {cls.INK} !important;
            }}

            /* All form labels (the small caption above each widget) */
            .stTextInput label, .stSelectbox label, .stRadio > label,
            .stCheckbox > label, .stSlider label, .stFileUploader label,
            .stNumberInput label {{
                color: {cls.INK} !important;
                font-weight: 500;
            }}

            /* File uploader dropzone */
            [data-testid="stFileUploader"] section {{
                background: white !important;
                border: 1px dashed {cls.BORDER} !important;
                border-radius: 10px !important;
            }}
            [data-testid="stFileUploader"] section * {{
                color: {cls.INK} !important;
            }}
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] section small * {{
                color: {cls.MUTED} !important;
            }}
            [data-testid="stFileUploader"] button {{
                background: {cls.PRIMARY} !important;
                color: white !important;
                border: none !important;
            }}

            /* Expander — header AND inner body text */
            [data-testid="stExpander"] details {{
                background: white !important;
                border: 1px solid {cls.BORDER} !important;
                border-radius: 10px !important;
            }}
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary * {{
                color: {cls.INK} !important;
                font-weight: 500;
            }}
            [data-testid="stExpander"] [data-testid="stExpanderDetails"],
            [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {{
                color: {cls.INK} !important;
            }}

            /* Metric */
            [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
                color: {cls.MUTED} !important;
            }}

            /* Captions */
            [data-testid="stCaptionContainer"], .stCaption {{
                color: {cls.MUTED} !important;
            }}

            /* Spinner text */
            [data-testid="stSpinner"] {{ color: {cls.PRIMARY} !important; }}

            /* Sidebar — force every text node readable */
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] .stMarkdown *,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] label *,
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {{
                color: {cls.INK} !important;
            }}
            /* but preserve our themed sidebar section labels (green) */
            section[data-testid="stSidebar"] .stMarkdown div[style*="{cls.PRIMARY}"] {{
                color: {cls.PRIMARY} !important;
            }}

            /* Alert boxes (st.info / st.error / st.warning) */
            div[data-baseweb="notification"] * {{
                color: {cls.INK} !important;
            }}

            /* Code blocks */
            code, pre {{ color: {cls.INK} !important; }}
        </style>
        """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Upload manager
# ----------------------------------------------------------------------

class UploadManager:
    """Persist uploaded PDFs to disk and produce stable cache keys."""

    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir

    def save(self, uploaded_file) -> Tuple[str, str, str]:
        """Save one PDF; return (path, digest, original_name)."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        raw = uploaded_file.getvalue()
        digest = hashlib.md5(raw).hexdigest()[:12]
        target = self.upload_dir / f"{digest}_{uploaded_file.name}"
        if not target.exists():
            target.write_bytes(raw)
        return str(target), digest, uploaded_file.name

    def save_many(self, uploaded_files) -> Tuple[Tuple[str, str, str], ...]:
        """Save a batch; return a hashable tuple of (path, digest, name)."""
        return tuple(self.save(f) for f in uploaded_files)


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
        "groq":       "Llama via Groq",
        "analysis":   "Analysis (Extractive + Llama side-by-side)",
    }
    METHOD_COLORS = {
        "tfidf":   "#1976D2",
        "minhash": "#E64A19",
        "simhash": "#2E7D32",
    }
    # Quick queries are tuned to handbook vocabulary so extractive mode
    # lands on the canonical chunk (verified against handbook PDF).
    SAMPLE_QUERIES = [
        "What CGPA must a student obtain to satisfactorily complete the degree requirement?",
        "What grade is awarded if a student fails a course?",
        "What attendance percentage is required to sit the End Semester Examination?",
        "How many courses can a student repeat to clear F or XF grades?",
        "Under what conditions is a student recommended for withdrawal?",
        "When is a student placed on probation?",
        "What is the XF grade and how is it cleared?",
        "What is the policy on deferment of a semester?",
        "What is the Rector's Gold Medal awarded for?",
        "How many courses can a student register for in a Summer Semester?",
    ]
    DEFAULT_TOP_K = 5

    @classmethod
    def chip(cls, method: str) -> str:
        color = cls.METHOD_COLORS.get(method, "#666")
        label = method.upper()
        return (
            f"<span class='chip' style='background:{color}1A;color:{color};"
            f"border:1px solid {color}40;'>{label}</span>"
        )


# ----------------------------------------------------------------------
# Hero & section helpers
# ----------------------------------------------------------------------

def render_hero() -> None:
    st.markdown("""
    <div class="nust-hero">
        <h1>📚 NUST Academic Policy QA</h1>
        <div class="tag">Scalable retrieval over the NUST Undergraduate Handbook —
            MinHash + LSH, SimHash, and TF-IDF, with optional LLM grounding.</div>
        <div class="badge-row">
            <span class="nust-badge gold">CS-404 · Big Data</span>
            <span class="nust-badge">MinHash + LSH</span>
            <span class="nust-badge">SimHash</span>
            <span class="nust-badge">TF-IDF Baseline</span>
            <span class="nust-badge">SON · MapReduce</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section(title: str, icon: str = "") -> None:
    icon_html = f"<span>{icon}</span>" if icon else ""
    st.markdown(
        f"<div class='nust-section'>{icon_html}<h3>{title}</h3>"
        f"<span class='rule'></span></div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        "<div class='nust-footer'>"
        "Built for CS-404 · NUST SEECS · "
        "Locality Sensitive Hashing · TF-IDF · LLM-grounded answers"
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

class Sidebar:
    """Renders the sidebar and returns user-selected settings."""

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg

    def render(self) -> Dict:
        with st.sidebar:
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;color:{Theme.PRIMARY};"
                f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;'>"
                f"Configuration</div>", unsafe_allow_html=True,
            )
            st.markdown("### ⚙️ Settings")

            uploaded_pdfs = st.file_uploader(
                "Handbook PDF(s)",
                type=["pdf"],
                accept_multiple_files=True,
                help=(
                    "Upload one or more PDFs (UG handbook, PG handbook, "
                    "policy docs). All uploaded files are indexed and "
                    "queried together as a single corpus."
                ),
            )

            st.markdown("&nbsp;", unsafe_allow_html=True)
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
                horizontal=False,
            )

            groq_api_key = ""
            if answer_mode in ("groq", "analysis"):
                groq_api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    help="Get a free key at console.groq.com — required for "
                         "Llama answers and Analysis mode.",
                )

            show_comparison = st.checkbox("Show all-method comparison", value=True)

            st.divider()
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;color:{Theme.PRIMARY};"
                f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;'>"
                f"Quick Queries</div>", unsafe_allow_html=True,
            )
            sample_clicked = None
            for q in self.cfg.SAMPLE_QUERIES:
                if st.button(q, use_container_width=True, key=f"sq_{hash(q)}"):
                    sample_clicked = q

        return {
            "uploaded_pdfs": uploaded_pdfs,
            "top_k": top_k,
            "method": method,
            "answer_mode": answer_mode,
            "groq_api_key": groq_api_key,
            "show_comparison": show_comparison,
            "sample_clicked": sample_clicked,
        }

    def show_index_metrics(self, n_chunks: int, avg_words: int, n_files: int) -> None:
        with st.sidebar:
            st.divider()
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;color:{Theme.PRIMARY};"
                f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;'>"
                f"Index Stats</div>", unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Files", n_files)
            c2.metric("Chunks", n_chunks)
            c3.metric("Avg words", avg_words)


# ----------------------------------------------------------------------
# Results view
# ----------------------------------------------------------------------

class ResultsView:
    """Renders retrieval results, evidence, and method comparison."""

    def __init__(self, cfg: UIConfig):
        self.cfg = cfg

    def answer_comparison(
        self,
        ext_answer: str,
        llm_answer: str,
        method: str,
        latency_ms: float,
    ) -> None:
        """Side-by-side extractive vs Llama answer cards."""
        col_ext, col_llm = st.columns(2)
        with col_ext:
            st.markdown(f"""
            <div class='nust-card answer' style='border-left-color:#1976D2;'>
                <div class='answer-label' style='color:#1976D2;'>Extractive</div>
                <div class='answer-body'>{ext_answer}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_llm:
            st.markdown(f"""
            <div class='nust-card answer' style='border-left-color:{Theme.PRIMARY};'>
                <div class='answer-label'>Llama (Groq)</div>
                <div class='answer-body'>{llm_answer}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='nust-card meta'>
                <div>Retrieval: {self.cfg.chip(method)}
                    <b style='margin-left:6px'>{self.cfg.METHOD_LABELS[method]}</b></div>
                <div>Latency: <b>{latency_ms:.2f} ms</b></div>
                <div>Mode: <b>Analysis · both answerers on identical retrieved chunks</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def answer_card(self, answer_text: str, method: str, latency_ms: float, mode: str) -> None:
        st.markdown(
            f"""
            <div class='nust-card answer'>
                <div class='answer-label'>Answer</div>
                <div class='answer-body'>{answer_text}</div>
            </div>
            <div class='nust-card meta'>
                <div>Retrieval: {self.cfg.chip(method)}
                    <b style='margin-left:6px'>{self.cfg.METHOD_LABELS[method]}</b></div>
                <div>Latency: <b>{latency_ms:.2f} ms</b></div>
                <div>Mode: <b>{self.cfg.ANSWER_MODE_LABELS[mode]}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def chunk_card(self, chunk: Dict, rank: int) -> None:
        method = chunk.get("method", "")
        score = chunk.get("score", 0)
        pages = f"Pages {chunk['start_page']}–{chunk['end_page']}"
        source = chunk.get("source", "")
        src_seg = f"  ·  📄 {source}" if source else ""
        header = f"#{rank}  ·  {pages}{src_seg}  ·  Score {score:.4f}  ·  {method.upper()}"
        with st.expander(header):
            meta_bits = []
            if source:
                meta_bits.append(f"Source: **{source}**")
            if chunk.get("section"):
                meta_bits.append(f"Section: {chunk['section']}")
            if meta_bits:
                st.caption("  ·  ".join(meta_bits))
            text = chunk["text"]
            st.markdown(text[:700] + ("…" if len(text) > 700 else ""))

    def comparison_table(self, results_all: Dict[str, Tuple[List[Dict], float]]) -> None:
        cols = st.columns(len(results_all))
        for col, (method, (results, latency)) in zip(cols, results_all.items()):
            color = self.cfg.METHOD_COLORS.get(method, "#999")
            with col:
                st.markdown(
                    f"""
                    <div class='nust-card' style='border-top:3px solid {color};'>
                        <div style='font-size:11px;font-weight:700;letter-spacing:1px;
                                    text-transform:uppercase;color:{color};'>
                            {method}
                        </div>
                        <div style='font-size:14px;font-weight:600;margin:4px 0 8px;'>
                            {self.cfg.METHOD_LABELS[method]}
                        </div>
                        <div style='font-size:12px;color:#6B6B6B;margin-bottom:10px;'>
                            ⏱ {latency:.2f} ms
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if results:
                    top = results[0]
                    st.markdown(
                        f"""
                        <div style='font-size:12px;color:#6B6B6B;'>Top score</div>
                        <div style='font-size:18px;font-weight:700;color:{color};
                                    margin-bottom:6px;'>{top['score']:.4f}</div>
                        <div style='font-size:12px;color:#6B6B6B;margin-bottom:8px;'>
                            Pages {top['start_page']}–{top['end_page']}
                        </div>
                        <div style='font-size:13px;line-height:1.5;color:#333;'>
                            {top['text'][:240]}…
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No results")
                st.markdown("</div>", unsafe_allow_html=True)

    def evidence(self, supporting_chunks: List[Dict]) -> None:
        for i, ev in enumerate(supporting_chunks, 1):
            src_seg = f"  ·  📄 {ev['source']}" if ev.get("source") else ""
            header = (f"Evidence #{i}  ·  Pages {ev['pages']}"
                      f"{src_seg}  ·  Score {ev['score']:.4f}")
            with st.expander(header):
                meta_bits = []
                if ev.get("source"):
                    meta_bits.append(f"Source: **{ev['source']}**")
                if ev.get("section"):
                    meta_bits.append(f"Section: {ev['section']}")
                if meta_bits:
                    st.caption("  ·  ".join(meta_bits))
                st.markdown(ev["text"])

    def build_stats_panel(self, stats: Dict) -> None:
        """SON / MapReduce build statistics card (per-file + timings)."""
        # ----- Per-file source cards -----
        per_file = stats["per_file"]
        n_cols = min(len(per_file), 3) or 1
        cols = st.columns(n_cols)
        for i, f in enumerate(per_file):
            with cols[i % n_cols]:
                st.markdown(f"""
                <div class='nust-card' style='padding:14px 16px;'>
                  <div style='font-size:11px;text-transform:uppercase;
                              color:{Theme.PRIMARY};font-weight:600;
                              letter-spacing:1px;'>📄 Source {i + 1}</div>
                  <div style='font-size:14px;font-weight:600;margin-top:4px;
                              word-break:break-word;line-height:1.35;'
                       title="{f['name']}">{f['name']}</div>
                  <div style='display:flex;gap:14px;margin-top:10px;
                              font-size:12px;color:{Theme.MUTED};
                              flex-wrap:wrap;'>
                    <span><b style='color:{Theme.INK};'>{f['n_chunks']}</b> chunks</span>
                    <span><b style='color:{Theme.INK};'>{f['avg_words']}</b> avg words</span>
                    <span><b style='color:{Theme.INK};'>{f['max_page']}</b> pages</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ----- SON / MapReduce timings -----
        mh, sh = stats["minhash"], stats["simhash"]
        st.markdown(f"""
        <div class='nust-card' style='margin-top:14px;padding:18px 22px;'>
          <div style='display:flex;justify-content:space-between;
                      align-items:baseline;margin-bottom:14px;flex-wrap:wrap;gap:10px;'>
            <div>
              <div style='font-size:11px;text-transform:uppercase;
                          color:{Theme.PRIMARY};font-weight:600;
                          letter-spacing:1.4px;'>SON · MapReduce build</div>
              <div style='font-size:15px;font-weight:600;margin-top:2px;'>
                {stats['n_chunks_total']} chunks · {stats['n_files']} file(s)
                · <span style='color:{Theme.PRIMARY};'>{stats['n_workers']} parallel workers</span>
              </div>
            </div>
            <div style='font-size:11px;color:{Theme.MUTED};
                        text-align:right;line-height:1.5;'>
              Phase 1 (Map): per-partition signatures<br>
              Phase 2 (Reduce): merge into global LSH bands
            </div>
          </div>
          <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px;'>
            <div style='border-left:3px solid #E64A19;padding-left:12px;'>
              <div style='font-size:11px;color:{Theme.MUTED};
                          text-transform:uppercase;letter-spacing:0.6px;'>MinHash + LSH</div>
              <div style='font-size:24px;font-weight:700;color:#E64A19;
                          margin-top:2px;'>{mh['total_time_s']}s</div>
              <div style='font-size:12px;color:{Theme.MUTED};margin-top:2px;'>
                Map <b style='color:{Theme.INK};'>{mh['map_time_s']}s</b>
                · Reduce <b style='color:{Theme.INK};'>{mh['reduce_time_s']}s</b>
              </div>
            </div>
            <div style='border-left:3px solid #2E7D32;padding-left:12px;'>
              <div style='font-size:11px;color:{Theme.MUTED};
                          text-transform:uppercase;letter-spacing:0.6px;'>SimHash</div>
              <div style='font-size:24px;font-weight:700;color:#2E7D32;
                          margin-top:2px;'>{sh['total_time_s']}s</div>
              <div style='font-size:12px;color:{Theme.MUTED};margin-top:2px;'>
                Map <b style='color:{Theme.INK};'>{sh['map_time_s']}s</b>
                · Reduce <b style='color:{Theme.INK};'>{sh['reduce_time_s']}s</b>
              </div>
            </div>
            <div style='border-left:3px solid #1976D2;padding-left:12px;'>
              <div style='font-size:11px;color:{Theme.MUTED};
                          text-transform:uppercase;letter-spacing:0.6px;'>TF-IDF (serial)</div>
              <div style='font-size:24px;font-weight:700;color:#1976D2;
                          margin-top:2px;'>{stats['tfidf_time_s']}s</div>
              <div style='font-size:12px;color:{Theme.MUTED};margin-top:2px;'>
                Baseline · single process
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


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
            initial_sidebar_state="expanded",
        )
        Theme.inject()
        render_hero()

    def _load_system(self, uploaded_pdfs) -> Tuple[Retriever, List[Dict], Dict]:
        file_specs = self.uploads.save_many(uploaded_pdfs)
        return _build_retriever(file_specs, str(self.data_dir))

    def _empty_state(self) -> None:
        st.markdown(f"""
        <div class='nust-card' style='text-align:center;padding:48px 24px;'>
            <div style='font-size:42px;margin-bottom:8px;'>📄</div>
            <div style='font-size:17px;font-weight:600;margin-bottom:4px;'>
                Upload one or more handbooks to begin
            </div>
            <div style='color:{Theme.MUTED};font-size:14px;line-height:1.6;'>
                Drop the NUST UG/PG handbook (and optionally any related policy PDFs)
                into the sidebar. Files are chunked separately and cached by content
                hash, then indexed together via a SON / MapReduce parallel build.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- main flow ----------

    def run(self) -> None:
        self._configure_page()
        settings = self.sidebar.render()

        uploaded = settings["uploaded_pdfs"]
        if not uploaded:
            self._empty_state()
            render_footer()
            st.stop()

        try:
            retriever, chunks, build_stats = self._load_system(uploaded)
        except Exception as e:
            st.error(f"Failed to load handbook(s): {e}")
            st.stop()

        avg_words = sum(c["word_count"] for c in chunks) // len(chunks)
        self.sidebar.show_index_metrics(
            len(chunks), avg_words, build_stats["n_files"]
        )

        # Build statistics card
        section("Indexed sources · SON / MapReduce build", "📦")
        self.results_view.build_stats_panel(build_stats)

        # ---------- query input ----------
        section("Ask a question", "💬")
        default_query = settings["sample_clicked"] or st.session_state.get("last_query", "")
        query = st.text_input(
            "Question",
            value=default_query,
            placeholder="e.g. What is the minimum CGPA to graduate?",
            label_visibility="collapsed",
        )
        col1, _ = st.columns([1, 5])
        search_clicked = col1.button("🔍  Search", type="primary", use_container_width=True)

        if not query:
            st.markdown(
                f"<div style='color:{Theme.MUTED};font-size:13px;margin-top:6px;'>"
                f"Type a question above or pick a quick query from the sidebar."
                f"</div>", unsafe_allow_html=True,
            )
            render_footer()
            return

        if not (search_clicked or settings["sample_clicked"]):
            render_footer()
            return

        st.session_state["last_query"] = query

        # ---------- retrieval ----------
        with st.spinner("Retrieving relevant chunks…"):
            results, latency = retriever.query(
                query, method=settings["method"], top_k=settings["top_k"]
            )
            results_all = (
                retriever.query_all(query, top_k=settings["top_k"])
                if settings["show_comparison"] else None
            )

        # ---------- answer generation ----------
        mode = settings["answer_mode"]
        groq_kwargs = {"max_context_chunks": settings["top_k"]}
        if settings.get("groq_api_key"):
            groq_kwargs["api_key"] = settings["groq_api_key"]

        if mode == "analysis":
            with st.spinner("Generating extractive + Llama answers in parallel…"):
                ext_data = make_answerer("extractive").generate(query, results)
                llm_data = make_answerer("groq", **groq_kwargs).generate(query, results)
            answer_data = ext_data  # use extractive's evidence as the single source
        else:
            with st.spinner("Generating answer…"):
                kwargs = groq_kwargs if mode == "groq" else {}
                answerer = make_answerer(mode, **kwargs)
                answer_data = answerer.generate(query, results)

        # ---------- display ----------
        section("Answer", "💡")
        if mode == "analysis":
            self.results_view.answer_comparison(
                ext_data["answer"],
                llm_data["answer"],
                settings["method"],
                latency,
            )
        else:
            self.results_view.answer_card(
                answer_data["answer"],
                settings["method"],
                latency,
                settings["answer_mode"],
            )

        section("Supporting evidence", "📄")
        self.results_view.evidence(answer_data["supporting_chunks"])

        section(f"Top-{settings['top_k']} retrieved chunks", "🔎")
        for rank, chunk in enumerate(results, 1):
            self.results_view.chunk_card(chunk, rank)

        if results_all is not None:
            section("All-method comparison (Top-1)", "📊")
            self.results_view.comparison_table(results_all)

        render_footer()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    QAApp().run()
