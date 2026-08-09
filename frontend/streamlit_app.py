import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.evaluation.question_generator import (
    generate_dataset_from_directory
)

from app.evaluation.evaluator import (
    run_evaluation
)

from app.evaluation.dynamic_evaluator import (
    STRATEGIES
)

from app.config import DOCS_DIR

import pandas as pd
import streamlit as st
import os
import sys
import shutil
import subprocess
import json
import ast
import time
from pathlib import Path

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVALUATION_DIR = DATA_DIR / "evaluation"
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------
# BACKEND IMPORT
# ---------------------------------------------------------

try:
    from app.rag.pipeline import ask
    BACKEND_AVAILABLE = True
except Exception as e:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(e)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocuMind — AI Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F8FAFC;
    color: #172033;
}

/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}

/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #1E1B4B 100%
    );

    border-right: none;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Sidebar text */

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

/* Logo */

.logo {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.logo span {
    color: #A78BFA;
}

.tagline {
    color: #CBD5E1 !important;
    font-size: 12px;
    margin-bottom: 30px;
}

/* Navigation title */

.nav-title {
    color: #94A3B8 !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin: 20px 0 8px 0;
}

/* ================================
   HERO
================================ */

.hero {
    padding: 34px 38px;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #4F46E5 0%,
            #7C3AED 55%,
            #06B6D4 100%
        );

    box-shadow:
        0 15px 35px rgba(79,70,229,0.20);

    margin-bottom: 28px;
    color: white;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1.8px;
    margin-bottom: 8px;
    color: white;
}

.hero-subtitle {
    color: #E0E7FF;
    font-size: 15px;
}

/* ================================
   METRIC CARDS
================================ */

.metric-card {
    padding: 22px;
    border-radius: 18px;

    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    box-shadow:
        0 6px 20px rgba(15,23,42,0.06);

    min-height: 125px;

    transition: 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 30px rgba(79,70,229,0.12);
}

.metric-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.metric-value {
    color: #312E81;
    font-size: 31px;
    font-weight: 800;
}

.metric-description {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 6px;
}

/* ================================
   SECTION TITLES
================================ */

.section-title {
    font-size: 21px;
    font-weight: 800;
    color: #172033;
    margin-top: 30px;
    margin-bottom: 16px;
}

/* ================================
   DOCUMENT CARDS
================================ */

.document-card {
    padding: 20px;
    border-radius: 17px;

    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    box-shadow:
        0 4px 15px rgba(15,23,42,0.05);

    margin-bottom: 12px;

    transition: 0.2s ease;
}

.document-card:hover {
    border-color: #818CF8;

    box-shadow:
        0 8px 25px rgba(79,70,229,0.10);
}

.document-name {
    color: #1E293B;
    font-weight: 700;
    font-size: 15px;
}

.document-meta {
    color: #64748B;
    font-size: 12px;
    margin-top: 6px;
}

/* ================================
   CHAT
================================ */

.user-message {
    background: #EEF2FF;

    border: 1px solid #C7D2FE;

    color: #312E81;

    padding: 17px 20px;

    border-radius: 18px 18px 5px 18px;

    margin: 12px 0 12px 18%;
}

.ai-message {
    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    color: #334155;

    padding: 20px;

    border-radius: 18px 18px 18px 5px;

    margin: 12px 18% 12px 0;

    box-shadow:
        0 5px 18px rgba(15,23,42,0.05);
}

.ai-label {
    color: #4F46E5;

    font-size: 12px;

    font-weight: 800;

    margin-bottom: 8px;
}

/* ================================
   SOURCE CARDS
================================ */

.source-card {
    padding: 15px 17px;

    border-radius: 14px;

    background: #F8FAFC;

    border: 1px solid #E2E8F0;

    margin: 8px 0;
}

.source-title {
    color: #1E293B;
    font-size: 13px;
    font-weight: 700;
}

.source-meta {
    color: #64748B;
    font-size: 11px;
    margin-top: 4px;
}

/* ================================
   STATUS
================================ */

.status-online {
    color: #34D399 !important;

    font-size: 12px;

    font-weight: 700;
}

.status-offline {
    color: #FB7185 !important;

    font-size: 12px;

    font-weight: 700;
}

/* ================================
   STREAMLIT BUTTONS
================================ */

.stButton > button {

    background: #4F46E5;

    color: white;

    border: none;

    border-radius: 10px;

    font-weight: 700;

    padding: 10px 18px;

    transition: 0.2s ease;

    box-shadow:
        0 4px 12px rgba(79,70,229,0.18);
}

.stButton > button:hover {

    background: #4338CA;

    color: white;

    transform: translateY(-1px);

    box-shadow:
        0 7px 18px rgba(79,70,229,0.25);
}

/* ================================
   INPUTS
================================ */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {

    background: #FFFFFF !important;

    color: #172033 !important;

    border-radius: 11px !important;

    border: 1px solid #CBD5E1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {

    border-color: #6366F1 !important;

    box-shadow:
        0 0 0 2px rgba(99,102,241,0.12) !important;
}

/* ================================
   FILE UPLOADER
================================ */

[data-testid="stFileUploader"] {

    background: #FFFFFF;

    border: 2px dashed #A5B4FC;

    border-radius: 16px;

    padding: 10px;
}

/* ================================
   SELECTBOX
================================ */

div[data-baseweb="select"] {

    background: white !important;

    border-radius: 10px !important;
}

/* ================================
   DATAFRAME
================================ */

[data-testid="stDataFrame"] {

    border-radius: 14px;

    overflow: hidden;

    border: 1px solid #E2E8F0;
}

/* ================================
   ALERTS
================================ */

[data-testid="stAlert"] {

    border-radius: 12px;
}

/* ================================
   CHAT INPUT
================================ */

[data-testid="stChatInput"] {

    border-radius: 14px;

    border: 1px solid #CBD5E1;
}

/* ================================
   SLIDER
================================ */

.stSlider [data-baseweb="slider"] {

    color: #4F46E5;
}

/* ================================
   RADIO
================================ */

[data-testid="stSidebar"] .stRadio label {

    color: #E2E8F0 !important;

    font-weight: 500;
}

/* ================================
   DIVIDERS
================================ */

hr {

    border-color: #E2E8F0;
}

/* ================================
   STATUS BADGE
================================ */

.status-badge {

    display: inline-block;

    padding: 6px 12px;

    border-radius: 20px;

    background: #DCFCE7;

    color: #15803D;

    font-size: 11px;

    font-weight: 700;
}

/* ================================
   EVALUATION COLORS
================================ */

.eval-good {

    background: #ECFDF5;

    border: 1px solid #A7F3D0;

    color: #047857;

    padding: 14px;

    border-radius: 12px;
}

.eval-warning {

    background: #FFFBEB;

    border: 1px solid #FDE68A;

    color: #B45309;

    padding: 14px;

    border-radius: 12px;
}

.eval-danger {

    background: #FEF2F2;

    border: 1px solid #FECACA;

    color: #B91C1C;

    padding: 14px;

    border-radius: 12px;
}

/* ================================
   HIDE STREAMLIT BRANDING
================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_documents():
    return sorted(DOCUMENTS_DIR.glob("*.pdf"))


def get_document_count():
    return len(get_documents())


def get_questions_count():
    try:
        if QUESTIONS_FILE.exists():
            with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data)
    except Exception:
        pass
    return 0


def run_command(command):
    """
    Run a project command and return stdout/stderr.
    """
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out."


def run_evaluation(strategy):
    """
    Run the existing evaluation script.
    """

    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--strategy",
        strategy
    ]

    return run_command(command)


def run_ingestion(strategy, reset=True):
    """
    Run existing ingestion pipeline.
    """

    command = [
        sys.executable,
        "scripts/ingest.py",
        "--strategy",
        strategy
    ]

    if reset:
        command.append("--reset")

    return run_command(command)


def parse_metrics(output):
    """
    Try to find a dictionary in evaluation output.
    """

    lines = output.strip().splitlines()

    for line in reversed(lines):
        line = line.strip()

        if line.startswith("{") and line.endswith("}"):
            try:
                return ast.literal_eval(line)
            except Exception:
                pass

    return None


def metric_value(metrics, key):
    if not metrics:
        return 0

    value = metrics.get(key, 0)

    try:
        return float(value)
    except Exception:
        return 0


def format_metric(value):
    if value <= 1:
        return f"{value * 100:.1f}%"
    return f"{value:.1f}"


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="logo">◈ Docu<span>Mind</span></div>
        <div class="tagline">
            AI Document Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-title">Workspace</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Documents",
            "Chat",
            "Evaluation Lab",
            "Experiments",
            "Analytics",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="nav-title">System</div>',
        unsafe_allow_html=True
    )

    if BACKEND_AVAILABLE:
        st.markdown(
            '<div class="status-online">● RAG Engine Online</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-offline">● RAG Engine Error</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.caption("DocuMind v2.0")
    st.caption("RAG • Semantic Search • Evaluation")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:15px;
    ">
        <div>
            <div style="
                font-size:12px;
                color:#64748b;
                text-transform:uppercase;
                letter-spacing:1.5px;
            ">
                AI Document Intelligence
            </div>
        </div>
        <div style="
            padding:8px 14px;
            border-radius:20px;
            background:rgba(52,211,153,0.08);
            border:1px solid rgba(52,211,153,0.15);
            color:#34d399;
            font-size:12px;
        ">
            ● System Ready
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Intelligent Document Search
            </div>
            <div class="hero-subtitle">
                Ask questions, retrieve evidence, and evaluate your RAG pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    documents = get_documents()
    document_count = len(documents)
    question_count = get_questions_count()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">DOCUMENTS</div>
                <div class="metric-value">{document_count}</div>
                <div class="metric-description">PDF files indexed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EVALUATION QUESTIONS</div>
                <div class="metric-value">{question_count}</div>
                <div class="metric-description">Ground-truth questions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">RETRIEVAL</div>
                <div class="metric-value">Top-K</div>
                <div class="metric-description">Semantic vector search</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">ENGINE</div>
                <div class="metric-value">RAG</div>
                <div class="metric-description">Retrieval augmented generation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Quick Start</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown("### 📄 Documents")
        st.write("Upload and index PDF documents.")
        if st.button("Manage Documents", use_container_width=True):
            st.session_state["page_override"] = "Documents"
            st.rerun()

    with q2:
        st.markdown("### 💬 Ask Questions")
        st.write("Chat with your documents using RAG.")
        if st.button("Open Document Chat", use_container_width=True):
            st.session_state["page_override"] = "Chat"
            st.rerun()

    with q3:
        st.markdown("### 🔬 Evaluate")
        st.write("Measure retrieval quality.")
        if st.button("Open Evaluation Lab", use_container_width=True):
            st.session_state["page_override"] = "Evaluation Lab"
            st.rerun()

    st.markdown(
        '<div class="section-title">Recent Documents</div>',
        unsafe_allow_html=True
    )

    if documents:

        for document in documents[:5]:

            size_kb = document.stat().st_size / 1024

            st.markdown(
                f"""
                <div class="document-card">
                    <div class="document-name">
                        📄 {document.name}
                    </div>
                    <div class="document-meta">
                        {size_kb:.1f} KB · PDF · Indexed
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("No documents uploaded yet. Go to Documents to add a PDF.")


# =========================================================
# DOCUMENTS
# =========================================================

elif page == "Documents":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📚 Document Library</div>
            <div class="hero-subtitle">
                Upload, manage and index your knowledge base.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Upload PDF")

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        destination = DOCUMENTS_DIR / uploaded_file.name

        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded `{uploaded_file.name}` successfully.")

    documents = get_documents()

    st.markdown("### Your Documents")

    if documents:

        for document in documents:

            col1, col2, col3 = st.columns([5, 2, 1])

            size_kb = document.stat().st_size / 1024

            with col1:
                st.markdown(
                    f"""
                    <div class="document-card">
                        <div class="document-name">
                            📄 {document.name}
                        </div>
                        <div class="document-meta">
                            {size_kb:.1f} KB · PDF
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if st.button(
                    "Re-index",
                    key=f"index_{document.name}"
                ):

                    with st.spinner("Indexing document..."):

                        code, stdout, stderr = run_ingestion(
                            "recursive",
                            reset=True
                        )

                    if code == 0:
                        st.success("Document indexed successfully.")
                    else:
                        st.error(stderr or stdout)

            with col3:
                if st.button(
                    "Delete",
                    key=f"delete_{document.name}"
                ):

                    try:
                        document.unlink()
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    else:
        st.info("Upload a PDF to create your document knowledge base.")


# =========================================================
# CHAT
# =========================================================

elif page == "Chat":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">💬 Document Chat</div>
            <div class="hero-subtitle">
                Ask questions and get grounded answers with source citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not BACKEND_AVAILABLE:
        st.error("RAG backend could not be loaded.")
        st.code(BACKEND_ERROR)
        st.stop()

    documents = get_documents()

    if not documents:
        st.warning("Please upload and index a PDF first.")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_document = st.selectbox(
            "Document",
            [d.name for d in documents]
        )

    with col2:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=10,
            value=5
        )

    strategy = st.selectbox(
        "Chunking strategy",
        [
            "recursive",
            "fixed",
            "sentence",
            "parent_child"
        ]
    )

    st.markdown("### Conversation")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ai-message">
                    <div class="ai-label">✦ DOCUMIND AI</div>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("Searching your knowledge base..."):

            start = time.time()

            try:
                result = ask(
                    question,
                    strategy,
                    top_k
                )

                elapsed = time.time() - start

            except Exception as e:

                st.error(f"RAG error: {e}")
                st.stop()

        # Handle existing result structure
        if isinstance(result, dict):

            answer = result.get(
                "answer",
                result.get("response", "")
            )

            contexts = result.get(
                "contexts",
                result.get("sources", [])
            )

        else:

            answer = str(result)
            contexts = []

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()

    if st.session_state.chat_history:

        st.markdown("### Retrieved Evidence")

        # Get last answer result again only if supported
        # Existing backend remains untouched.


# =========================================================
# EVALUATION LAB
# =========================================================

# ============================================================
# RAG EVALUATION LAB
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔬 RAG Evaluation Lab
    </div>

    <div class="section-subtitle">
        Dynamically generate evaluation questions from your
        documents and benchmark different chunking strategies.
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Evaluation controls
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    questions_per_page = st.number_input(
        "Questions per page",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:

    max_questions = st.number_input(
        "Maximum questions",
        min_value=2,
        max_value=30,
        value=10
    )

with col3:

    st.metric(
        "Strategies",
        len(STRATEGIES)
    )


# ------------------------------------------------------------
# Run button
# ------------------------------------------------------------

run_evaluation_button = st.button(
    "⚡ Generate Questions & Run All Experiments",
    use_container_width=True
)


if run_evaluation_button:

    # ========================================================
    # STEP 1 — GENERATE QUESTIONS
    # ========================================================

    try:

        with st.spinner(
            "🤖 AI is analyzing your documents..."
        ):

            dataset = generate_dataset_from_directory(
                DOCS_DIR,
                questions_per_page=int(
                    questions_per_page
                ),
                max_questions=int(
                    max_questions
                )
            )

        if not dataset:

            st.error(
                "No evaluation questions were generated."
            )

            st.stop()

        st.success(
            f"✓ Generated {len(dataset)} dynamic questions"
        )

        # ----------------------------------------------------
        # Show questions
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 AI-Generated Evaluation Questions"
        )

        for index, item in enumerate(
            dataset,
            start=1
        ):

            with st.expander(
                f"Q{index} · {item['question']}"
            ):

                st.write(
                    "**Ground Truth:**"
                )

                for source in item[
                    "expected_sources"
                ]:

                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )

        # ====================================================
        # STEP 2 — EVALUATE STRATEGIES
        # ====================================================

        st.markdown(
            "### 🔬 Chunking Experiments"
        )

        results = {}

        progress = st.progress(
            0
        )

        status = st.empty()

        for index, strategy in enumerate(
            STRATEGIES
        ):

            status.markdown(
                f"🔄 Running **{strategy}**..."
            )

            try:

                summary, rows = run_evaluation(
                    strategy=strategy,
                    dataset=dataset
                )

                results[strategy] = {
                    "summary": summary,
                    "rows": rows
                }

            except Exception as exc:

                results[strategy] = {
                    "summary": {},
                    "rows": [],
                    "error": str(exc)
                }

            progress.progress(
                (index + 1) /
                len(STRATEGIES)
            )

        status.success(
            "✓ All chunking experiments completed."
        )

        # ====================================================
        # STEP 3 — PERFORMANCE TABLE
        # ====================================================

        st.markdown(
            "### 📊 Performance Comparison"
        )

        comparison = []

        for strategy, result in results.items():

            summary = result.get(
                "summary",
                {}
            )

            comparison.append(
                {
                    "Strategy": strategy.replace(
                        "_",
                        " "
                    ).title(),

                    "Hit@1": summary.get(
                        "hit@1",
                        0
                    ),

                    "Hit@3": summary.get(
                        "hit@3",
                        0
                    ),

                    "Hit@5": summary.get(
                        "hit@5",
                        0
                    ),

                    "MRR": summary.get(
                        "mrr",
                        0
                    )
                }
            )

        comparison_df = pd.DataFrame(
            comparison
        )

        if not comparison_df.empty:

            display_df = comparison_df.copy()

            for column in [
                "Hit@1",
                "Hit@3",
                "Hit@5",
                "MRR"
            ]:

                display_df[column] = (
                    display_df[column] * 100
                ).round(1).astype(str) + "%"

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # STEP 4 — BEST STRATEGY
        # ====================================================

        valid_results = {
            strategy: result["summary"]
            for strategy, result in results.items()
            if result.get("summary")
        }

        if valid_results:

            best_strategy = max(
                valid_results,
                key=lambda strategy:
                    valid_results[strategy].get(
                        "mrr",
                        0
                    )
            )

            best_summary = valid_results[
                best_strategy
            ]

            st.markdown(
                "### 🏆 Best Retrieval Strategy"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Strategy",
                    best_strategy.replace(
                        "_",
                        " "
                    ).title()
                )

            with c2:

                st.metric(
                    "MRR",
                    f"{best_summary.get('mrr', 0):.1%}"
                )

            with c3:

                st.metric(
                    "Hit@1",
                    f"{best_summary.get('hit@1', 0):.1%}"
                )

            st.success(
                f"🏆 **{best_strategy.replace('_', ' ').title()}** "
                f"achieved the highest MRR on the dynamically "
                f"generated evaluation dataset."
            )

    except Exception as exc:

        st.error(
            f"❌ Evaluation failed: {exc}"
        )

# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📈 Analytics</div>
            <div class="hero-subtitle">
                Monitor your RAG system and retrieval experiments.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    documents = get_documents()

    total_size = sum(
        d.stat().st_size for d in documents
    ) / 1024

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Documents",
            len(documents)
        )

    with c2:
        st.metric(
            "Knowledge Base Size",
            f"{total_size:.1f} KB"
        )

    with c3:
        st.metric(
            "Evaluation Questions",
            get_questions_count()
        )

    st.markdown("### System Architecture")

    st.code(
        """
PDF Documents
      ↓
Text Extraction
      ↓
Advanced Chunking
      ↓
Embedding Model
      ↓
ChromaDB Vector Store
      ↓
Semantic Retrieval
      ↓
Top-K Context
      ↓
LLM
      ↓
Grounded Answer + Sources
      ↓
Retrieval Evaluation
      ↓
Hit@K + MRR
        """,
        language="text"
    )

    st.markdown("### RAG Metrics")

    metrics_data = {
        "Metric": [
            "Retrieval",
            "Generation",
            "Evaluation",
            "Vector Database"
        ],
        "Technology": [
            "Semantic Search",
            "LLM",
            "Hit@K / MRR",
            "ChromaDB"
        ],
        "Status": [
            "Active",
            "Active",
            "Active",
            "Active"
        ]
    }

    import pandas as pd

    st.dataframe(
        pd.DataFrame(metrics_data),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SETTINGS
# =========================================================

elif page == "Settings":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">⚙️ Settings</div>
            <div class="hero-subtitle">
                Configure your RAG workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Retrieval")

    top_k_setting = st.slider(
        "Default Top-K",
        1,
        10,
        5
    )

    st.markdown("### Chunking")

    default_strategy = st.selectbox(
        "Default Chunking Strategy",
        [
            "recursive",
            "fixed",
            "sentence",
            "parent_child"
        ]
    )

    st.markdown("### LLM")

    provider = os.getenv(
        "LLM_PROVIDER",
        "openai"
    )

    st.text_input(
        "LLM Provider",
        value=provider,
        disabled=True
    )

    st.markdown("### Embeddings")

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )

    st.text_input(
        "Embedding Model",
        value=embedding_model,
        disabled=True
    )

    st.info(
        "Configuration values are loaded from your project's .env file."
    )