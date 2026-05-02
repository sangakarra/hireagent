"""
HireAgent — Streamlit UI
Run: streamlit run src/hireagent/ui/app.py
"""
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the src/ package root is importable when running via `streamlit run`
_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Load ANTHROPIC_API_KEY — local .env first, Streamlit Cloud secrets as fallback.
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["general"]["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

from hireagent.agents import run_analysis
from hireagent.rag.document_loader import load_and_chunk_pdf
from hireagent.rag.pipeline import ingest_resume

# ── Constants ────────────────────────────────────────────────────────────────
HISTORY_DIR = Path("data/history")
HISTORY_FILE = HISTORY_DIR / "history.json"
CHROMA_DIR = Path("data/chroma_db")


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HireAgent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Layout ---- */
    [data-testid="stAppViewContainer"] { background: #f0f4f8; }
    .block-container { padding-top: 1.75rem !important; max-width: 1100px; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] small { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155; }
    [data-testid="stSidebar"] .stFileUploader label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.04); border: 1px dashed #475569;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #2563eb; color: white !important; border: none;
        font-weight: 600; border-radius: 8px; transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #1d4ed8; }
    [data-testid="stSidebar"] .stButton > button:active { background: #1e40af; }
    [data-testid="stSidebar"] .stAlert { background: rgba(255,255,255,0.05); }

    /* ---- Main header ---- */
    .ha-title {
        font-size: 2.6rem; font-weight: 800; letter-spacing: -0.025em;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 60%, #0ea5e9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0; line-height: 1.15;
    }
    .ha-subtitle {
        font-size: 1rem; color: #64748b; margin-top: 0.3rem; margin-bottom: 1.75rem;
    }

    /* ---- Card ---- */
    .ha-card {
        background: #ffffff; border-radius: 14px; padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.04);
        border: 1px solid #e2e8f0;
    }
    .ha-card-title {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 14px;
    }

    /* ---- Skill badges ---- */
    .badge-wrap { display: flex; flex-wrap: wrap; gap: 7px; padding: 2px 0; }
    .badge {
        display: inline-block; background: #dcfce7; color: #15803d;
        border: 1px solid #86efac; padding: 4px 14px; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.01em;
    }
    .badge-empty { color: #94a3b8; font-size: 0.875rem; font-style: italic; }

    /* ---- Gap table ---- */
    .gap-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    .gap-table th {
        text-align: left; padding: 9px 14px; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8;
        background: #f8fafc; border-bottom: 1.5px solid #e2e8f0;
    }
    .gap-table th:first-child { border-radius: 6px 0 0 0; }
    .gap-table th:last-child  { border-radius: 0 6px 0 0; }
    .gap-table td { padding: 11px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; color: #1e293b; }
    .gap-table tr:last-child td { border-bottom: none; }
    .gap-table tbody tr:hover td { background: #f8fafc; }
    .sev-critical {
        display: inline-block; background: #fee2e2; color: #dc2626;
        padding: 3px 11px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
    }
    .sev-partial {
        display: inline-block; background: #fef9c3; color: #a16207;
        padding: 3px 11px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
    }
    .sev-nice-to-have, .sev-nice {
        display: inline-block; background: #f1f5f9; color: #64748b;
        padding: 3px 11px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
    }

    /* ---- Score gauge text (inside SVG) ---- */
    .gauge-wrap { text-align: center; padding: 6px 0 0; overflow: hidden; }

    /* ---- Cover letter textarea ---- */
    .stTextArea textarea {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: 0.925rem !important; line-height: 1.75 !important;
        color: #1e293b !important; background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important; border-radius: 8px !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 0.9rem; padding: 8px 18px;
        border-radius: 8px 8px 0 0; color: #64748b;
    }
    .stTabs [aria-selected="true"] { color: #2563eb !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: #2563eb !important; }

    /* ---- Footer ---- */
    .ha-footer {
        text-align: center; padding: 28px 0 16px; color: #94a3b8;
        font-size: 0.8rem; border-top: 1px solid #e2e8f0; margin-top: 36px;
        letter-spacing: 0.01em;
    }

    /* ---- Sidebar status chips ---- */
    .chip {
        display: inline-block; padding: 4px 12px; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600; margin: 3px 0; line-height: 1.4;
    }
    .chip-green { background: rgba(34,197,94,.15); color: #4ade80; }
    .chip-blue  { background: rgba(59,130,246,.15); color: #93c5fd; }

    /* Sidebar brand */
    .sb-brand {
        font-size: 1.45rem; font-weight: 800; color: #f1f5f9;
        letter-spacing: -0.015em; margin-bottom: 2px;
    }
    .sb-tag { font-size: 0.78rem; color: #64748b; }

    /* Expander label */
    .streamlit-expanderHeader { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Persistence ───────────────────────────────────────────────────────────────

def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_history(history: list) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


# ── Rendering helpers ─────────────────────────────────────────────────────────

def gauge_html(score: float) -> str:
    """Render an SVG arc-based gauge for a 0–10 score."""
    cx, cy, r = 160, 148, 118
    # theta_screen = 180° + 180° × (score/10), sweeping clockwise through the top
    theta = math.radians(180.0 + 180.0 * (score / 10.0))
    end_x = cx + r * math.cos(theta)
    end_y = cy + r * math.sin(theta)
    sx, sy = cx - r, cy
    ex, ey = cx + r, cy

    if score < 4:
        color, label = "#dc2626", "Poor Match"
    elif score < 7:
        color, label = "#d97706", "Moderate Match"
    else:
        color, label = "#16a34a", "Strong Match"

    if score <= 0.05:
        score_arc = ""
    elif score >= 9.95:
        # Full semicircle — avoid degenerate arc by tiny offset
        score_arc = (
            f'<path d="M {sx},{sy} A {r},{r} 0 0,1 {ex - 0.01},{ey + 0.01}" '
            f'fill="none" stroke="{color}" stroke-width="22" stroke-linecap="round"/>'
        )
    else:
        score_arc = (
            f'<path d="M {sx},{sy} A {r},{r} 0 0,1 {end_x:.2f},{end_y:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="22" stroke-linecap="round"/>'
        )

    return f"""
    <div class="gauge-wrap">
      <svg viewBox="0 0 320 180" width="100%"
           style="max-width:280px;display:block;margin:0 auto;overflow:visible"
           xmlns="http://www.w3.org/2000/svg" aria-label="Match score {score:.1f} out of 10">
        <!-- Background track -->
        <path d="M {sx},{sy} A {r},{r} 0 0,1 {ex},{ey}"
              fill="none" stroke="#e2e8f0" stroke-width="22" stroke-linecap="round"/>
        <!-- Score arc -->
        {score_arc}
        <!-- Score value (inside arc) -->
        <text x="{cx}" y="{cy - 24}" text-anchor="middle"
              font-family="system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
              font-size="62" font-weight="800" fill="{color}">{score:.1f}</text>
        <!-- out of 10 -->
        <text x="{cx}" y="{cy + 14}" text-anchor="middle"
              font-family="system-ui,-apple-system,sans-serif"
              font-size="16" fill="#94a3b8">out of 10</text>
      </svg>
      <div style="font-family:system-ui,-apple-system,sans-serif;font-size:1.05rem;
                  font-weight:700;color:{color};margin-top:10px;letter-spacing:0.01em">
        {label}
      </div>
    </div>"""


def skill_badges_html(skills: list) -> str:
    if not skills:
        return '<span class="badge-empty">None identified</span>'
    badges = "".join(f'<span class="badge">{s}</span>' for s in skills)
    return f'<div class="badge-wrap">{badges}</div>'


def gaps_table_html(gaps: list) -> str:
    _sev_css = {
        "critical": "sev-critical",
        "partial": "sev-partial",
        "nice-to-have": "sev-nice-to-have",
        "nice": "sev-nice",
    }
    rows = []
    for g in gaps:
        skill = g.get("skill", "")
        raw_sev = g.get("severity", "nice-to-have").lower().replace(" ", "-")
        css = _sev_css.get(raw_sev, "sev-nice-to-have")
        label = raw_sev.replace("-", " ").title()
        rows.append(
            f"<tr><td>{skill}</td>"
            f'<td><span class="{css}">{label}</span></td></tr>'
        )
    body = "\n".join(rows)
    return f"""
    <table class="gap-table">
      <thead><tr><th>Skill Gap</th><th>Severity</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def role_card_html(requirements: dict) -> str:
    title = requirements.get("title", "")
    company = requirements.get("company", "")
    yoe = requirements.get("years_experience")
    lines = []
    if title:
        lines.append(f'<div style="font-size:1.05rem;font-weight:700;color:#1e293b">{title}</div>')
    if company and company.lower() not in ("", "unknown"):
        lines.append(f'<div style="color:#64748b;margin-top:2px">at <strong>{company}</strong></div>')
    if yoe:
        lines.append(
            f'<div style="margin-top:8px;font-size:0.85rem;color:#94a3b8">'
            f'🗓 {yoe}+ years experience required</div>'
        )
    inner = "\n".join(lines) if lines else '<span style="color:#94a3b8">Details parsed from job text</span>'
    return f'<div class="ha-card"><div class="ha-card-title">Role</div>{inner}</div>'


def extract_job_meta(state: dict) -> tuple[str, str]:
    req = state.get("requirements", {})
    company = req.get("company", "Unknown") or "Unknown"
    title = req.get("title", "") or ""
    if not title:
        first_line = state.get("job_text", "").strip().splitlines()[0][:60] if state.get("job_text") else ""
        title = first_line + ("…" if len(first_line) == 60 else "")
    return company, title


def _gap_icon(severity: str) -> str:
    return {"critical": "🔴", "partial": "🟡"}.get(severity.lower().replace(" ", "-"), "⚪")


# ── Session state ─────────────────────────────────────────────────────────────

def _chroma_exists() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

if "resume_ingested" not in st.session_state:
    st.session_state.resume_ingested = _chroma_exists()
if "resume_chunk_count" not in st.session_state:
    st.session_state.resume_chunk_count = 0
if "ingested_filename" not in st.session_state:
    st.session_state.ingested_filename = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "result_cached" not in st.session_state:
    st.session_state.result_cached = False
if "history" not in st.session_state:
    st.session_state.history = load_history()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-brand">🎯 HireAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">AI-powered job fit analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        '<span style="font-size:.85rem;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:.06em">Resume</span>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDF is chunked, embedded with all-MiniLM-L6-v2, and stored in a local ChromaDB collection.",
    )

    chunks_missing = st.session_state.resume_chunk_count == 0
    if uploaded is not None or chunks_missing:
        btn_label = "Re-ingest Resume" if st.session_state.resume_ingested else "⬆ Ingest Resume"
        btn_disabled = uploaded is None
        if st.button(
            btn_label,
            use_container_width=True,
            type="primary",
            disabled=btn_disabled,
            help="Upload a resume PDF above first" if btn_disabled else None,
        ):
            with st.spinner("Chunking & embedding…"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    chunks = load_and_chunk_pdf(tmp_path)
                    ingest_resume(tmp_path)
                    st.session_state.resume_ingested = True
                    st.session_state.resume_chunk_count = len(chunks)
                    st.session_state.ingested_filename = uploaded.name
                    st.success("Resume ingested!", icon="✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    st.markdown("---")

    if st.session_state.resume_ingested:
        fname = st.session_state.ingested_filename or "resume.pdf"
        n_chunks = st.session_state.resume_chunk_count
        st.markdown(
            f"""
            <div class="chip chip-green">✓ Resume active</div><br>
            <div style="margin-top:8px">
              <span class="chip chip-blue">📄 {fname}</span><br>
              <span class="chip chip-blue">📊 {n_chunks} chunks</span><br>
              <span class="chip chip-blue">🤖 all-MiniLM-L6-v2</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#64748b;font-size:.875rem;padding:4px 0">'
            "Upload a resume PDF to get started.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
            1. **Upload** your resume — it's chunked and embedded locally.\n
            2. **Paste** a job description and hit Analyze.\n
            3. Claude **scores** your fit, maps **skill gaps**, and drafts a **cover letter** for strong matches (score ≥ 6).
            """,
            unsafe_allow_html=True,
        )


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown('<h1 class="ha-title">HireAgent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="ha-subtitle">Paste a job description · get your match score, gap analysis, and a tailored cover letter.</p>',
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_analyze, tab_history = st.tabs(["🔍  Analyze Job", "📋  History"])


# ═══════════════════════ Tab 1 — Analyze Job ══════════════════════════════════

with tab_analyze:
    job_text = st.text_area(
        "Job Description",
        height=220,
        placeholder=(
            "Paste the full job posting here — title, responsibilities, required skills, etc.\n\n"
            "The more detail you provide, the more accurate the analysis."
        ),
        label_visibility="collapsed",
    )

    col_btn, col_warn = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button(
            "Analyze →",
            type="primary",
            use_container_width=True,
            disabled=not job_text.strip(),
        )
    with col_warn:
        if not st.session_state.resume_ingested:
            st.warning("Upload and ingest a resume in the sidebar first.", icon="⬅")

    if analyze_clicked and job_text.strip():
        if not st.session_state.resume_ingested:
            st.error("No resume ingested. Upload a PDF in the sidebar and click Ingest Resume.")
        else:
            with st.spinner("Running analysis… this takes 20–40 s while Claude processes the job."):
                try:
                    result = run_analysis(job_text)
                    st.session_state.analysis_result = result
                    st.session_state.result_cached = result.get("_cached", False)

                    if not result.get("error"):
                        company, title = extract_job_meta(result)
                        entry = {
                            "id": datetime.now().isoformat(),
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "company": company,
                            "title": title,
                            "score": result["match_score"],
                            "analysis": {k: result[k] for k in (
                                "job_text", "requirements", "match_score",
                                "matched_skills", "gaps", "cover_letter", "error",
                            )},
                        }
                        st.session_state.history.insert(0, entry)
                        save_history(st.session_state.history)

                except Exception as exc:
                    st.error(f"Analysis pipeline failed: {exc}")
                    st.session_state.analysis_result = None

    # ── Results ──────────────────────────────────────────────────────────────
    result = st.session_state.analysis_result
    if result is not None:
        if result.get("error"):
            st.error(f"**Analysis error:** {result['error']}")
        else:
            st.divider()
            if st.session_state.result_cached:
                st.info(
                    "Served from cache — same resume and job description as a previous run. "
                    "No API calls were made.",
                    icon="⚡",
                )
            score = result["match_score"]

            # Row 1: Score gauge + Matched skills + Role info
            col_left, col_right = st.columns([1, 2], gap="large")

            with col_left:
                st.markdown(
                    f'<div class="ha-card"><div class="ha-card-title">Match Score</div>'
                    f'{gauge_html(score)}</div>',
                    unsafe_allow_html=True,
                )

            with col_right:
                st.markdown(
                    f'<div class="ha-card"><div class="ha-card-title">Matched Skills</div>'
                    f'{skill_badges_html(result["matched_skills"])}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(role_card_html(result.get("requirements", {})), unsafe_allow_html=True)

            # Row 2: Skill Gaps
            gaps = result.get("gaps", [])
            if gaps:
                st.markdown(
                    f'<div class="ha-card"><div class="ha-card-title">Skill Gaps</div>'
                    f'{gaps_table_html(gaps)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div style="font-size:.875rem;font-weight:600;color:#475569;'
                    'margin:.5rem 0 .25rem">Remediation suggestions</div>',
                    unsafe_allow_html=True,
                )
                for gap in gaps:
                    sev = gap.get("severity", "nice-to-have").lower().replace(" ", "-")
                    icon = _gap_icon(sev)
                    with st.expander(f"{icon}  {gap['skill']}"):
                        st.write(gap.get("suggestion", "No suggestion available."))
            else:
                st.markdown(
                    '<div class="ha-card"><div class="ha-card-title">Skill Gaps</div>'
                    '<span style="color:#16a34a;font-weight:600">✓ No significant gaps — strong match!</span></div>',
                    unsafe_allow_html=True,
                )

            # Row 3: Cover Letter
            st.markdown('<div style="margin-top:6px"></div>', unsafe_allow_html=True)
            cover = result.get("cover_letter", "")
            if cover:
                st.markdown(
                    '<div class="ha-card-title" style="margin-top:8px">Cover Letter</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Generated because your score is ≥ 6 — copy, personalize, then send.")
                st.text_area(
                    "cover_letter_out",
                    value=cover,
                    height=340,
                    label_visibility="collapsed",
                    key="cl_analyze",
                )
            else:
                st.info(
                    "Cover letter is only generated for matches with score ≥ 6. "
                    "Address the skill gaps above to improve your fit.",
                    icon="✍️",
                )


# ═══════════════════════ Tab 2 — History ══════════════════════════════════════

with tab_history:
    history = st.session_state.history

    if not history:
        st.info(
            "No analyses yet. Run your first job analysis in the Analyze tab.",
            icon="📭",
        )
    else:
        df = pd.DataFrame([
            {
                "Date": h["date"],
                "Company": h["company"],
                "Role": h["title"],
                "Score": round(float(h["score"]), 1),
            }
            for h in history
        ])

        def _score_style(val: float) -> str:
            if val < 4:
                return "background:#fee2e2;color:#dc2626;font-weight:700"
            if val < 7:
                return "background:#fef9c3;color:#a16207;font-weight:700"
            return "background:#dcfce7;color:#15803d;font-weight:700"

        try:
            styled = df.style.format({"Score": "{:.1f}"}).map(_score_style, subset=["Score"])
        except AttributeError:
            styled = df.style.format({"Score": "{:.1f}"}).applymap(_score_style, subset=["Score"])

        st.markdown(
            '<div style="font-size:.875rem;color:#64748b;margin-bottom:.5rem">'
            "Select a row to see the full analysis.</div>",
            unsafe_allow_html=True,
        )

        try:
            event = st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
            )
            selected_rows = event.selection.rows if hasattr(event, "selection") else []
        except Exception:
            st.dataframe(styled, use_container_width=True, hide_index=True)
            selected_rows = []
            st.caption("(Row selection requires Streamlit ≥ 1.35 — upgrade with `pip install -U streamlit`.)")

        if selected_rows:
            idx = selected_rows[0]
            entry = history[idx]
            h_result = entry["analysis"]

            st.divider()
            h_col_left, h_col_right = st.columns([1, 2], gap="large")

            with h_col_left:
                company_label = entry["company"] if entry["company"].lower() != "unknown" else ""
                header = entry["title"] or company_label or "Analysis"
                st.markdown(
                    f'<div class="ha-card">'
                    f'<div class="ha-card-title">{entry["date"]}</div>'
                    f'<div style="font-weight:700;color:#1e293b;font-size:1rem;margin-bottom:4px">{header}</div>'
                    f'{gauge_html(h_result["match_score"])}</div>',
                    unsafe_allow_html=True,
                )

            with h_col_right:
                st.markdown(
                    f'<div class="ha-card"><div class="ha-card-title">Matched Skills</div>'
                    f'{skill_badges_html(h_result["matched_skills"])}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(role_card_html(h_result.get("requirements", {})), unsafe_allow_html=True)

            h_gaps = h_result.get("gaps", [])
            if h_gaps:
                st.markdown(
                    f'<div class="ha-card"><div class="ha-card-title">Skill Gaps</div>'
                    f'{gaps_table_html(h_gaps)}</div>',
                    unsafe_allow_html=True,
                )
                for gap in h_gaps:
                    sev = gap.get("severity", "nice-to-have").lower().replace(" ", "-")
                    with st.expander(f"{_gap_icon(sev)}  {gap['skill']}"):
                        st.write(gap.get("suggestion", ""))

            h_cover = h_result.get("cover_letter", "")
            if h_cover:
                st.markdown(
                    '<div class="ha-card-title" style="margin-top:8px">Cover Letter</div>',
                    unsafe_allow_html=True,
                )
                st.text_area(
                    "hist_cover_letter",
                    value=h_cover,
                    height=300,
                    label_visibility="collapsed",
                    key="cl_history",
                )


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="ha-footer">'
    "Built with LangChain + LangGraph + ChromaDB&nbsp;&nbsp;|&nbsp;&nbsp;Powered by Claude"
    "</div>",
    unsafe_allow_html=True,
)
