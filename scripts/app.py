"""
Resume Ranker & Feedback System
================================
Multi-page Streamlit app combining BERT (BGE) cosine similarity with
Llama 3.1 reasoning to rank and evaluate resumes against multiple JDs.

Run:  streamlit run scripts/app.py
"""

import streamlit as st
import pandas as pd
import json
import time
import base64
from io import BytesIO

from pdf_processor import process_pdf, set_contact_model
from embedding_engine import batch_analyze, multi_jd_analyze, get_device, MODEL_NAME
from llama_ranker import rank_resumes, get_detailed_feedback


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Ranker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Consistent color palette (Streamlit-friendly, no dark-on-dark clashes)
# ---------------------------------------------------------------------------
# Primary:    #0f62fe  (blue-600)
# Surface:    #ffffff  (white)
# Header bg:  #f0f4ff  (blue-50)
# Header txt: #1e293b  (slate-800)
# Border:     #e2e8f0  (slate-200)
# Muted text: #64748b  (slate-500)
# Hover row:  #f8fafc  (slate-50)
# Badges:     green/blue/amber/red tints

st.markdown("""
<style>
    .block-container { max-width: 1200px; padding-top: 2rem; }

    /* ---- Candidate table ---- */
    .candidate-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        font-size: 0.92rem;
        border: 1px solid #e2e8f0;
    }
    .candidate-table thead th {
        background: #f0f4ff;
        color: #1e293b;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
        white-space: nowrap;
        border-bottom: 2px solid #c7d2fe;
    }
    .candidate-table tbody td {
        padding: 12px 16px;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: middle;
        color: #334155;
    }
    .candidate-table tbody tr:hover {
        background: #f8fafc;
    }
    .candidate-table tbody tr:last-child td {
        border-bottom: none;
    }

    /* Rank column highlight for top-3 */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px; height: 30px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .rank-1 { background: #fef3c7; color: #92400e; }
    .rank-2 { background: #e0e7ff; color: #3730a3; }
    .rank-3 { background: #ffe4e6; color: #9f1239; }
    .rank-other { background: #f1f5f9; color: #475569; }

    /* Recommendation badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-strong { background: #dcfce7; color: #166534; }
    .badge-good   { background: #dbeafe; color: #1e40af; }
    .badge-partial{ background: #fef3c7; color: #92400e; }
    .badge-weak   { background: #fee2e2; color: #991b1b; }

    /* Job-fit chip */
    .job-chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #bfdbfe;
    }

    /* Score bar */
    .score-bar-bg {
        background: #e2e8f0;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    /* Summary stat cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stat-card h2 {
        margin: 0; font-size: 2rem; color: #0f62fe;
    }
    .stat-card p {
        margin: 4px 0 0; color: #64748b; font-size: 0.85rem;
    }

    /* Contact line in table */
    .contact-line {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 2px;
    }
    .contact-line a {
        color: #64748b;
        text-decoration: none;
    }
    .contact-line a:hover {
        color: #0f62fe;
        text-decoration: underline;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key in [
    "jds_data", "resumes_data", "multi_results", "llm_result",
    "analysis_done", "saved_jds", "resume_pdfs",
]:
    if key not in st.session_state:
        if key in ("saved_jds",):
            st.session_state[key] = []
        elif key == "resume_pdfs":
            st.session_state[key] = {}
        elif key == "analysis_done":
            st.session_state[key] = False
        else:
            st.session_state[key] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_rank_html(rank) -> str:
    r = int(rank) if isinstance(rank, (int, float)) else 0
    cls = f"rank-{r}" if r in (1, 2, 3) else "rank-other"
    return f'<span class="rank-badge {cls}">{rank}</span>'


def get_badge_html(rec: str) -> str:
    rec_lower = rec.lower().replace(" ", "")
    if "strong" in rec_lower:
        return f'<span class="badge badge-strong">{rec}</span>'
    elif "good" in rec_lower:
        return f'<span class="badge badge-good">{rec}</span>'
    elif "partial" in rec_lower:
        return f'<span class="badge badge-partial">{rec}</span>'
    elif rec in ("N/A", "", "n/a"):
        # Should not appear -- signals a matching bug
        return '<span class="badge badge-weak">PENDING</span>'
    else:
        return f'<span class="badge badge-weak">{rec}</span>'


def score_to_recommendation(pct: float) -> str:
    """Generate a recommendation label from a match percentage."""
    if pct >= 70:
        return "STRONG MATCH"
    elif pct >= 50:
        return "GOOD MATCH"
    elif pct >= 30:
        return "PARTIAL MATCH"
    else:
        return "WEAK MATCH"


def get_score_bar_html(score: float) -> str:
    if score >= 70:
        color = "#16a34a"
    elif score >= 50:
        color = "#2563eb"
    elif score >= 35:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f'''
    <div style="display:flex;align-items:center;gap:8px;">
        <div class="score-bar-bg" style="flex:1;">
            <div class="score-bar-fill" style="width:{min(score,100):.0f}%;background:{color};"></div>
        </div>
        <span style="font-weight:600;font-size:0.88rem;min-width:45px;color:#334155;">{score:.1f}%</span>
    </div>
    '''


def safe_float(val, default: float = 0.0) -> float:
    """Convert any value to a float safely. Handles strings like '72%', None, 'N/A'."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.strip().rstrip("%").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return default
    return default


def resolve_job(llm_job: str | None, bert_job: str | None) -> str:
    """
    Return the best available job name for a candidate.
    Priority: LLM assignment > BERT best-fit > 'Unassigned'.
    Rejects garbage values like 'N/A', empty, None, or 'null'.
    """
    _invalid = {None, "", "N/A", "n/a", "null", "None", "Unassigned", "unknown", "Unknown"}
    if llm_job and llm_job.strip() not in _invalid:
        return llm_job.strip()
    if bert_job and bert_job.strip() not in _invalid:
        return bert_job.strip()
    return "Unassigned"


def deduplicate_candidates(table_rows: list) -> list:
    """
    Ensure each candidate appears exactly once.
    Deduplicates by BOTH filename AND candidate name (to catch cases where
    the LLM creates separate entries for the same person).
    Keeps the entry with the highest overall_pct (best job fit).
    """
    # First pass: group by filename
    by_file: dict[str, list] = {}
    for row in table_rows:
        fname = row["filename"]
        by_file.setdefault(fname, []).append(row)

    # Second pass: also group by normalized name to catch same-name dupes
    # (in case LLM generated multiple entries for the same person)
    by_name: dict[str, list] = {}
    for row in table_rows:
        norm_name = row["name"].strip().lower()
        if norm_name and norm_name not in ("unknown", "n/a"):
            by_name.setdefault(norm_name, []).append(row)

    # Build dedup map: for each filename, pick the row with highest overall_pct
    best_per_file: dict[str, dict] = {}
    for fname, rows in by_file.items():
        best_per_file[fname] = max(rows, key=lambda r: r["overall_pct"])

    # Also check name-based groups -- if multiple filenames share a name,
    # keep only the best scoring one
    used_names: set[str] = set()
    used_files: set[str] = set()
    result = []

    # Sort by overall_pct descending so we always pick the best version first
    sorted_best = sorted(best_per_file.values(), key=lambda r: r["overall_pct"], reverse=True)

    for row in sorted_best:
        norm_name = row["name"].strip().lower()
        fname = row["filename"]

        # Skip if we already have this person (by name or filename)
        if fname in used_files:
            continue
        if norm_name and norm_name not in ("unknown", "n/a") and norm_name in used_names:
            continue

        result.append(row)
        used_files.add(fname)
        if norm_name and norm_name not in ("unknown", "n/a"):
            used_names.add(norm_name)

    # Re-rank sequentially after dedup
    for i, row in enumerate(result, 1):
        row["rank"] = i

    return result


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Resume Ranker")
    st.caption(f"Embedding: {MODEL_NAME}")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Detailed Review"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    llama_model = st.selectbox(
        "Llama Model (Ollama)",
        ["llama3.1", "llama3.1:8b", "llama3.1:70b", "llama3.2", "llama3.2:3b"],
        index=0,
    )

    # Sync the contact extraction model with the selected Llama model
    set_contact_model(llama_model)

    st.divider()

    # Device status
    device = get_device()
    if device == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"GPU: {gpu_name}")
    else:
        st.warning(
            "Running on CPU\n\n"
            "```\npip install --pre torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu128\n```"
        )

    st.divider()
    st.caption("100% local. No data leaves your machine.")


# =========================================================================
# PAGE 1: DASHBOARD
# =========================================================================
if page == "Dashboard":

    st.header("Upload & Analyze")

    # ----- JOB DESCRIPTIONS -----
    st.subheader("Job Descriptions")
    st.caption("Upload one or more JD PDFs. They persist for this session so you can reuse them across runs.")

    col_jd_upload, col_jd_saved = st.columns([1, 1], gap="large")

    with col_jd_upload:
        jd_files = st.file_uploader(
            "Upload JD PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="jd_upload",
        )
        if jd_files and st.button("Save JDs", type="secondary"):
            new_jds = []
            for jf in jd_files:
                jd_data = process_pdf(BytesIO(jf.read()))
                jf.seek(0)
                jd_entry = {
                    "name": jf.name.replace(".pdf", "").replace(".PDF", ""),
                    "text": jd_data.get("cleaned_text", ""),
                    "word_count": jd_data.get("word_count", 0),
                    "sections": jd_data.get("sections", {}),
                }
                existing_names = [j["name"] for j in st.session_state["saved_jds"]]
                if jd_entry["name"] not in existing_names:
                    new_jds.append(jd_entry)
            st.session_state["saved_jds"].extend(new_jds)
            if new_jds:
                st.success(f"Saved {len(new_jds)} new JD(s).")
            else:
                st.info("All JDs already saved.")

    with col_jd_saved:
        st.markdown("**Saved Job Descriptions**")
        if st.session_state["saved_jds"]:
            for i, jd in enumerate(st.session_state["saved_jds"]):
                st.markdown(f"- **{jd['name']}** ({jd['word_count']} words)")
            if st.button("Clear all saved JDs", type="secondary"):
                st.session_state["saved_jds"] = []
                st.rerun()
        else:
            st.info("No JDs saved yet. Upload and save JDs on the left.")

    st.divider()

    # ----- RESUMES -----
    st.subheader("Resumes")
    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_upload",
    )
    if resume_files:
        st.success(f"{len(resume_files)} resume(s) uploaded")

    st.divider()

    # ----- ANALYSIS CONTROLS -----
    can_run = len(st.session_state.get("saved_jds", [])) > 0 and len(resume_files or []) > 0

    run_clicked = st.button(
        "Analyze & Rank",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    # ----- RUN ANALYSIS -----
    if run_clicked:
        saved_jds = st.session_state["saved_jds"]

        # Step 1: Extract resume text
        with st.status("Extracting text from resumes (Tesseract + AI contact extraction)...", expanded=True) as status:
            t0 = time.time()
            resumes_data = []
            resume_pdfs = {}
            for rf in resume_files:
                raw_bytes = rf.read()
                rf.seek(0)
                resume_pdfs[rf.name] = base64.b64encode(raw_bytes).decode("utf-8")
                rdata = process_pdf(BytesIO(raw_bytes))
                rdata["filename"] = rf.name
                resumes_data.append(rdata)
                if rdata.get("error"):
                    st.warning(f"{rf.name}: extraction issue -- {rdata['error']}")
                st.write(
                    f"{rf.name}: {rdata.get('word_count', 0)} words | "
                    f"{rdata.get('name', 'Unknown')} | "
                    f"{rdata.get('email', 'N/A')} | "
                    f"{rdata.get('phone', 'N/A')}"
                )
            st.session_state["resumes_data"] = resumes_data
            st.session_state["resume_pdfs"] = resume_pdfs
            status.update(label=f"Text extracted ({time.time()-t0:.1f}s)", state="complete")

        # Step 2: Multi-JD BERT analysis
        with st.status("Computing BERT similarity (BGE) across all JDs...", expanded=True) as status:
            t0 = time.time()
            resume_dicts = [
                {
                    "filename": r.get("filename", "unknown"),
                    "sections": r.get("sections", {"full_text": r.get("cleaned_text", "")}),
                    "word_count": r.get("word_count", 0),
                    "name": r.get("name", "Unknown"),
                    "email": r.get("email", "N/A"),
                    "phone": r.get("phone", "N/A"),
                    "experience": r.get("experience", "N/A"),
                }
                for r in resumes_data
            ]
            jd_inputs = [{"name": jd["name"], "text": jd["text"]} for jd in saved_jds]
            multi_results = multi_jd_analyze(resume_dicts, jd_inputs)
            st.session_state["multi_results"] = multi_results
            status.update(label=f"BERT analysis done ({time.time()-t0:.1f}s)", state="complete")

        # Step 3: Llama ranking
        with st.status("Llama is analyzing and ranking resumes...", expanded=True) as status:
            st.write("This may take a minute depending on model size and number of resumes.")
            t0 = time.time()
            resume_for_llm = [
                {
                    "filename": r.get("filename", "unknown"),
                    "cleaned_text": r.get("cleaned_text", ""),
                    "name": r.get("name", "Unknown"),
                    "email": r.get("email", "N/A"),
                    "phone": r.get("phone", "N/A"),
                    "experience": r.get("experience", "N/A"),
                }
                for r in resumes_data
            ]
            llm_result = rank_resumes(
                resumes=resume_for_llm,
                model=llama_model,
                jds=jd_inputs,
                best_fit=multi_results["best_fit"],
                per_jd_scores=multi_results["per_jd"],
            )
            st.session_state["llm_result"] = llm_result
            st.session_state["llama_model"] = llama_model
            st.session_state["analysis_done"] = True
            status.update(label=f"Llama ranking done ({time.time()-t0:.1f}s)", state="complete")

    # =====================================================================
    # RESULTS DISPLAY
    # =====================================================================
    if st.session_state.get("analysis_done") and st.session_state.get("llm_result"):
        multi_results = st.session_state["multi_results"]
        llm_result = st.session_state["llm_result"]
        resumes_data = st.session_state["resumes_data"]
        resume_pdfs = st.session_state.get("resume_pdfs", {})
        best_fit = multi_results.get("best_fit", {})

        if "error" in llm_result:
            st.error(f"LLM Error: {llm_result['error']}")
            if llm_result.get("raw_response"):
                with st.expander("Raw LLM response (for debugging)"):
                    st.code(llm_result["raw_response"], language="json")
            st.warning(
                "Tip: Llama sometimes produces malformed JSON, especially with many resumes. "
                "Try running the analysis again, or switch to a smaller model (llama3.2:3b) "
                "which tends to produce cleaner structured output."
            )
            # Even if LLM failed, show BERT-only results as fallback
            st.info("Showing BERT-only results below (no LLM reasoning).")
            llm_ranking = []
        else:
            llm_ranking = llm_result.get("final_ranking", [])

        # ---- Match LLM entries to resumes ----
        # The LLM may use different filenames, candidate names, or indices.
        # We use a scoring approach: for each uploaded resume, score every LLM
        # entry and pick the best match (above a minimum threshold).

        def _norm(s: str) -> str:
            """Lowercase, strip, remove extension."""
            s = s.lower().strip()
            for ext in (".pdf", ".docx", ".doc", ".txt"):
                if s.endswith(ext):
                    s = s[: -len(ext)].strip()
            return s

        def _match_score(resume_fname: str, resume_name: str, llm_entry: dict) -> int:
            """Return a match score (higher = better). 0 = no match."""
            score = 0
            llm_fname = _norm(llm_entry.get("filename", ""))
            llm_name = llm_entry.get("name", "").lower().strip()
            r_fname = _norm(resume_fname)
            r_name = resume_name.lower().strip()

            # Exact filename match
            if r_fname and llm_fname and r_fname == llm_fname:
                score += 100
            # Partial filename overlap
            elif r_fname and llm_fname:
                if r_fname in llm_fname or llm_fname in r_fname:
                    score += 60
            # Candidate name match
            if r_name and llm_name and r_name not in ("unknown", "n/a"):
                if r_name == llm_name:
                    score += 80
                elif r_name in llm_name or llm_name in r_name:
                    score += 40
            # LLM filename matches resume name
            if r_name and llm_fname and r_name in llm_fname:
                score += 30
            # Resume filename matches LLM name
            if r_fname and llm_name and llm_name in r_fname:
                score += 30
            return score

        # Build best match for each resume
        used_llm_indices: set[int] = set()
        resume_to_llm: dict[str, dict] = {}

        for r_data in resumes_data:
            fname = r_data.get("filename", "unknown")
            cand_name = r_data.get("name", "Unknown")
            best_score = 0
            best_idx = -1

            for idx, entry in enumerate(llm_ranking):
                if idx in used_llm_indices:
                    continue
                ms = _match_score(fname, cand_name, entry)
                if ms > best_score:
                    best_score = ms
                    best_idx = idx

            if best_score >= 30 and best_idx >= 0:
                resume_to_llm[fname] = llm_ranking[best_idx]
                used_llm_indices.add(best_idx)

        # ---- Build candidate rows ----
        all_table_rows = []

        for r_data in resumes_data:
            fname = r_data.get("filename", "unknown")
            bf = best_fit.get(fname, {})
            # Also try fuzzy BERT lookup
            if not bf:
                for key in best_fit:
                    if _norm(key) == _norm(fname) or _norm(fname) in _norm(key) or _norm(key) in _norm(fname):
                        bf = best_fit[key]
                        break
            llm_entry = resume_to_llm.get(fname, {})

            # -- Scores --
            bert_weighted = safe_float(bf.get("score", bf.get("weighted_score", bf.get("raw_cosine_score", 0))))
            bert_skills = safe_float(bf.get("skills_score", 0))
            bert_exp = safe_float(bf.get("experience_score", 0))

            skills_pct = bert_skills if bert_skills > 0 else bert_weighted
            exp_pct = bert_exp if bert_exp > 0 else bert_weighted
            overall_pct = safe_float(llm_entry.get("overall_match_pct", 0))
            if overall_pct <= 0:
                overall_pct = bert_weighted

            # Job assignment
            candidate_job = resolve_job(
                llm_entry.get("best_fit_job"),
                bf.get("jd"),
            )

            # Recommendation: use LLM if available, otherwise derive from score
            llm_rec = llm_entry.get("recommendation", "")
            if not llm_rec or llm_rec.strip().upper() in ("N/A", "NULL", "NONE", "UNKNOWN", ""):
                rec = score_to_recommendation(overall_pct)
            else:
                rec = llm_rec.strip()

            all_table_rows.append({
                "rank": llm_entry.get("rank", 0),
                "name": llm_entry.get("name") or r_data.get("name", "Unknown"),
                "email": r_data.get("email", "N/A"),
                "phone": r_data.get("phone", "N/A"),
                "filename": fname,
                "experience": r_data.get("experience", "N/A"),
                "best_fit_job": candidate_job,
                "overall_pct": overall_pct,
                "skills_pct": skills_pct,
                "exp_pct": exp_pct,
                "recommendation": rec,
                "feedback": llm_entry.get("feedback", ""),
                "has_pdf": fname in resume_pdfs,
            })

        # Deduplicate (one entry per person)
        all_table_rows = deduplicate_candidates(all_table_rows)

        # Debug info (collapsed by default)
        with st.expander("Debug: raw data (expand to troubleshoot)", expanded=False):
            st.write(f"**Uploaded resumes:** {[r.get('filename') for r in resumes_data]}")
            st.write(f"**LLM returned** {len(llm_ranking)} ranking entries")
            st.write(f"**LLM filenames:** {[e.get('filename') for e in llm_ranking]}")
            st.write(f"**LLM names:** {[e.get('name') for e in llm_ranking]}")
            st.write(f"**BERT best_fit keys:** {list(best_fit.keys())}")
            st.write(f"**Matched LLM entries:** {len(resume_to_llm)}/{len(resumes_data)} resumes")
            st.write(f"**Rows after dedup:** {len(all_table_rows)}")
            for row in all_table_rows:
                matched = "LLM" if row["filename"] in resume_to_llm else "BERT-only"
                st.write(f"  {row['filename']}: [{matched}] job={row['best_fit_job']}, "
                         f"overall={row['overall_pct']:.1f}%, skills={row['skills_pct']:.1f}%, "
                         f"exp={row['exp_pct']:.1f}%, status=**{row['recommendation']}**")

        # ----- FILTER CONTROLS -----
        st.divider()
        st.subheader("Filters")
        st.caption("Adjust thresholds or filter by job to narrow down candidates.")

        # Job filter -- collect from the rows we just built
        all_jobs = sorted(set(
            row["best_fit_job"]
            for row in all_table_rows
            if row["best_fit_job"] not in ("N/A", "Unassigned", "")
        ))
        has_unassigned = any(row["best_fit_job"] == "Unassigned" for row in all_table_rows)
        all_jobs_options = ["All Jobs"] + all_jobs
        if has_unassigned:
            all_jobs_options.append("Unassigned")

        fc0, fc1, fc2, fc3 = st.columns(4)
        with fc0:
            job_filter = st.selectbox(
                "Filter by Job",
                all_jobs_options,
                index=0,
                help="Show only candidates matched to this job role.",
            )
        with fc1:
            min_skills = st.slider("Min Skills %", 0, 100, 0, step=5)
        with fc2:
            min_experience = st.slider("Min Experience %", 0, 100, 0, step=5)
        with fc3:
            min_overall = st.slider("Min Overall Match %", 0, 100, 0, step=5)

        # ------ Apply filters ------
        table_rows = []
        for row in all_table_rows:
            # Job filter: "All Jobs" passes everything
            if job_filter != "All Jobs" and row["best_fit_job"] != job_filter:
                continue
            # Score filters (safe_float already ensured these are floats)
            if row["skills_pct"] < min_skills:
                continue
            if row["exp_pct"] < min_experience:
                continue
            if row["overall_pct"] < min_overall:
                continue
            table_rows.append(row)

        # Apply top N from session (set during analysis) or show all
        result_top_n = st.session_state.get("result_top_n", len(table_rows))
        if len(table_rows) > 1:
            result_top_n = st.slider(
                "Show top N candidates",
                min_value=1,
                max_value=len(table_rows),
                value=min(result_top_n, len(table_rows)),
                key="result_top_n_slider",
            )
            st.session_state["result_top_n"] = result_top_n
        table_rows = table_rows[:result_top_n]

        # ----- SUMMARY STATS -----
        st.divider()
        total_analyzed = len(all_table_rows)
        showing = len(table_rows)
        strong = sum(1 for r in table_rows if "strong" in r["recommendation"].lower())
        good = sum(1 for r in table_rows if "good" in r["recommendation"].lower())

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f'''<div class="stat-card"><h2>{total_analyzed}</h2><p>Resumes Analyzed</p></div>''', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'''<div class="stat-card"><h2>{showing}</h2><p>Showing (filtered)</p></div>''', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'''<div class="stat-card"><h2>{strong}</h2><p>Strong Matches</p></div>''', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'''<div class="stat-card"><h2>{good}</h2><p>Good Matches</p></div>''', unsafe_allow_html=True)

        # ----- CANDIDATE TABLE -----
        st.divider()
        st.subheader("Top Candidates")

        if table_rows:
            rows_html = ""
            for r in table_rows:
                badge = get_badge_html(r["recommendation"])
                score_bar = get_score_bar_html(r["overall_pct"])
                rank_html = get_rank_html(r["rank"])

                # Candidate name -- hyperlinked to PDF download if available
                if r["has_pdf"]:
                    b64 = resume_pdfs.get(r["filename"], "")
                    name_html = (
                        f'<a href="data:application/pdf;base64,{b64}" '
                        f'download="{r["filename"]}" '
                        f'style="color:#0f62fe;text-decoration:none;font-weight:600;" '
                        f'title="Click to download resume PDF">'
                        f'{r["name"]}</a>'
                    )
                else:
                    name_html = f'<span style="font-weight:600;color:#1e293b;">{r["name"]}</span>'

                # Contact sub-line: email and phone
                contact_parts = []
                if r["email"] != "N/A":
                    contact_parts.append(
                        f'<a href="mailto:{r["email"]}">{r["email"]}</a>'
                    )
                if r["phone"] != "N/A":
                    contact_parts.append(
                        f'<a href="tel:{r["phone"]}">{r["phone"]}</a>'
                    )
                contact_html = (
                    " &middot; ".join(contact_parts)
                    if contact_parts
                    else '<span style="color:#94a3b8;">No contact info</span>'
                )

                rows_html += f"""
                <tr>
                    <td style="text-align:center;">{rank_html}</td>
                    <td>
                        <div>{name_html}</div>
                        <div class="contact-line">{contact_html}</div>
                    </td>
                    <td style="white-space:nowrap;">{r['experience']}</td>
                    <td><span class="job-chip">{r['best_fit_job']}</span></td>
                    <td style="min-width:180px;">{score_bar}</td>
                    <td style="text-align:center;font-weight:500;">{r['skills_pct']:.0f}%</td>
                    <td style="text-align:center;font-weight:500;">{r['exp_pct']:.0f}%</td>
                    <td>{badge}</td>
                </tr>"""

            table_html = f"""
            <table class="candidate-table">
                <thead>
                    <tr>
                        <th style="width:50px;text-align:center;">Rank</th>
                        <th style="min-width:220px;">Candidate</th>
                        <th>Exp.</th>
                        <th>Best Fit Job</th>
                        <th style="min-width:180px;">Overall Match</th>
                        <th style="text-align:center;">Skills</th>
                        <th style="text-align:center;">Exp %</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("No candidates match the current filter criteria. Try lowering the thresholds or selecting 'All Jobs'.")

        # ----- QUICK FEEDBACK (collapsed) -----
        st.divider()
        st.subheader("Quick Feedback")
        st.caption("Expand any candidate to see Llama's brief assessment. For full analysis, go to Detailed Review.")

        for r in table_rows:
            if r["feedback"]:
                with st.expander(f"{r['name']} -- {r['recommendation']}"):
                    st.markdown(r["feedback"])

        # ----- SUMMARY -----
        summary = llm_result.get("summary", "")
        if summary:
            st.divider()
            st.subheader("Hiring Summary")
            st.markdown(summary)


# =========================================================================
# PAGE 2: DETAILED REVIEW
# =========================================================================
elif page == "Detailed Review":

    if not st.session_state.get("analysis_done"):
        st.info("Run an analysis from the Dashboard first to see detailed results here.")
        st.stop()

    multi_results = st.session_state["multi_results"]
    llm_result = st.session_state["llm_result"]
    resumes_data = st.session_state["resumes_data"]
    saved_jds = st.session_state["saved_jds"]

    if "error" in llm_result:
        st.error(f"LLM Error: {llm_result['error']}")
        st.stop()

    ranking = llm_result.get("final_ranking", [])
    best_fit = multi_results.get("best_fit", {})

    st.header("Detailed Review")
    st.caption("In-depth analysis: section-level BERT scores, head-to-head comparisons, and per-candidate deep-dive feedback.")

    # ==================================================================
    # Section-level BERT scores per JD
    # ==================================================================
    st.subheader("Section-Level BERT Scores")

    jd_tab_names = [jd["name"] for jd in saved_jds]
    if jd_tab_names:
        jd_tabs = st.tabs(jd_tab_names)
        for jd_idx, jd_tab in enumerate(jd_tabs):
            with jd_tab:
                jd_name = jd_tab_names[jd_idx]
                jd_results = multi_results["per_jd"].get(jd_name, [])

                for sr in jd_results:
                    with st.expander(f"{sr['filename']} -- Weighted: {sr['weighted_score']}% | Raw: {sr['raw_cosine_score']}%"):
                        if sr.get("section_scores"):
                            for sec, score in sorted(sr["section_scores"].items(), key=lambda x: -x[1]):
                                col_lbl, col_bar = st.columns([1, 3])
                                with col_lbl:
                                    st.markdown(f"**{sec}**")
                                with col_bar:
                                    st.progress(min(score / 100.0, 1.0))
                                    st.caption(f"{score:.1f}%")
                        else:
                            st.info("No section-level scores available for this resume.")

    st.divider()

    # ==================================================================
    # Head-to-Head Comparisons
    # ==================================================================
    comparisons = llm_result.get("pairwise_comparisons", [])
    if comparisons:
        st.subheader("Head-to-Head Comparisons")
        st.caption("Llama's reasoned pairwise analysis for every adjacent pair in the ranking.")

        for comp in comparisons:
            higher = comp.get("higher_ranked", "?")
            lower = comp.get("lower_ranked", "?")
            reasoning = comp.get("reasoning", "No reasoning provided.")

            with st.expander(f"{higher}  vs  {lower}"):
                col_h, col_l = st.columns(2)
                with col_h:
                    st.markdown(f"**Higher Ranked:** {higher}")
                    bf_h = best_fit.get(higher, {})
                    if bf_h:
                        st.caption(f"Best fit: {bf_h.get('jd', 'N/A')} ({bf_h.get('score', 0):.1f}%)")
                with col_l:
                    st.markdown(f"**Lower Ranked:** {lower}")
                    bf_l = best_fit.get(lower, {})
                    if bf_l:
                        st.caption(f"Best fit: {bf_l.get('jd', 'N/A')} ({bf_l.get('score', 0):.1f}%)")

                st.markdown("---")
                st.markdown(f"**Reasoning:** {reasoning}")

    st.divider()

    # ==================================================================
    # Per-Resume Strengths / Weaknesses / Missing
    # ==================================================================
    st.subheader("Per-Candidate Breakdown")

    for entry in ranking:
        fname = entry.get("filename", "Unknown")
        name = entry.get("name", "Unknown")
        rec = entry.get("recommendation", "N/A")
        badge = get_badge_html(rec)

        with st.expander(f"#{entry.get('rank', '?')} -- {name} ({fname}) -- {rec}"):
            detail_job = resolve_job(entry.get("best_fit_job"), best_fit.get(fname, {}).get("jd"))
            st.markdown(f"**Overall Match:** {entry.get('overall_match_pct', 'N/A')}%")
            st.markdown(f"**Best Fit Job:** {detail_job}")
            if entry.get("job_fit_reasoning"):
                st.markdown(f"**Job Fit Reasoning:** {entry['job_fit_reasoning']}")

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("**Strengths**")
                for s in entry.get("strengths", []):
                    st.markdown(f"- {s}")
            with col_w:
                st.markdown("**Weaknesses**")
                for w in entry.get("weaknesses", []):
                    st.markdown(f"- {w}")

            missing = entry.get("missing_requirements", [])
            if missing:
                st.markdown("**Missing Requirements**")
                for m in missing:
                    st.markdown(f"- {m}")

            if entry.get("feedback"):
                st.info(f"**Feedback:** {entry['feedback']}")

    st.divider()

    # ==================================================================
    # Deep-dive feedback per resume (on-demand from Llama)
    # ==================================================================
    st.subheader("Deep-Dive AI Feedback")
    st.caption("Request an in-depth analysis for any candidate. Llama will generate a comprehensive review.")

    jd_for_feedback = st.selectbox(
        "Analyze against JD:",
        [jd["name"] for jd in saved_jds],
        index=0,
    )
    jd_text_for_feedback = next(
        (jd["text"] for jd in saved_jds if jd["name"] == jd_for_feedback), ""
    )

    tabs = st.tabs([f"{r.get('name', 'Unknown')} ({r.get('filename', '?')[:20]})" for r in resumes_data])
    for i, tab in enumerate(tabs):
        with tab:
            cache_key = f"detailed_fb_{resumes_data[i].get('filename', i)}_{jd_for_feedback}"
            if cache_key in st.session_state:
                st.markdown(st.session_state[cache_key])
            else:
                if st.button(f"Generate detailed feedback", key=f"fb_{i}_{jd_for_feedback}"):
                    with st.spinner("Llama is writing detailed feedback..."):
                        fb = get_detailed_feedback(
                            jd_text=jd_text_for_feedback,
                            resume_text=resumes_data[i].get("cleaned_text", ""),
                            resume_filename=resumes_data[i].get("filename", "unknown"),
                            model=st.session_state.get("llama_model", "llama3.1"),
                        )
                        st.session_state[cache_key] = fb
                        st.markdown(fb)

    # ==================================================================
    # Raw data explorer
    # ==================================================================
    with st.expander("Raw data (debug)"):
        st.json({
            "bert_multi_results": {
                "best_fit": multi_results.get("best_fit", {}),
                "per_jd_summary": {
                    jd_name: [
                        {"filename": r["filename"], "weighted": r["weighted_score"], "raw": r["raw_cosine_score"]}
                        for r in results
                    ]
                    for jd_name, results in multi_results.get("per_jd", {}).items()
                },
            },
            "llm_result": llm_result,
        })
