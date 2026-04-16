"""
Semantic similarity engine using sentence-transformers and cosine similarity
for resume-to-job-description matching.

Model: BAAI/bge-base-en-v1.5  (768-dim, ~440 MB)
  - Significantly outperforms all-MiniLM-L6-v2 on retrieval/similarity benchmarks
    (MTEB avg ~63.5 vs ~56.3 for MiniLM)
  - Still small enough to run comfortably on an RTX 5060 (8 GB VRAM)
  - Uses instruction-prefixed encoding for queries (improves retrieval accuracy)

Device strategy
---------------
1. Try CUDA with a real tensor operation to confirm compatible kernels.
2. Fall back to CPU automatically if CUDA kernels don't match the arch.
"""

import os
import warnings
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-base-en-v1.5"
# BGE models benefit from an instruction prefix on the *query* side
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Robust device detection
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        a = torch.randn(2, 2, device="cuda")
        _ = a @ a
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[embedding_engine] GPU detected: {gpu_name}")
        return "cuda"
    except RuntimeError as exc:
        warnings.warn(
            f"\n{'='*70}\n"
            f"CUDA detected but kernel launch failed:\n  {exc}\n\n"
            f"FIX: pip install --pre torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/nightly/cu128\n\n"
            f"Falling back to CPU.\n"
            f"{'='*70}\n"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"


_device: str | None = None

def get_device() -> str:
    global _device
    if _device is None:
        _device = _resolve_device()
    return _device


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = get_device()
        print(f"[embedding_engine] Loading {MODEL_NAME} on '{device}' ...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_embedding(text: str, is_query: bool = False) -> np.ndarray:
    """
    Return embedding for the given text.
    For BGE models, the *query* should be prefixed with QUERY_PREFIX
    while *documents* (resume sections) are encoded as-is.
    """
    model = get_model()
    if is_query:
        text = QUERY_PREFIX + text
    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)


def compute_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity (0-100 scale)."""
    a = emb_a.reshape(1, -1)
    b = emb_b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0]) * 100


# ---------------------------------------------------------------------------
# Section-level analysis
# ---------------------------------------------------------------------------

SECTION_WEIGHTS = {
    "Skills":                  0.30,
    "Technical Skills":        0.30,
    "Experience":              0.25,
    "Work Experience":         0.25,
    "Professional Experience": 0.25,
    "Education":               0.15,
    "Projects":                0.15,
    "Summary":                 0.10,
    "Objective":               0.10,
    "Profile":                 0.10,
    "Certifications":          0.10,
    "Header":                  0.05,
    "full_text":               1.00,
}


def analyze_resume(resume_sections: dict, jd_text: str) -> dict:
    """
    Compare each resume section against the full JD.
    Returns per-section scores, weighted score, and raw cosine score.
    """
    jd_emb = compute_embedding(jd_text, is_query=True)

    section_scores = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for section_name, section_text in resume_sections.items():
        if not section_text.strip():
            continue
        sec_emb = compute_embedding(section_text, is_query=False)
        score = compute_cosine_similarity(sec_emb, jd_emb)
        section_scores[section_name] = round(score, 2)

        weight = SECTION_WEIGHTS.get(section_name, 0.05)
        weighted_sum += score * weight
        weight_sum += weight

    weighted_overall = round(weighted_sum / weight_sum, 2) if weight_sum else 0.0

    full_text = " ".join(resume_sections.values())
    full_emb = compute_embedding(full_text, is_query=False)
    raw_overall = round(compute_cosine_similarity(full_emb, jd_emb), 2)

    # Compute per-category scores for filtering
    skills_score = 0.0
    experience_score = 0.0
    education_score = 0.0

    skills_keys = {"Skills", "Technical Skills"}
    exp_keys = {"Experience", "Work Experience", "Professional Experience"}
    edu_keys = {"Education"}

    for k, v in section_scores.items():
        if k in skills_keys:
            skills_score = max(skills_score, v)
        elif k in exp_keys:
            experience_score = max(experience_score, v)
        elif k in edu_keys:
            education_score = max(education_score, v)

    return {
        "section_scores": section_scores,
        "weighted_score": weighted_overall,
        "raw_cosine_score": raw_overall,
        "skills_score": round(skills_score, 2),
        "experience_score": round(experience_score, 2),
        "education_score": round(education_score, 2),
    }


def batch_analyze(resumes: list[dict], jd_text: str) -> list[dict]:
    """
    Analyze a list of resume dicts against a single JD.
    Returns a list sorted by weighted_score descending.
    """
    results = []
    for resume in resumes:
        analysis = analyze_resume(resume["sections"], jd_text)
        results.append({
            "filename": resume["filename"],
            "word_count": resume.get("word_count", 0),
            "name": resume.get("name", "Unknown"),
            "email": resume.get("email", "N/A"),
            "experience": resume.get("experience", "N/A"),
            **analysis,
        })
    results.sort(key=lambda r: r["weighted_score"], reverse=True)
    return results


def multi_jd_analyze(resumes: list[dict], jds: list[dict]) -> dict:
    """
    Analyze every resume against every JD.
    Returns:
      {
        "per_jd": { jd_name: [sorted results], ... },
        "best_fit": { resume_filename: { "jd": ..., "score": ... }, ... }
      }
    """
    per_jd = {}
    # Track best JD per resume
    best_fit = {}

    for jd in jds:
        jd_name = jd["name"]
        jd_results = batch_analyze(resumes, jd["text"])
        per_jd[jd_name] = jd_results

        for res in jd_results:
            fname = res["filename"]
            if fname not in best_fit or res["weighted_score"] > best_fit[fname]["score"]:
                best_fit[fname] = {
                    "jd": jd_name,
                    "score": res["weighted_score"],
                    "skills_score": res["skills_score"],
                    "experience_score": res["experience_score"],
                    "education_score": res["education_score"],
                    "section_scores": res["section_scores"],
                    "raw_cosine_score": res["raw_cosine_score"],
                }

    return {"per_jd": per_jd, "best_fit": best_fit}
