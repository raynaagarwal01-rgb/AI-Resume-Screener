"""
LLM-powered ranking and feedback engine using Llama 3.1 via Ollama.
Supports multi-JD matching: each candidate is ranked against all JDs
and assigned to their best-fit role.
"""

import json
import re
import ollama


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert technical recruiter and hiring manager with 20+ years 
of experience evaluating candidates. You provide precise, evidence-based 
assessments grounded in the actual content of each resume relative to a 
specific job description.

RULES:
- Base every claim on concrete evidence from the resume text.
- Never fabricate skills or experience not present in the resume.
- When comparing resumes, cite specific differences (e.g. "Resume A lists 
  5 years of Python vs Resume B's 2 years").
- Be direct and constructive in feedback; avoid vague praise."""


def _build_multi_jd_ranking_prompt(
    jds: list[dict],
    resumes: list[dict],
    best_fit: dict,
    per_jd_scores: dict,
) -> str:
    """
    Build the prompt for multi-JD ranking.
    Each resume gets its best-fit JD based on BERT scores,
    and Llama validates/overrides with reasoning.
    """
    parts = []

    # JOB DESCRIPTIONS
    for i, jd in enumerate(jds):
        parts.append("=" * 60)
        parts.append(f"JOB DESCRIPTION {i+1}: {jd['name']}")
        parts.append("=" * 60)
        parts.append(jd["text"][:5000])
        parts.append("")

    # RESUMES
    for i, resume in enumerate(resumes):
        fname = resume["filename"]
        bf = best_fit.get(fname, {})
        parts.append("-" * 60)
        parts.append(f"RESUME {i+1}: {fname}")
        parts.append(f"  Name: {resume.get('name', 'Unknown')}")
        parts.append(f"  Email: {resume.get('email', 'N/A')}")
        parts.append(f"  Experience: {resume.get('experience', 'N/A')}")
        parts.append(f"  BERT Best-Fit Job: {bf.get('jd', 'N/A')} ({bf.get('score', 0):.1f}%)")

        # Show scores per JD
        for jd_name, jd_results in per_jd_scores.items():
            for r in jd_results:
                if r["filename"] == fname:
                    parts.append(f"  Score vs '{jd_name}': {r['weighted_score']}% (skills: {r['skills_score']}%, exp: {r['experience_score']}%)")
                    break
        parts.append("-" * 60)
        words = resume["cleaned_text"].split()
        text = " ".join(words[:2000])
        if len(words) > 2000:
            text += "\n[...truncated...]"
        parts.append(text)
        parts.append("")

    # INSTRUCTIONS
    jd_names_str = json.dumps([jd["name"] for jd in jds])
    parts.append("=" * 60)
    parts.append("INSTRUCTIONS")
    parts.append("=" * 60)
    parts.append(f"""
Analyze these resumes against ALL job descriptions above.
Available job roles: {jd_names_str}

You MUST return a valid JSON object with this exact structure and nothing else:

{{
  "final_ranking": [
    {{
      "rank": 1,
      "filename": "<filename>",
      "name": "<candidate name>",
      "best_fit_job": "<which JD this candidate is best suited for>",
      "job_fit_reasoning": "<why this job is the best fit for this candidate>",
      "overall_match_pct": <number 0-100>,
      "recommendation": "STRONG MATCH" | "GOOD MATCH" | "PARTIAL MATCH" | "WEAK MATCH",
      "strengths": ["strength 1", "strength 2", ...],
      "weaknesses": ["weakness 1", "weakness 2", ...],
      "missing_requirements": ["requirement 1", ...],
      "feedback": "<2-3 sentence actionable feedback>"
    }}
  ],
  "pairwise_comparisons": [
    {{
      "higher_ranked": "<filename>",
      "lower_ranked": "<filename>",
      "reasoning": "<specific reasons citing evidence from both resumes>"
    }}
  ],
  "summary": "<overall analysis paragraph>"
}}

CRITICAL RULES:
- "final_ranking" MUST contain EXACTLY ONE entry per resume. Do NOT create
  multiple entries for the same person. Each candidate appears ONCE with
  their single best-fit job.
- "best_fit_job" MUST be one of these exact job names: {jd_names_str}
  DO NOT use "N/A", "null", "Unknown", or any other value. Every single candidate
  MUST be assigned to exactly one job from that list. Pick the best match even if
  the fit is weak -- there is no "unassigned" option.
- The BERT scores are one signal; use your own judgment about actual qualifications.
- For pairwise_comparisons, compare EVERY adjacent pair in your final ranking.
- Be specific: cite skills, years, projects, education from the resumes.
- Return ONLY the JSON object. No markdown fences, no extra text.
""")

    return "\n".join(parts)


def _build_single_jd_ranking_prompt(
    jd_text: str,
    resumes: list[dict],
    similarity_results: list[dict],
) -> str:
    """Build prompt for single-JD mode (backward compatible)."""
    parts = []
    parts.append("=" * 60)
    parts.append("JOB DESCRIPTION")
    parts.append("=" * 60)
    parts.append(jd_text[:6000])
    parts.append("")

    for i, resume in enumerate(resumes):
        sim = similarity_results[i] if i < len(similarity_results) else {}
        parts.append("-" * 60)
        parts.append(f"RESUME {i+1}: {resume['filename']}")
        parts.append(f"  Name: {resume.get('name', 'Unknown')}")
        parts.append(f"  Email: {resume.get('email', 'N/A')}")
        parts.append(f"  Experience: {resume.get('experience', 'N/A')}")
        parts.append(f"  BERT Weighted Score: {sim.get('weighted_score', 'N/A')}%")
        parts.append(f"  BERT Raw Score: {sim.get('raw_cosine_score', 'N/A')}%")
        if sim.get("section_scores"):
            parts.append(f"  Section scores: {json.dumps(sim['section_scores'])}")
        parts.append("-" * 60)
        words = resume["cleaned_text"].split()
        text = " ".join(words[:2000])
        if len(words) > 2000:
            text += "\n[...truncated...]"
        parts.append(text)
        parts.append("")

    parts.append("=" * 60)
    parts.append("INSTRUCTIONS")
    parts.append("=" * 60)
    parts.append("""
Analyze these resumes against the job description above.
Return a valid JSON object with this exact structure and nothing else:

{
  "final_ranking": [
    {
      "rank": 1,
      "filename": "<filename>",
      "name": "<candidate name>",
      "best_fit_job": "<job title from the JD>",
      "overall_match_pct": <number 0-100>,
      "recommendation": "STRONG MATCH" | "GOOD MATCH" | "PARTIAL MATCH" | "WEAK MATCH",
      "strengths": ["strength 1", "strength 2", ...],
      "weaknesses": ["weakness 1", "weakness 2", ...],
      "missing_requirements": ["requirement 1", ...],
      "feedback": "<2-3 sentence actionable feedback>"
    }
  ],
  "pairwise_comparisons": [
    {
      "higher_ranked": "<filename>",
      "lower_ranked": "<filename>",
      "reasoning": "<specific reasons citing evidence from both resumes>"
    }
  ],
  "summary": "<overall analysis paragraph>"
}

CRITICAL RULES:
- "best_fit_job" MUST be set to the job title from the JD above. Never use "N/A" or "null".
- overall_match_pct is YOUR assessment, not just the BERT score.
- Include pairwise_comparisons for EVERY adjacent pair.
- Be specific: cite skills, years, projects, education.
- Return ONLY the JSON object.
""")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def rank_resumes(
    jd_text: str = "",
    resumes: list[dict] = None,
    similarity_results: list[dict] = None,
    model: str = "llama3.1",
    # Multi-JD params
    jds: list[dict] = None,
    best_fit: dict = None,
    per_jd_scores: dict = None,
) -> dict:
    """
    Send resumes + JD(s) + BERT scores to Llama for reasoned ranking.
    Supports both single-JD and multi-JD modes.
    """
    if jds and best_fit and per_jd_scores:
        user_prompt = _build_multi_jd_ranking_prompt(jds, resumes, best_fit, per_jd_scores)
    else:
        sim_lookup = {s["filename"]: s for s in (similarity_results or [])}
        ordered_sims = [sim_lookup.get(r["filename"], {}) for r in resumes]
        user_prompt = _build_single_jd_ranking_prompt(jd_text, resumes, ordered_sims)

    MAX_RETRIES = 2
    last_error = ""
    last_raw = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # On retry, append a correction message with the previous bad output
            if attempt > 0 and last_raw:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed as valid JSON. "
                        "Please return ONLY a valid JSON object, no markdown fences, "
                        "no extra text before or after. Make sure all strings are "
                        "double-quoted, no trailing commas, and all brackets are balanced. "
                        "Return the same analysis but as clean JSON."
                    ),
                })

            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0.2 if attempt > 0 else 0.3,
                    "num_predict": 4096,
                    "top_p": 0.9,
                },
                # Request JSON format if Ollama supports it
                format="json" if attempt > 0 else None,
            )
            raw = response["message"]["content"].strip()
            result = _parse_response(raw)

            # If parsing succeeded (no error key), return it
            if "error" not in result:
                return result

            # Parsing failed -- save for retry
            last_error = result.get("error", "")
            last_raw = raw
            print(f"[llama_ranker] Attempt {attempt + 1} parse failed: {last_error}")

        except ollama.ResponseError as e:
            return {"error": f"Ollama error: {e}", "raw_response": ""}
        except Exception as e:
            last_error = str(e)
            last_raw = ""
            print(f"[llama_ranker] Attempt {attempt + 1} exception: {e}")

    return {"error": f"Failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}", "raw_response": last_raw}


def _parse_response(raw: str) -> dict:
    """
    Robust JSON extractor that handles the many ways Llama can mangle JSON:
      - Markdown fences (```json ... ```)
      - Leading/trailing prose around the JSON
      - Trailing commas before } or ]
      - Single-quoted strings
      - Unescaped newlines inside string values
      - Truncated JSON (attempts brace-balancing)
    """

    # ---- Step 1: Strip markdown code fences ----
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # ---- Step 2: Try direct parse ----
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ---- Step 3: Find the outermost { ... } block ----
    # Llama sometimes adds prose before/after the JSON
    start = cleaned.find("{")
    if start == -1:
        return {"error": "No JSON object found in LLM response.", "raw_response": raw}

    # Find matching closing brace via brace-counting
    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        # Truncated JSON -- try to close it
        snippet = cleaned[start:]
        snippet += "}" * depth  # close all open braces
        # Also close any open arrays
        open_brackets = snippet.count("[") - snippet.count("]")
        if open_brackets > 0:
            snippet += "]" * open_brackets + "}" * open_brackets
    else:
        snippet = cleaned[start : end + 1]

    # ---- Step 4: Fix common Llama JSON errors ----
    # Remove trailing commas before } or ]
    snippet = re.sub(r",\s*([}\]])", r"\1", snippet)

    # Replace single quotes with double quotes (but not inside double-quoted strings)
    # This is a best-effort heuristic
    if '"' not in snippet and "'" in snippet:
        snippet = snippet.replace("'", '"')

    # Remove control characters that break JSON (newlines inside strings)
    # Replace literal newlines inside string values with \\n
    def _fix_string_newlines(m: re.Match) -> str:
        s = m.group(0)
        inner = s[1:-1]  # strip outer quotes
        inner = inner.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{inner}"'

    snippet = re.sub(r'"(?:[^"\\]|\\.)*"', _fix_string_newlines, snippet, flags=re.DOTALL)

    # ---- Step 5: Try parsing the cleaned snippet ----
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # ---- Step 6: Last resort -- line-by-line repair ----
    # Try removing lines that cause parse errors
    lines = snippet.split("\n")
    for attempt in range(min(10, len(lines))):
        try:
            return json.loads("\n".join(lines))
        except json.JSONDecodeError as e:
            # Try removing the offending line
            if hasattr(e, "lineno") and 0 < e.lineno <= len(lines):
                bad_line = lines[e.lineno - 1]
                # Don't remove structural lines
                if bad_line.strip() not in ("{", "}", "[", "]", "},", "],"):
                    lines.pop(e.lineno - 1)
                    continue
            break

    return {"error": "Could not parse LLM response as JSON after all repair attempts.", "raw_response": raw}


# ---------------------------------------------------------------------------
# Individual detailed feedback
# ---------------------------------------------------------------------------

def get_detailed_feedback(
    jd_text: str,
    resume_text: str,
    resume_filename: str,
    model: str = "llama3.1",
) -> str:
    """Get in-depth feedback for a single resume against a JD."""
    prompt = f"""Analyze this resume against the job description and provide detailed, 
actionable feedback.

JOB DESCRIPTION:
{jd_text[:4000]}

RESUME ({resume_filename}):
{resume_text[:4000]}

Provide:
1. Overall match assessment (percentage and category)
2. Top 5 strengths relevant to this role
3. Top 5 gaps or weaknesses
4. Specific suggestions to improve this resume for this role
5. Keywords from the JD missing from the resume
6. Section-by-section analysis (Skills, Experience, Education, Projects)

Be specific and cite evidence from both documents."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 3072},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error generating feedback: {e}"
