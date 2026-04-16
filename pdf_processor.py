"""
PDF text extraction and preprocessing module.
Uses Tesseract OCR (via pytesseract) with pdf2image for robust PDF parsing,
including scanned documents and image-based PDFs.

Falls back to pdfplumber for digitally-born PDFs if Tesseract yields no text.
Contact extraction uses Llama (via Ollama) as the primary method with regex fallback.
"""

import json
import re
import tempfile
from io import BytesIO
from pathlib import Path

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

try:
    import ollama as _ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_text_tesseract(pdf_bytes: bytes) -> str:
    """
    Convert each PDF page to an image, then OCR with Tesseract.
    Returns the concatenated text from all pages.
    """
    images = convert_from_bytes(pdf_bytes, dpi=300)
    text_parts = []
    for img in images:
        page_text = pytesseract.image_to_string(img, lang="eng")
        if page_text and page_text.strip():
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_pdfplumber(pdf_file) -> str:
    """Fallback: use pdfplumber for digitally-born PDFs."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception:
        return ""


def extract_text(pdf_file) -> str:
    """
    Primary extraction: Tesseract OCR.
    Fallback: pdfplumber (for digital PDFs where OCR may be less accurate).
    """
    # Read raw bytes for Tesseract
    if isinstance(pdf_file, BytesIO):
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
    elif isinstance(pdf_file, bytes):
        pdf_bytes = pdf_file
    else:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)

    text = extract_text_tesseract(pdf_bytes)

    # If Tesseract returned very little, try pdfplumber as fallback
    if len(text.strip().split()) < 20:
        fallback = extract_text_pdfplumber(pdf_file)
        if len(fallback.strip().split()) > len(text.strip().split()):
            text = fallback

    return text


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def extract_sections(text: str) -> dict:
    """
    Attempt to split a resume into common sections.
    Returns a dict with section names as keys.
    If no headings are found, returns {"full_text": text}.
    """
    heading_pattern = re.compile(
        r"^(SUMMARY|OBJECTIVE|EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|"
        r"EDUCATION|SKILLS|TECHNICAL SKILLS|CERTIFICATIONS|PROJECTS|"
        r"ACHIEVEMENTS|AWARDS|PUBLICATIONS|LANGUAGES|INTERESTS|REFERENCES|"
        r"CONTACT|PROFILE|ABOUT ME|QUALIFICATIONS)\b",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return {"full_text": text}

    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(0).strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[section_name] = text[start:end].strip()

    # Include any text before the first heading as "Header"
    if matches[0].start() > 0:
        sections["Header"] = text[: matches[0].start()].strip()

    return sections


# ---------------------------------------------------------------------------
# AI-powered contact extraction (Llama via Ollama)
# ---------------------------------------------------------------------------

_CONTACT_MODEL = "llama3.1"  # can be overridden via set_contact_model()


def set_contact_model(model: str):
    """Allow the app to set which Ollama model to use for contact extraction."""
    global _CONTACT_MODEL
    _CONTACT_MODEL = model


def extract_contact_info_ai(text: str) -> dict | None:
    """
    Use Llama to extract name, email, and phone from resume text.
    Returns dict with keys name/email/phone, or None if it fails.
    Only sends the first ~1500 chars (the header area of a resume).
    """
    if not _HAS_OLLAMA:
        return None

    # Only send the top portion -- contact info is always at the top
    header_text = text[:1500]

    prompt = f"""Extract the candidate's contact information from this resume header.
Return ONLY a JSON object with exactly these three keys, nothing else:
{{"name": "Full Name", "email": "email@example.com", "phone": "+1 555-123-4567"}}

Rules:
- "name" must be the person's full name (first and last). Never return a company name.
- "email" must be a valid email address found in the text, or "N/A" if not found.
- "phone" must be the phone number exactly as written, or "N/A" if not found.
- Return ONLY the JSON object. No explanation, no markdown fences.

RESUME TEXT:
{header_text}"""

    try:
        response = _ollama.chat(
            model=_CONTACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 200},
            format="json",
        )
        raw = response["message"]["content"].strip()

        # Clean markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)

        data = json.loads(raw.strip())
        result = {
            "name": data.get("name", "").strip() or "Unknown",
            "email": data.get("email", "").strip() or "N/A",
            "phone": data.get("phone", "").strip() or "N/A",
        }

        # Sanity checks -- if AI returned garbage, reject it
        if len(result["name"]) < 2 or len(result["name"]) > 80:
            result["name"] = "Unknown"
        if result["email"] != "N/A" and "@" not in result["email"]:
            result["email"] = "N/A"
        if result["phone"] != "N/A":
            digits = re.sub(r"\D", "", result["phone"])
            if len(digits) < 7 or len(digits) > 15:
                result["phone"] = "N/A"

        # Only return if we got at least a name
        if result["name"] != "Unknown":
            return result
        return None

    except Exception as e:
        print(f"[pdf_processor] AI contact extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Regex-based contact extraction (fallback)
# ---------------------------------------------------------------------------

def extract_contact_info_regex(text: str) -> dict:
    """
    Best-effort extraction of candidate name, email and phone from resume text.
    """
    info = {"name": "Unknown", "email": "N/A", "phone": "N/A"}

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        info["email"] = email_match.group(0)

    # Phone -- common formats:
    #   +1 (555) 123-4567, 555-123-4567, (555)1234567, +44 7700 900000, 555.123.4567
    phone_patterns = [
        r"\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}",
        r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}",
    ]
    for pat in phone_patterns:
        phone_match = re.search(pat, text)
        if phone_match:
            candidate = phone_match.group(0).strip()
            # Sanity: must have at least 7 digits
            digits = re.sub(r"\D", "", candidate)
            if 7 <= len(digits) <= 15:
                info["phone"] = candidate
                break

    # Name heuristic: first non-empty line that looks like a name
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        # Skip lines that look like emails, phones, urls, or section headers
        if re.search(r"@|http|www\.|^\d|^\(?\d{3}", line):
            continue
        if re.match(r"^(SUMMARY|OBJECTIVE|EXPERIENCE|EDUCATION|SKILLS|CONTACT|PROFILE)", line, re.IGNORECASE):
            continue
        # If it's 2-4 words and mostly alpha, treat as name
        words = line.split()
        if 1 <= len(words) <= 5 and all(re.match(r"^[A-Za-z.\-']+$", w) for w in words):
            info["name"] = line
            break

    return info


# ---------------------------------------------------------------------------
# Experience extraction (years)
# ---------------------------------------------------------------------------

def extract_experience_years(text: str) -> str:
    """
    Attempt to find total years of experience from text.
    Returns a string like '5 years' or 'N/A'.
    """
    # Look for explicit mentions like "10+ years", "5 years of experience"
    patterns = [
        r"(\d{1,2}\+?)\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
        r"(\d{1,2}\+?)\s*(?:years?|yrs?)\s+(?:in|of)",
        r"experience\s*[:\-]?\s*(\d{1,2}\+?)\s*(?:years?|yrs?)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} years"

    # Count date ranges in experience sections
    date_ranges = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|19\d{2}|present|current)",
        text, re.IGNORECASE,
    )
    if date_ranges:
        import datetime
        current_year = datetime.datetime.now().year
        total = 0
        for start, end in date_ranges:
            s = int(start)
            e = current_year if end.lower() in ("present", "current") else int(end)
            total += max(e - s, 0)
        if total > 0:
            return f"~{total} years"

    return "N/A"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _empty_result(error_msg: str = "") -> dict:
    """Return a safe empty result dict with all expected keys."""
    return {
        "raw_text": "",
        "cleaned_text": "",
        "sections": {"full_text": ""},
        "char_count": 0,
        "word_count": 0,
        "name": "Unknown",
        "email": "N/A",
        "phone": "N/A",
        "experience": "N/A",
        "error": error_msg,
    }


def process_pdf(pdf_file) -> dict:
    """Full pipeline: extract -> clean -> section-split -> contact info."""
    try:
        raw = extract_text(pdf_file)
    except Exception as e:
        result = _empty_result(f"Text extraction failed: {e}")
        print(f"[pdf_processor] Extraction error: {e}")
        return result

    try:
        cleaned = clean_text(raw)
        sections = extract_sections(cleaned)

        # AI-first contact extraction, regex fallback
        contact = extract_contact_info_ai(cleaned)
        if contact is None:
            contact = extract_contact_info_regex(cleaned)

        experience = extract_experience_years(cleaned)
    except Exception as e:
        result = _empty_result(f"Processing failed: {e}")
        result["raw_text"] = raw
        print(f"[pdf_processor] Processing error: {e}")
        return result

    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "sections": sections,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "name": contact.get("name", "Unknown"),
        "email": contact.get("email", "N/A"),
        "phone": contact.get("phone", "N/A"),
        "experience": experience,
    }
