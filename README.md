# Resume Ranking & Feedback System

A powerful, locally-run AI-powered resume screening and ranking system that combines semantic similarity (BERT) with advanced reasoning (Llama 3.1) to rank candidates and provide detailed hiring insights. Designed for HR teams to evaluate multiple resumes against one or more job descriptions efficiently.

## 🎯 Features

### Core Functionality
- **Multi-JD Support**: Upload multiple job descriptions once and evaluate every candidate against all roles
- **Smart Candidate Deduplication**: Automatically deduplicates candidates who appear multiple times, keeping only their best-fit role
- **BERT-Based Embeddings**: Uses `BAAI/bge-base-en-v1.5` (768-dim, MTEB ~63.5) for semantic matching with section-level scoring (Skills, Experience, Education, etc.)
- **Llama-Powered Reasoning**: Llama 3.1 via Ollama performs final ranking with detailed pairwise comparisons and evidence-based feedback
- **AI Contact Extraction**: Llama extracts candidate name, email, and phone directly from resume text
- **Tesseract OCR**: Handles both digital PDFs and scanned/image-based resumes with automatic fallback to pdfplumber for clean PDFs
- **Advanced Filtering**: Filter by job role, minimum skills %, experience %, and overall match %
- **Dual-Page UI**: Dashboard for quick glance candidate list; Detailed Review for in-depth analysis

### Dashboard Features
- **Top Candidates Table** with hyperlinked resume PDFs for instant download
- **Contact Info Display**: Name (linked), email, phone number visible at a glance
- **Score Visualization**: Progress bars for overall match, section-level scores
- **Job Assignment**: Each candidate matched to their best-fit role
- **Quick Feedback**: Expandable candidate assessments
- **Summary Stats**: Total analyzed, filtered count, strong/good matches
- **Real-time Filtering**: Adjust thresholds and instantly see filtered results

### Detailed Review Page
- **Per-JD BERT Scores**: Section-level breakdowns (Skills, Experience, Education, etc.)
- **Head-to-Head Comparisons**: Pairwise analysis of adjacent candidates with reasoning
- **Strengths & Weaknesses**: Per-candidate deep-dive analysis from Llama
- **Missing Requirements**: Gap analysis for each candidate
- **Flexible Analysis**: Select any saved JD to re-analyze candidates

## 🏗️ Architecture

### Component Overview

```
resume-ranking/
├── scripts/
│   ├── app.py                 # Main Streamlit dashboard (2-page app)
│   ├── pdf_processor.py       # PDF extraction (Tesseract + pdfplumber)
│   ├── embedding_engine.py    # BERT embeddings (BGE-base-en-v1.5)
│   ├── llama_ranker.py        # LLM ranking & reasoning (Ollama)
│   ├── requirements.txt       # Python dependencies
│   └── run.sh                 # Installation & launch script
└── README.md                  # This file
```

### Data Flow

```
1. PDF Upload → Tesseract/pdfplumber extraction
                        ↓
2. Contact Info → AI (Llama) extraction + Regex fallback
                        ↓
3. Section Split → Skills, Experience, Education, etc.
                        ↓
4. BERT Embedding → BGE embeddings for each section + full text
                        ↓
5. Cosine Similarity → Score vs all JD sections (weighted)
                        ↓
6. LLM Ranking → Llama receives resumes + BERT scores
                        ↓
7. Final Ranking → JSON with rank, recommendation, feedback, job match
                        ↓
8. Deduplication → One entry per candidate (highest overall %)
                        ↓
9. Dashboard Display → Filtered table with scores, recommendations, contact info
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** with `llama3.1` model downloaded
- **Tesseract OCR** and **Poppler** utilities
- **NVIDIA GPU** (RTX 5060 or better recommended for BERT inference)

### Installation

#### 1. Clone and Navigate
```bash
cd /path/to/resume-ranking
```

#### 2. Auto Install (Recommended)
```bash
cd scripts
chmod +x run.sh
./run.sh
```

This script will:
- Check for Tesseract and Poppler; provide install instructions if missing
- Create a Python virtual environment
- Install dependencies from `requirements.txt`
- Launch the Streamlit app

#### 3. Manual Installation

**Install System Dependencies:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Download Poppler from: https://blog.alivate.com.au/poppler-windows/
- Add both to PATH

**Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

**Ensure Ollama is Running:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model (if not already done)
ollama pull llama3.1
```

**Launch the App:**
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### Dashboard (Main Page)

#### Step 1: Add Job Descriptions
1. Click **"Add New Job"** button
2. Upload a JD PDF or paste text
3. Click **Save JD** to add to session
4. Repeat for additional roles (supports 2-5 JDs)

#### Step 2: Upload Resumes
1. Click **"Upload Resumes"** file uploader
2. Select multiple PDFs (both digital and scanned supported)
3. Click **Extract & Analyze**
4. System extracts text, contact info, and sections

#### Step 3: Run Analysis
1. All JDs are matched against all resumes via BERT
2. Llama ranks candidates and assigns best-fit jobs
3. Results appear below with filters applied

#### Step 4: Filter & Sort
- **Filter by Job**: Narrow to a specific role
- **Min Skills %**: Filter out candidates below skill threshold
- **Min Experience %**: Filter by experience level
- **Min Overall Match %**: Set overall quality floor
- **Show top N**: Limit displayed results

#### Step 5: Explore Results
- Click any **candidate name** to download their resume PDF
- View **contact info** (email, phone) inline
- See **quick feedback** by expanding each row
- Navigate to **Detailed Review** for deep analysis

### Detailed Review Page

1. Select a candidate from the dropdown
2. View **BERT Section Scores**: Skills, Experience, Education breakdown
3. Read **Head-to-Head Comparisons**: Why candidate A ranks higher than B
4. Review **Strengths & Weaknesses**: Specific evidence-based assessment
5. Check **Missing Requirements**: Gaps vs job description
6. Change JD selector at top to re-analyze against different roles

## ⚙️ Configuration

### BERT Model Selection

To change the embedding model (default: `BAAI/bge-base-en-v1.5`):

Edit `embedding_engine.py`:
```python
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Alternative (438MB)
# or
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # Smaller (80MB)
```

### Llama Model Selection

To use a different Ollama model:

In the Streamlit sidebar, select from the dropdown. Tested models:
- `llama3.1` (8B, ~5GB VRAM) - **Recommended** for this task
- `llama3.2:3b` (3B, ~2GB VRAM) - Smaller, cleaner JSON output
- `llama2` (7B/13B) - Older, less structured output

### Section Weights (BERT Scoring)

Edit `embedding_engine.py` `calculate_weighted_similarity()`:
```python
weights = {
    "full_text": 0.20,
    "skills": 0.30,          # Skills heavily weighted
    "experience": 0.25,      # Experience second
    "education": 0.15,
    "summary": 0.10,
}
```

### GPU Fallback

If your GPU isn't detected or CUDA fails:
1. App automatically falls back to CPU
2. Check sidebar for device status
3. To force CPU: `os.environ["CUDA_VISIBLE_DEVICES"] = ""`
4. To fix GPU: Install PyTorch nightly with CUDA 12.8 support:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 🔧 Troubleshooting

### "No kernel image is available"
Your PyTorch build lacks CUDA kernels for your GPU architecture (e.g., RTX 5060 / Blackwell sm_120).
**Fix**: Install nightly PyTorch:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### "tesseract is not installed or cannot be found"
Tesseract binary not in PATH.
**Fix**:
- **Ubuntu**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Add Tesseract installation folder to PATH, then restart Streamlit

### "LLM could not parse JSON"
Llama returned malformed JSON.
**Fixes**:
1. Try running analysis again (may be transient)
2. Switch to smaller model: `ollama pull llama3.2:3b`
3. Check Debug info in dashboard for raw LLM output

### No candidates showing in dashboard (all filters "No match")
Most likely: LLM filename/name matching failure.
**Debug Steps**:
1. Expand **"Debug: raw data"** section in Dashboard
2. Check if resumes were extracted (should list filenames)
3. Check if LLM matched them (should show count > 0)
4. Lower all filter sliders to 0% to test
5. Look for filename/name mismatches in debug output

### Status showing "PENDING" instead of recommendation
Indicator that LLM entry wasn't found; system fell back to score-based recommendation.
**Check**:
- Debug output shows match score for that resume
- Overall % should still be accurate
- This is a graceful fallback, not an error

### "Poppler not found" when using digital PDFs
pdfplumber fallback for text extraction requires Poppler.
**Fix**:
- **Ubuntu**: `sudo apt install poppler-utils`
- **macOS**: `brew install poppler`
- **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases

## 📊 System Performance

### Benchmarks (RTX 5060, 16GB RAM, llama3.1:8b)

| Task | Time | Notes |
|------|------|-------|
| Extract 5 PDFs | 10-15s | Tesseract OCR + pdfplumber fallback |
| BERT encode 5 resumes | 3-5s | 768-dim embeddings + cosine similarity |
| LLM rank 5 resumes vs 2 JDs | 30-45s | Llama 3.1 inference |
| Full analysis cycle | 50-70s | Total end-to-end |
| Dashboard render (100 rows) | 1-2s | After analysis complete |

**Scalability**: Tested with up to 50 resumes and 4 JDs (~3-4 min total).

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] LinkedIn/ATS format import
- [ ] Export results to CSV/PDF report
- [ ] Candidate comparison matrix
- [ ] Save/load analysis sessions
- [ ] Fine-tuned BERT model for HR domain
- [ ] Multi-language support
- [ ] REST API wrapper

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Sentence-Transformers** for BAAI/BGE embeddings
- **Ollama** for local LLM inference
- **Streamlit** for the interactive UI framework
- **Tesseract** for robust OCR
- **PyPDF/pdfplumber** for PDF handling

## 📧 Support

For issues, questions, or feature requests, open an issue on GitHub or check the Debug panel in the app for detailed logs.

---

**Last Updated**: April 2026  
**Status**: Production Ready  
**Python**: 3.10+  
**Tested On**: RTX 5060, 16GB RAM, Ubuntu 22.04 / macOS 13+ / Windows 11
