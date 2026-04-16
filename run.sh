#!/usr/bin/env bash
# ============================================================
# Resume Ranker - Setup & Run
# ============================================================
# Prerequisites:
#   1. Python 3.10+
#   2. Ollama installed and running (https://ollama.com)
#   3. Tesseract OCR installed:
#        Ubuntu/Debian:  sudo apt install tesseract-ocr
#        macOS:          brew install tesseract
#        Windows:        https://github.com/UB-Mannheim/tesseract/wiki
#   4. Poppler (for pdf2image):
#        Ubuntu/Debian:  sudo apt install poppler-utils
#        macOS:          brew install poppler
#        Windows:        https://github.com/oschwartz10612/poppler-windows
#   5. An NVIDIA GPU with CUDA drivers (optional, for faster BERT inference)
#
# This script will:
#   - Install Python dependencies
#   - Check Tesseract is installed
#   - Pull the Llama 3.1 model if not already available
#   - Launch the Streamlit app
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================="
echo " Resume Ranker - Setup"
echo "=============================="

# 1. Install Python dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# 2. Check Tesseract
echo ""
echo "[2/4] Checking Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "WARNING: Tesseract is not installed."
    echo "Install it:"
    echo "  Ubuntu/Debian:  sudo apt install tesseract-ocr"
    echo "  macOS:          brew install tesseract"
    echo "  Windows:        https://github.com/UB-Mannheim/tesseract/wiki"
    echo ""
    echo "The app will fall back to pdfplumber for digital PDFs,"
    echo "but scanned PDFs will not be readable."
else
    echo "Tesseract found: $(tesseract --version | head -1)"
fi

# 3. Check Ollama and pull model
echo ""
echo "[3/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed."
    echo "Install it from https://ollama.com and restart this script."
    exit 1
fi

if ! ollama list &> /dev/null; then
    echo "ERROR: Ollama service is not running."
    echo "Start it with: ollama serve"
    exit 1
fi

echo "Pulling llama3.1 model (skip if already downloaded)..."
ollama pull llama3.1

# 4. Launch Streamlit
echo ""
echo "[4/4] Launching Resume Ranker..."
echo "Open http://localhost:8501 in your browser."
echo ""
streamlit run app.py --server.port 8501 --server.headless true
