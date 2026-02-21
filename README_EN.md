# Exam Grader

> [Русская версия](README.md)

Automated exam answer sheet recognition and grading system. Upload scanned PDF forms and an answer key to get a scored Excel report with embedded cell preview images.

## How It Works

1. **Upload** — a multi-page PDF of scanned answer sheets and an Excel file with correct answers.
2. **Choose recognition mode:**
   - **CLIP (local)** — each cell is recognized separately by a CLIP model running locally. No API key required.
   - **OpenAI Vision API** — the full page image is sent to gpt-5.2. Requires an OpenAI API key.
3. **Alignment** — each page is aligned to a reference template using SIFT and homography.
4. **Recognition** — answers are extracted by the chosen method.
5. **Grading** — recognized answers are compared to the key; scores use configurable weights.
6. **Export** — results are saved to Excel with per-question scores, totals, and embedded cell previews.

## Project Structure

```
streamlit-exam-form-app/
├── start.bat                       # One-click launcher (Windows)
├── build_portable.bat              # Build portable version (no Python install)
├── app.py                          # Streamlit UI entry point
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Python dependencies
│
├── exam_grader/                    # Main package
│   ├── config.py                   # Constants, paths, thresholds
│   ├── models/                     # Domain data models
│   │   ├── form.py                 # Form — top-level container with rows
│   │   ├── row.py                  # Row — line of cells (answer, correction, metadata)
│   │   ├── cell.py                 # Cell — single input with recognized value
│   │   └── symbol.py               # Symbol — detected character in a cell
│   ├── recognition/                # Recognition pipeline
│   │   ├── pipeline.py             # FormRecognition — orchestrator (CLIP & OpenAI)
│   │   ├── digit_classifier.py     # CLIP-based digit recognizer (lazy-loaded)
│   │   └── openai_recognizer.py    # OpenAI Vision API recognition
│   ├── processing/                 # Image processing
│   │   └── image_processing.py     # Alignment, border cleanup, symbol centering
│   ├── grading/                    # Answer checking and scoring
│   │   └── grader.py               # Score calculation, answer normalization
│   └── export/                     # Output generation
│       └── excel.py                # Excel export with embedded images
│
├── data/                           # Reference data
│   ├── template_raw.jpg            # Form template for alignment
│   ├── rows_data_new_format.json   # Cell coordinates
│   ├── ref_pics/                   # Reference digit images for CLIP
│   └── answers.xlsx                 # Sample answer key
│
└── notebooks/                      # Development and experiments
    ├── test_pipeline.ipynb         # Step-by-step CLIP pipeline test
    └── test_openai_pipeline.ipynb  # OpenAI pipeline test
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Streamlit | Web interface for upload and results |
| PDF | PyMuPDF (fitz) | High-resolution PDF to image conversion |
| Image processing | OpenCV, scikit-image | Alignment, thresholding, contour detection |
| Recognition (local) | CLIP (transformers, PyTorch) | Cell-by-cell handwritten digit recognition |
| Recognition (cloud) | OpenAI Vision API (optional) | Whole-page recognition via gpt-5.2 |
| Data | pandas | Answer comparison and scoring |
| Excel export | xlsxwriter, openpyxl | Formatted output with embedded images |

## Quick Start (Windows)

**Requirements:** [Python 3.10+](https://www.python.org/downloads/) — check "Add Python to PATH" during installation.

Double-click **`start.bat`** — it will create a virtual environment, install dependencies, and open the app in your browser.

### Portable version (no Python install)

Run **`build_portable.bat`** — it downloads a portable Python and installs all dependencies into the `python/` folder. After that, `start.bat` will use this local Python; no system-wide Python installation is needed.

## Manual Installation

```bash
git clone <repository-url>
cd streamlit-exam-form-app

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
pip install -e .
```

To enable OpenAI Vision API (optional):

```bash
pip install -e ".[openai]"
```

## Usage

```bash
streamlit run app.py
```

1. Choose recognition mode: **CLIP (local)** or **OpenAI Vision API**.
2. If using OpenAI, enter your API key.
3. Enter the exam version number.
4. Upload the scanned PDF file.
5. Upload the Excel file with correct answers (column `Вариант` and columns `1`–`10`).
6. Click **Распознать** (Recognize).
7. Download the resulting Excel file.

## Answer Key Format

| Вариант | 1 | 2 | 3 | ... | 10 |
|---------|---|---|---|-----|----|
| 1 | 275 | 38 | 14 | ... | 3.5 |
| 2 | 100 | 42 | 7 | ... | 2.0 |

## Scoring

Points per question are configured in `exam_grader/config.py`:

- Question 1: 0.5 points
- Questions 2–9: 1.0 point each
- Question 10: 1.5 points
- Maximum: 10.0 points

A question is marked correct if the original answer **or** the correction matches the answer key.
