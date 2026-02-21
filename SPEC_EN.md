# Exam Grader — Technical Specification v1

> [Русская версия](SPEC.md)

## **1. Overview**

An automated system for recognizing handwritten exam answer sheets and grading them against a provided answer key. It processes scanned PDF forms through computer vision and ML pipelines and produces scored Excel reports.

### **Key Objectives**

- Automate the manual exam grading workflow.
- Provide **two recognition backends**: local CLIP model (offline) and OpenAI Vision API (cloud).
- Generate detailed Excel reports with embedded cell preview images for verification.
- Support correction fields — students can override their original answers.

### **Project Philosophy**

- Single-package architecture — simple to understand and deploy.
- Lazy model loading — ML models load on first use, not at import time.
- Configuration as constants — all thresholds, paths, and scoring weights in one file.
- Clear separation of concerns — models, recognition, processing, grading, and export are independent modules.

## **2. Architecture**

### **2.1 Single-Package Architecture**

```
exam_grader/
├── config.py           # Configuration constants
├── models/             # Domain data structures
├── recognition/        # Recognition pipeline
├── processing/         # Image processing utilities
├── grading/            # Answer checking and scoring
└── export/             # Excel output generation
```

### **2.2 Module Responsibilities**

| Module | Purpose | Imports From |
|--------|---------|--------------|
| `config` | Constants, paths, thresholds | stdlib only |
| `models` | Form, Row, Cell, Symbol classes | `config`, `processing`, `recognition` |
| `processing` | Alignment, styling, PDF conversion | `config` |
| `recognition` | CLIP & OpenAI recognition, orchestration | `config`, `models` |
| `grading` | Answer comparison, score calculation | `config` |
| `export` | Excel generation with embedded images | `config`, `models` |

### **2.3 Data Flow**

Two recognition paths share the same grading and export pipeline:

```
                                  ┌─── CLIP Mode ─────────────────────┐
                                  │  Style → Cell OCR →                │
PDF Upload → Page Images → Align ─┤  Empty/Minus/Comma detection       ├→ Grading → Excel
                                  │                                    │
                                  ├─── OpenAI Mode ────────────────────┤
                                  │  Send image → GPT-4o → Parse JSON  │
                                  └────────────────────────────────────┘
```

## **3. Technology Stack**

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.10+ | Type hints, f-strings |
| UI | Streamlit | Web UI for file upload and results |
| PDF | PyMuPDF (fitz) | High-resolution PDF page rendering |
| Image processing | OpenCV | SIFT alignment, thresholding, contour detection |
| Image analysis | scikit-image | Otsu thresholding for digit binarization |
| ML framework | PyTorch | CLIP model inference |
| ML model | CLIP ViT-B/32 (MNIST fine-tuned) | Handwritten digit embedding comparison (local) |
| Cloud OCR | OpenAI Vision API (optional) | Whole-page recognition via GPT-4o |
| Data | pandas | DataFrame operations, answer merging |
| Validation | Pydantic | Data model validation |
| Excel output | xlsxwriter | Excel with embedded images and formulas |
| Excel reading | openpyxl | Reading answer key Excel files |
| Image conversion | Pillow | Image format conversions |

## **4. Detailed Project Structure**

```
streamlit-exam-form-app/
│
├── start.bat                          # One-click launcher (Windows)
├── build_portable.bat                 # Portable build script
├── app.py                             # Streamlit entry point (thin orchestrator)
├── pyproject.toml                     # Python packaging
├── requirements.txt                   # Dependencies
├── README.md                          # User docs (Russian)
├── README_EN.md                       # User docs (English)
├── SPEC.md                            # This spec (Russian)
├── SPEC_EN.md                         # This spec (English)
├── .gitignore
│
├── exam_grader/                       # MAIN PACKAGE
│   ├── __init__.py                    # Public API re-exports
│   ├── config.py                      # Configuration constants
│   │
│   ├── models/                        # Domain models
│   │   ├── __init__.py
│   │   ├── form.py                    # Form: top-level container, orchestrates recognition
│   │   ├── row.py                     # Row: line of cells with metadata
│   │   ├── cell.py                    # Cell: single input cell
│   │   └── symbol.py                 # Symbol: detected character
│   │
│   ├── recognition/                   # Recognition pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                # FormRecognition: end-to-end (CLIP & OpenAI)
│   │   ├── digit_classifier.py        # DigitClassifier: CLIP-based, lazy-loaded
│   │   └── openai_recognizer.py       # OpenAI Vision API recognition
│   │
│   ├── processing/                    # Image processing
│   │   ├── __init__.py
│   │   └── image_processing.py        # Alignment, borders, symbol detection, PDF utils
│   │
│   ├── grading/                       # Grading logic
│   │   ├── __init__.py
│   │   └── grader.py                  # Answer checking, scoring, normalization
│   │
│   └── export/                        # Output generation
│       ├── __init__.py
│       └── excel.py                   # Excel export with images and formulas
│
├── data/                              # REFERENCE DATA
│   ├── template_raw.jpg               # Clean form template for SIFT alignment
│   ├── rows_data_new_format.json      # Cell coordinates per row
│   ├── answers.xlsx                    # Sample correct answers
│   ├── ref_pics/                      # Reference digit images for CLIP
│   │   ├── zero.png ... nine.png      # Digit references
│   │   ├── comma.png, minus.png       # Special character references
│   │   └── empty.png                  # Empty cell reference
│   └── valid_format/                  # Sample input PDFs
│
└── notebooks/                         # DEVELOPMENT
    ├── test_pipeline.ipynb            # Step-by-step CLIP pipeline test
    ├── test_openai_pipeline.ipynb     # OpenAI pipeline test & CLIP comparison
    ├── prepare_json_new_format.ipynb  # Cell coordinate preparation
    ├── allign_squares.ipynb           # Alignment experiments
    └── ...                            # Other test/dev notebooks
```

## **5. Module Specifications**

### **5.1 Configuration (`config.py`)**

Central store for all constants. No runtime state.

**Path constants:**
- `TEMPLATE_PATH` — reference form image for alignment
- `CELL_COORDS_PATH` — JSON file with cell coordinates
- `REF_PICS_DIR` — directory with reference digit images
- `SAVED_EXCELS_DIR` — directory for saved results

**Processing thresholds:**
- `PDF_ZOOM` (6.0) — PDF render scale (~432 DPI)
- `EMPTY_CELL_THRESHOLD` (0.01) — pixel volume ratio below which a cell is empty
- `SYMBOL_MIN_HEIGHT_RATIO` (0.4) — minimum symbol height relative to cell height

**Business rules:**
- `SCORING_WEIGHTS` — points per question (Q1: 0.5, Q2–9: 1.0, Q10: 1.5)
- `NUM_QUESTIONS` (10) — number of questions on the form
- `CELLS_PER_ROW` — number of input cells per row type

**ML configuration:**
- `CLIP_VISION_MODEL_NAME` — fine-tuned CLIP model identifier
- `CLIP_INPUT_SIZE` (128) — image size for CLIP input

### **5.2 Models (`models/`)**

**Class hierarchy:**

```
Form
 ├── rows: dict[str, Row]     # 23 rows total
 │    ├── "date"               # 8 cells
 │    ├── "user_id"            # 8 cells
 │    ├── "version"            # 4 cells
 │    ├── "answer1"..."answer10"     # 9 cells each
 │    └── "correction1"..."correction10"  # 9 cells each
 │
 Row
 ├── cells: list[Cell]
 ├── user_answers: list[str | None]
 ├── correct_answers: list[str]
 └── row_image: np.ndarray | None
 │
 Cell
 ├── x, y, w, h: int | None     # Coordinates on aligned image
 ├── user_value: str | None      # Recognized value
 └── correct_value: str | None   # Expected value
```

**Design decision:** Rows are stored in `dict[str, Row]` keyed by name (e.g. `form.rows["answer3"]`) instead of dynamic attributes. This is type-safe, iterable, and avoids `setattr`/`getattr`.

### **5.3 Recognition (`recognition/`)**

**`FormRecognition.run_pipeline(mode, api_key, openai_model)`** supports two modes:

**CLIP mode (`mode="clip"`, default):**

1. Load cell coordinates from JSON
2. Load scanned image and template
3. Align image to template (SIFT + homography)
4. Style image (white borders, inversion, symbol centering)
5. Load correct answers from DataFrame
6. Run OCR pipeline (digit recognition, empty detection, special chars)
7. Generate row preview images

**OpenAI mode (`mode="openai"`):**

1. Load coordinates, image, and template (same as CLIP)
2. Align image to template
3. Load correct answers
4. Send full aligned page image to OpenAI Vision API (GPT-4o)
5. Parse structured JSON response and populate `Form` via `set_answers_from_dict()`
6. Generate row preview images

Downstream grading and export pipeline is identical for both modes.

**`DigitClassifier`** (CLIP only) uses lazy loading:
- Models are not loaded at import
- First `predict()` call triggers model load
- Uses CLIP ViT-B/32 fine-tuned on MNIST
- Compares cell image embedding to 10 precomputed reference embeddings
- Returns digit label with highest cosine similarity

**`openai_recognizer`** (OpenAI only):
- Encodes page image as base64 and sends to GPT-4o with structured prompt
- Expects JSON with keys: `date`, `user_id`, `version`, `answer_1`–`answer_10`, `correction_1`–`correction_10`
- Normalizes response to internal row names (e.g. `answer_1` → `answer1`, strips dots from dates)
- Lazy-imports `openai` — only required when OpenAI mode is selected

### **5.4 Processing (`processing/`)**

**Alignment pipeline:**
1. SIFT feature detection on template and scanned image
2. Brute-force matching with cross-check
3. Homography estimation via RANSAC
4. Perspective warp to align scanned image to template coordinates

**Image styling pipeline:**
1. Draw white borders around cell boundaries to isolate content
2. Invert image (white-on-black for contour detection)
3. For each cell: find symbol contour, scale to fill cell, center

### **5.5 Grading (`grading/`)**

**Answer checking logic:**
- A question is correct if `correct_answer == user_answer` OR `correct_answer == user_correction`
- Answers are normalized: commas to dots, convert to float then back to string
- Final output replaces original answers with corrections where corrections exist

### **5.6 Export (`export/`)**

**Excel output includes:**
- Date, participant code, version
- Per-question answers with cell preview images
- Per-question scores and total score (as Excel SUM formula)

## **6. Configuration Management**

All configuration lives in `exam_grader/config.py`. No environment variables or `.env` files — the app is designed for local single-user use.

**Path resolution:** Paths are computed relative to `config.py` using `pathlib.Path`, making the package relocatable.

## **7. Development Setup**

### **7.1 Initial Setup**

```bash
git clone <repository-url>
cd streamlit-exam-form-app
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

### **7.2 Running the Application**

```bash
streamlit run app.py
```

### **7.3 Code Quality**

```bash
pip install -e ".[dev]"

ruff check exam_grader/
ruff format exam_grader/
pytest
```

## **8. Design Patterns & Conventions**

### **8.1 Lazy Loading**

Heavy ML models and optional dependencies load on demand:

- **CLIP model** — wrapped in `DigitClassifier` with module-level convenience function. Models load on first `predict()` call.
- **OpenAI client** — `openai` is imported inside `recognize_form_with_openai()`, so it is only required when OpenAI mode is selected.

```python
_classifier: DigitClassifier | None = None

def predict_digit(image: np.ndarray) -> tuple[str, np.ndarray]:
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier.predict(image)
```

### **8.2 Type Hints**

All public functions and class attributes have type annotations. `from __future__ import annotations` is used for forward references and `X | Y` union syntax on Python 3.10+.

### **8.3 Coding Style**

- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Constants:** `UPPER_SNAKE_CASE` in `config.py`
- **Comments:** English only; explain non-obvious intent only
- **Docstrings:** On all public classes and functions
- **Line length:** 100 characters (in `pyproject.toml`)
- **Imports:** Sorted by isort (via Ruff)

### **8.4 Module Boundaries**

```
config         ← stdlib only
processing     ← config, external libs
recognition    ← config, models, processing
models         ← config, processing, recognition
grading        ← config, external libs
export         ← config, models
```

Feature modules (`grading`, `export`) do not import from each other.
