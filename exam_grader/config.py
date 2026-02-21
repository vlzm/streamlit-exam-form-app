from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

TEMPLATE_PATH = DATA_DIR / "template_raw.jpg"
CELL_COORDS_PATH = DATA_DIR / "rows_data_new_format.json"
REF_PICS_DIR = DATA_DIR / "ref_pics"
SAVED_EXCELS_DIR = PROJECT_ROOT / "saved_excels"

PDF_ZOOM = 6.0
ALIGNMENT_SCALE = 1.0

# Minimum pixel volume ratio to consider a cell non-empty
EMPTY_CELL_THRESHOLD = 0.01

# Minimum symbol bounding box dimensions (pixels)
SYMBOL_MIN_SIZE = 7

# Minimum symbol volume ratio relative to cell area
SYMBOL_MIN_VOLUME_RATIO = 0.05

# Symbol height must be at least this fraction of cell height
SYMBOL_MIN_HEIGHT_RATIO = 0.4

NUM_QUESTIONS = 10

# Points awarded per question
SCORING_WEIGHTS: dict[int, float] = {
    1: 0.5,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 1.5,
}

# Number of cells per row type
CELLS_PER_ROW: dict[str, int] = {
    "date": 8,
    "user_id": 8,
    "version": 4,
}
DEFAULT_CELLS_PER_ROW = 9

ROW_NAMES: list[str] = (
    ["date", "user_id", "version"]
    + [f"answer{i}" for i in range(1, NUM_QUESTIONS + 1)]
    + [f"correction{i}" for i in range(1, NUM_QUESTIONS + 1)]
)

# Number of cells to include in row preview image
ROW_IMAGE_CELLS: dict[str, int] = {
    "date": 8,
}
DEFAULT_ROW_IMAGE_CELLS = 5

# Mapping from internal names to Russian display names for Excel output
COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "date": "Дата",
    "user_id": "Код участника",
    "version": "Вариант",
    **{f"answer{i}": f"Задание {i}" for i in range(1, NUM_QUESTIONS + 1)},
    **{f"correction{i}": f"Замена {i}" for i in range(1, NUM_QUESTIONS + 1)},
}

# CLIP digit recognition label mapping
DIGIT_LABELS: dict[str, str] = {
    "comma": ",",
    "minus": "-",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "zero": "0",
    "empty": "",
}

REF_DIGIT_NAMES: list[str] = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
]

CLIP_VISION_MODEL_NAME = "tanganke/clip-vit-base-patch32_mnist"
CLIP_PROCESSOR_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INPUT_SIZE = 128
