from __future__ import annotations

import json
from typing import TYPE_CHECKING

import cv2
import numpy as np

from exam_grader.config import (
    DEFAULT_ROW_IMAGE_CELLS,
    EMPTY_CELL_THRESHOLD,
    NUM_QUESTIONS,
    ROW_IMAGE_CELLS,
    ROW_NAMES,
)
from exam_grader.models.row import Row
from exam_grader.processing.image_processing import align_image_pipeline, style_image
from exam_grader.recognition.digit_classifier import predict_digit

if TYPE_CHECKING:
    import pandas as pd


class Form:
    """
    Represents a single exam answer form with rows for metadata
    (date, user_id, version), answers (answer1-10), and corrections (correction1-10).
    """

    def __init__(self, version_input: int | None = None) -> None:
        self.rows: dict[str, Row] = {name: Row(name) for name in ROW_NAMES}
        self.image: np.ndarray | None = None
        self.raw_image: np.ndarray | None = None
        self.template: np.ndarray | None = None
        self.version_input = version_input

    def __repr__(self) -> str:
        rows_repr = ", ".join(f"{name}={row}" for name, row in self.rows.items())
        return f"Form({rows_repr})"

    # --- Data loading ---

    def load_meta_from_json(self, json_path: str) -> None:
        """Load cell coordinates for all rows from a JSON configuration file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row_name, row in self.rows.items():
            if row_name in data:
                row.load_data(data[row_name])

    def load_correct_answers(self, answers: pd.DataFrame) -> None:
        """Load correct answers from a DataFrame filtered by the form's version."""
        version = int(self.version_input)
        filtered = answers[answers["Вариант"] == version].reset_index(drop=True)
        for i in range(1, NUM_QUESTIONS + 1):
            filtered[i] = filtered[i].astype(str)

        for i in range(1, NUM_QUESTIONS + 1):
            answer_row = self.rows[f"answer{i}"]
            correction_row = self.rows[f"correction{i}"]
            correct_answer = filtered.iloc[0, i]

            answer_row.correct_answers.append(correct_answer)
            correction_row.correct_answers.append(correct_answer)

            answer_chars = self._number_to_char_list(correct_answer)
            for j, char in enumerate(answer_chars):
                answer_row.cells[j].correct_value = char
                correction_row.cells[j].correct_value = char

    def load_image(self, image: np.ndarray) -> None:
        """Set the scanned form image."""
        self.image = image
        self.raw_image = image.copy()

    def load_template(self, template_path: str) -> None:
        """Load the reference template image for alignment."""
        self.template = cv2.imread(template_path)

    # --- Image processing ---

    def align_form(self, scale_factor: float = 1.0) -> None:
        """Align the scanned image to the template using feature matching."""
        aligned = align_image_pipeline(self.image, self.template, scale_factor)
        self.image = aligned.copy()
        self.raw_image = aligned.copy()

    def style_image(self) -> None:
        """Apply border whitening, inversion, and symbol centering to all cells."""
        cells = []
        for row in self.rows.values():
            for cell in row.cells:
                cells.append((cell.x, cell.y, cell.w, cell.h))
        self.image = style_image(self.image, cells)

    # --- OCR recognition ---

    def get_user_answers_pipeline(self) -> None:
        """Run the full answer recognition pipeline: OCR, empty detection, special chars."""
        self._recognize_digits()
        self._detect_empty_cells()
        self._detect_all_minuses()
        self._detect_all_commas()
        self._collect_row_answers()

    def _recognize_digits(self) -> None:
        """Run CLIP-based digit recognition on every cell."""
        for row in self.rows.values():
            for cell in row.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell_image = cv2.cvtColor(self.image[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
                predicted, pred_input = predict_digit(cell_image)
                if predicted == "1 (with a thin vertical line)":
                    predicted = "1"
                if predicted == "7 (with a flat top part)":
                    predicted = "7"
                cell.user_value = predicted
                cell.cell_pred_input = pred_input

    def _detect_empty_cells(self) -> None:
        """Mark cells with very low pixel volume as empty."""
        for row in self.rows.values():
            for cell in row.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell_image = self.image[y : y + h, x : x + w]
                volume = np.sum(cell_image)
                if volume < EMPTY_CELL_THRESHOLD * 255 * w * h:
                    cell.user_value = None

    def _extract_cell_image(self, cell) -> np.ndarray:
        """Extract the image region corresponding to a cell."""
        x, y, w, h = cell.x, cell.y, cell.w, cell.h
        return self.image[y : y + h, x : x + w]

    def _detect_minuses_in_row(self, row: Row) -> None:
        """Detect minus sign: first cell empty, second cell has content."""
        img0 = self._extract_cell_image(row.cells[0])
        img1 = self._extract_cell_image(row.cells[1])
        if np.sum(img0) == 0 and np.sum(img1) > 0:
            row.cells[0].user_value = "-"

    def _detect_commas_in_row(self, row: Row) -> None:
        """Detect commas: an empty cell between two filled cells."""
        sums = [np.sum(self._extract_cell_image(cell)) for cell in row.cells]
        for i in range(1, len(row.cells) - 1):
            if sums[i] == 0 and sums[i - 1] > 0 and sums[i + 1] > 0:
                row.cells[i].user_value = ","

    def _detect_all_minuses(self) -> None:
        """Detect minus signs in all answer and correction rows."""
        for i in range(1, NUM_QUESTIONS + 1):
            self._detect_minuses_in_row(self.rows[f"answer{i}"])
            self._detect_minuses_in_row(self.rows[f"correction{i}"])

    def _detect_all_commas(self) -> None:
        """Detect commas in all answer and correction rows."""
        for i in range(1, NUM_QUESTIONS + 1):
            self._detect_commas_in_row(self.rows[f"answer{i}"])
            self._detect_commas_in_row(self.rows[f"correction{i}"])

    def _collect_row_answers(self) -> None:
        """Aggregate individual cell values into row-level answer lists."""
        for row in self.rows.values():
            row.user_answers = [cell.user_value for cell in row.cells]
            row.correct_answers = [cell.correct_value for cell in row.cells]

    # --- Row image generation ---

    def _get_row_image(self, row: Row) -> np.ndarray:
        """Generate a concatenated preview image from a row's first N cells."""
        num_cells = ROW_IMAGE_CELLS.get(row.row_name, DEFAULT_ROW_IMAGE_CELLS)
        cell_images = []
        for i, cell in enumerate(row.cells):
            if i >= num_cells:
                break
            x, y, w, h = cell.x, cell.y, cell.w, cell.h
            cell_images.append(self.raw_image[y : y + h, x : x + w])
        row_image = np.concatenate(cell_images, axis=1)
        return cv2.resize(row_image, (64, 16), interpolation=cv2.INTER_AREA)

    def set_row_images(self) -> None:
        """Generate preview images for all rows. Uses correction image if correction exists."""
        for name in ("user_id", "version", "date"):
            self.rows[name].row_image = self._get_row_image(self.rows[name])

        for i in range(1, NUM_QUESTIONS + 1):
            answer_row = self.rows[f"answer{i}"]
            correction_row = self.rows[f"correction{i}"]
            answer_row.row_image = self._get_row_image(answer_row)
            correction_row.row_image = self._get_row_image(correction_row)
            if correction_row.cells[0].user_value is not None:
                answer_row.row_image = correction_row.row_image.copy()

    # --- External answer injection (for OpenAI mode) ---

    def set_answers_from_dict(self, data: dict[str, str]) -> None:
        """
        Populate cell values and row answers from a flat dictionary
        (e.g. from OpenAI recognition). Each value string is split into
        individual characters and distributed across cells.
        """
        for row_name in ROW_NAMES:
            if row_name not in data:
                continue
            value = data[row_name]
            row = self.rows[row_name]
            chars: list[str | None] = list(value) if value else []
            num_cells = len(row.cells)
            padded = chars[:num_cells] + [None] * max(0, num_cells - len(chars))
            for j, cell in enumerate(row.cells):
                cell.user_value = padded[j]
            row.user_answers = list(padded)

    # --- Properties ---

    @property
    def answer_rows(self) -> list[Row]:
        return [self.rows[f"answer{i}"] for i in range(1, NUM_QUESTIONS + 1)]

    @property
    def correction_rows(self) -> list[Row]:
        return [self.rows[f"correction{i}"] for i in range(1, NUM_QUESTIONS + 1)]

    # --- Utilities ---

    @staticmethod
    def _number_to_char_list(number: str | float) -> list[str]:
        """Convert a number to a list of characters, replacing '.' with ','."""
        return list(str(number).replace(".", ","))
