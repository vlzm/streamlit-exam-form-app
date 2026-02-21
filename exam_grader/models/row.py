from __future__ import annotations

import numpy as np

from exam_grader.config import CELLS_PER_ROW, DEFAULT_CELLS_PER_ROW
from exam_grader.models.cell import Cell


class Row:
    """Row of cells within a form (e.g. a single answer line or metadata field)."""

    def __init__(self, row_name: str) -> None:
        self.row_name = row_name
        self.x: int | None = None
        self.y: int | None = None
        self.w: int | None = None
        self.h: int | None = None
        num_cells = CELLS_PER_ROW.get(row_name, DEFAULT_CELLS_PER_ROW)
        self.cells: list[Cell] = [Cell(row_name) for _ in range(num_cells)]
        self.correct_answers: list[str] = []
        self.user_answers: list[str | None] = []
        self.row_image: np.ndarray | None = None

    def get_contour(self) -> tuple[int, int, int, int] | None:
        """Compute the bounding box encompassing all cells in this row."""
        if not self.cells:
            return None

        x_min = min(c.x for c in self.cells)
        y_min = min(c.y for c in self.cells)
        x_max = max(c.x + c.w for c in self.cells)
        y_max = max(c.y + c.h for c in self.cells)

        self.x = x_min
        self.y = y_min - 15
        self.w = (x_max - x_min) + 15
        self.h = (y_max - y_min) + 15
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def load_data(self, row_dict: dict) -> None:
        """Load cell coordinates from a dictionary (parsed from JSON config)."""
        for i in range(len(self.cells)):
            cell_key = f"cell{i}"
            if cell_key in row_dict:
                self.cells[i].load_from_dict(row_dict[cell_key])

    def __repr__(self) -> str:
        return f"Row('{self.row_name}', cells={self.cells})"
