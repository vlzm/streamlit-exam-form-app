from __future__ import annotations

import numpy as np


class Cell:
    """Single cell within a form row, holding recognized and correct values."""

    def __init__(self, row_name: str) -> None:
        self.row_name = row_name
        self.cell_number: int | None = None
        self.x: int | None = None
        self.y: int | None = None
        self.w: int | None = None
        self.h: int | None = None
        self.user_value: str | None = None
        self.correct_value: str | None = None
        self.cell_image: np.ndarray | None = None
        self.cell_pred_input: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"Cell(row_name={self.row_name}, cell_number={self.cell_number}, "
            f"x={self.x}, y={self.y}, w={self.w}, h={self.h}, "
            f"user_value={self.user_value})"
        )

    def load_from_dict(self, cell_dict: dict) -> None:
        """Populate cell coordinates and metadata from a JSON-loaded dictionary."""
        self.x = cell_dict.get("x")
        self.y = cell_dict.get("y")
        self.w = cell_dict.get("w")
        self.h = cell_dict.get("h")
        self.user_value = cell_dict.get("user_value")
        self.cell_number = cell_dict.get("cell_number")
