from __future__ import annotations

import numpy as np


class Symbol:
    """Recognized symbol within a cell, with local coordinates and image crop."""

    def __init__(
        self,
        x: int | None,
        y: int | None,
        w: int | None,
        h: int | None,
        value: str | None = None,
        symbol_image: np.ndarray | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.value = value
        self.symbol_image = symbol_image

    def __repr__(self) -> str:
        return f"Symbol(x={self.x}, y={self.y}, w={self.w}, h={self.h}, value={self.value})"
