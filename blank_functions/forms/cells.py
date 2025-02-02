from blank_functions.forms.symbols import Symbol
import cv2
import numpy as np

class Cell:
    def __init__(self, row_name):
        """
        Класс, представляющий одну клетку.
        
        :param row_name: Название строки, к которой относится клетка
        """
        self.row_name = row_name
        self.cell_number = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.value = None
        self.cell_number = None
        self.symbols = []
        self.cell_image = None

    def __repr__(self):
        return (f"Cell(row_name={self.row_name}, cell_number={self.cell_number}, "
                f"x={self.x}, y={self.y}, w={self.w}, h={self.h}, "
                f"value={self.value}, symbols={self.symbols})")

    def detect_symbol_in_cell(self, cell_image):
        self.symbols = []
        x_cell, y_cell = int(self.x or 0), int(self.y or 0)
        w_cell, h_cell = int(self.w or 0), int(self.h or 0)

        if w_cell == 0 or h_cell == 0:
            self.symbols.append(Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None))
            return

        cell_img = cv2.medianBlur(cell_image[y_cell:y_cell + h_cell, x_cell:x_cell + w_cell], 3).copy()
        _, thresh = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY_INV)

        if np.max(thresh) == 0:
            self.symbols.append(Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None))
            return

        non_zero_coords = np.where(thresh != 0)
        x_min, y_min = np.min(non_zero_coords[1]), np.min(non_zero_coords[0])
        x_max, y_max = np.max(non_zero_coords[1]), np.max(non_zero_coords[0])

        symbol_crop = thresh[y_min:y_max, x_min:x_max].copy()
        symbol_crop_vol_per_pixel = np.sum(symbol_crop) / (w_cell * h_cell)

        if x_max - x_min > 7 and y_max - y_min > 7 and symbol_crop_vol_per_pixel > 0.05:
            self.symbols.append(Symbol(x=x_min, y=y_min, w=x_max-x_min, h=y_max-y_min, value=None, symbol_image=symbol_crop))
        else:
            self.symbols.append(Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None))

    def load_from_dict(self, cell_dict):
        """
        Заполняем поля клетки из словаря (обычно загружается из JSON).
        """
        self.x = cell_dict.get("x")
        self.y = cell_dict.get("y")
        self.w = cell_dict.get("w")
        self.h = cell_dict.get("h")
        self.value = cell_dict.get("value")
        self.cell_number = cell_dict.get("cell_number")
