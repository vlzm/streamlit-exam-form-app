from blank_functions.forms.cells import Cell

class Row:
    def __init__(self, row_name):
        self.row_name = row_name
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        # Сразу создаём 10 объектов Cell
        # (каждому можно проставить cell_number = i)
        if row_name == "subject":
            self.cells = [Cell(row_name) for _ in range(10)]
        elif row_name == "user_id":
            self.cells = [Cell(row_name) for _ in range(8)]
        elif row_name == "version":
            self.cells = [Cell(row_name) for _ in range(4)]
        else:
            self.cells = [Cell(row_name) for _ in range(9)]
        self.correct_answers = []

    def get_contour(self):
        """
        Возвращает (x, y, w, h) — bounding box по всей строке,
        делая полный проход по cells.
        """
        if not self.cells:
            return None  # или (0,0,0,0)

        x_coords = [cell.x for cell in self.cells]
        y_coords = [cell.y for cell in self.cells]
        x_right_edges = [cell.x + cell.w for cell in self.cells]
        y_bottom_edges = [cell.y + cell.h for cell in self.cells]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_right_edges)
        y_max = max(y_bottom_edges)

        w = x_max - x_min
        h = y_max - y_min
        self.x = x_min
        self.y = y_min - 15
        self.w = w + 15
        self.h = h + 15
        return (x_min, y_min, w, h)

    def load_data(self, row_dict):
        """
        row_dict ожидается вида:
        {
          "cell1": { "x": ..., "y": ... },
          "cell2": { "x": ..., "y": ... },
          ...
        }
        """
        for i in range(10):
            cell_key = f"cell{i}"  # cell1..cell10
            if cell_key in row_dict:
                self.cells[i].load_from_dict(row_dict[cell_key])

    def __repr__(self):
        return f"Row('{self.row_name}', cells={self.cells})"