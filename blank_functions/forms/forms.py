from blank_functions.forms.rows import Row
import cv2
import json
from blank_functions.utils.image_processing import place_row_image_into_form, align_image_pipeline, recalculate_cell, style_image


class Form:
    # Определяем все названия строк, которые хотим использовать
    ROW_NAMES = (
        ["subject", "user_id", "version"]
        + [f"answer{i}" for i in range(1, 11)]
        + [f"correction{i}" for i in range(1, 11)]
    )

    def __init__(self, version_input = None):
        """
        contours — возможно, вам нужно передавать сюда какие-то контуры или
        иной массив данных, но для упрощённого примера можно оставить как есть.
        """
        # Вместо того, чтобы вручную создавать self.subject, self.user_id, ...
        # делаем это в цикле.

        for row_name in self.ROW_NAMES:
            setattr(self, row_name, Row(row_name))
        
        self.image = None
        self.template = None
        self.answer_minus_list = []
        self.correction_minus_list = []
        self.version_input = version_input


    def __repr__(self):
        # Тоже можно собрать строку динамически
        # (но если вам удобнее - оставьте вручную)
        rows_repr = []
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            rows_repr.append(f"{row_name}={row_obj}")
        rows_repr_str = ", ".join(rows_repr)
        return f"Form({rows_repr_str})"

    def load_meta_from_json(self, json_path):
        """
        Читаем файл JSON и заполняем все поля (строки).
        Структура JSON должна быть такая, чтобы для каждого row
        (subject, user_id, version, answer1..answer10, correction1..correction10)
        были данные по cell1..cellN (зависит от Row).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row_name in self.ROW_NAMES:
            if row_name in data:
                row_obj = getattr(self, row_name)  # Получаем экземпляр Row
                row_obj.load_data(data[row_name])

    @staticmethod
    def number_to_list(number):
        # Преобразуем число в строку
        number_str = str(number)

        # Преобразуем строку в список символов
        result = list(number_str.replace('.', ','))

        return result

    def load_correct_answers(self, answers):
        version = int(self.version_input)
        answers = answers[answers['Вариант'] == version].reset_index(drop=True)

        for i in range(1, 11):
            answers[i] = answers[i].astype(str)

        # Append correct answers and update cell values
        for i in range(1, 11):
            answer_attr = getattr(self, f'answer{i}')
            correct_answer = answers.iloc[0, i]
            answer_attr.correct_answers.append(correct_answer)
            
            answers_cells = self.number_to_list(correct_answer)
            for j, cell_value in enumerate(answers_cells):
                answer_attr.cells[j].value = cell_value

    def get_symbals_from_image(self):
        image = self.image
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                cell.detect_symbol_in_cell(image)

    def get_sybmol_row(self):
        original_image = self.image
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            place_row_image_into_form(row_obj, original_image)

        # cv2.imwrite("modified_form.png", original_image)

    def remove_cells_lines(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                scale = 40
                scale_2 = 15
                scale_3 = 10
                scale_4 = 12
                self.image[y - scale : y + scale_3, x - scale_2 : x + w + scale_2] = 255 # нижняя горизонтальная линия
                self.image[y + h - scale_3 : y + h + scale, x - scale_2 : x + w + scale_2] = 255 # верхняя горизонтальная лини
                self.image[y - scale_2 : y + h + scale_2, x - scale_4 : x + scale_4] = 255 # левая вертикальная линия
                self.image[y - scale_2 : y + h + scale_2, x + w - scale_4 : x + w + scale_4] = 255 # правая вертикальная линия
        # cv2.imwrite("removed_cells_lines.png", self.image)

    def load_image(self, image):
        self.image = image
    
    def load_template(self, template_path):
        self.template = cv2.imread(template_path)

    def align_form(self, scale_factor = 0.25):
        aligned_image = align_image_pipeline(self.image, self.template, scale_factor)
        self.image = aligned_image.copy()
    
    def recalculate_cells(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell.x, cell.y, cell.w, cell.h = recalculate_cell(self.image, (x, y, w, h))

    def style_image(self):
        cells = []
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cells.append((x, y, w, h))
        self.image = style_image(self.image, cells)

    def visualize_form(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite("visualized_form.png", self.image)
    
    def get_rows_contour(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            row_obj.get_contour()

    def get_symbol_values(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                if cell.symbols[0].symbol_image is not None:
                    cell.symbols[0].value = cell.symbols[0].get_highest_similarity()

    def get_minus_lists(self):
        for row_name in [f"answer{i}" for i in range(1, 11)]:
            row_obj = getattr(self, row_name)
            if row_obj.cells[0].symbols[0].symbol_image is not None:
                symbol_img_shape = row_obj.cells[0].symbols[0].symbol_image.shape
                if symbol_img_shape[0]/symbol_img_shape[1] < 0.75:
                    self.answer_minus_list.append(-1)
                else:
                    self.answer_minus_list.append(1)
            else:
                self.answer_minus_list.append(1)
        
        for row_name in [f"correction{i}" for i in range(1, 11)]:
            row_obj = getattr(self, row_name)
            if row_obj.cells[0].symbols[0].symbol_image is not None:
                symbol_img_shape = row_obj.cells[0].symbols[0].symbol_image.shape
                if symbol_img_shape[0]/symbol_img_shape[1] < 0.75:
                    self.correction_minus_list.append(-1)
                else:
                    self.correction_minus_list.append(1)
            else:
                self.correction_minus_list.append(1)

    @property
    def answer_rows(self):
        return [getattr(self, f"answer{i}") for i in range(1, 11)]
    
    @property
    def correction_rows(self):
        return [getattr(self, f"correction{i}") for i in range(1, 11)]