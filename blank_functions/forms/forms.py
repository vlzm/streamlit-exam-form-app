from blank_functions.forms.rows import Row
import cv2
import numpy as np
import json
from blank_functions.utils.image_processing import place_row_image_into_form, align_image_pipeline, recalculate_cell, style_image
from blank_functions.forms.model import predict_digit
import matplotlib.pyplot as plt
import pickle
class Form:
    # Определяем все названия строк, которые хотим использовать
    ROW_NAMES = (
        ["date", "user_id", "version"]
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
        self.raw_image = None
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
            correction_attr = getattr(self, f'correction{i}')
            correct_answer = answers.iloc[0, i]
            answer_attr.correct_answers.append(correct_answer)
            correction_attr.correct_answers.append(correct_answer)
            

            answers_cells = self.number_to_list(correct_answer)
            for j, cell_value in enumerate(answers_cells):
                answer_attr.cells[j].correct_value = cell_value
                correction_attr.cells[j].correct_value = cell_value

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
        self.raw_image = image.copy()
    
    def load_template(self, template_path):
        self.template = cv2.imread(template_path)

    def align_form(self, scale_factor = 0.25):
        aligned_image = align_image_pipeline(self.image, self.template, scale_factor)
        self.image = aligned_image.copy()
        self.raw_image = aligned_image.copy()
    
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

    def get_user_answers(self):
        for row_name in self.ROW_NAMES:
            for cell in getattr(self, row_name).cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell_image = self.image[y:y+h, x:x+w]
                cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
                predicted_digit, cell_pred_input = predict_digit(cell_image)
                if predicted_digit == '1 (with a thin vertical line)':
                    predicted_digit = '1'
                if predicted_digit == '7 (with a flat top part)':
                    predicted_digit = '7'
                cell.user_value = predicted_digit
                cell.cell_pred_input = cell_pred_input

    


    def get_user_answers_pipeline(self):
        self.get_user_answers()
        # self.check_digits_and_replace()
        self.get_empty_cells()
        self.get_correct_minuses()
        self.get_correct_commas()
        self.get_user_answers_rows()



    def check_digits_and_replace(self):
        for i in range(1, 11):
            answer_row = getattr(self, f"answer{i}")
            correction_row = getattr(self, f"correction{i}")
            
            check_flag_answer = True
            check_flag_correction = True
            
            # check answer row
            for cell in answer_row.cells:
                if cell.correct_value in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    if cell.user_value != cell.correct_value:   
                        check_flag_answer = False
                        break
            if check_flag_answer:
                for cell in answer_row.cells:
                    cell.user_value = cell.correct_value    
            
            # check correction row
            for cell in correction_row.cells:
                if cell.correct_value in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    if cell.user_value != cell.correct_value:   
                        check_flag_correction = False
                        break
            if check_flag_correction:
                for cell in correction_row.cells:
                    cell.user_value = cell.correct_value


    def get_user_answers_rows(self):
        for row_name in self.ROW_NAMES:
            row_values = []
            row_correct_values = []
            row = getattr(self, row_name)
            for cell in row.cells:
                row_values.append(cell.user_value)
                row_correct_values.append(cell.correct_value)
            row.user_answers = row_values
            row.correct_answers = row_correct_values

    def get_empty_cells(self):
        for row_name in self.ROW_NAMES:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell_image = self.image[y:y+h, x:x+w]
                volume = np.sum(cell_image)
                if volume < 0.01 * 255 * w * h:
                    cell.user_value = None

    def extract_cell_image(self, image, cell):
        x, y, w, h = cell.x, cell.y, cell.w, cell.h
        return self.image[y:y+h, x:x+w]

    def get_largest_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def should_mark_as_minus(self, image, contour):
        if contour is None:
            return False
        x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
        contour_volume = cv2.contourArea(contour)
        # image_volume = image.shape[0] * image.shape[1]
        # ratio = contour_volume / image_volume
        h_cell = image.shape[1]
        return h_contour < 0.4*h_cell    

    def should_mark_as_comma(self, image, contour):
        if contour is None:
            return False
        x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
        contour_volume = cv2.contourArea(contour)
        # image_volume = image.shape[0] * image.shape[1]
        # ratio = contour_volume / image_volume
        h_cell = image.shape[1]
        return h_contour > 0.4*h_cell



    def process_row_minuses(self, row):
        cell0 = row.cells[0]
        cell0_image = self.extract_cell_image(self.image, cell0)
        cell1 = row.cells[1]
        cell1_image = self.extract_cell_image(self.image, cell1)
        if np.sum(cell0_image) == 0 and np.sum(cell1_image) > 0:
            cell0.user_value = '-'


    def process_row_commas(self, row):
        cell0 = row.cells[0]
        cell0_image = self.extract_cell_image(self.image, cell0)
        cell1 = row.cells[1]
        cell1_image = self.extract_cell_image(self.image, cell1)
        cell2 = row.cells[2]
        cell2_image = self.extract_cell_image(self.image, cell2)
        cell3 = row.cells[3]
        cell3_image = self.extract_cell_image(self.image, cell3)
        cell4 = row.cells[4]
        cell4_image = self.extract_cell_image(self.image, cell4)
        cell5 = row.cells[5]
        cell5_image = self.extract_cell_image(self.image, cell5)
        cell6 = row.cells[6]
        cell6_image = self.extract_cell_image(self.image, cell6)
        cell7 = row.cells[7]
        cell7_image = self.extract_cell_image(self.image, cell7)
        cell8 = row.cells[8]
        cell8_image = self.extract_cell_image(self.image, cell8)

        if np.sum(cell1_image) == 0 and np.sum(cell0_image) > 0 and np.sum(cell2_image) > 0:
            cell1.user_value = ','

        if np.sum(cell2_image) == 0 and np.sum(cell1_image) > 0 and np.sum(cell3_image) > 0:
            cell2.user_value = ','

        if np.sum(cell3_image) == 0 and np.sum(cell2_image) > 0 and np.sum(cell4_image) > 0:
            cell3.user_value = ','

        if np.sum(cell4_image) == 0 and np.sum(cell3_image) > 0 and np.sum(cell5_image) > 0:
            cell4.user_value = ','

        if np.sum(cell5_image) == 0 and np.sum(cell4_image) > 0 and np.sum(cell6_image) > 0:
            cell5.user_value = ','

        if np.sum(cell6_image) == 0 and np.sum(cell5_image) > 0 and np.sum(cell7_image) > 0:
            cell6.user_value = ','

        if np.sum(cell7_image) == 0 and np.sum(cell6_image) > 0 and np.sum(cell8_image) > 0:
            cell7.user_value = ','


    def get_correct_minuses(self):
        for i in range(1, 11):
            self.process_row_minuses(getattr(self, f"answer{i}"))
        for i in range(1, 11):
            self.process_row_minuses(getattr(self, f"correction{i}"))

    def get_correct_commas(self):
        for i in range(1, 11):
            self.process_row_commas(getattr(self, f"answer{i}"))
        for i in range(1, 11):
            self.process_row_commas(getattr(self, f"correction{i}"))

    def get_row_image(self, answer):
        num_cells = 5
        if answer.row_name == "date":
            num_cells = 8
        row_images = list()
        for cell, i in zip(answer.cells, range(1, 11)):
            if i <=num_cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cell_image = self.raw_image[y:y+h, x:x+w]
                row_images.append(cell_image)
        # объединить изображения в одно
        row_image = np.concatenate(row_images, axis=1)
        row_image = cv2.resize(row_image, (64, 16), interpolation=cv2.INTER_AREA)
        return row_image

    def set_row_images(self):
        for row in ["user_id", "version", "date"]:
            cur_row = getattr(self, row)
            # save to pickle
            with open(f"{row}.pkl", "wb") as f:
                pickle.dump(cur_row, f)
            getattr(self, row).row_image = self.get_row_image(getattr(self, row))
        for j in range(1, 11):
            answer_row = getattr(self, f"answer{j}")

            correction_row = getattr(self, f"correction{j}")
            answer_row.row_image = self.get_row_image(answer_row)
            correction_row.row_image = self.get_row_image(correction_row)
            if correction_row.cells[0].user_value != None:
                answer_row.row_image = correction_row.row_image.copy()





    @property
    def answer_rows(self):
        return [getattr(self, f"answer{i}") for i in range(1, 11)]
    
    @property
    def correction_rows(self):
        return [getattr(self, f"correction{i}") for i in range(1, 11)]