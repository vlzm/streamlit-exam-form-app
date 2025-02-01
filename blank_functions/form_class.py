import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import time
class Symbol:
    """
    Класс символа с координатами и распознанным значением.
    Координаты (x, y, w, h) можно трактовать как локальные
    относительно выреза ячейки, или как глобальные — 
    по вашему выбору.
    """
    def __init__(self, x, y, w, h, value=None, symbol_image=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.value = value
        self.symbol_image = symbol_image
    
    @staticmethod
    def prepare_ref_pics():
        one_pic = cv2.imread("ref_pics/one_pic.png", cv2.IMREAD_GRAYSCALE)
        two_pic = cv2.imread("ref_pics/two_pic.png", cv2.IMREAD_GRAYSCALE)
        three_pic = cv2.imread("ref_pics/three_pic.png", cv2.IMREAD_GRAYSCALE)
        four_pic = cv2.imread("ref_pics/four_pic.png", cv2.IMREAD_GRAYSCALE)
        five_pic = cv2.imread("ref_pics/five_pic.png", cv2.IMREAD_GRAYSCALE)
        six_pic = cv2.imread("ref_pics/six_pic.png", cv2.IMREAD_GRAYSCALE)
        seven_pic = cv2.imread("ref_pics/seven_pic.png", cv2.IMREAD_GRAYSCALE)
        eight_pic = cv2.imread("ref_pics/eight_pic.png", cv2.IMREAD_GRAYSCALE)
        nine_pic = cv2.imread("ref_pics/nine_pic.png", cv2.IMREAD_GRAYSCALE)
        zero_pic = cv2.imread("ref_pics/zero_pic.png", cv2.IMREAD_GRAYSCALE)
        minus_pic = cv2.imread("ref_pics/minus_pic.png", cv2.IMREAD_GRAYSCALE)
        comma_pic = cv2.imread("ref_pics/comma_pic.png", cv2.IMREAD_GRAYSCALE)

        ref_dict = {
            "1": one_pic,
            "2": two_pic,
            "3": three_pic,
            "4": four_pic,
            "5": five_pic,
            "6": six_pic,
            "7": seven_pic,
            "8": eight_pic,
            "9": nine_pic,
            "0": zero_pic,
            "-": minus_pic,
            ",": comma_pic,
        }
        return ref_dict

    def get_highest_similarity(self):
        ref_dict = Symbol.prepare_ref_pics()
        max_similarity = 0
        max_similarity_symbol = None
        for k, v in ref_dict.items():
            # Загружаем изображения
            template_image = cv2.resize(v, (self.symbol_image.shape[1], self.symbol_image.shape[0]))
            # Вычисляем SSIM
            similarity, _ = ssim(self.symbol_image, template_image, full=True)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_symbol = k
        return max_similarity_symbol

    def __repr__(self):
        return f"Symbol(x={self.x}, y={self.y}, w={self.w}, h={self.h}, value={self.value})"

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
    
    # def detect_symbol_in_cell_last(self, image):
    #     symbol_crop = np.zeros((1, 1))

    #     x_cell = int(self.x or 0)
    #     y_cell = int(self.y or 0)
    #     w_cell = int(self.w or 0)
    #     h_cell = int(self.h or 0)

    #     cell_img = image[y_cell:y_cell + h_cell, x_cell:x_cell + w_cell]
    #     # _, thresh = cv2.threshold(cell_img, 180, 255, cv2.THRESH_BINARY)
    #     # cell_img = thresh
    #     self.cell_image = cell_img.copy()



    #     gray = cell_img
    #     _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    #     # plt.imshow(thresh, cmap='gray')
    #     # plt.show()

    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    #     self.symbols = []
    #     if len(contours) > 0:
    #         # Предположим, что самый крупный контур — это наш символ
    #         cnt = max(contours, key=cv2.contourArea)
    #         x_sym, y_sym, w_sym, h_sym = cv2.boundingRect(cnt)
    #         if w_sym > 5 and h_sym > 5:

    #             # OCR (опционально)
    #             symbol_crop = gray[y_sym:y_sym+h_sym, x_sym:x_sym+w_sym].copy()
    #             symbol = Symbol(x=x_sym, y=y_sym, w=w_sym, h=h_sym, value=None, symbol_image=symbol_crop)
    #             self.symbols.append(symbol)
    #         else:
    #             self.symbols = [Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None)]
    #     else:
    #         self.symbols = [Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None)]

    def detect_symbol_in_cell(self, image):
        symbol_crop = np.zeros((1, 1))
        self.symbols = []
        x_cell = int(self.x or 0)
        y_cell = int(self.y or 0)
        w_cell = int(self.w or 0)
        h_cell = int(self.h or 0)
        if w_cell == 0 or h_cell == 0:
            self.symbols.append(Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None))
        else:
            cell_img = image[y_cell:y_cell + h_cell, x_cell:x_cell + w_cell]
            cell_img = cv2.medianBlur(cell_img, 3).copy()  

            _, thresh = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY_INV)
            if np.max(thresh) == 0:
                self.symbols.append(Symbol(x=None, y=None, w=None, h=None, value=None, symbol_image=None))
                return
            x_min = np.min(np.where(thresh != 0)[1])
            y_min = np.min(np.where(thresh != 0)[0])
            x_max = np.max(np.where(thresh != 0)[1])
            y_max = np.max(np.where(thresh != 0)[0])

            symbol_crop = thresh[y_min:y_max, x_min:x_max].copy()
            symbol_crop_vol = np.sum(symbol_crop)
            symbol_crop_vol_per_pixel = symbol_crop_vol / ((w_cell) * (h_cell))
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

class Form:
    # Определяем все названия строк, которые хотим использовать
    ROW_NAMES = (
        ["subject", "user_id", "version"]
        + [f"answer{i}" for i in range(1, 11)]
        + [f"correction{i}" for i in range(1, 11)]
    )

    def __init__(self, contours=None):
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
            # print(data) 

        # Вместо повторяющегося if "answer1" in data: self.answer1.load_data(...)
        # можно пройтись по всем row_name:
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
        version = self.version.cells[0].value
        version = 1
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
        for row_name in ([f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]):
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                cell.detect_symbol_in_cell(image)

    def get_sybmol_row(self):
        original_image = self.image
        for row_name in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
            row_obj = getattr(self, row_name)
            place_row_image_into_form(row_obj, original_image)

        # cv2.imwrite("modified_form.png", original_image)

    def remove_cells_lines(self):
        for row_name in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
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
        aligned_image = get_aligned_pic(self.template, self.image, scale_factor)
        self.image = aligned_image.copy()

    def visualize_form(self):
        for row_name in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
            row_obj = getattr(self, row_name)
            for cell in row_obj.cells:
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite("visualized_form.png", self.image)
    
    def get_rows_contour(self):
        for row_name in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
            row_obj = getattr(self, row_name)
            row_obj.get_contour()

    def get_symbol_values(self):
        for row_name in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
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
        
    def run_pipeline(self, image, template_path, json_path, answers):
        form = Form()
        form.load_meta_from_json(json_path)
        form.load_correct_answers(answers)
        form.load_image(image)
        form.load_template(template_path)
        form.align_form(scale_factor = 0.25)
        form.remove_cells_lines()
        form.get_symbals_from_image()
        form.get_rows_contour()
        # form.get_symbol_values()
        # form.get_minus_lists()
        form.get_sybmol_row()

        return form
def rebuild_row_image(row_obj, original_image, symbol_spacing=0):
    """
    Собираем "слитное" изображение строки, но сохраняем исходное
    вертикальное смещение каждого символа внутри row_obj.cells.
    
    Предполагается, что symbol.y хранит "верх" символа *локально* в пределах ячейки,
    но можно и глобально хранить, нужно только быть согласованным.
    """
    import cv2
    import numpy as np

    symbols_to_place = []
    
    for cell in row_obj.cells:
        # Глобальные координаты ячейки (или локальные, зависит от вашей логики).
        cell_x, cell_y = int(cell.x), int(cell.y)
        cell_img = original_image[cell_y:cell_y+cell.h, cell_x:cell_x+cell.w]
        for sym in cell.symbols:
            if sym.symbol_image is None:
                continue
            sx, sy, sw, sh = int(sym.x), int(sym.y), int(sym.w), int(sym.h)
            # Вырез символа:
            sym_img = cell_img[sy:sy+sh, sx:sx+sw+10]
            
            # Координата Y в глобальной системе row'а:
            # т.к. sy - локальная внутри cell, надо сложить с cell_y, чтобы получить глобальную.
            # Но если row_obj.cells идут подряд, вы можете трактовать "row_y_min" как минимум из всех ячеек.
            # Для демонстрации запишем "глобальную" верхнюю точку символа:
            global_y = cell_y + sy
            
            symbols_to_place.append((sym_img, global_y))

    if not symbols_to_place:
        return None

    # Сортируем символы по порядку cells? 
    # Вопрос: нужно ли сначала сортировать по ячейкам (или уже отсортированы) 
    # и внутри ячейки — по x. Предположим, что уже упорядочено (или сделайте manual sort).

    # Найдём minY и maxY+H по всем символам:
    min_y = min(g_y for (_, g_y) in symbols_to_place)
    # max_y = max(g_y + sym_img.shape[0] for (sym_img, g_y) in symbols_to_place)

    # Теперь собираем всё в "одну строку" по горизонтали, но с учётом вертикального смещения.
    # Заведём current_x = 0, будем шагать по каждому символу.
    
    # Определим высоту финального row_image, как разницу между min_y и max из (g_y + h).
    max_bottom = max(g_y + s.shape[0] for (s, g_y) in symbols_to_place)
    total_height = max_bottom - min_y

    # Считаем суммарную ширину (просто сумма ширин символов)
    
    total_sym_width = sum(s.shape[1] for (s, _) in symbols_to_place)
    num_symbols = len(symbols_to_place)
    total_spacing = symbol_spacing * (num_symbols - 1) if num_symbols > 1 else 0
    # print('total_spacing', total_spacing)
    total_width = total_sym_width + total_spacing
    # print('total_width', total_width)
    

    row_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)  # белый BGR
    # transform row_image to gray
    row_image = cv2.cvtColor(row_image, cv2.COLOR_BGR2GRAY)
    current_x = 0
    for sym_img, g_y in symbols_to_place:
        h_sym, w_sym = sym_img.shape[:2]
        # "верх" символа внутри row_image 
        # = (g_y - min_y), т.е. сохраняем сдвиг относительно верхней границы row_image
        offset_y = g_y - min_y

        # Вырезаем ROI:
        roi = row_image[offset_y:offset_y + h_sym, current_x:current_x + w_sym]
        if roi.shape[:2] == sym_img.shape[:2]:
            row_image[offset_y:offset_y + h_sym, current_x:current_x + w_sym] = sym_img
        else:
            # на всякий случай, если что-то не совпало
            hh = min(roi.shape[0], sym_img.shape[0])
            ww = min(roi.shape[1], sym_img.shape[1])
            row_image[offset_y:offset_y + hh, current_x:current_x + ww] = sym_img[:hh,:ww]

        current_x += w_sym  # сдвигаем вправо для следующего символа

    return row_image

def place_row_image_into_form(row_obj, original_image):
    """
    Берём row_image из rebuild_row_image(...) и вставляем в bounding box строки,
    закрасив предварительно область белым.
    """
    # 1. Считаем bounding box всей строки (пробегаем по клеткам, берём min/max x и y).
    # Для наглядности реализуем тут же (или используем row_obj.get_contour()).
    x_coords = [int(c.x) for c in row_obj.cells]
    y_coords = [int(c.y) for c in row_obj.cells]
    x_maxs = [int(c.x + c.w) for c in row_obj.cells]
    y_maxs = [int(c.y + c.h) for c in row_obj.cells]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_maxs)
    y_max = max(y_maxs)

    # 2. Собираем картинку строки
    row_image = rebuild_row_image(row_obj, original_image)
    # plt.imshow(row_image, cmap='gray')
    # plt.show()


    if row_image is None:
        return  # Нет символов, ничего вставлять

    # 3. Заливаем исходный bounding box белым
    cv2.rectangle(
        original_image,
        (x_min, y_min),
        (x_max, y_max),
        (255, 255, 255),
        thickness=-1
    )

    # 4. Вырезаем «целевой регион» (ROI) в original_image, куда будем помещать row_image
    target_width = x_max - x_min
    target_height = y_max - y_min

    # Если row_image больше или меньше по размеру, решаем, хотим ли «подгонять»
    row_h, row_w = row_image.shape[:2]

    # Для примера: подгоним row_image по ширине (сохраняя соотношение сторон)
    scale = target_width / row_w
    new_w = int(row_w * scale)
    new_h = int(row_h * scale)

    # Проверим, не вылезет ли он по высоте
    if new_h > target_height:
        # если вылазит — придётся ещё масштабировать по высоте
        scale2 = target_height / new_h
        new_w = int(new_w * scale2)
        new_h = int(new_h * scale2)

    # Масштабируем
    row_image_resized = cv2.resize(row_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # row_image_resized = row_image.copy()
    # Теперь размещаем (начиная с (x_min, y_min))
    # Если хотим по центру — можно сдвинуть, если хотим в левом верхнем углу bounding box — без сдвига
    # Возьмём левый верх:
    paste_x = x_min
    paste_y = y_min

    # увеличиваеи контраст
    # row_image_resized = cv2.convertScaleAbs(row_image_resized, alpha=1.1, beta=0)
    # Вставляем row_image_resized в original_image
    # Нужно не выйти за границы массива:
    region = original_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
    if region.shape[:2] == row_image_resized.shape[:2]:
        row_obj.row_image = row_image_resized.copy()
        original_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = row_image_resized
        # row_image_output = row_image_resized.copy()
    else:
        print(1)
        # На всякий случай проверка (если что-то не так с размерами)
        hh = min(region.shape[0], row_image_resized.shape[0])
        ww = min(region.shape[1], row_image_resized.shape[1])
        row_obj.row_image = row_image_resized[:hh, :ww].copy()
        original_image[paste_y:paste_y+hh, paste_x:paste_x+ww] = row_image_resized[:hh, :ww]
        # row_image_output = row_image_resized[:hh, :ww].copy()

    # plt.imshow(row_image_output, cmap='gray')
    # row_obj.row_image = row_image_output

def get_aligned_pic(template, filled, scale_factor = 1):
    start_time = time.time()
    # get template shapes
    # resize filled by scale_factor
    # get filled sizes
    filled_height_orig, filled_width_orig = filled.shape[:2]
    # resize filled by scale_factor
    filled = cv2.resize(filled, (int(filled_width_orig * scale_factor), int(filled_height_orig * scale_factor)), interpolation=cv2.INTER_AREA)
    filled_height, filled_width = filled.shape[:2]
    template = cv2.resize(template, (filled_width, filled_height), interpolation=cv2.INTER_AREA)
    template_height, template_width = template.shape[:2]
    time_1 = time.time()
    scale_x = 170
    scale_y = 170
    scale_w = 60
    scale_h = 60

    corners_rects = [
        (260 - scale_x, 170 - scale_y, 100 + 1*scale_w + 2*scale_x, 100 + 1*scale_h + 2*scale_y),
        (3230 - scale_x, 170 - scale_y, 100 + 1*scale_w + 2*scale_x, 100 + 1*scale_h + 2*scale_y),  # правый верх
        (260 - scale_x, 4810 - scale_y, 100 + 1*scale_w + 2*scale_x, 100 + 1*scale_h + 2*scale_y), # левый низ
        (3230 - scale_x, 4810 - scale_y, 100 + 1*scale_w + 2*scale_x, 100 + 1*scale_h + 2*scale_y) # правый низ

    ]
    # Создаём маску
    mask = np.zeros(template.shape[:2], dtype=np.uint8)

    # Рисуем белые прямоугольники там, где хотим искать ключевые точки
    for (x, y, w_c, h_c) in corners_rects:
        cv2.rectangle(mask, (x, y), (x+w_c, y+h_c), 255, -1)
    time_2 = time.time()
    # Нахождение ключевых точек и дескрипторов
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(filled, None)
    time_3 = time.time()

    # Сопоставление точек
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    # Используем лучшие сопоставления
    good_matches = matches
    time_4 = time.time()
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    time_5 = time.time()

    # Вычисление матрицы гомографии
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    time_6 = time.time()


    # Применение перспективного преобразования
    h, w = template.shape[:2]
    aligned_image = cv2.warpPerspective(filled, matrix, (w, h))
    time_7 = time.time()

    end_time = time.time()
    print(f"Время выполнения 1: {end_time - start_time:.2f} секунд")
    print(f"Время выполнения 2: {time_1 - start_time:.2f} секунд")
    print(f"Время выполнения 3: {time_2 - time_1:.2f} секунд")
    print(f"Время выполнения 4: {time_3 - time_2:.2f} секунд")
    print(f"Время выполнения 5: {time_4 - time_3:.2f} секунд")
    print(f"Время выполнения 6: {time_5 - time_4:.2f} секунд")
    print(f"Время выполнения 7: {time_6 - time_5:.2f} секунд")
    print(f"Время выполнения 8: {time_7 - time_6:.2f} секунд")

    # get back to original size
    aligned_image = cv2.resize(aligned_image, (filled_width_orig, filled_height_orig), interpolation=cv2.INTER_AREA)

    # save aligned_image
    cv2.imwrite("aligned_image.jpg", aligned_image)

    return aligned_image


def get_aligned_pic_last(template_img, cur_pic, scale_factor = 0.25):
    corners_template = np.float32([
        (260 , 170),  # Верхний левый
        (3230 , 170),  # Верхний правый
        (260 , 4810),  # Нижний левый

        (3230, 4810)  # Нижний правый
    ])
    H, W = template_img.shape[:2]
    cut_pic = cv2.resize(cur_pic, (W, H), interpolation=cv2.INTER_AREA)

    def get_contours(image):
        # Загрузка изображения

        # Инвертируем изображение (пунктирные линии должны быть белыми)
        _, binary = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
        # print(binary.shape)

        # Применяем размытие, чтобы убрать шум
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        # print(blurred.shape)

        # Используем морфологию для выделения прямоугольных структур
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=1)
        # print(processed.shape)


        # Поиск контуров
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: x[0][0][1])
        contours = sorted(contours, key=lambda x: x[0][0][0])
        contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

        # Копия изображения для отображения результатов
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print(output.shape)

        # Фильтруем и выделяем только прямоугольные блоки
        contours_valid = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Условие для фильтрации блоков по размеру
            if 100 < w < 150 and 100 < h < 200:  # Подстраивайте размеры под бланк
                contours_valid.append(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return contours_valid, output

    # Функция для нахождения квадратов в cut_pic
    def find_squares(image, template_corners):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image

        contours, output = get_contours(gray)
        # print(len(contours))
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # visualize the contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        global test_gray
        test_gray = gray

        global test_output
        test_output = output
        found_corners = []
        for cnt in contours:


            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)

            # Фильтруем по размеру и положению
            for tx, ty in template_corners:
                # print(tx, ty, center[0], center[1])
                if abs(tx - center[0]) < 250 and abs(ty - center[1]) < 200:
                    # print(tx, ty, center[0], center[1])
                    # print("found", tx, ty, center[0], center[1])
                    found_corners.append(center)
                    break






        # sort the found_corners by x and y
        found_corners = sorted(found_corners, key=lambda x: x[0])
        found_corners = sorted(found_corners, key=lambda x: x[1])
        print('len', len(found_corners))
        return np.float32(found_corners) if len(found_corners) == 4 else None




    # Находим квадраты в cut_pic
    corners_cut = find_squares(cut_pic, corners_template)

    if corners_cut is not None:
        # Вычисляем матрицу преобразования
        matrix = cv2.getPerspectiveTransform(corners_cut, corners_template)
        
        # Применяем преобразование
        aligned_cut_pic = cv2.warpPerspective(cut_pic, matrix, (template_img.shape[1], template_img.shape[0]))

        # Сохраняем или показываем результат
        # cv2.imwrite("aligned_cut_pic.jpg", aligned_cut_pic)

    else:
        print("Не удалось найти все 4 квадрата в cut_pic")

    return aligned_cut_pic



# def get_contours(image_path):
#     # Загрузка изображения
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Инвертируем изображение (пунктирные линии должны быть белыми)
#     _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

#     # Применяем размытие, чтобы убрать шум
#     blurred = cv2.GaussianBlur(binary, (5, 5), 0)

#     # Используем морфологию для выделения прямоугольных структур
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # Поиск контуров
#     contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda x: x[0][0][1])
#     contours = sorted(contours, key=lambda x: x[0][0][0])
#     contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

#     # Копия изображения для отображения результатов
#     output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#     # Фильтруем и выделяем только прямоугольные блоки
#     contours_valid = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         # Условие для фильтрации блоков по размеру
#         if 100 < w < 150 and 100 < h < 200:  # Подстраивайте размеры под бланк
#             contours_valid.append(contour)
#             cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     return contours_valid, output

def visualize_row(contours, image_path):
    # Копия изображения для отображения результатов
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Фильтруем и выделяем только прямоугольные блоки
    contours_valid = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Условие для фильтрации блоков по размеру
        if 100 < w < 150 and 100 < h < 200:  # Подстраивайте размеры под бланк
            contours_valid.append(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output