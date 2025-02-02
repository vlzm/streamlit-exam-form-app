import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        # print(1)
        # На всякий случай проверка (если что-то не так с размерами)
        hh = min(region.shape[0], row_image_resized.shape[0])
        ww = min(region.shape[1], row_image_resized.shape[1])
        row_obj.row_image = row_image_resized[:hh, :ww].copy()
        original_image[paste_y:paste_y+hh, paste_x:paste_x+ww] = row_image_resized[:hh, :ww]
        # row_image_output = row_image_resized[:hh, :ww].copy()

    # plt.imshow(row_image_output, cmap='gray')
    # row_obj.row_image = row_image_output


def get_aligned_pic(template, filled, scale_factor = 1):

    filled_height_orig, filled_width_orig = filled.shape[:2]
    # resize filled by scale_factor
    filled = cv2.resize(filled, (int(filled_width_orig * scale_factor), int(filled_height_orig * scale_factor)), interpolation=cv2.INTER_AREA)
    filled_height, filled_width = filled.shape[:2]
    template = cv2.resize(template, (filled_width, filled_height), interpolation=cv2.INTER_AREA)
    template_height, template_width = template.shape[:2]
 
    # Нахождение ключевых точек и дескрипторов
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(filled, None)

    # Сопоставление точек
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Используем лучшие сопоставления
    good_matches = matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Вычисление матрицы гомографии
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Применение перспективного преобразования
    h, w = template.shape[:2]
    aligned_image = cv2.warpPerspective(filled, matrix, (w, h))

    # get back to original size
    aligned_image = cv2.resize(aligned_image, (filled_width_orig, filled_height_orig), interpolation=cv2.INTER_AREA)

    return aligned_image


# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################
# #######################################

def correct_image_rotation(image, debug=False, display=True):
    """
    Reads an image from the given path, detects nearly horizontal lines to estimate the skew angle,
    rotates the image to correct the skew, and returns the rotated image.

    Parameters:
        image_path (str): Path to the input image.
        debug (bool): If True, prints detected angles and computed skew.
        display (bool): If True, displays the rotated image using matplotlib.

    Returns:
        rotated (numpy.ndarray): The rotated image.
    """


    # 1) Read the original image and convert it to grayscale
    img = image.copy()
    # plt.imshow(img, cmap='gray')
    # plt.title("original image")
    # plt.axis("off")
    # plt.show()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # 2) Preprocess: apply Gaussian blur and thresholding to create a binary image
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blur, 150, 255, cv2.THRESH_BINARY_INV)

    # 3) Edge detection using Canny
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # 4) Use Hough Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # 5) Compute the median angle of lines that are near horizontal (±10°)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Convert theta to a degree-based angle relative to the horizontal axis
            angle_deg = (theta * 180 / np.pi) - 90
            if debug:
                print(f"Detected line angle: {angle_deg:.2f}°")
            if abs(angle_deg) < 10:
                angles.append(angle_deg)

    median_angle = np.median(angles) if angles else 0
    if debug:
        print(f"Computed skew (median angle): {median_angle:.2f}°")

    # 6) Rotate the image using the negative of the computed skew angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # if display:
    #     plt.imshow(rotated, cmap='gray')
    #     plt.title("Rotated Image")
    #     plt.axis("off")
    #     plt.show()

    return rotated


def convert_to_gray(image):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

def preprocess_threshold(gray):
    """
    Apply thresholding, blurring, and morphological closing on a grayscale image
    to enhance cell detection.
    """
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=3)

    # plt.figure(figsize=(7, 14))
    # plt.imshow(processed, cmap='gray')
    # plt.title("processed")
    # plt.axis("off")
    # plt.show()
    return processed

def detect_cells(gray, processed_thresh):
    """
    Find contours on the processed threshold image, filter them based on expected
    cell sizes, and draw preliminary bounding boxes on the grayscale image.
    Returns a sorted list of cell coordinates (x, y, w, h).
    """
    contours, _ = cv2.findContours(processed_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by top-left coordinate (y then x)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 100 < w < 150 and 100 < h < 200:  # Adjust thresholds as needed
        # if 180 < w < 150 and 100 < h < 200:  # Adjust thresholds as needed
            cells.append((x, y, w, h))
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return sorted(cells, key=lambda b: (b[1], b[0]))

def create_visualization_image(image, cells):
    """
    Convert the processed grayscale image to a color image and redraw cell
    bounding boxes for clarity.
    """
    visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in cells:
        cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return visual

def draw_white_borders(image, cells):
    """
    Draw white border lines around each detected cell region for enhanced emphasis.
    """
    for (x, y, w, h) in cells:
        scale = 10
        scale_2 = 10
        scale_3 = 3
        scale_4 = 10
        image[y - scale : y + scale_3, x - scale_2 : x + w + scale_2] = 255  # lower horizontal line
        image[y + h - scale_3 : y + h + scale, x - scale_2 : x + w + scale_2] = 255  # upper horizontal line
        image[y - scale_2 : y + h + scale_2, x - scale_4 : x + scale_4] = 255      # left vertical line
        image[y - scale_2 : y + h + scale_2, x + w - scale_4 : x + w + scale_4] = 255  # right vertical line
    return image

def denoise_cells(image, cells):
    """
    For each cell region, apply binarization and denoising (erosion, median blur,
    and Gaussian blur) to refine the cell image.
    """
    for (x, y, w, h) in cells:
        cell_roi = image[y:y+h, x:x+w]
        _, cell_bin = cv2.threshold(cell_roi, 245, 255, cv2.THRESH_BINARY_INV)
        local_kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(cell_bin, local_kernel, iterations=1)
        denoised = cv2.medianBlur(eroded, 9)
        denoised = cv2.GaussianBlur(denoised, (9, 9), 0)
        image[y:y+h, x:x+w] = denoised
    return image


# def align_image_pipeline(image, template, scale_factor = 0.25):
#     rotated = correct_image_rotation(image)
#     # rotated = image.copy()

#     aligned_image_1 = get_aligned_pic(template, rotated, scale_factor)
#     # gray = convert_to_gray(aligned_image_1)
#     gray = aligned_image_1
#     processed_thresh = preprocess_threshold(gray)

#     # Detect cells and draw initial bounding boxes on the gray image
#     cells = detect_cells(gray, processed_thresh)

#     # Create a color visualization image and redraw the bounding boxes
#     processed_image = create_visualization_image(gray, cells)

#     # Draw white borders for extra clarity and denoise each cell region
#     processed_image = draw_white_borders(processed_image, cells)
#     processed_image = denoise_cells(processed_image, cells)

#     # Create a DataFrame with cell coordinates
#     df_cells = pd.DataFrame(cells, columns=["x", "y", "w", "h"])

#     plt.figure(figsize=(10, 20))
#     plt.imshow(processed_image)
#     plt.title("aligned image")
#     plt.axis("off")
#     plt.show()

#     return processed_image

def refine_cell_contour(aligned_image, cell_bbox, margin=10, debug=False):
    """
    Уточняет bounding box клетки в заданной области (ROI), расширенной на margin пикселей.

    Параметры:
        aligned_image (numpy.ndarray): выровненное изображение (желательно в grayscale).
        cell_bbox (tuple): шаблонный bounding box клетки в формате (x, y, w, h).
        margin (int): запас, на который расширяется область поиска.
        debug (bool): если True, выводит отладочную информацию.

    Возвращает:
        refined_bbox (tuple): уточнённый bounding box клетки.
    """
    x, y, w, h = cell_bbox
    # Определяем ROI с запасом
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, aligned_image.shape[1])
    y1 = min(y + h + margin, aligned_image.shape[0])
    
    roi = aligned_image[y0:y1, x0:x1]
    
    # Если изображение цветное – переводим в grayscale
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
    # Пороговое преобразование и сглаживание ROI для выделения контура
    _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(roi_thresh, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    roi_thresh = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # if debug:
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(roi_thresh, cmap='gray')
    #     plt.title("ROI после порогового преобразования")
    #     plt.axis("off")
    #     plt.show()
    
    # Поиск контуров в ROI
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Если найден хотя бы один контур – выбираем тот, чей центр ближе всего к центру шаблонного bbox
    refined_bbox = cell_bbox  # по умолчанию оставляем исходные координаты
    if contours:
        best_contour = None
        best_distance = float('inf')
        expected_center = (w / 2, h / 2)  # центр в системе координат ROI
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            center_cnt = (rx + rw / 2, ry + rh / 2)
            distance = np.linalg.norm(np.array(center_cnt) - np.array(expected_center))

            volume = rw * rh
            volume_condition = 100 < volume < 200   
            if distance < best_distance and volume_condition:
                best_distance = distance
                best_contour = cnt
        if best_contour is not None:
            rx, ry, rw, rh = cv2.boundingRect(best_contour)
            # Пересчитываем координаты контура относительно исходного изображения
            refined_bbox = (x0 + rx, y0 + ry, rw, rh)
            if debug:
                print(f"Исходный bbox: {cell_bbox}")
                print(f"Уточнённый bbox: {refined_bbox}")
            # Для отладки визуализируем найденный контур
            roi_vis = cv2.cvtColor(roi_thresh, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(roi_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(roi_vis)
            # plt.title("Найденный контур в ROI")
            # plt.axis("off")
            # plt.show()
    else:
        if debug:
            print("Контур не найден в ROI, остаёмся на шаблонном bbox.")

    return refined_bbox

def align_image_pipeline(image, template, scale_factor=0.25, debug=True):
    """
    Общий пайплайн выравнивания изображения по шаблону с уточнением координат клеток.

    Параметры:
        image (numpy.ndarray): исходное изображение.
        template: объект шаблона, содержащий информацию для выравнивания и список клеток 
                  (например, template.cells должен возвращать список bbox'ов клеток).
        scale_factor (float): коэффициент масштабирования для выравнивания.


    Возвращает:
        processed_image (numpy.ndarray): итоговое изображение с отмеченными клетками.
    """

    
    # 2. Выравнивание изображения по шаблону
    aligned_image = get_aligned_pic(template, image, scale_factor)
    
    rotated = correct_image_rotation(aligned_image)

    # if debug:
    #     plt.figure(figsize=(7, 7))
    #     plt.imshow(aligned_image, cmap='gray')
    #     plt.title("aligned image")
    #     plt.axis("off")
    #     plt.show()

    
    return rotated
    
def recalculate_cell(aligned_image, template_cell, margin=20, debug=False):
    
    # 5. Для каждой клетки уточняем координаты, используя ROI
    x, y, w, h = refine_cell_contour(aligned_image, template_cell, margin, debug)

    # if debug:
    #     plt.figure(figsize=(7, 7))
    #     plt.imshow(aligned_image, cmap='gray')
    #     plt.title("recalculated cell")
    #     plt.axis("off")
    #     plt.show()

    return x, y, w, h

def style_image(aligned_image, cells, debug=True):
    # 6. Создаём визуализацию – переводим в цветное изображение и рисуем bounding box'ы
    processed_image = create_visualization_image(aligned_image, cells)
    # plt.imshow(processed_image, cmap='gray')
    # plt.title("processed_image image")
    # plt.axis("off")
    # plt.show()
    
    # 7. Рисуем дополнительные белые рамки для лучшей наглядности
    processed_image = draw_white_borders(aligned_image, cells)
    # plt.imshow(processed_image, cmap='gray')
    # plt.title("processed_image image")
    # plt.axis("off")
    # plt.show()
    
    # 8. Применяем денойзинг для улучшения качества выделения клеток
    processed_image = denoise_cells(aligned_image, cells)

    if debug:
        plt.figure(figsize=(10, 20))
        plt.imshow(processed_image, cmap='gray')
        plt.title("styled image")
        plt.axis("off")
        plt.show()
    
    return processed_image