from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Загружаем видовую модель, обученную для MNIST (весы для vision части)
vision_model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_mnist').to(device)

# Загружаем оригинальный CLIP-модель (мультимодальную)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Подгружаем веса видовой модели (MNIST) в multimodal модель CLIP
clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict())

# Определяем текстовые метки для цифр от 1 до 9
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Загружаем процессор для корректной предобработки изображений и текстов
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    # Define the rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image


def deskew_image(image):
    """
    Функция выравнивания изображения с цифрой (белая цифра на черном фоне) с помощью моментов.
    
    Аргументы:
        image: исходное изображение в градациях серого (grayscale).
        
    Возвращает:
        Выравненное изображение.
    """
    # Приводим изображение к бинарному виду.
    # Если изображение уже бинарное, этот шаг можно опустить.
    # Порог 128 выбирается как среднее значение между 0 и 255.
    
    # Вычисляем пространственные моменты бинарного изображения

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(image)
    
    # Если центральный момент второго порядка по y (mu02) очень мал, считать наклон равным 0
    if abs(moments["mu02"]) < 1e-2:
        return image.copy()
    
    # Вычисляем угол наклона (в радианах)
    # Используем функцию arctan2, которая учитывает знак числителя и знаменателя.
    skew_rad = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
    # Переводим угол в градусы
    skew_deg = np.degrees(skew_rad)
    
    # Для компенсации наклона поворачиваем изображение на -skew_deg градусов.
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_deg, 1.0)
    
    # Поворачиваем изображение. Используем INTER_LINEAR для интерполяции
    # и BORDER_REPLICATE, чтобы избежать появления чёрных границ.
    deskewed = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    
    deskewed = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    return deskewed

def predict_digit(image, labels):
    # Обрабатываем изображение и текстовые метки.
    # Процессор создаст словарь с ключом "pixel_values" для изображения и "input_ids"/"attention_mask" для текста.
    # resize image to 224x224
    # image = cv2.resize(image, (32, 32))
    # rotate image
    # image = rotate_image(image, 15)
    if np.sum(image) == 0:
        return None, image
    
    kernel = np.ones((3,3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    # image = deskew_image(image)
    # image = cv2.resize(image, (28, 28))
    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap='gray')
    plt.show()

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)


    # Получаем выходы модели без вычисления градиентов
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # logits_per_image — это матрица сходства между изображением и каждым из текстовых описаний.
        logits_per_image = outputs.logits_per_image  # размер: (1, число меток)
        # Преобразуем логиты в вероятности по меткам.
        probs = logits_per_image.softmax(dim=1)

    # Определяем индекс метки с наибольшей вероятностью
    pred_idx = probs.argmax(dim=1).item()
    predicted_digit = labels[pred_idx]
    # print('predicted digit', predicted_digit)
    return predicted_digit, image