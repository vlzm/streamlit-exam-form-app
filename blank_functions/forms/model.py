from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import filters
from skimage.color import rgb2gray
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Загружаем видовую модель, обученную для MNIST (весы для vision части)
# vision_model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_mnist').to(device)

# # Загружаем оригинальный CLIP-модель (мультимодальную)
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# # Подгружаем веса видовой модели (MNIST) в multimodal модель CLIP
# clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict())

# # Определяем текстовые метки для цифр от 1 до 9
# labels = ["0", "1 (with a thin vertical line)", "2", "3", "4", "5", "6", "7 (with a horizontal stroke)", "8", "9"]

# # Загружаем процессор для корректной предобработки изображений и текстов
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     # Define the rotation matrix
#     center = (w // 2, h // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
#     return rotated_image


# def deskew_image(image):
#     """
#     Функция выравнивания изображения с цифрой (белая цифра на черном фоне) с помощью моментов.
    
#     Аргументы:
#         image: исходное изображение в градациях серого (grayscale).
        
#     Возвращает:
#         Выравненное изображение.
#     """
#     # Приводим изображение к бинарному виду.
#     # Если изображение уже бинарное, этот шаг можно опустить.
#     # Порог 128 выбирается как среднее значение между 0 и 255.
    
#     # Вычисляем пространственные моменты бинарного изображения

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     moments = cv2.moments(image)
    
#     # Если центральный момент второго порядка по y (mu02) очень мал, считать наклон равным 0
#     if abs(moments["mu02"]) < 1e-2:
#         return image.copy()
    
#     # Вычисляем угол наклона (в радианах)
#     # Используем функцию arctan2, которая учитывает знак числителя и знаменателя.
#     skew_rad = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
#     # Переводим угол в градусы
#     skew_deg = np.degrees(skew_rad)
    
#     # Для компенсации наклона поворачиваем изображение на -skew_deg градусов.
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, skew_deg, 1.0)
    
#     # Поворачиваем изображение. Используем INTER_LINEAR для интерполяции
#     # и BORDER_REPLICATE, чтобы избежать появления чёрных границ.
#     deskewed = cv2.warpAffine(image, M, (w, h),
#                               flags=cv2.INTER_LINEAR,
#                               borderMode=cv2.BORDER_REPLICATE)
    
#     deskewed = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
#     return deskewed
    

# def predict_digit(image, labels = ["0", "1 (with a thin vertical line)", "2", "3", "4", "5", "6", "7 (with a flat top part)", "8", "9"]):
#     if np.sum(image) == 0:
#         return None, image
    
#     kernel = np.ones((3,3), np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)

#     image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)

#     inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)


#     # Получаем выходы модели без вычисления градиентов
#     with torch.no_grad():
#         outputs = clip_model(**inputs)
#         # logits_per_image — это матрица сходства между изображением и каждым из текстовых описаний.
#         logits_per_image = outputs.logits_per_image  # размер: (1, число меток)
#         # Преобразуем логиты в вероятности по меткам.
#         probs = logits_per_image.softmax(dim=1)

#     # Определяем индекс метки с наибольшей вероятностью
#     pred_idx = probs.argmax(dim=1).item()
#     predicted_digit = labels[pred_idx]
#     # plt.figure(figsize=(3, 3))
#     # plt.imshow(image)
#     # plt.title(f'predicted digit: {predicted_digit}')
#     # plt.show()
#     # print('predicted digit', predicted_digit)
#     return predicted_digit, image

string_to_digit = {
    'comma': ',',
    'minus': '-',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'zero': '0',
    'empty': ''
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_mnist').to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict())

# Load reference images
# ref_images_dict = {name: cv2.resize(cv2.imread(f"C:/Users/zamko/Documents/mom_project/repo/data/ref_pics/{name}.png"), (16, 16))
#                    for name in ['comma', 'minus', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']}

ref_images_dict = {name: cv2.resize(cv2.imread(f"C:/Users/zamko/Documents/mom_project/repo/data/ref_pics/{name}.png"), (28, 28), interpolation=cv2.INTER_AREA)
                   for name in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'minus', 'comma', 'empty']}

# for key, ref_embedding in ref_images_dict.items():
#     _, ref_image = cv2.threshold(ref_embedding, 200, 255, cv2.THRESH_BINARY_INV)
#     ref_images_dict[key] = ref_image

ref_embeddings = {}
for key, image1 in ref_images_dict.items():
    image1_preprocess = processor(images=image1, return_tensors="pt")['pixel_values'].to(device)
    ref_embeddings[key] = clip_model.get_image_features(image1_preprocess)



def predict_digit(image):
    # image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image2 = image
    # kernel = np.ones((2,2), np.uint8)
    # image2 = cv2.erode(image2, kernel, iterations=2)
    # kernel = np.ones((5,5), np.uint8)
    # image2 = cv2.erode(image2, kernel, iterations=1)
    # image2 = cv2.resize(image2, (28, 28), interpolation=cv2.INTER_AREA)
    # image2 = cv2.bitwise_not(image2)
    # _, image2 = cv2.threshold(image2, 200, 255, cv2.THRESH_BINARY_INV)
    img = rgb2gray(image2)
    threshold = filters.threshold_otsu(img)  # Автоматический порог
    binary = img > threshold  # Делаем маску
    binary_cv2 = (binary * 255).astype(np.uint8)
    binary_cv2 = cv2.cvtColor(binary_cv2, cv2.COLOR_GRAY2BGR)
    image2 = binary_cv2
    image2 = cv2.resize(image2, (28, 28), interpolation=cv2.INTER_AREA)

    image2_preprocess = processor(images=image2, return_tensors="pt")['pixel_values'].to(device)
    image2_embedding = clip_model.get_image_features(image2_preprocess)

    # Calculate similarity scores
    scores_dict = {key: torch.nn.functional.cosine_similarity(ref_embedding, image2_embedding)
                   for key, ref_embedding in ref_embeddings.items()}

    # Find the label with the highest similarity score
    scores_dict_max_name = max(scores_dict, key=scores_dict.get)
    predicted_digit = string_to_digit[scores_dict_max_name]
    
    return predicted_digit, image2
    # print(scores_dict_max_name)
    # plt.figure(figsize=(1, 1))  
    # plt.imshow(image2)
    # plt.axis('off')
    # plt.show()