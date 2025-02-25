from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import filters
from skimage.color import rgb2gray
from blank_functions.paths import path_to_ref_pics

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


ref_images_dict = {name: cv2.resize(cv2.imread(f"{path_to_ref_pics}/{name}.png"), (128, 128), interpolation=cv2.INTER_AREA)
                   for name in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']}

ref_embeddings = {}
for key, image1 in ref_images_dict.items():
    image1_preprocess = processor(images=image1, return_tensors="pt")['pixel_values'].to(device)
    ref_embeddings[key] = clip_model.get_image_features(image1_preprocess)



def predict_digit(image):
    image2 = image
    img = rgb2gray(image2)
    threshold = filters.threshold_otsu(img)  # Автоматический порог
    binary = img > threshold  # Делаем маску
    binary_cv2 = (binary * 255).astype(np.uint8)
    binary_cv2 = cv2.cvtColor(binary_cv2, cv2.COLOR_GRAY2BGR)
    image2 = binary_cv2
    image2 = cv2.resize(image2, (128, 128), interpolation=cv2.INTER_AREA)

    image2_preprocess = processor(images=image2, return_tensors="pt")['pixel_values'].to(device)
    image2_embedding = clip_model.get_image_features(image2_preprocess)

    # Calculate similarity scores
    scores_dict = {key: torch.nn.functional.cosine_similarity(ref_embedding, image2_embedding)
                   for key, ref_embedding in ref_embeddings.items()}

    # Find the label with the highest similarity score
    scores_dict_max_name = max(scores_dict, key=scores_dict.get)
    predicted_digit = string_to_digit[scores_dict_max_name]
    
    return predicted_digit, image2