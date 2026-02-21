from __future__ import annotations

import cv2
import numpy as np
import torch
from skimage import filters
from skimage.color import rgb2gray
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from exam_grader.config import (
    CLIP_INPUT_SIZE,
    CLIP_PROCESSOR_MODEL_NAME,
    CLIP_VISION_MODEL_NAME,
    DIGIT_LABELS,
    REF_DIGIT_NAMES,
    REF_PICS_DIR,
)


class DigitClassifier:
    """
    CLIP-based handwritten digit recognizer. Compares cell images against
    reference digit embeddings using cosine similarity.

    Models are loaded lazily on first prediction to avoid blocking import.
    """

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._ref_embeddings: dict[str, torch.Tensor] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vision_model = CLIPVisionModel.from_pretrained(CLIP_VISION_MODEL_NAME).to(
            self._device
        )
        self._processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR_MODEL_NAME)
        self._model = CLIPModel.from_pretrained(CLIP_PROCESSOR_MODEL_NAME).to(
            self._device
        )
        self._model.vision_model.load_state_dict(
            vision_model.vision_model.state_dict()
        )

        ref_images = {
            name: cv2.resize(
                cv2.imread(str(REF_PICS_DIR / f"{name}.png")),
                (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            for name in REF_DIGIT_NAMES
        }

        self._ref_embeddings = {}
        for name, img in ref_images.items():
            pixels = self._processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].to(self._device)
            self._ref_embeddings[name] = self._get_image_embedding(pixels)

    def _get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features as a plain tensor.
        Handles both old transformers (returns Tensor) and new versions
        (returns BaseModelOutputWithPooling).
        """
        output = self._model.get_image_features(pixel_values=pixel_values)
        if isinstance(output, torch.Tensor):
            return output
        vision_out = self._model.vision_model(pixel_values=pixel_values)
        pooled = vision_out[1]
        return self._model.visual_projection(pooled)

    def predict(self, image: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Predict the digit in a cell image.

        Args:
            image: RGB cell image.

        Returns:
            Tuple of (predicted character, preprocessed image used for prediction).
        """
        self._ensure_loaded()

        gray = rgb2gray(image)
        threshold = filters.threshold_otsu(gray)
        binary = (gray > threshold).astype(np.uint8) * 255
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(
            binary_bgr,
            (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        pixels = self._processor(images=resized, return_tensors="pt")[
            "pixel_values"
        ].to(self._device)
        embedding = self._get_image_embedding(pixels)

        scores = {
            name: torch.nn.functional.cosine_similarity(ref_emb, embedding)
            for name, ref_emb in self._ref_embeddings.items()
        }
        best_match = max(scores, key=scores.get)
        return DIGIT_LABELS[best_match], resized


_classifier: DigitClassifier | None = None


def predict_digit(image: np.ndarray) -> tuple[str, np.ndarray]:
    """Module-level convenience function for digit prediction with lazy initialization."""
    global _classifier
    if _classifier is None:
        _classifier = DigitClassifier()
    return _classifier.predict(image)
