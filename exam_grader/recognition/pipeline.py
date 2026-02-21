from __future__ import annotations

import numpy as np
import pandas as pd

from exam_grader.models.form import Form


class FormRecognition:
    """Orchestrates the full form processing pipeline from image to recognized answers."""

    def __init__(
        self,
        image: np.ndarray,
        template_path: str,
        json_path: str,
        answers: pd.DataFrame,
        version: int,
    ) -> None:
        self.image = image
        self.template_path = template_path
        self.json_path = json_path
        self.answers = answers
        self.version = version

    def __repr__(self) -> str:
        return (
            f"FormRecognition(template_path={self.template_path}, "
            f"json_path={self.json_path}, version={self.version})"
        )

    def run_pipeline(
        self,
        mode: str = "clip",
        api_key: str | None = None,
        openai_model: str = "gpt-4o",
    ) -> Form:
        """
        Execute the recognition pipeline and return the processed Form.

        Args:
            mode: "clip" for local CLIP-based OCR, "openai" for OpenAI Vision API.
            api_key: OpenAI API key (required when mode="openai").
            openai_model: OpenAI model name (default: "gpt-4o").
        """
        if mode == "openai":
            return self._run_openai_pipeline(api_key, openai_model)
        return self._run_clip_pipeline()

    def _run_clip_pipeline(self) -> Form:
        """Cell-by-cell recognition using local CLIP model."""
        form = Form(self.version)
        form.load_meta_from_json(self.json_path)
        form.load_image(self.image)
        form.load_template(self.template_path)
        form.align_form(scale_factor=1.0)
        form.style_image()
        form.load_correct_answers(self.answers)
        form.get_user_answers_pipeline()
        form.set_row_images()
        return form

    def _run_openai_pipeline(self, api_key: str, openai_model: str) -> Form:
        """
        Whole-image recognition using OpenAI Vision API.
        Still aligns the image and generates row previews for Excel export.
        """
        from exam_grader.recognition.openai_recognizer import recognize_form_with_openai

        recognized = recognize_form_with_openai(self.image, api_key, model=openai_model)

        form = Form(self.version)
        form.load_meta_from_json(self.json_path)
        form.load_image(self.image)
        form.load_template(self.template_path)
        form.align_form(scale_factor=1.0)
        form.load_correct_answers(self.answers)
        form.set_answers_from_dict(recognized)
        form.set_row_images()
        return form
