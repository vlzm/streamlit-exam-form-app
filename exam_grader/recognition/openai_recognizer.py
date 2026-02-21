"""OpenAI Vision-based form recognition as an alternative to cell-by-cell CLIP OCR."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

from exam_grader.config import NUM_QUESTIONS

RECOGNITION_PROMPT = """\
Recognize this exam answer sheet image. Extract all handwritten values \
and return them as JSON with exactly the following structure:
{
  "date": "12.03.2025",
  "user_id": "1234",
  "version": "1",
  "answer_1": "275",
  "answer_2": "38",
  "answer_3": "14",
  "answer_4": "3478",
  "answer_5": "102",
  "answer_6": "3",
  "answer_7": "0,5",
  "answer_8": "18",
  "answer_9": "",
  "answer_10": "",
  "correction_1": "-275",
  "correction_2": "",
  "correction_3": "",
  "correction_4": "",
  "correction_5": "",
  "correction_6": "",
  "correction_7": "-3",
  "correction_8": "",
  "correction_9": "",
  "correction_10": ""
}

Rules:
- Pay special attention to minus signs (-) and commas (,) — don't miss them!
- Use comma as decimal separator (e.g. "0,5" not "0.5").
- Leave empty string "" for empty/unfilled fields.
- The left column contains answers ("Ответы"), the right column contains corrections ("Замена").
- Return ONLY valid JSON, nothing else.\
"""


def _encode_image(image: np.ndarray) -> str:
    """Convert a numpy image to a base64-encoded PNG string."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def recognize_form_with_openai(
    image: np.ndarray,
    api_key: str,
    model: str = "gpt-5.2",
    prompt: str = RECOGNITION_PROMPT,
) -> dict[str, str]:
    """
    Send a form image to OpenAI Vision API and return recognized values.

    Args:
        image: Grayscale or RGB page image as numpy array.
        api_key: OpenAI API key.
        model: OpenAI model name (default: gpt-5.2).
        prompt: Recognition prompt.

    Returns:
        Normalized dict with keys: date, user_id, version, answer1..10, correction1..10.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = _encode_image(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    print(response)
    raw = json.loads(response.choices[0].message.content)
    return _normalize_response(raw)


def _normalize_response(raw: dict[str, Any]) -> dict[str, str]:
    """
    Map OpenAI response keys to the internal format expected by
    Form.set_answers_from_dict() and prepare_form_dict().
    """
    result: dict[str, str] = {}

    # Date: strip non-digit characters so it fits into 8 cells (DDMMYYYY)
    date_raw = str(raw.get("date", ""))
    result["date"] = "".join(c for c in date_raw if c.isdigit())

    result["user_id"] = str(raw.get("user_id", raw.get("registration_number", "")))
    result["version"] = str(raw.get("version", raw.get("case", "")))

    for i in range(1, NUM_QUESTIONS + 1):
        result[f"answer{i}"] = str(raw.get(f"answer_{i}", raw.get(f"answer{i}", "")))
        result[f"correction{i}"] = str(
            raw.get(f"correction_{i}", raw.get(f"correction{i}", ""))
        )

    return result
