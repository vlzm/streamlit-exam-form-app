from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image
from xlsxwriter.utility import xl_rowcol_to_cell

from exam_grader.config import COLUMN_DISPLAY_NAMES, NUM_QUESTIONS

if TYPE_CHECKING:
    from exam_grader.models.form import Form


def prepare_form_dict(form: Form) -> dict[str, str]:
    """
    Extract recognized values from a Form into a flat dictionary
    with concatenated cell values per row.
    """
    result: dict[str, str] = {}
    row_names = (
        ["date", "user_id", "version"]
        + [f"answer{i}" for i in range(1, NUM_QUESTIONS + 1)]
        + [f"correction{i}" for i in range(1, NUM_QUESTIONS + 1)]
    )
    for name in row_names:
        row = form.rows[name]
        result[name] = "".join(v for v in row.user_answers if v is not None)
    return result


def transform_to_dataframe(data: dict[str, str]) -> pd.DataFrame:
    """
    Convert a form data dictionary into a single-row DataFrame
    with Russian display column names.
    """
    df = pd.DataFrame(data, index=[0])
    column_order = (
        ["date", "user_id", "version"]
        + [f"answer{i}" for i in range(1, NUM_QUESTIONS + 1)]
        + [f"correction{i}" for i in range(1, NUM_QUESTIONS + 1)]
    )
    df = df[column_order]
    df.rename(columns=COLUMN_DISPLAY_NAMES, inplace=True)
    return df


def add_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty placeholder columns for row preview images in the Excel output."""
    result = df.copy()
    result["Картинка код участника"] = ""
    result["Картинка вариант"] = ""
    result["Картинка Дата"] = ""
    for i in range(1, NUM_QUESTIONS + 1):
        result[f"Картинка ответа {i}"] = ""

    ordered_cols = [
        "Дата", "Картинка Дата",
        "Код участника", "Картинка код участника",
        "Вариант", "Картинка вариант",
    ]
    for i in range(1, NUM_QUESTIONS + 1):
        ordered_cols += [f"Задание {i}", f"Картинка ответа {i}"]
    for i in range(1, NUM_QUESTIONS + 1):
        ordered_cols.append(f"Начисленные баллы {i}")
    ordered_cols.append("Начисленные баллы сумма")

    return result[ordered_cols]


def save_to_excel(df: pd.DataFrame, form_dict: dict[int, Form]) -> io.BytesIO:
    """
    Export results to an Excel file with embedded row preview images
    and a SUM formula for total score.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        worksheet = writer.sheets["Sheet1"]

        for row_idx in range(len(df)):
            form = form_dict[row_idx]
            excel_row = row_idx + 1

            _insert_row_image(worksheet, df, excel_row, "Картинка Дата", form.rows["date"])
            _insert_row_image(
                worksheet, df, excel_row, "Картинка код участника", form.rows["user_id"]
            )
            _insert_row_image(
                worksheet, df, excel_row, "Картинка вариант", form.rows["version"]
            )

            for i in range(1, NUM_QUESTIONS + 1):
                _insert_row_image(
                    worksheet, df, excel_row,
                    f"Картинка ответа {i}", form.rows[f"answer{i}"],
                )

            col_start = df.columns.get_loc("Начисленные баллы 1")
            col_end = df.columns.get_loc(f"Начисленные баллы {NUM_QUESTIONS}")
            cell_start = xl_rowcol_to_cell(excel_row, col_start)
            cell_end = xl_rowcol_to_cell(excel_row, col_end)
            worksheet.write_formula(excel_row, col_end + 1, f"=SUM({cell_start}:{cell_end})")

    output.seek(0)
    return output


def _insert_row_image(worksheet, df: pd.DataFrame, excel_row: int, col_name: str, row) -> None:
    """Insert a row's preview image into the specified Excel cell."""
    col_idx = df.columns.get_loc(col_name)
    img_data = io.BytesIO()
    img = Image.fromarray(row.row_image)
    img.save(img_data, format="PNG")
    img_data.seek(0)
    worksheet.insert_image(excel_row, col_idx, "preview.png", {"image_data": img_data})
