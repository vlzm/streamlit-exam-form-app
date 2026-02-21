from __future__ import annotations

import pandas as pd

from exam_grader.config import NUM_QUESTIONS, SCORING_WEIGHTS


def get_correct_answers(correct_answers_path: bytes | str) -> pd.DataFrame:
    """
    Load and normalize correct answers from an Excel file.
    Converts all answer columns to string representation via float normalization.
    """
    df = pd.read_excel(correct_answers_path)
    for col in df.columns:
        if col != "Вариант":
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float).astype(str)
    df["Вариант"] = df["Вариант"].astype(str)
    df.rename(
        columns={i: f"Правильный ответ {i}" for i in range(1, NUM_QUESTIONS + 1)},
        inplace=True,
    )
    return df


def postprocess_raw_output(
    df: pd.DataFrame, correct_answers: pd.DataFrame, version: int
) -> pd.DataFrame:
    """Merge recognized answers with correct answers and normalize data types."""
    result = df.copy()
    result["Вариант"] = str(int(version))
    result["Дата"] = result["Дата"].str.upper()

    for i in range(1, NUM_QUESTIONS + 1):
        result[f"Задание {i}"] = result[f"Задание {i}"].replace(",", ".", regex=True)
        result[f"Задание {i}"] = result[f"Задание {i}"].replace("", "nan").astype(float)
        result[f"Замена {i}"] = result[f"Замена {i}"].replace(",", ".", regex=True)
        result[f"Замена {i}"] = result[f"Замена {i}"].replace("", "nan").astype(float)

    for col in result.columns:
        result[col] = result[col].astype(str)

    return pd.merge(result, correct_answers, on="Вариант", how="left")


def check_answers(total_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare recognized answers against correct answers and calculate scores.
    A question is correct if either the original answer or the correction matches.
    """
    for i in range(1, NUM_QUESTIONS + 1):
        correct = total_df[f"Правильный ответ {i}"]
        answer = total_df[f"Задание {i}"]
        correction = total_df[f"Замена {i}"]
        is_correct = (correct == answer) | (correct == correction)
        total_df[f"Начисленные баллы {i}"] = is_correct.apply(
            lambda x, q=i: SCORING_WEIGHTS[q] if x else 0
        )

    score_cols = [f"Начисленные баллы {i}" for i in range(1, NUM_QUESTIONS + 1)]
    total_df["Начисленные баллы сумма"] = total_df[score_cols].sum(axis=1)
    return total_df


def final_styling(total_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns, replace original answers with corrections where available,
    and clean up NaN display values.
    """
    cols: list[str] = ["Дата", "Код участника", "Вариант"]
    for i in range(1, NUM_QUESTIONS + 1):
        cols += [f"Задание {i}", f"Замена {i}"]
    for i in range(1, NUM_QUESTIONS + 1):
        cols.append(f"Начисленные баллы {i}")
    cols.append("Начисленные баллы сумма")

    result = total_df[cols].copy()

    for i in range(1, NUM_QUESTIONS + 1):
        result[f"Задание {i}"] = result[f"Замена {i}"].where(
            result[f"Замена {i}"] != "nan", result[f"Задание {i}"]
        )
        result = result.drop(columns=[f"Замена {i}"])

    for col in result.columns:
        result[col] = result[col].replace("nan", "")

    return result
