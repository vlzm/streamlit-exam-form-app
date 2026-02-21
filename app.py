import datetime
import os
import traceback

import pandas as pd
import streamlit as st

from exam_grader import (
    FormRecognition,
    add_image_columns,
    check_answers,
    final_styling,
    get_correct_answers,
    get_page_image_from_pdf,
    get_pdf_page_count,
    postprocess_raw_output,
    prepare_form_dict,
    save_to_excel,
    transform_to_dataframe,
)
from exam_grader.config import CELL_COORDS_PATH, SAVED_EXCELS_DIR, TEMPLATE_PATH

st.title("Распознавание экзаменационных бланков")
st.write("Загрузите PDF файл, нажмите 'Распознать', и получите результат в формате Excel.")

recognition_mode = st.radio(
    "Способ распознавания",
    options=["CLIP (локальная модель)", "OpenAI Vision API"],
    horizontal=True,
)
use_openai = recognition_mode == "OpenAI Vision API"

openai_api_key = ""
if use_openai:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

cur_version = st.text_input("Введите номер варианта", type="default")
uploaded_pdf = st.file_uploader("Загрузите PDF файл", type=["pdf"])
uploaded_answers = st.file_uploader(
    "Загрузите Excel файл с правильными ответами", type=["xlsx"]
)

SAVED_EXCELS_DIR.mkdir(exist_ok=True)

if uploaded_pdf and cur_version:
    if use_openai and not openai_api_key:
        st.warning("Введите OpenAI API Key для использования OpenAI Vision API.")
    elif st.button("Распознать"):
        try:
            pdf_bytes = uploaded_pdf.read()
            answers_bytes = uploaded_answers.read()

            if not pdf_bytes or not answers_bytes:
                if not pdf_bytes:
                    st.error("Загруженный PDF файл пуст.")
                if not answers_bytes:
                    st.error("Загруженный Excel файл пуст.")
            else:
                version = int(cur_version)
                answers = pd.read_excel(answers_bytes)
                num_pages = get_pdf_page_count(pdf_bytes)
                st.write(f"Страниц: {num_pages}")

                mode = "openai" if use_openai else "clip"
                df_global = pd.DataFrame()
                form_dict = {}

                progress = st.progress(0, text="Распознавание...")
                for i in range(num_pages):
                    progress.progress(
                        (i + 1) / num_pages,
                        text=f"Страница {i + 1} из {num_pages}...",
                    )
                    cur_pic = get_page_image_from_pdf(pdf_bytes, i)
                    recognition = FormRecognition(
                        image=cur_pic,
                        template_path=str(TEMPLATE_PATH),
                        json_path=str(CELL_COORDS_PATH),
                        answers=answers,
                        version=version,
                    )
                    form = recognition.run_pipeline(
                        mode=mode,
                        api_key=openai_api_key if use_openai else None,
                    )
                    cur_dict = prepare_form_dict(form)
                    df_current = transform_to_dataframe(cur_dict)
                    df_global = pd.concat([df_global, df_current]).reset_index(drop=True)
                    form_dict[i] = form

                correct_answers = get_correct_answers(answers_bytes)
                df_processed = postprocess_raw_output(df_global, correct_answers, version)
                df_checked = check_answers(df_processed)
                df_styled = final_styling(df_checked)
                df_styled = add_image_columns(df_styled)
                excel_data = save_to_excel(df_styled, form_dict)

                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Results_variant_{version}_{date_time}.xlsx"

                st.success("Распознавание завершено!")
                st.download_button(
                    label="Скачать Excel файл",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                with open(SAVED_EXCELS_DIR / filename, "wb") as f:
                    f.write(excel_data.getvalue())

        except Exception:
            st.error("Произошла ошибка! Подробности записаны в консоль.")
            traceback.print_exc()

# Saved files history sidebar
st.sidebar.header("История сохранённых Excel файлов")
if SAVED_EXCELS_DIR.exists():
    for saved_file in os.listdir(SAVED_EXCELS_DIR):
        full_path = SAVED_EXCELS_DIR / saved_file
        with open(full_path, "rb") as f:
            file_bytes = f.read()
        st.sidebar.download_button(
            label=saved_file,
            data=file_bytes,
            file_name=saved_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=saved_file,
        )
