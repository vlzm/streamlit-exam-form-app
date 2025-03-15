import streamlit as st
import pandas as pd
import fitz
import cv2
from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI
from pydantic import BaseModel
import json
import numpy as np
import traceback
from typing import Optional
import time
import datetime
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Border, Side, Font

from blank_functions.forms.form_recognition import FormRecognition
from blank_functions.ui.ui_functions import get_pic_from_pdf, save_to_excel, get_correct_answers, postprocess_raw_output, check_answers, final_styling, extract_text_from_image, transform_json_to_dataframe
from blank_functions.ui.ui_functions import promt, prepare_cur_dict, reorder_cols


# UI Streamlit
st.title("Распознавание экзаменационных бланков")
st.write("Загрузите PDF файл, нажмите 'Распознать', и получите результат в формате Excel.")

cur_version = st.text_input("Введите номер варианта", type="default")
uploaded_pdf = st.file_uploader("Загрузите PDF файл", type=["pdf"])
uploaded_answers = st.file_uploader("Загрузите Excel файл с правильными ответами", type=["xlsx"])

SAVE_FOLDER = "saved_excels"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if uploaded_pdf and cur_version:
    if st.button("Распознать"):
        try:
            # Основной код обработки PDF
            pdf_bytes = uploaded_pdf.read()
            st.write("Размер файла PDF:", len(pdf_bytes))
            answers_bytes = uploaded_answers.read()
            answers = pd.read_excel(answers_bytes)
            st.write("Размер файла Excel:", len(answers_bytes))
            if not pdf_bytes or not answers_bytes:
                if not pdf_bytes:
                    st.error("Загруженный файл пуст. Пожалуйста, выберите корректный PDF файл.")
                if not answers_bytes:
                    st.error("Загруженный файл пуст. Пожалуйста, выберите корректный Excel файл с правильными ответами.")
            else:
                cur_version = int(cur_version)
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = pdf_document.page_count
                st.write("Количество страниц:", pdf_document.page_count)

                df_global = pd.DataFrame()
                form_dict = {}
                for i in range(0, num_pages):
                    cur_pic = get_pic_from_pdf(pdf_bytes, i, zoom=6.0)
                    form = FormRecognition(
                        image = cur_pic,
                        template_path = 'data/template_raw.jpg',
                        json_path = 'data/rows_data_new_format.json',
                        answers = answers,
                        version = cur_version)

                    form = form.run_pipeline()
                    cur_dict = prepare_cur_dict(form)
                    df_current = transform_json_to_dataframe(cur_dict)
                    df_global = pd.concat([df_global, df_current]).reset_index(drop=True)
                    form_dict[i] = form

                correct_answers = get_correct_answers(answers_bytes)
                df_global_processed = postprocess_raw_output(df_global, correct_answers, cur_version)
                df_global_answers = check_answers(df_global_processed)
                df_global_styled = final_styling(df_global_answers)
                df_global_styled = reorder_cols(df_global_styled)
                excel_data = save_to_excel(df_global_styled, form_dict)
              

                # Добавление имени файла
                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                st.success("Распознавание завершено!")
                st.download_button(
                    label="Скачать Excel файл",
                    data=excel_data,
                    file_name=f"Results_variant_{cur_version}_{date_time}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                # сохраняем файл в папку
                with open(f"saved_excels/Results_variant_{cur_version}_{date_time}.xlsx", "wb") as f:
                    f.write(excel_data.getvalue())

        except Exception as e:
            st.error("Произошла ошибка! Подробности записаны в консоль.")
            # Лог ошибки в терминал
            traceback.print_exc()

# Раздел истории сохранённых файлов
st.sidebar.header("История сохранённых Excel файлов")
saved_files = os.listdir(SAVE_FOLDER)
for saved_file in saved_files:
    full_path = os.path.join(SAVE_FOLDER, saved_file)
    with open(full_path, "rb") as f:
        file_bytes = f.read()
    st.sidebar.download_button(
        label=saved_file,
        data=file_bytes,
        file_name=saved_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=saved_file  # уникальный ключ для каждой кнопки
    )