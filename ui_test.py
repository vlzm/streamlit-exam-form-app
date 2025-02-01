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

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Border, Side, Font

from form_class import Form

from ui_functions import get_pic_from_pdf, save_to_excel, get_correct_answers, postprocess_raw_output, check_answers, final_styling, extract_text_from_image, transform_json_to_dataframe
from ui_functions import promt

# UI Streamlit
st.title("Распознавание экзаменационных бланков")
st.write("Загрузите PDF файл, нажмите 'Распознать', и получите результат в формате Excel.")

api_key = st.text_input("Введите ваш OpenAI API ключ:", type="password")

uploaded_pdf = st.file_uploader("Загрузите PDF файл", type=["pdf"])
uploaded_answers = st.file_uploader("Загрузите Excel файл с правильными ответами", type=["xlsx"])

if uploaded_pdf and api_key:
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
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = pdf_document.page_count
                st.write("Количество страниц:", pdf_document.page_count)

                df_global = pd.DataFrame()

                for i in range(num_pages):
                    st.write(f"Обрабатываем страницу {i + 1}...")
                    cur_pic = get_pic_from_pdf(pdf_bytes, i)
                    form = Form()
                    form = form.run_pipeline(
                        image = cur_pic,
                        template_path = "template.jpg",
                        json_path = "rows_data.json",
                        answers = answers)  
                    cur_pic_adjusted = form.image
                    answer_minus_list = form.answer_minus_list
                    correction_minus_list = form.correction_minus_list
                    parsed_json = extract_text_from_image(api_key, cur_pic_adjusted, promt)
                    df_current = transform_json_to_dataframe(parsed_json, answer_minus_list, correction_minus_list, form.answer1.row_image)
                    df_global = pd.concat([df_global, df_current]).reset_index(drop=True)
                correct_answers = get_correct_answers(answers_bytes)
                df_global_processed = postprocess_raw_output(df_global, correct_answers)
                df_global_answers = check_answers(df_global_processed)
                df_global_styled = final_styling(df_global_answers)
                temp_df = pd.DataFrame([['Предмет', 'Код участника', 'Вариант', "Задание 1", "Картинка ответа 1", "Задание 2", "Картинка ответа 2", "Задание 3", "Картинка ответа 3", "Задание 4", "Картинка ответа 4", "Задание 5", "Картинка ответа 5", "Задание 6", "Картинка ответа 6", "Задание 7", "Картинка ответа 7", "Задание 8", "Картинка ответа 8", "Задание 9", "Картинка ответа 9", "Задание 10", "Картинка ответа 10", 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 10]], columns=df_global_styled.columns)
                temp_df.columns = df_global_styled.columns
                df_global_styled = pd.concat([temp_df, df_global_styled], ignore_index=True)
                
                excel_data = save_to_excel(df_global_styled)

                # Добавление имени файла
                st.success("Распознавание завершено!")
                st.download_button(
                    label="Скачать Excel файл",
                    data=excel_data,
                    file_name="Formatted_Data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error("Произошла ошибка! Подробности записаны в консоль.")
            # Лог ошибки в терминал
            traceback.print_exc()
