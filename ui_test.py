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


# Ваши существующие функции
class ResearchPaperExtraction(BaseModel):
    subject: str
    participant_code: str
    variant: str
    answers_1: float
    answers_2: float
    answers_3: float
    answers_4: float
    answers_5: float
    answers_6: float
    answers_7: float
    answers_8: float
    answers_9: float
    answers_10: float

def encode_image(image):
    image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def extract_text_from_image(api_key, image):
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image)
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract the text and numeric data from the provided image of a completed form. The form contains sequences of cells, with each cell containing a single character. The characters can include letters, digits, the minus sign, commas, and other punctuation marks. Ensure to capture each character exactly as it appears in its respective cell and maintain the order in which the cells are arranged on the form."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            response_format={
                    'type': 'json_schema',
                    'json_schema':
                        {
                            "name": "whocares",
                            "schema": ResearchPaperExtraction.model_json_schema()
                        }},
        )
    return json.loads(response.choices[0].message.content)

def transform_json_to_dataframe(parsed_json):
    df = pd.DataFrame([parsed_json])
    order = ['subject', 'participant_code', 'variant', 'answers_1', 'answers_2', 'answers_3', 'answers_4', 'answers_5',
             'answers_6', 'answers_7', 'answers_8', 'answers_9', 'answers_10']
    df = df[order]
    rename_dict = {
        'subject': 'Предмет',
        'participant_code': 'Код участника',
        'variant': 'Вариант',
        'answers_1': 'Ответ 1',
        'answers_2': 'Ответ 2',
        'answers_3': 'Ответ 3',
        'answers_4': 'Ответ 4',
        'answers_5': 'Ответ 5',
        'answers_6': 'Ответ 6',
        'answers_7': 'Ответ 7',
        'answers_8': 'Ответ 8',
        'answers_9': 'Ответ 9',
        'answers_10': 'Ответ 10'
    }
    df.rename(columns=rename_dict, inplace=True)
    return df

def get_pic_from_pdf(pdf_stream, index):
    """
    Получает изображение страницы из PDF
    :param pdf_stream: поток PDF файла
    :param index: индекс страницы
    :return: серое изображение страницы
    """
    # Открываем PDF из байтового потока
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    page = pdf_document.load_page(index)  # Загружаем страницу
    pix = page.get_pixmap()  # Преобразуем страницу в изображение
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return img_gray

# UI Streamlit
st.title("Распознавание экзаменационных бланков")
st.write("Загрузите PDF файл, нажмите 'Распознать', и получите результат в формате Excel.")

api_key = st.text_input("Введите ваш OpenAI API ключ:", type="password")

uploaded_pdf = st.file_uploader("Загрузите PDF файл", type=["pdf"])
st.write("Файл загружен, начинаем обработку...")

if uploaded_pdf and api_key:
    if st.button("Распознать"):
        try:
            # Основной код обработки PDF
            pdf_bytes = uploaded_pdf.read()
            st.write("Размер файла:", len(pdf_bytes))
            if not pdf_bytes:
                st.error("Загруженный файл пуст. Пожалуйста, выберите корректный PDF файл.")
            else:
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                num_pages = pdf_document.page_count
                st.write("Количество страниц:", pdf_document.page_count)

                df_global = pd.DataFrame()

                for i in range(num_pages):
                    st.write(f"Обрабатываем страницу {i + 1}...")
                    cur_pic = get_pic_from_pdf(pdf_bytes, i)
                    parsed_json = extract_text_from_image(api_key, cur_pic)
                    df_current = transform_json_to_dataframe(parsed_json)
                    df_global = pd.concat([df_global, df_current]).reset_index(drop=True)

                # Сохранение результатов в Excel
                output = BytesIO()
                df_global.to_excel(output, index=False, sheet_name="Результаты")
                output.seek(0)

                # Добавление имени файла
                filename = "results.xlsx"
                b64 = base64.b64encode(output.read()).decode()

                # Формирование ссылки с атрибутом download
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Скачать файл {filename}</a>'
                st.success("Распознавание завершено!")
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error("Произошла ошибка! Подробности записаны в консоль.")
            # Лог ошибки в терминал
            traceback.print_exc()
