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
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Border, Side, Font
import io
from blank_functions.forms.form_recognition import FormRecognition

promt = """"You are tasked with extracting information from an image of a completed exam answer sheet and converting it into a structured JSON format. The answer sheet has the following structure:

Top Section:

Document Title: Labeled as "БЛАНК ОТВЕТОВ."
date Name: Handwritten in a grid labeled "МАТЕМАТИКА."
Participant Code: A field containing four cells where numbers are written.
Version Number: A field containing four cells where numbers are written.
Instructions: Indicates that answers can be integers or finite decimal fractions with 1-2 decimal places. It specifies that units of measurement and periods should not be included after the answer.
Example: The example answer "-0,9" is provided.
Middle Section:

Filling Instructions: Explains that filling should be done using a black gel or capillary pen with digits of a specific standard. Acceptable symbols are listed as: ", -1234567890."
Example of Valid Symbols: Includes symbols such as ", - 1 2 3 4 5 6 7 8 9 0 . "
Main Section: Divided into two large columns:

Left Column: "Ответы к заданиям"

Numbered from 1 to 10.
Each question is accompanied by a grid of cells to record the answer.
Examples of filled answers:
№1: "275"
№2: "38"
№3: "14"
№4: "3478"
№5: "102"
№6: "3"
№7: "0,5"
№8: "18"
Right Column: "Замена ошибочных ответов на задания"

Numbered similarly to the left column (1–10).
Intended for recording corrected answers for mistakes.
Examples of corrections:
№1: Corrected answer "-275."
№7: Corrected answer "-3."
Your task:

Extract all the information from the described answer sheet and format it in a JSON structure.
Ensure the JSON includes all relevant details such as date name, participant code, version number, and both the answers and corrections for questions.
Example JSON Structure:
{
  "date_name": "МАТЕМАТИКА",
  "participant_code": "1234",
  "version_number": "5678",
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

Be careful! Pay special attention to the minus sign and the comma separately. Don't miss them!
"""

#
def encode_image(image):
    image = Image.fromarray(image)

    # Save the image to a BytesIO buffer in PNG format
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the PNG image to base64
    test_pic_bytes = buffer.getvalue()
    base64_encoded = base64.b64encode(test_pic_bytes).decode('utf-8')

    return base64_encoded

def extract_text_from_image(api_key, image, prompt):
    client = OpenAI(api_key=api_key)

    # Getting the base64 string
    base64_image = encode_image(image)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
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

    json_response = json.loads(response.choices[0].message.content)
    return json_response

def get_pic_from_pdf(pdf_stream, index, zoom=6.0):
    """
    Получает изображение страницы из PDF с высоким качеством
    :param pdf_stream: поток PDF файла
    :param index: индекс страницы
    :param zoom: коэффициент увеличения (по умолчанию 4.0 для ~600 DPI)
    :return: изображение в формате NumPy (серое)
    """
    # Открываем PDF из байтового потока
    start_time = time.time()
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    time_1 = time.time()
    page = pdf_document.load_page(index)  # Загружаем страницу
    time_2 = time.time()
    # Матрица для увеличения качества (увеличение DPI)
    mat = fitz.Matrix(zoom, zoom)
    time_3 = time.time()
    # Преобразуем страницу в изображение
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)  # Улучшенное цветовое пространство
    time_4 = time.time()
    # Конвертируем Pixmap в Pillow Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    time_5 = time.time()
    # Опционально: сохраняем в PNG для максимального качества перед конвертацией
    # img.save("original_image.png", format="PNG")
    time_6 = time.time()
    # Конвертируем в оттенки серого с максимальным качеством
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    time_7 = time.time()

    return img_gray

def get_correct_answers(correct_answers_path):
    correct_answers = pd.read_excel(correct_answers_path)
    for col in correct_answers.columns:
        if col != 'Вариант':
            correct_answers[col] = correct_answers[col].astype(str)
            correct_answers[col] = correct_answers[col].apply(lambda x: x.replace(',', '.'))
            correct_answers[col] = correct_answers[col].astype(float)
            correct_answers[col] = correct_answers[col].astype(str)
    correct_answers['Вариант'] = correct_answers['Вариант'].astype(str)
    correct_answers.rename(columns={1: "Правильный ответ 1", 2: "Правильный ответ 2", 3: "Правильный ответ 3", 4: "Правильный ответ 4", 5: "Правильный ответ 5", 6: "Правильный ответ 6", 7: "Правильный ответ 7", 8: "Правильный ответ 8", 9: "Правильный ответ 9", 10: "Правильный ответ 10"}, inplace=True)
    return correct_answers

def postprocess_raw_output(df_global_fin, correct_answers, version):

    df_global_fin['Вариант'] = version

    df_global_fin['Дата'] = df_global_fin['Дата'].str.upper()
    df_global_fin['Вариант'] = df_global_fin['Вариант'].astype(int)
    df_global_fin['Вариант'] = df_global_fin['Вариант'].astype(str)
    for i in range(1, 11):
        df_global_fin[f'Задание {i}'] = df_global_fin[f'Задание {i}'].replace(',', '.', regex=True)
        df_global_fin[f'Задание {i}'] = df_global_fin[f'Задание {i}'].replace('', 'nan').astype(float)
        df_global_fin[f'Замена {i}'] = df_global_fin[f'Замена {i}'].replace(',', '.', regex=True)
        df_global_fin[f'Замена {i}'] = df_global_fin[f'Замена {i}'].replace('', 'nan').astype(float)
    for col in df_global_fin.columns:
        df_global_fin[col] = df_global_fin[col].astype(str)

    total_df = pd.merge(df_global_fin, correct_answers, on="Вариант", how="left")

    return total_df


class ResearchPaperExtraction(BaseModel):
    date_name: str
    participant_code: int
    version_number: int
    answer_1: Optional[float]
    answer_2: Optional[float]
    answer_3: Optional[float]
    answer_4: Optional[float]
    answer_5: Optional[float]
    answer_6: Optional[float]
    answer_7: Optional[float]
    answer_8: Optional[float]
    answer_9: Optional[float]
    answer_10: Optional[float]
    correction_1: Optional[float]
    correction_2: Optional[float]
    correction_3: Optional[float]
    correction_4: Optional[float]
    correction_5: Optional[float]
    correction_6: Optional[float]
    correction_7: Optional[float]
    correction_8: Optional[float]
    correction_9: Optional[float]
    correction_10: Optional[float]


def transform_json_to_dataframe(parsed_json):
    df = pd.DataFrame(parsed_json, index=[0])
    order = ['date', 'user_id', 'version', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5', 'answer6', 'answer7', 'answer8', 'answer9', 'answer10', 'correction1', 'correction2', 'correction3', 'correction4', 'correction5', 'correction6', 'correction7', 'correction8', 'correction9', 'correction10']
    df = df[order]

    rename_dict = {
    'date': 'Дата',
    'user_id': 'Код участника',
    'version': 'Вариант',
    'answer1': 'Задание 1',
    'answer2': 'Задание 2',
    'answer3': 'Задание 3',
    'answer4': 'Задание 4',
    'answer5': 'Задание 5',

    'answer6': 'Задание 6',
    'answer7': 'Задание 7',
    'answer8': 'Задание 8',
    'answer9': 'Задание 9',
    'answer10': 'Задание 10',
    'correction1': 'Замена 1',
    'correction2': 'Замена 2',
    'correction3': 'Замена 3',

    'correction4': 'Замена 4',
    'correction5': 'Замена 5',
    'correction6': 'Замена 6',
    'correction7': 'Замена 7',
    'correction8': 'Замена 8',
    'correction9': 'Замена 9',
    'correction10': 'Замена 10',

    'row_image1': 'Картинка ответа 1',
    'row_image2': 'Картинка ответа 2',
    'row_image3': 'Картинка ответа 3',
    'row_image4': 'Картинка ответа 4',
    'row_image5': 'Картинка ответа 5',
    'row_image6': 'Картинка ответа 6',

    'row_image7': 'Картинка ответа 7',
    'row_image8': 'Картинка ответа 8',
    'row_image9': 'Картинка ответа 9',
    'row_image10': 'Картинка ответа 10'

    }

    df.rename(columns=rename_dict, inplace=True)
    return df

def prepare_cur_dict(form_dict):
    cur_dict = {}
    for row_name in ["date", "user_id", "version"]:
        row = getattr(form_dict, row_name)
        user_answers = row.user_answers
        cur_dict[row_name] = user_answers


    for row_name in [f"answer{i}" for i in range(1, 11)]:
        row = getattr(form_dict, row_name)
        user_answers = row.user_answers
        cur_dict[row_name] = user_answers


    for row_name in [f"correction{i}" for i in range(1, 11)]:
        row = getattr(form_dict, row_name)
        user_answers = row.user_answers
        cur_dict[row_name] = user_answers


    for key in ["date", "user_id", "version"] + [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
        cur_dict[key] = "".join([x for x in cur_dict[key] if x is not None])


    # for key in [f"answer{i}" for i in range(1, 11)] + [f"correction{i}" for i in range(1, 11)]:
    #     if cur_dict[key] != '':
    #         cur_dict[key] = str(float(cur_dict[key].replace(',', '.')))

    # cur_dict['date'] = 'МАТЕМАТИКА'

    return cur_dict

def check_answers(total_df):

    scores_dict = {
    1: 0.5,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1.5
    }
    # for i in range(1, 11):
    #     right_answer_v2 = total_df[f'Правильный ответ {i}'].apply(lambda x: x.replace('.', '1') + '.0')
    #     print(right_answer_v2)
    #     print(total_df[f'Задание {i}'])
    #     print(total_df[f'Замена {i}'])
    #     if right_answer_v2.values[0] == total_df[f'Задание {i}'].values[0]:
    #         total_df[f'Задание {i}'] = total_df[f'Правильный ответ {i}']
    #     if right_answer_v2.values[0] == total_df[f'Замена {i}'].values[0]:
    #         total_df[f'Замена {i}'] = total_df[f'Правильный ответ {i}']

    for i in range(1, 11):
        total_df[f'Начисленные баллы {i}'] = ((total_df[f'Правильный ответ {i}'] == total_df[f'Задание {i}']) | (total_df[f'Правильный ответ {i}'] == total_df[f'Замена {i}'])).replace(True, 'Верно').replace(False, 'Неверно').apply(lambda x: scores_dict[i] if x == 'Верно' else 0)

    total_df['Начисленные баллы сумма'] = total_df[[f'Начисленные баллы {i}' for i in range(1, 11)]].sum(axis=1)

    return total_df

def final_styling(total_df):
    reorder_cols_list = ['Дата', 'Код участника', 'Вариант']
    for i in range(1, 11):  
        reorder_cols_list.append(f'Задание {i}')
        reorder_cols_list.append(f'Замена {i}')
        # reorder_cols_list.append(f'Картинка ответа {i}')

    for i in range(1, 11):
        reorder_cols_list.append(f'Начисленные баллы {i}')
    reorder_cols_list.append('Начисленные баллы сумма')
    total_df = total_df[reorder_cols_list]

    for i in range(1, 11):
        total_df.loc[:, f'Задание {i}'] = total_df[f'Замена {i}'].where(total_df[f'Замена {i}'] != 'nan', total_df[f'Задание {i}'])
        total_df = total_df.drop(columns=[f'Замена {i}'])

    for col in total_df.columns:
        total_df[col] = total_df[col].replace('nan', '')

    return total_df

def save_to_excel_last(df_global_styled, file_name="Formatted_Data.xlsx"):
    columns = ["Дата", "Код участника", "Вариант", "Задание 1", "Картинка ответа 1", "Задание 2", "Картинка ответа 2", "Задание 3", "Картинка ответа 3", "Задание 4", "Картинка ответа 4", "Задание 5", "Картинка ответа 5", "Задание 6", "Картинка ответа 6", "Задание 7", "Картинка ответа 7", "Задание 8", "Картинка ответа 8", "Задание 9", "Картинка ответа 9", "Задание 10", "Картинка ответа 10", "Начисленные баллы 1", "Начисленные баллы 2", "Начисленные баллы 3", "Начисленные баллы 4", "Начисленные баллы 5", "Начисленные баллы 6", "Начисленные баллы 7", "Начисленные баллы 8", "Начисленные баллы 9", "Начисленные баллы 10"]

    # Save to Excel with formatting
    wb = Workbook()
    ws = wb.active
    ws.title = "Общая таблица"

    # Write the header
    header = ["Дата", "Код участника", "Вариант", "Ответы", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "Баллы", None, None, None, None, None, None, None, None, None, "Всего"]
    ws.append(header)
    ws.merge_cells(start_row=1, start_column=4, end_row=1, end_column=23)
    ws.merge_cells(start_row=1, start_column=24, end_row=1, end_column=33)
    ws.merge_cells(start_row=1, start_column=1, end_row=3, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=3, end_column=2)
    ws.merge_cells(start_row=1, start_column=3, end_row=3, end_column=3)
    ws.merge_cells(start_row=1, start_column=34, end_row=2, end_column=34)

    # Write the subheader
    ws.append(columns)

    # Write the data
    for row in dataframe_to_rows(df_global_styled, index=False, header=False):
        ws.append(row)

    # Align headers
    for cell in ws[1]:
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Create a border for the entire dataframe
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.border = thin_border

    # Make the first three rows bold
    for row in ws.iter_rows(min_row=1, max_row=3):
        for cell in row:
            cell.font = Font(bold=True)

    for col in range(4, 24):
        ws.merge_cells(start_row=2, start_column=col, end_row=3, end_column=col)

    # Save to a BytesIO object
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def save_to_excel_simple(df_global_styled, file_name="Formatted_Data.xlsx"):
    columns = ["Дата", "Код участника", "Вариант", "Задание 1", "Задание 2", "Задание 3", "Задание 4", "Задание 5", "Задание 6", "Задание 7", "Задание 8", "Задание 9", "Задание 10", "Начисленные баллы 1", "Начисленные баллы 2", "Начисленные баллы 3", "Начисленные баллы 4", "Начисленные баллы 5", "Начисленные баллы 6", "Начисленные баллы 7", "Начисленные баллы 8", "Начисленные баллы 9", "Начисленные баллы 10"]

    # Save to Excel with formatting
    wb = Workbook()
    ws = wb.active
    ws.title = "Общая таблица"

    # Write the subheader
    ws.append(columns)

    # Write the data
    for row in dataframe_to_rows(df_global_styled, index=False, header=False):
        ws.append(row)

    # Align headers
    for cell in ws[1]:
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Create a border for the entire dataframe
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.border = thin_border

    # Make the first three rows bold
    for row in ws.iter_rows(min_row=1, max_row=3):
        for cell in row:
            cell.font = Font(bold=True)

    # Save to a BytesIO object
    output = BytesIO()
    wb.save(output)
    output.seek(0)

    # Save the BytesIO object to a file for verification
    with open(f"{file_name}", "wb") as f:
        f.write(output.getbuffer())
    return output

def save_to_excel_local(df_global_styled, form_dict):
    with pd.ExcelWriter('output_with_images.xlsx', engine='xlsxwriter') as writer:
        # Сохраняем сам DataFrame на лист (например, Sheet1)
        df_global_styled.to_excel(writer, sheet_name='Sheet1', index=False)
        
        # Получаем объект worksheet, чтобы работать с картинками
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']


        # Теперь итерируемся по строкам DF
        for row_idx in range(len(df_global_styled)):
            form = form_dict[row_idx]
            excel_row = row_idx + 1

            # Вставка изображений
            excel_col = df_global_styled.columns.get_loc('Картинка Дата')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'date').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера
            
            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )

            excel_col = df_global_styled.columns.get_loc(f'Картинка код участника')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'user_id').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера
            
            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )

            excel_col = df_global_styled.columns.get_loc(f'Картинка вариант')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'version').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера

            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )
            
            for i in range(1, 11):
                excel_col = df_global_styled.columns.get_loc(f'Картинка ответа {i}')
                img_data = io.BytesIO()
                img = Image.fromarray(getattr(form, f'answer{i}').row_image)
                img.save(img_data, format='PNG')
                img_data.seek(0)  # переходим в начало буфера
                
                worksheet.insert_image(
                    excel_row, 
                    excel_col,
                    "some_name.png", 
                    {'image_data': img_data}
                )


def save_to_excel(df_global_styled, form_dict):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Сохраняем сам DataFrame на лист (например, Sheet1)
        df_global_styled.to_excel(writer, sheet_name='Sheet1', index=False)
        
        # Получаем объект worksheet, чтобы работать с картинками
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']


        # Теперь итерируемся по строкам DF
        for row_idx in range(len(df_global_styled)):
            form = form_dict[row_idx]
            excel_row = row_idx + 1

            # Вставка изображений
            excel_col = df_global_styled.columns.get_loc('Картинка Дата')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'date').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера
            
            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )

            excel_col = df_global_styled.columns.get_loc(f'Картинка код участника')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'user_id').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера
            
            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )

            excel_col = df_global_styled.columns.get_loc(f'Картинка вариант')
            img_data = io.BytesIO()
            img = Image.fromarray(getattr(form, 'version').row_image)
            img.save(img_data, format='PNG')
            img_data.seek(0)  # переходим в начало буфера

            worksheet.insert_image(
                excel_row, 
                excel_col,
                "some_name.png", 
                {'image_data': img_data}
            )
            
            for i in range(1, 11):
                excel_col = df_global_styled.columns.get_loc(f'Картинка ответа {i}')
                img_data = io.BytesIO()
                img = Image.fromarray(getattr(form, f'answer{i}').row_image)
                img.save(img_data, format='PNG')
                img_data.seek(0)  # переходим в начало буфера
                
                worksheet.insert_image(
                    excel_row, 
                    excel_col,
                    "some_name.png", 
                    {'image_data': img_data}
                )


        writer.close()

    output.seek(0)
    return output



def reorder_cols(df_global_styled):
    df_global_styled[f'Картинка код участника'] = ''
    df_global_styled[f'Картинка вариант'] = ''
    df_global_styled[f'Картинка Дата'] = ''

    for i in range(1, 11):
        df_global_styled[f'Картинка ответа {i}'] = ''


    reorder_cols_list = ['Дата', 'Картинка Дата', 'Код участника', 'Картинка код участника', 'Вариант', 'Картинка вариант']
    for i in range(1, 11):  
        reorder_cols_list.append(f'Задание {i}')
        reorder_cols_list.append(f'Картинка ответа {i}')



    for i in range(1, 11):
        reorder_cols_list.append(f'Начисленные баллы {i}')

    reorder_cols_list.append('Начисленные баллы сумма')

    df_global_styled = df_global_styled[reorder_cols_list]
    return df_global_styled